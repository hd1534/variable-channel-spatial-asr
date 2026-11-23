#!/usr/bin/env python3
"""Train, validate and evaluate Track C (Qwen venv).

    python3 -m trackc.train --config configs/track_c.json --data-root DATA --run-dir RUNS/track_c \\
        --alignments ALIGN/train-clean-100.jsonl ALIGN/dev-clean.jsonl --bank-mirror STORE/rirbanks
    python3 -m trackc.train ... --eval            # the best checkpoint on eval-t2 and eval-t2-static

Training sessions are word-aligned two-talker sessions (:mod:`trackc.scenes`)
from the same rooms, noise and channel subsets as the other systems; the
validation sessions are the other systems' validation talkers (dev-clean minus
the eval split).  A run directory follows the training engine's rules:
checkpoints every ``ckpt_every`` steps or ``ckpt_every_sec`` seconds (only the
trainable tensors, LoRA + adapter + head), a resumed run refuses another config
and, unless ``--allow-code-change``, another commit; ``best.pt`` holds the best
validation step (lowest cpWER), ``summary.json`` the final validation.

Scores come from the official scorer (:mod:`evaluate`), as ``full_audio``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, "") and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import evaluate  # noqa: E402
import spec  # noqa: E402
from trackc import text as tt  # noqa: E402
from trackc.model import TrackCConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODE = "full_audio"
TRACK = 2


def fingerprint(cfg: TrackCConfig) -> str:
    """Everything that changes what is learned (cadence and step budget left out, as in the engine)."""
    d = cfg.to_dict()
    for key in ("max_steps", "val_every", "ckpt_every", "ckpt_every_sec", "log_every", "final_val_sessions",
                "max_new_tokens"):
        d.pop(key)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


# -- scoring ---------------------------------------------------------------------------
def score(records: list[dict[str, Any]], truth: list[dict[str, Any]], group: Callable[[str], str] | None = None) -> dict:
    """Official ``full_audio`` metrics overall and, with ``group(case_id)``, per group."""
    def one(recs: list[dict], rows: list[dict]) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            p, g = Path(tmp) / "p.jsonl", Path(tmp) / "g.jsonl"
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
            g.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            return evaluate.evaluate_files(p, g, TRACK)["scopes"][MODE]["metrics"]

    out: dict[str, Any] = {"all": one(records, truth)}
    if group is not None:
        names = sorted({group(r["case_id"]) for r in truth}, reverse=True)
        out["by_M"] = {n: one([r for r in records if group(r["case_id"]) == n],
                              [r for r in truth if group(r["case_id"]) == n]) for n in names}
    return out


def case_outputs(case_id: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Prediction records of one case from the model's text and frame classes."""
    return tt.records(case_id, tt.grid_slots(result["classes"]), tt.parse_tagged(result["text"]), MODE)


def run_is_complete(run_dir: str | Path, cfg: TrackCConfig) -> bool:
    """A finished ``summary.json`` for this config at its full step budget (no torch needed)."""
    path = Path(run_dir) / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return summary.get("fingerprint") == fingerprint(cfg) and (summary.get("trained_steps") or 0) >= cfg.max_steps \
        and (Path(run_dir) / "best.pt").is_file()


def eval_is_complete(run_dir: str | Path, cfg: TrackCConfig, sets: list[str], limit: int = 0) -> bool:
    """Scores of every set and variant for the current ``best.pt``."""
    try:
        summary = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    variants = ["full"] + (["no_space"] if cfg.use_spatial else [])
    for set_name in sets:
        for variant in variants:
            try:
                rep = json.loads((Path(run_dir) / "eval" / set_name / variant / "score.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            if (rep.get("fingerprint"), rep.get("best_step"), rep.get("limit")) != \
                    (fingerprint(cfg), summary.get("best_step"), limit):
                return False
    return True


# -- the run ---------------------------------------------------------------------------
class TrackCRun:
    def __init__(self, cfg: TrackCConfig, args: argparse.Namespace) -> None:
        import torch

        from training.config import load_config_dict
        from training.corpus import build_corpora
        from training.engine import Checkpoints, git_commit, needed_banks, pick_device
        from training.rirbank import build_bank
        from trackc.scenes import aligned_corpus, load_alignments

        self.cfg, self.args = cfg, args
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = pick_device(args.device)
        self.fp = fingerprint(cfg)
        self.commit = git_commit()
        self.ckpts = Checkpoints(self.run_dir, 2)
        self._check_config()
        self.data_cfg = load_config_dict({"name": cfg.name, "data": cfg.data}).data
        self.alignments = load_alignments(args.alignments)
        train, val = build_corpora(self.data_cfg, args.data_root)
        self.train_corpus, self.val_corpus = aligned_corpus(train, self.alignments), aligned_corpus(val, self.alignments)
        banks = needed_banks(load_config_dict({"name": cfg.name, "data": cfg.data}))
        root = Path(args.data_root) / "rirbanks"
        self.train_bank = str(build_bank(banks["train"], root, mirror=args.bank_mirror))
        self.val_bank = str(build_bank(banks["val"], root, mirror=args.bank_mirror))
        self.log(f"[trackc] {cfg.name}: spatial={cfg.use_spatial} device={self.device} train speakers="
                 f"{len(self.train_corpus)} val speakers={len(self.val_corpus)} aligned utterances={len(self.alignments)}")
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        self.model = self._load_model()
        params = [p for p in self.model.parameters() if p.requires_grad]
        from training.engine import lr_factor

        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_factor(s, cfg.warmup_steps, max(cfg.max_steps, 1)))
        self.step, self.best = 0, math.inf

    # -- plumbing ----------------------------------------------------------------------
    def log(self, message: str) -> None:
        print(message, flush=True)

    def _event(self, **record: Any) -> None:
        record = {"step": self.step, "time": round(time.time(), 1), **record}
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _check_config(self) -> None:
        from training.engine import atomic_write_text

        path = self.run_dir / "config.json"
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("fingerprint") != self.fp:
            raise SystemExit(f"{self.run_dir} was started with another Track C config; resume with the same config "
                             "or pick a new --run-dir")
        atomic_write_text(path, json.dumps({"fingerprint": self.fp, "commit": self.commit, "system": "track_c",
                                            "config": self.cfg.to_dict()}, indent=1))

    def _load_model(self):  # noqa: ANN202
        from trackc.model import load, log_parameter_counts

        model = load(self.cfg, self.device)
        self.log(f"[trackc] parameters: {log_parameter_counts(model)}")
        return model

    def _state(self) -> dict[str, Any]:
        from training.engine import _rng_state

        return {"step": self.step, "best": self.best, "fingerprint": self.fp, "commit": self.commit,
                "model": self.model.trainable_state(), "opt": self.opt.state_dict(),
                "sched": self.sched.state_dict(), "rng": _rng_state()}

    def _resume(self) -> None:
        from training.engine import _restore_rng_state

        state = self.ckpts.load_latest(self.device, self.log)
        if state is None:
            return
        if state["fingerprint"] != self.fp:
            raise SystemExit(f"{self.ckpts.dir} belongs to another config; refusing to resume")
        if state.get("commit") != self.commit and not self.args.allow_code_change and state["step"] < self.cfg.max_steps:
            raise SystemExit(f"the checkpoint was trained with commit {state.get('commit')} but the code is "
                             f"{self.commit}; pass --allow-code-change to continue on the new code")
        self.model.load_trainable(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.sched.load_state_dict(state["sched"])
        _restore_rng_state(state["rng"])
        self.step, self.best = int(state["step"]), float(state["best"])
        self.log(f"[trackc] resumed at step {self.step} (best cpWER {self.best:.4f})")

    # -- training ----------------------------------------------------------------------
    def fit(self) -> None:
        import torch
        from torch.utils.data import DataLoader

        from training.data import to_device
        from training.render import render_batch
        from trackc.scenes import TextTrainStream, collate

        cfg = self.cfg
        self._resume()
        if self.step >= cfg.max_steps:
            self.log(f"[trackc] already at step {self.step} >= max_steps {cfg.max_steps}")
            return
        stream = TextTrainStream(self.data_cfg, self.train_corpus, self.train_bank, self.alignments, cfg.seed,
                                 start_step=self.step)
        loader = DataLoader(stream, batch_size=cfg.batch_size, num_workers=self.data_cfg.num_workers,
                            collate_fn=collate, pin_memory=self.device == "cuda",
                            persistent_workers=self.data_cfg.num_workers > 0)
        self.model.train()
        window: dict[str, list[float]] = {}
        last_ckpt, micro = time.time(), 0
        t0 = time.time()
        for batch in loader:
            batch = render_batch(to_device(batch, self.device))
            loss, stats = self.model.loss(batch)
            (loss / cfg.grad_accum).backward()
            for k, v in stats.items():
                window.setdefault(k, []).append(v)
            micro += 1
            if micro % cfg.grad_accum:
                continue
            norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], cfg.grad_clip)
            if not torch.isfinite(norm):
                self.opt.zero_grad(set_to_none=True)
                self.log(f"[trackc] non-finite gradient at step {self.step}; skipped")
                continue
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()
            self.step += 1
            window.setdefault("grad_norm", []).append(float(norm))
            if self.step % cfg.log_every == 0 or self.step == 1:
                means = {k: round(float(np.mean(v)), 4) for k, v in window.items()}
                means["step_sec"] = round((time.time() - t0) / (cfg.log_every if self.step > 1 else 1), 3)
                self._event(kind="train", lr=self.sched.get_last_lr()[0], **means)
                self.log(f"[train] step {self.step} " + " ".join(f"{k}={v}" for k, v in means.items()))
                window, t0 = {}, time.time()
            done = self.step >= cfg.max_steps
            if self.step % cfg.val_every == 0 or done:
                self._validate_and_keep_best()
                self.model.train()
            if self.step % cfg.ckpt_every == 0 or time.time() - last_ckpt >= cfg.ckpt_every_sec or done:
                self.ckpts.save(self._state(), self.step)
                last_ckpt = time.time()
            if done:
                break

    # -- validation --------------------------------------------------------------------
    def validate(self, n_sessions: int, masks: tuple[str, ...]) -> dict[str, Any]:
        import torch
        from torch.utils.data import DataLoader

        from training.data import to_device, with_mask
        from training.official import truth_row
        from training.render import render_batch
        from trackc.scenes import TextValSessions, collate

        self.model.eval()
        loader = DataLoader(TextValSessions(self.data_cfg, self.val_corpus, self.val_bank, self.alignments, n_sessions),
                            batch_size=1, num_workers=min(2, self.data_cfg.num_workers), collate_fn=collate)
        records, truth, truncated = [], [], 0
        with torch.no_grad():
            for full in loader:
                full = render_batch(to_device(full, self.device))
                index, meta = int(full["index"][0]), full["meta"][0]
                for bits in masks:
                    case_id = f"trackc_val_{index:04d}__m{bits}"
                    result = self.model.transcribe(with_mask(full, bits))[0]
                    truncated += result["truncated"]
                    records += case_outputs(case_id, result)
                    labels = full["labels_eval"][0].cpu().tolist()
                    truth.append(truth_row(case_id, labels, {int(k): v for k, v in meta["talker_text"].items()}))
        report = score(records, truth, group=lambda cid: str(cid.rsplit("__m", 1)[1].count("1")))
        report["n_cases"], report["truncated_outputs"] = len(truth), truncated
        return report

    def _validate_and_keep_best(self) -> None:
        from training.engine import atomic_save

        report = self.validate(self.cfg.val_sessions, self.cfg.val_masks)
        cpwer = float(report["all"]["cpwer"])
        improved = cpwer < self.best
        if improved:
            self.best = cpwer
            atomic_save({"step": self.step, "fingerprint": self.fp, "model": self.model.trainable_state(),
                         "val": report}, self.run_dir / "best.pt")
        self._event(kind="val", improved=improved, **report)
        by_m = {m: (round(v["speaker_attribution_accuracy"], 4), round(v["cpwer"], 4)) for m, v in report["by_M"].items()}
        self.log(f"[val] step {self.step} cpWER={cpwer:.4f} spk_acc={report['all']['speaker_attribution_accuracy']:.4f} "
                 f"by_M(acc, cpWER)={by_m} truncated={report['truncated_outputs']}" + ("  *best*" if improved else ""))

    def _load_best(self) -> int:
        import torch

        best = torch.load(self.run_dir / "best.pt", map_location=self.device, weights_only=False)
        if best.get("fingerprint") != self.fp:
            raise SystemExit("best.pt belongs to another config")
        self.model.load_trainable(best["model"])
        return int(best["step"])

    def finalize(self, started: float) -> dict[str, Any]:
        import torch

        from training.engine import atomic_write_text

        best_step = self._load_best()
        self.log(f"[trackc] final validation with best.pt (step {best_step})")
        report = self.validate(self.cfg.final_val_sessions, tuple(spec.EVAL_MASKS))
        summary = {"run": self.run_dir.name, "config": self.cfg.name, "system": "track_c", "observation_mode": MODE,
                   "use_spatial": self.cfg.use_spatial, "trained_steps": self.step, "best_step": best_step,
                   "fingerprint": self.fp, "commit": self.commit, "device": self.device,
                   "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
                   "gpu_max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device == "cuda" else None,
                   "wall_sec": round(time.time() - started, 1), "val": report}
        atomic_write_text(self.run_dir / "summary.json", json.dumps(summary, indent=1, ensure_ascii=False))
        return summary

    # -- the eval sets -----------------------------------------------------------------
    def evaluate_sets(self, sets: list[str], limit: int = 0) -> dict[str, Any]:
        """``full`` and, for the spatial model, ``no_space`` (adapter off) on each eval set, by M."""
        import torch

        import eval_t2
        from training.engine import atomic_write_text

        best_step = self._load_best()
        self.model.eval()
        variants = ["full"] + (["no_space"] if self.cfg.use_spatial else [])
        results: dict[str, Any] = {"best_step": best_step}
        for set_name in sets:
            set_dir = REPO / "data" / set_name
            gt, meta = eval_t2.read_set(set_dir)
            sessions: dict[str, list[dict]] = {}
            for row in gt:
                sessions.setdefault(row["session_id"], []).append(row)
            order = list(sessions)[: limit or None]
            keep = {r["case_id"] for sid in order for r in sessions[sid]}
            truth = [r for r in gt if r["case_id"] in keep]
            for variant in variants:
                out_dir = self.run_dir / "eval" / set_name / variant
                out_dir.mkdir(parents=True, exist_ok=True)
                records, truncated = [], 0
                for i, sid in enumerate(order):
                    rows = sessions[sid]
                    x = torch.from_numpy(eval_t2.read_session(set_dir, sid))
                    pos = np.zeros((spec.N_CHANNELS_MAX, 2), dtype=np.float32)
                    for row in rows:
                        m = meta[row["case_id"]]
                        pos[m["active_channels"]] = np.asarray(m["mic_positions_xy_m"], dtype=np.float32)
                    masks = torch.tensor([[b == "1" for b in row["mask_bits"]] for row in rows])
                    batch = {"audio": (x[None] * masks[:, :, None].to(x.dtype)).to(self.device),
                             "mask": masks.to(self.device),
                             "mic_xy": torch.from_numpy(pos)[None].expand(len(rows), -1, -1).contiguous().to(self.device),
                             "no_space": variant == "no_space"}
                    with torch.no_grad():
                        outs = self.model.transcribe(batch)
                    for row, result in zip(rows, outs):
                        truncated += result["truncated"]
                        records += case_outputs(row["case_id"], result)
                    if (i + 1) % 20 == 0 or i + 1 == len(order):
                        self.log(f"  [{set_name}/{variant}] {i + 1}/{len(order)} sessions")
                n_mic = {r["case_id"]: str(len(meta[r["case_id"]]["active_channels"])) for r in truth}
                report = score(records, truth, group=lambda cid: n_mic[cid])
                report.update({"observation_mode": MODE, "variant": variant, "set": set_name, "best_step": best_step,
                               "n_cases": len(truth), "truncated_outputs": truncated, "limit": limit,
                               "fingerprint": self.fp, "commit": self.commit})
                (out_dir / "predictions.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
                atomic_write_text(out_dir / "score.json", json.dumps(report, indent=1, ensure_ascii=False))
                cells = " | ".join(f"M={m} acc {v['speaker_attribution_accuracy']:.3f} cpWER {v['cpwer']:.3f}"
                                   for m, v in report["by_M"].items())
                self.log(f"[{set_name}] {variant:9s} {cells}")
                results[f"{set_name}/{variant}"] = report
        return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--alignments", nargs="+", required=True)
    parser.add_argument("--bank-mirror")
    parser.add_argument("--device")
    parser.add_argument("--allow-code-change", action="store_true")
    parser.add_argument("--eval", action="store_true", help="score best.pt on the eval sets instead of training")
    parser.add_argument("--sets", nargs="+", default=["eval-t2", "eval-t2-static"])
    parser.add_argument("--limit", type=int, default=0, help="--eval: first N sessions per set (smoke runs)")
    parser.add_argument("--override", nargs="*", default=[], help="key=value (JSON values), e.g. max_steps=20")
    args = parser.parse_args(argv)
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    for item in args.override:
        key, _, value = item.partition("=")
        raw[key] = json.loads(value)
    cfg = TrackCConfig.from_dict(raw)
    run = TrackCRun(cfg, args)
    started = time.time()
    if args.eval:
        run.evaluate_sets(args.sets, args.limit)
        return 0
    run.fit()
    summary = run.finalize(started)
    print("===== VC-SASR TRACK C SUMMARY (paste this back) =====")
    print(json.dumps({k: v for k, v in summary.items()}, ensure_ascii=False))
    print("===== END =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
