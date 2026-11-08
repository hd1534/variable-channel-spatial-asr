"""Training, validation and checkpointing for one run directory.

A run directory holds everything needed to resume or audit a run:

``config.json``       the resolved config and its fingerprint
``ckpt/``             ``step_XXXXXXX.pt`` (model + loss + optimiser + scheduler +
                      scaler + RNG state) every ``train.ckpt_every`` steps or
                      ``train.ckpt_every_sec`` seconds, the newest
                      ``train.ckpt_keep`` kept, listed with size and SHA-256 in
                      ``ckpt/LATEST.json``
``best.pt``           the weights with the best validation Spk Acc so far
``metrics.jsonl``     one line per log / validation event
``summary.json``      final validation of the best weights: tuned thresholds,
                      Spk Acc per M, per N and per mask, gate statistics
``val_cases.jsonl``   per-case results with the facts H1-H4 are analysed by
                      (M, N, voice similarity of the pair, spatial separability)

Colab can drop a runtime at any moment, and Google Drive uploads in the
background, so the newest file on Drive may be missing or cut short after a
crash.  Every write goes through :func:`atomic_save` (temp file +
``os.replace``), and :meth:`Run.fit` resumes from the newest checkpoint whose
size and SHA-256 still match its ``LATEST.json`` entry, falling back to the one
before.  A resumed run refuses a config whose fingerprint differs from the one
it was started with, and code from another commit unless asked to
(``--allow-code-change``); every commit a run has used is recorded.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

import spec
from models import SYSTEMS, build_system, needs_voice
from models.common import build_voice_encoder, causal_vad, reference_audio, reference_channel, srp_embeddings, voice_embeddings
from models.slots import (accuracy_near_changes, best_assignment_accuracy, causal_ceiling, centroid_accuracy,
                          online_slots, theta_grid, tune_theta)
from training.config import VAL_PROTOCOL_VERSION, RunConfig, config_fingerprint, config_to_dict
from training.corpus import build_corpora
from training.data import TrainStream, ValSessions, collate, to_device, with_mask
from training.render import render_batch
from training.loss import SessionCentroidLoss
from training.official import official_speaker_check, track_for
from training.rirbank import BankSpec, build_bank

REPO = Path(__file__).resolve().parent.parent


def _rng_state() -> dict[str, Any]:
    """Capture all random generators needed to resume training unchanged."""
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a checkpoint's CPU and available CUDA random generators."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (out.stdout.strip() + ("+dirty" if dirty else "")) or None
    except (OSError, subprocess.SubprocessError):
        return None


def atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


class Checkpoints:
    """``ckpt/step_XXXXXXX.pt`` files listed newest first in ``ckpt/LATEST.json``."""

    def __init__(self, run_dir: Path, keep: int) -> None:
        self.dir = run_dir / "ckpt"
        self.index = self.dir / "LATEST.json"
        self.keep = max(1, keep)

    def entries(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def save(self, state: dict[str, Any], step: int) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"step_{step:07d}.pt"
        atomic_save(state, path)
        entry = {"file": path.name, "step": step, "bytes": path.stat().st_size, "sha256": sha256_of(path)}
        entries = [entry] + [e for e in self.entries() if e["file"] != path.name]
        for old in entries[self.keep :]:
            (self.dir / old["file"]).unlink(missing_ok=True)
        atomic_write_text(self.index, json.dumps(entries[: self.keep], indent=1))

    def load_latest(self, map_location: str, log: Callable[[str], None]) -> dict[str, Any] | None:
        for entry in self.entries():
            path = self.dir / entry["file"]
            try:
                if path.stat().st_size != entry["bytes"] or sha256_of(path) != entry["sha256"]:
                    raise ValueError("size or SHA-256 differs from LATEST.json")
                return torch.load(path, map_location=map_location, weights_only=False)
            except Exception as exc:  # noqa: BLE001 - any unreadable checkpoint falls back to the previous one
                log(f"[run] skipping checkpoint {path.name}: {exc}")
        return None


def tdoa_separability(mic_xy: np.ndarray, mask: np.ndarray, azimuths_deg: list[float]) -> float | None:
    """Smallest, over talker pairs, of the largest far-field TDOA difference (in
    samples) over the active microphone pairs.  ``< 1`` means the two talkers
    are spatially indistinguishable at this mask (e.g. mirror images at M = 2).

    >>> xy = np.array([[0.1, 0.0], [-0.1, 0.0], [0, 0.1], [0, -0.1]])
    >>> tdoa_separability(xy, np.array([1, 1, 0, 0], bool), [60.0, -60.0])  # mirror about the pair axis
    0.0
    >>> tdoa_separability(xy, np.array([1, 0, 0, 0], bool), [60.0, -60.0]) is None
    True
    """
    active = np.flatnonzero(mask)
    if len(active) < 2 or len(azimuths_deg) < 2:
        return None
    u = np.stack([np.cos(np.radians(azimuths_deg)), np.sin(np.radians(azimuths_deg))], axis=1)
    delays = -(mic_xy[active] @ u.T) / spec.SPEED_OF_SOUND_M_S * spec.SAMPLE_RATE_HZ  # [mics, talkers]
    pair_tdoa = np.stack([delays[j] - delays[i] for i, j in itertools.combinations(range(len(active)), 2)])
    worst = min(
        float(np.abs(pair_tdoa[:, a] - pair_tdoa[:, b]).max())
        for a, b in itertools.combinations(range(len(azimuths_deg)), 2)
    )
    return round(worst, 6)


def _print(message: str) -> None:
    print(message, flush=True)


def needed_banks(cfg: RunConfig) -> dict[str, BankSpec]:
    """The room banks a run reads: validation always, training only for trainable systems."""
    engine = "fallback_ism" if cfg.data.synthetic and cfg.data.rir_engine == "pyroomacoustics" else cfg.data.rir_engine
    shape = {"variants": cfg.data.bank_variants, "jitter_m": tuple(cfg.data.jitter_m)}
    banks = {"val": BankSpec(cfg.data.val_seed + 1_000_003, cfg.data.n_rooms_val, cfg.data.rir_sec,
                             cfg.data.rt60_range_sec, engine, **shape)}
    if SYSTEMS[cfg.model.name].trainable:
        banks["train"] = BankSpec(cfg.train.seed, cfg.data.n_rooms_train, cfg.data.rir_sec, cfg.data.rt60_range_sec,
                                  engine, **shape)
    return banks


class Run:
    def __init__(self, cfg: RunConfig, run_dir: str | Path, data_root: str | Path, device: str | None = None,
                 workers: int | None = None, model_cache: str | None = None, bank_mirror: str | None = None,
                 allow_code_change: bool = False, overrides: dict[str, Any] | None = None,
                 log: Callable[[str], None] = _print) -> None:
        self.cfg = cfg
        self.overrides = dict(overrides or {})
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = pick_device(device)
        self.workers = cfg.data.num_workers if workers is None else workers
        self.log = log
        self.fingerprint = config_fingerprint(cfg)
        self.commit = git_commit()
        self.commits: list[str | None] = [self.commit]
        self.allow_code_change = allow_code_change
        self.ckpts = Checkpoints(self.run_dir, cfg.train.ckpt_keep)
        self._check_config()

        torch.manual_seed(cfg.train.seed)
        random.seed(cfg.train.seed)
        np.random.seed(cfg.train.seed % (2**32))

        trainable = SYSTEMS[cfg.model.name].trainable
        self.train_corpus, self.val_corpus = build_corpora(cfg.data, data_root, need_train=trainable)
        bank_root = Path(data_root) / "rirbanks"
        banks = {name: str(build_bank(spec_, bank_root, log=log, mirror=bank_mirror))
                 for name, spec_ in needed_banks(cfg).items()}  # the untrained rungs never see a training session
        self.train_bank, self.val_bank = banks.get("train"), banks["val"]
        self.log(f"[run] {cfg.name}: system={cfg.model.name} state={cfg.model.state} device={self.device} "
                 f"train speakers={len(self.train_corpus) if self.train_corpus else 0} val speakers={len(self.val_corpus)}")

        self.voice = build_voice_encoder(cfg.model, self.device, savedir=model_cache) if needs_voice(cfg) else None
        self.system = build_system(cfg, self.voice).to(self.device)
        self.val_cache: dict[tuple, torch.Tensor] = {}
        self.step = 0
        self.best = -math.inf
        if self.system.trainable:
            self.loss_fn = SessionCentroidLoss(contrast_weight=cfg.train.contrast_weight).to(self.device)
            self.pos_loss_fn = SessionCentroidLoss(contrast_weight=cfg.train.contrast_weight).to(self.device)
            params = (list(self.system.parameters()) + list(self.loss_fn.parameters())
                      + list(self.pos_loss_fn.parameters()))
            self.opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
            warm, total = cfg.train.warmup_steps, max(cfg.train.max_steps, 1)

            def lr_at(step: int) -> float:
                if step < warm:
                    return (step + 1) / warm
                progress = min(1.0, (step - warm) / max(1, total - warm))
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

            self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_at)
            self.amp_dtype = None
            if cfg.train.amp and self.device == "cuda":
                # bf16 natively from Ampere (capability 8) on; torch.cuda.is_bf16_supported() also says
                # yes to the slow emulated bf16 of a T4, which gets float16 with loss scaling instead.
                major, _ = torch.cuda.get_device_capability()
                self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
            n_params = sum(p.numel() for p in self.system.parameters() if p.requires_grad)
            self.log(f"[run] trainable parameters: {n_params:,}; mixed precision: {self.amp_dtype}")

    # -- bookkeeping -------------------------------------------------------------
    def _check_config(self) -> None:
        path = self.run_dir / "config.json"
        if path.is_file():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("fingerprint") != self.fingerprint:
                raise SystemExit(
                    f"{self.run_dir} was started with config fingerprint {old.get('fingerprint')}, this config is "
                    f"{self.fingerprint}. Resume with the same config or pick a new --run-dir."
                )
        atomic_write_text(path, json.dumps({"fingerprint": self.fingerprint, "commit": git_commit(),
                                            "config": config_to_dict(self.cfg)}, indent=1, ensure_ascii=False))

    def _event(self, **record: Any) -> None:
        record = {"step": self.step, "time": round(time.time(), 1), **record}
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _state(self) -> dict[str, Any]:
        """Capture model, optimizer, progress and random state for a resumable checkpoint."""
        return {
            "step": self.step,
            "best": self.best,
            "fingerprint": self.fingerprint,
            "commit": self.commit,
            "commits": self.commits,
            "amp_dtype": str(self.amp_dtype),
            "model": self.system.state_dict(),
            "loss": self.loss_fn.state_dict(),
            "pos_loss": self.pos_loss_fn.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": _rng_state(),
        }

    def _resume(self) -> None:
        """Restore training state and refresh compatible best weights under the current protocol."""
        state = self.ckpts.load_latest(self.device, self.log)
        if state is None:
            self._restore_best_selection()
            return
        if state["fingerprint"] != self.fingerprint:
            raise SystemExit(f"{self.ckpts.dir} belongs to another config; refusing to resume")
        if state.get("commit") != self.commit and not self.allow_code_change:
            raise SystemExit(
                f"the checkpoint was trained with commit {state.get('commit')} but the code is {self.commit}. "
                "Check out that commit to resume, or pass --allow-code-change to continue on the new code "
                "(both commits are then recorded)."
            )
        self.commits = list(dict.fromkeys([*state.get("commits", [state.get("commit")]), self.commit]))
        self.system.load_state_dict(state["model"])
        self.loss_fn.load_state_dict(state["loss"])
        self.pos_loss_fn.load_state_dict(state["pos_loss"])
        self.opt.load_state_dict(state["opt"])
        self.sched.load_state_dict(state["sched"])
        if state["scaler"]:  # a disabled scaler (bf16 or CPU) saves nothing; a T4 session then starts fresh
            self.scaler.load_state_dict(state["scaler"])
        if state.get("amp_dtype") != str(self.amp_dtype):
            self.log(f"[run] note: mixed precision changes from {state.get('amp_dtype')} to {self.amp_dtype} on resume")
        self.step = state["step"]
        _restore_rng_state(state["rng"])
        self._restore_best_selection()
        self.log(f"[run] resumed at step {self.step} (best selection score {self.best:.4f}, commits {self.commits})")

    def _restore_best_selection(self) -> None:
        """Keep current scores or revalidate older, model-compatible best weights."""
        best: dict[str, Any] | None = self._load_best("cpu", require_current_protocol=False)
        self.best = -math.inf
        if best is None:
            return
        if best.get("val_protocol_version") == VAL_PROTOCOL_VERSION and "select_score" in best.get("val", {}):
            self.best = float(best["val"]["select_score"])
        else:
            self._revalidate_best(best)

    def _revalidate_best(self, previous: dict[str, Any]) -> None:
        """Compare old best and latest weights with fresh scores, preserving the resumed training state.

        The original file is replaced only after both candidates have been
        scored, so an interrupted migration is retried on the next resume.
        """
        model_state: dict[str, Any] = copy.deepcopy(self.system.state_dict())
        loss_state: dict[str, Any] = copy.deepcopy(self.loss_fn.state_dict())
        step: int = self.step
        training: bool = self.system.training
        rng: dict[str, Any] = _rng_state()
        self.log(f"[run] revalidating best.pt from protocol {previous.get('val_protocol_version')!r}")
        try:
            self.system.load_state_dict(previous["model"])
            if "loss" in previous:
                self.loss_fn.load_state_dict(previous["loss"])
            self.step = int(previous["step"])
            old_result: dict[str, Any] = self.validate(quick=True)
            old_result["select_score"] = self._selection_score(old_result)
            candidate: dict[str, Any] = {
                **previous, "val_protocol_version": VAL_PROTOCOL_VERSION, "val": old_result,
                "loss": copy.deepcopy(self.loss_fn.state_dict()),
            }
            self.system.load_state_dict(model_state)
            self.loss_fn.load_state_dict(loss_state)
            self.step = step
            latest_result: dict[str, Any] = self.validate(quick=True)
            latest_result["select_score"] = self._selection_score(latest_result)
            if latest_result["select_score"] > old_result["select_score"]:
                candidate = self._best_checkpoint(latest_result)
            atomic_save(candidate, self.run_dir / "best.pt")
            self.best = float(candidate["val"]["select_score"])
            self._event(kind="val_protocol_migration", previous_protocol=previous.get("val_protocol_version"),
                        val_protocol_version=VAL_PROTOCOL_VERSION, old_best_score=old_result["select_score"],
                        latest_score=latest_result["select_score"], best_step=candidate["step"])
        finally:
            self.system.load_state_dict(model_state)
            self.loss_fn.load_state_dict(loss_state)
            self.step = step
            self.system.train(training)
            _restore_rng_state(rng)

    # -- training ----------------------------------------------------------------
    def fit(self) -> None:
        cfg = self.cfg
        if not self.system.trainable:
            self.log(f"[run] {cfg.model.name} has nothing to train; only its thresholds are tuned")
            return
        self._resume()
        if self.step >= cfg.train.max_steps:
            if self._load_best("cpu") is None:
                self._validate_and_keep_best()
            self.log(f"[run] already at step {self.step} >= max_steps {cfg.train.max_steps}")
            return
        stream = TrainStream(cfg.data, self.train_corpus, self.train_bank, cfg.train.seed, start_step=self.step)
        loader = DataLoader(stream, batch_size=cfg.train.batch_size, num_workers=self.workers, collate_fn=collate,
                            pin_memory=self.device == "cuda", persistent_workers=self.workers > 0,
                            # ~5 MB per session in flight (dry talkers, variant map, room responses)
                            prefetch_factor=2 if self.workers > 0 else None)
        self.system.train()
        window: dict[str, list[float]] = {}
        skipped = 0
        consecutive_skips = 0
        last_ckpt = time.time()
        t_data = time.time()
        for batch in loader:
            data_sec = time.time() - t_data
            t0 = time.time()
            batch = render_batch(to_device(batch, self.device))
            batch["voice_keep"] = batch["labels"] >= 0
            with torch.autocast(device_type=self.device, dtype=self.amp_dtype or torch.float32,
                                enabled=self.amp_dtype is not None):
                z = self.system.embed(batch)
            loss_spk, stats = self.loss_fn(z.float(), batch["labels"])
            loss_pos, pos_stats = self.pos_loss_fn(self.system.last_spatial.float(), batch["pos_labels"])
            loss = loss_spk + cfg.train.position_loss_weight * loss_pos
            if not torch.isfinite(loss):
                skipped += 1
                consecutive_skips += 1
                self.log(f"[train] non-finite loss at step {self.step}; batch skipped ({skipped} so far)")
                if consecutive_skips >= cfg.train.max_consecutive_skips:
                    raise RuntimeError(f"aborting after {consecutive_skips} consecutive non-finite batches at step {self.step}")
                t_data = time.time()
                continue
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.system.parameters(), cfg.train.grad_clip)
            if not torch.isfinite(grad_norm):
                skipped += 1
                consecutive_skips += 1
                self.opt.zero_grad(set_to_none=True)
                self.scaler.update()
                self.log(f"[train] non-finite grad_norm at step {self.step}; batch skipped ({skipped} so far)")
                if consecutive_skips >= cfg.train.max_consecutive_skips:
                    raise RuntimeError(f"aborting after {consecutive_skips} consecutive non-finite batches at step {self.step}")
                t_data = time.time()
                continue
            consecutive_skips = 0
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            self.step += 1

            for key, value in (("loss_spk", loss_spk.item()), ("acc_spk", stats["acc"]), ("loss_pos", loss_pos.item()),
                               ("acc_pos", pos_stats["acc"]), ("theta_learned", self.loss_fn.threshold()),
                               ("grad_norm", float(grad_norm)), ("data_sec", data_sec), ("step_sec", time.time() - t0)):
                window.setdefault(key, []).append(value)
            gate = getattr(self.system, "last_gate", None)
            if gate is not None:
                speech = batch["labels"] >= 0
                m_count = batch["mask"].sum(1)
                for M in range(1, spec.N_CHANNELS_MAX + 1):
                    sel = speech & (m_count == M)[:, None]
                    if sel.any():
                        window.setdefault(f"gate_M{M}", []).append(float(gate[sel].mean()))

            if self.step % cfg.train.log_every == 0 or self.step == 1:
                means = {k: round(float(np.nanmean(v)), 4) for k, v in window.items()}
                self._event(kind="train", lr=self.sched.get_last_lr()[0], skipped=skipped, **means)
                self.log(f"[train] step {self.step} " + " ".join(f"{k}={v}" for k, v in means.items()))
                window = {}
            done = self.step >= cfg.train.max_steps
            if self.step % cfg.train.val_every == 0 or done:
                self._validate_and_keep_best()
                self.system.train()
            # after validation, so a checkpoint never predates the best.pt of its own step
            if self.step % cfg.train.ckpt_every == 0 or time.time() - last_ckpt >= cfg.train.ckpt_every_sec or done:
                self.ckpts.save(self._state(), self.step)
                last_ckpt = time.time()
            if done:
                break
            t_data = time.time()

    def _load_best(self, device: str, *, require_current_protocol: bool = True) -> dict[str, Any] | None:
        """Read compatible weights, optionally requiring a current-protocol selection score."""
        path = self.run_dir / "best.pt"
        if not path.is_file():
            return None
        try:
            best = torch.load(path, map_location=device, weights_only=False)
        except Exception as exc:  # noqa: BLE001
            self.log(f"[run] ignoring unreadable best.pt: {exc}")
            return None
        if not isinstance(best, dict) or best.get("fingerprint") != self.fingerprint or not {"model", "step"} <= best.keys():
            return None
        if require_current_protocol and (
            best.get("val_protocol_version") != VAL_PROTOCOL_VERSION or "select_score" not in best.get("val", {})
        ):
            return None
        return best

    @staticmethod
    def _selection_score(result: dict[str, Any]) -> float:
        """Select on official two-slot M>=2 accuracy, with the general-N diagnostic as fallback."""
        by_m = result.get("spk_acc_n2_cap2_by_M") or result["spk_acc_by_M"]
        multi: list[float] = [value for microphones, value in by_m.items() if int(microphones) >= 2]
        return round(float(np.mean(multi)), 4) if multi else result.get("spk_acc_n2_cap2", result["spk_acc"])

    def _best_checkpoint(self, result: dict[str, Any]) -> dict[str, Any]:
        """Describe the current weights and their validation result for an immediate atomic save."""
        return {"step": self.step, "fingerprint": self.fingerprint, "val_protocol_version": VAL_PROTOCOL_VERSION,
                "model": self.system.state_dict(), "loss": self.loss_fn.state_dict(), "val": result}

    def _validate_and_keep_best(self) -> None:
        """Validate current weights and atomically retain an improved selection score."""
        result: dict[str, Any] = self.validate(quick=True)
        result["select_score"] = self._selection_score(result)
        by_m = result.get("spk_acc_n2_cap2_by_M") or result["spk_acc_by_M"]
        score = result["select_score"]
        improved = score > self.best
        if improved:
            self.best = score
            atomic_save(self._best_checkpoint(result), self.run_dir / "best.pt")
        self._event(kind="val", improved=improved, **{k: v for k, v in result.items() if k != "cases"})
        extra = f" position_acc={result['position_acc_by_M']}" if "position_acc_by_M" in result else ""
        self.log(f"[val] step {self.step} select(M>=2,cap2)={score:.4f} spk_acc_n2_cap2={result.get('spk_acc_n2_cap2', result['spk_acc']):.4f} "
                 f"by_M(cap2)={by_m} theta_official_by_M={result.get('theta_official_by_M', result['theta_by_M'])}{extra}"
                 + ("  *best*" if improved else ""))

    # -- validation --------------------------------------------------------------
    def _cached(self, kind: str, batch: dict[str, Any], key_extra: Any, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
        keys = [(kind, int(i), key_extra(b) if callable(key_extra) else key_extra, batch.get("val_kind", "main"))
                for b, i in enumerate(batch["index"])]
        if not all(k in self.val_cache for k in keys):
            values = compute()
            for b, k in enumerate(keys):
                self.val_cache[k] = values[b].detach().cpu()
        return torch.stack([self.val_cache[k] for k in keys]).to(self.device)

    def collect_cases(self, kind: str = "main") -> list[dict[str, Any]]:
        """Embed every validation session under every mask of ``data.val_masks``.

        ``kind`` is a :func:`training.scenes.val_recipe` condition: ``main``
        (what thresholds and checkpoints are chosen on), ``static`` or
        ``colocated`` (the same sessions under other conditions).
        """
        cfg = self.cfg
        loader = DataLoader(ValSessions(cfg.data, self.val_corpus, self.val_bank, kind=kind),
                            batch_size=max(1, min(8, cfg.train.batch_size)), num_workers=self.workers, collate_fn=collate)
        cases: list[dict[str, Any]] = []
        self.system.eval()
        with torch.no_grad():
            for full in loader:
                full = render_batch(to_device(full, self.device))
                full["val_kind"] = kind
                for mask_bits in cfg.data.val_masks:
                    batch = with_mask(full, mask_bits)
                    cache: dict[str, torch.Tensor] = {}
                    if self.voice is not None:
                        refs = reference_channel(batch["mask"]).tolist()
                        cache["voice"] = self._cached("voice", batch, lambda b: refs[b],
                                                      lambda: voice_embeddings(self.voice, batch))
                    if cfg.model.name in ("space_only", "naive_fusion"):
                        cache["space"] = self._cached("space", batch, mask_bits,
                                                      lambda: srp_embeddings(batch, cfg.model.srp_temperature))
                    z = self.system.embed(batch, cache=cache).float()
                    speech = causal_vad(reference_audio(batch), cfg.slots.vad_rel_db, cfg.slots.vad_floor_dbfs)
                    gate = getattr(self.system, "last_gate", None)
                    spatial = getattr(self.system, "last_spatial", None)
                    for b in range(z.shape[0]):
                        meta = full["meta"][b]
                        case = {
                            "index": int(full["index"][b]),
                            "kind": kind,
                            "mask": mask_bits,
                            "M": mask_bits.count("1"),
                            "N": int(full["n_speakers"][b]),
                            "emb": z[b].cpu().numpy(),
                            "speech": speech[b].cpu().numpy(),
                            "labels": full["labels"][b].cpu().numpy(),
                            "pos_labels": full["pos_labels"][b].cpu().numpy(),
                            "labels_eval": full["labels_eval"][b].cpu().numpy(),
                            "meta": meta,
                            "tdoa_sep": tdoa_separability(full["mic_xy"][b].cpu().numpy().astype(np.float64),
                                                          np.array([c == "1" for c in mask_bits]),
                                                          meta["source_azimuth_deg"]),
                        }
                        for key in ("voice", "space"):
                            if key in cache:
                                case[key] = cache[key][b].float().cpu().numpy()
                        if spatial is not None:
                            case["spatial"] = spatial[b].float().cpu().numpy()
                        if gate is not None:
                            lab = case["labels"] >= 0
                            case["gate"] = float(gate[b].cpu().numpy()[lab].mean()) if lab.any() else None
                        cases.append(case)
        return cases

    # Thresholds are tuned per number of active microphones: the similarity scale of
    # every fused system moves with M (at M = 1 the space term is a constant), so one
    # shared threshold would hand some M an artificial deficit.
    def _tune_by_M(self, cases: list[dict[str, Any]], cap: int, emb_key: str = "emb") -> dict[int, dict[str, Any]]:
        thetas = theta_grid(self.cfg.slots.theta_grid)
        return {M: tune_theta([c for c in cases if c["M"] == M], thetas, cap, emb_key=emb_key)
                for M in sorted({c["M"] for c in cases})}

    @staticmethod
    def _accuracy_by_M(cases: list[dict[str, Any]], theta_by_M: dict[int, Any], cap: int,
                       emb_key: str = "emb") -> dict[str, float]:
        buckets: dict[int, list[float]] = {}
        for c in cases:
            pred = online_slots(c[emb_key], c["speech"], [theta_by_M[c["M"]]["theta"]], cap)[0]
            acc = best_assignment_accuracy(c["labels_eval"], pred)
            if acc is not None:
                buckets.setdefault(c["M"], []).append(acc)
        return {str(M): round(float(np.mean(v)), 4) for M, v in sorted(buckets.items())}

    def _tune_fusion_weights(self, cases: list[dict[str, Any]], cap: int | None = None,
                             space_key: str = "space") -> dict[str, Any]:
        """Per M, the spatial weight (and with it the threshold) with the best Spk Acc under ``cap``."""
        cap = cap or self.cfg.slots.max_slots
        start, stop, step = self.cfg.slots.weight_grid
        weights = np.round(np.arange(start, stop + step / 2, step), 6)
        thetas = theta_grid(self.cfg.slots.theta_grid)
        chosen: dict[int, float] = {}
        chosen_theta: dict[int, float] = {}
        best_acc: dict[int, float] = {}
        curves: dict[str, list[float]] = {}
        for M in sorted({c["M"] for c in cases}):
            group = [c for c in cases if c["M"] == M]
            curve = []
            thetas_m = []
            for w in weights:
                for c in group:
                    v = c["voice"] / np.maximum(np.linalg.norm(c["voice"], axis=-1, keepdims=True), 1e-12)
                    c["emb_w"] = np.concatenate([math.sqrt(w) * c[space_key], math.sqrt(1 - w) * v], axis=-1)
                tuned = tune_theta(group, thetas, cap, emb_key="emb_w")
                curve.append(tuned["acc"])
                thetas_m.append(tuned["theta"])
            idx = int(np.argmax(curve))
            chosen[M] = float(weights[idx])
            chosen_theta[M] = float(thetas_m[idx])
            best_acc[M] = round(float(curve[idx]), 4)
            curves[str(M)] = [round(a, 4) for a in curve]
        for c in cases:
            c.pop("emb_w", None)
        return {
            "weights": weights.tolist(),
            "spk_acc_by_M": curves,
            "best_by_M": {str(m): w for m, w in chosen.items()},
            "best_acc_by_M": {str(m): a for m, a in best_acc.items()},
            "theta_by_M": {str(m): t for m, t in chosen_theta.items()},
        }

    def validate(self, quick: bool = False, cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        cfg = self.cfg
        result: dict[str, Any] = {}
        cap = cfg.slots.max_slots
        cases = cases if cases is not None else self.collect_cases()
        two = [c for c in cases if c["N"] == 2]
        cap2 = min(spec.MAX_SPEAKERS_PER_TRACK[track_for(2)], cap)
        if cfg.model.name == "naive_fusion" and cfg.model.fusion_weight is None and not quick:
            tune_cases = two if two else cases
            tune_cap = cap2 if two else cap
            result["weight_search"] = self._tune_fusion_weights(tune_cases, cap=tune_cap)
            if two:
                result["weight_search_cap4"] = self._tune_fusion_weights(cases, cap=cap)
            self.system.weights_by_M.update({int(m): w for m, w in result["weight_search"]["best_by_M"].items()})
            for c in cases:  # same embedding the system now produces, without re-running it
                w = self.system.weights_by_M[c["M"]]
                v = c["voice"] / np.maximum(np.linalg.norm(c["voice"], axis=-1, keepdims=True), 1e-12)
                c["emb"] = np.concatenate([math.sqrt(w) * c["space"], math.sqrt(1 - w) * v], axis=-1)
        if cfg.model.name == "naive_fusion":
            result["fusion_weight_by_M"] = {str(m): w for m, w in sorted(self.system.weights_by_M.items())}

        theta_by_M = self._tune_by_M(cases, cap)
        theta_official = self._tune_by_M(two, cap2) if two else {}
        for c in cases:
            c["theta"] = theta_by_M[c["M"]]["theta"]
            c["pred"] = online_slots(c["emb"], c["speech"], [c["theta"]], cap)[0]
            c["spk_acc"] = best_assignment_accuracy(c["labels_eval"], c["pred"])
            c["centroid_acc"] = centroid_accuracy(c["emb"], c["labels"])
            c["causal_ceiling"] = causal_ceiling(c["labels"], c["labels_eval"])
            c["acc_after_change"], c["acc_steady"] = accuracy_near_changes(c["labels_eval"], c["pred"])
            spoken = c["pred"][c["speech"]]
            c["n_slots"] = int(len(np.unique(spoken))) if spoken.size else 0
            if "spatial" in c:
                c["position_acc"] = centroid_accuracy(c["spatial"], c["pos_labels"])

        def mean_of(key: str, group: Callable[[dict], Any]) -> dict[str, float]:
            buckets: dict[str, list[float]] = {}
            for c in cases:
                if c.get(key) is not None:
                    buckets.setdefault(str(group(c)), []).append(c[key])
            return {g: round(float(np.mean(v)), 4) for g, v in sorted(buckets.items())}

        valid = [c["spk_acc"] for c in cases if c["spk_acc"] is not None]
        result.update({
            "theta_by_M": {str(m): t["theta"] for m, t in theta_by_M.items()},
            "spk_acc": round(float(np.mean(valid)), 4) if valid else float("nan"),
            "centroid_acc": round(float(np.mean([c["centroid_acc"] for c in cases if c["centroid_acc"] is not None])), 4),
            "spk_acc_by_M": mean_of("spk_acc", lambda c: c["M"]),
            "spk_acc_by_N": mean_of("spk_acc", lambda c: c["N"]),
            "centroid_acc_by_M": mean_of("centroid_acc", lambda c: c["M"]),
            "causal_ceiling_by_N": mean_of("causal_ceiling", lambda c: c["N"]),
            "n_cases": len(cases),
        })
        if any("position_acc" in c for c in cases):
            result["position_acc_by_M"] = mean_of("position_acc", lambda c: c["M"])
        if two:
            result["theta_official_by_M"] = {str(m): t["theta"] for m, t in theta_official.items()}
            result["spk_acc_n2_cap2_by_M"] = self._accuracy_by_M(two, theta_official, cap2)
            per_case = [best_assignment_accuracy(c["labels_eval"],
                                                 online_slots(c["emb"], c["speech"], [theta_official[c["M"]]["theta"]], cap2)[0])
                        for c in two]
            # a mean over cases, like evaluate.py's speaker_attribution_accuracy
            result["spk_acc_n2_cap2"] = round(float(np.mean([a for a in per_case if a is not None])), 4)
        if quick:
            return result
        result["spk_acc_by_mask"] = mean_of("spk_acc", lambda c: c["mask"])
        result["spk_acc_after_change_by_M"] = mean_of("acc_after_change", lambda c: c["M"])
        result["spk_acc_steady_by_M"] = mean_of("acc_steady", lambda c: c["M"])
        result["slot_count_error_by_N"] = {
            g: round(float(np.mean([c["n_slots"] - c["N"] for c in cases if str(c["N"]) == g])), 3)
            for g in sorted({str(c["N"]) for c in cases})
        }
        gates = [c for c in cases if c.get("gate") is not None]
        if gates:
            result["gate_by_M"] = {str(M): round(float(np.mean([c["gate"] for c in gates if c["M"] == M])), 4)
                                   for M in sorted({c["M"] for c in gates})}
        if two:
            # The same two-talker cases, per-M thresholds and slot cap, scored by evaluate.py itself.
            result["official_n2"] = official_speaker_check(
                cases, lambda c: online_slots(c["emb"], c["speech"], [theta_official[c["M"]]["theta"]], cap2)[0],
                keep_dir=self.run_dir / "official_n2",
            )
        result["theta_by_M_full"] = theta_by_M
        result["cases"] = cases
        return result

    def _voice_reference(self, cases: list[dict[str, Any]], theta_by_M: dict[int, Any] | None = None) -> tuple[dict, dict]:
        """Voice-only on the same cases (``v`` alone), with its own per-M thresholds."""
        for c in cases:
            c["emb_voice"] = c["voice"] / np.maximum(np.linalg.norm(c["voice"], axis=-1, keepdims=True), 1e-12)
        theta_by_M = theta_by_M or self._tune_by_M(cases, self.cfg.slots.max_slots, emb_key="emb_voice")
        return theta_by_M, self._accuracy_by_M(cases, theta_by_M, self.cfg.slots.max_slots, emb_key="emb_voice")

    def _proposed_diagnostics(self, cases: list[dict[str, Any]], theta_by_M: dict[int, Any]) -> dict[str, Any]:
        """What the ladder alone cannot separate, on the same validation sessions.

        * ``voice_only`` / ``paired_gain_over_voice_by_M``: the voice-only rung on the
          very same cases, so the gain is a paired difference;
        * ``learned_space_only``: the spatial embedding alone (gate held at 1) --
          against SRP-PHAT space-only this isolates the learned representation;
        * ``fixed_gate``: the learned embeddings mixed with one weight per M tuned
          here -- against the full model this isolates what the per-chunk gate adds;
        * ``static``: the same sessions with talkers at rest and no noise (the
          eval's static set; thresholds re-tuned there);
        * ``colocated``: the same sessions with every talker at one position, where
          space carries no information: the spatial part should separate talkers
          only at chance, and the model should not fall below voice-only.
        """
        cap = self.cfg.slots.max_slots
        two = [c for c in cases if c["N"] == 2]
        cap2 = min(spec.MAX_SPEAKERS_PER_TRACK[track_for(2)], cap)
        out: dict[str, Any] = {}
        voice_theta, voice_acc = self._voice_reference(cases)
        model_acc = self._accuracy_by_M(cases, theta_by_M, cap)
        out["voice_only_by_M"] = voice_acc
        out["paired_gain_over_voice_by_M"] = {m: round(model_acc[m] - voice_acc[m], 4) for m in model_acc}
        for c in cases:
            c["emb_space"] = c["spatial"]
        space_theta = self._tune_by_M(cases, cap, emb_key="emb_space")
        out["learned_space_only_by_M"] = self._accuracy_by_M(cases, space_theta, cap, emb_key="emb_space")
        # Like theta_official_by_M for the full model: what eval_t2.py needs to run
        # this variant on the eval set without the validation data at hand.
        if two:
            out["learned_space_only_theta_official_by_M"] = {
                str(m): t["theta"] for m, t in self._tune_by_M(two, cap2, emb_key="emb_space").items()}
        fixed_cap4 = self._tune_fusion_weights(cases, cap=cap, space_key="spatial")
        out["fixed_gate_by_M"] = fixed_cap4["best_acc_by_M"]
        out["fixed_gate_weight_by_M"] = dict(fixed_cap4["best_by_M"])
        if two:
            fixed_off = self._tune_fusion_weights(two, cap=cap2, space_key="spatial")
            out["fixed_gate_weight_cap4_by_M"] = fixed_cap4["best_by_M"]
            out["fixed_gate_weight_by_M"] = fixed_off["best_by_M"]
            out["fixed_gate_n2_cap2_by_M"] = fixed_off["best_acc_by_M"]
            out["fixed_gate_theta_official_by_M"] = fixed_off["theta_by_M"]

        static = self.collect_cases("static")
        static_theta = self._tune_by_M(static, cap)
        _, static_voice = self._voice_reference(static)
        static_model = self._accuracy_by_M(static, static_theta, cap)
        out["static"] = {"spk_acc_by_M": static_model, "voice_only_by_M": static_voice,
                         "paired_gain_over_voice_by_M": {m: round(static_model[m] - static_voice[m], 4) for m in static_model}}

        colocated = self.collect_cases("colocated")
        _, coloc_voice = self._voice_reference(colocated, voice_theta)
        coloc_model = self._accuracy_by_M(colocated, theta_by_M, cap)

        def spatial_talker_acc(rows: list[dict[str, Any]]) -> dict[str, float]:
            buckets: dict[str, list[float]] = {}
            for c in rows:
                acc = centroid_accuracy(c["spatial"], c["labels"])
                if acc is not None:
                    buckets.setdefault(str(c["M"]), []).append(acc)
            return {m: round(float(np.mean(v)), 4) for m, v in sorted(buckets.items())}

        out["colocated"] = {
            "spk_acc_by_M": coloc_model,
            "voice_only_by_M": coloc_voice,
            "paired_gain_over_voice_by_M": {m: round(coloc_model[m] - coloc_voice[m], 4) for m in coloc_model},
            "spatial_part_talker_acc_by_M": spatial_talker_acc(colocated),
            "spatial_part_talker_acc_own_positions_by_M": spatial_talker_acc(cases),
            "chance_approx": round(float(np.mean([1.0 / c["N"] for c in colocated])), 4),
        }
        return out

    def _clean_voice_centroids(self) -> dict[str, np.ndarray]:
        """Per validation talker, the mean frozen-ECAPA embedding of clean 3 s pieces of
        their speech (no room), for binning talker pairs by voice similarity (H3)."""
        out: dict[str, np.ndarray] = {}
        rng = random.Random(f"clean-voice:{self.cfg.data.val_seed}")
        for speaker in self.val_corpus.speakers:
            utts = rng.sample(list(self.val_corpus.by_speaker[speaker]), min(4, len(self.val_corpus.by_speaker[speaker])))
            pieces = []
            for utt in utts:
                x = self.val_corpus.read(utt, 0, 3 * spec.SAMPLE_RATE_HZ)
                pieces.append(np.pad(x, (0, max(0, 3 * spec.SAMPLE_RATE_HZ - len(x))))[: 3 * spec.SAMPLE_RATE_HZ])
            # the encoder is fed 1 s chunks, like everywhere else: three per piece
            chunks = torch.as_tensor(np.stack(pieces), dtype=torch.float32, device=self.device).reshape(-1, spec.SAMPLE_RATE_HZ)
            emb = self.voice(chunks, torch.full((chunks.shape[0],), spec.SAMPLE_RATE_HZ, device=self.device)).cpu().numpy()
            mean = emb.mean(0)
            out[speaker] = mean / max(float(np.linalg.norm(mean)), 1e-12)
        return out

    # -- final report ------------------------------------------------------------
    def finalize(self, started: float) -> dict[str, Any]:
        best_step = None
        state = self._load_best(self.device) if self.system.trainable else None
        if state is not None:
            self.system.load_state_dict(state["model"])
            if "loss" in state:
                self.loss_fn.load_state_dict(state["loss"])
            best_step = state["step"]
            self.log(f"[run] final validation with best.pt (step {best_step})")
        result = self.validate(quick=False)
        cases = result.pop("cases")
        theta_full = result.pop("theta_by_M_full")
        if self.cfg.model.name == "proposed":
            result["diagnostics"] = self._proposed_diagnostics(cases, theta_full)
        clean = self._clean_voice_centroids() if self.voice is not None and not self.cfg.data.synthetic else {}
        with open(self.run_dir / "val_cases.jsonl.tmp", "w", encoding="utf-8") as fh:
            for c in cases:
                meta = c["meta"]
                ids = meta["speaker_ids"]
                pair_cos = None
                if len(ids) >= 2 and all(i in clean for i in ids):
                    pair_cos = round(max(float(clean[a] @ clean[b]) for a, b in itertools.combinations(ids, 2)), 4)
                az = meta["source_azimuth_deg"]
                az_sep = None
                if len(az) >= 2:
                    az_sep = round(min(abs((a - b + 180.0) % 360.0 - 180.0) for a, b in itertools.combinations(az, 2)), 2)
                fh.write(json.dumps({
                    "index": c["index"], "mask": c["mask"], "M": c["M"], "N": c["N"], "theta": c["theta"],
                    "spk_acc": c["spk_acc"], "acc_after_change": c["acc_after_change"], "acc_steady": c["acc_steady"],
                    "centroid_acc": c["centroid_acc"], "causal_ceiling": c["causal_ceiling"], "n_slots": c["n_slots"],
                    "position_acc": c.get("position_acc"), "gate": c.get("gate"),
                    "tdoa_sep_samples": c["tdoa_sep"], "azimuth_separation_deg": az_sep,
                    "clean_voice_cos_between_talkers": pair_cos,
                    "sexes": [self.val_corpus.sex.get(i) for i in ids],
                    **meta,
                }, ensure_ascii=False) + "\n")
        os.replace(self.run_dir / "val_cases.jsonl.tmp", self.run_dir / "val_cases.jsonl")
        summary = {
            "run": self.run_dir.name,
            "config": self.cfg.name,
            "system": self.cfg.model.name,
            "state": self.cfg.model.state if self.cfg.model.name == "proposed" else None,
            "voice_encoder": self.cfg.model.voice_encoder if self.voice is not None else None,
            "trained_steps": self.step,
            "best_step": best_step,
            "train_speakers": len(self.train_corpus) if self.train_corpus else None,
            "val_speakers": len(self.val_corpus),
            "val_sessions": self.cfg.data.n_val_sessions,
            "val_rooms": self.cfg.data.n_rooms_val,
            "overrides": self.overrides,
            "fingerprint": self.fingerprint,
            "val_protocol_version": VAL_PROTOCOL_VERSION,
            "commit": self.commit,
            "commits": self.commits,
            "amp_dtype": str(getattr(self, "amp_dtype", None)) if self.system.trainable else None,
            "theta_learned": round(self.loss_fn.threshold(), 4) if self.system.trainable else None,
            "bank_engine_version": json.loads((Path(self.val_bank) / "bank.json").read_text(encoding="utf-8")).get("engine_version"),
            "device": self.device,
            "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
            "gpu_max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.device == "cuda" else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "wall_sec": round(time.time() - started, 1),
            "synthetic_data": self.cfg.data.synthetic,
            **result,
        }
        atomic_write_text(self.run_dir / "summary.json", json.dumps(summary, indent=1, ensure_ascii=False))
        self._event(kind="final", **{k: v for k, v in summary.items() if k not in ("weight_search", "diagnostics")})
        return summary


def run_is_complete(run_dir: str | Path, cfg: RunConfig) -> bool:
    """Whether ``run_dir`` already holds a finished, matching ``summary.json`` (and ``best.pt``) for ``cfg``."""
    run = Path(run_dir)
    summary_path = run / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    fp = config_fingerprint(cfg)
    if (
        summary.get("fingerprint") != fp
        or summary.get("system") != cfg.model.name
        or summary.get("val_protocol_version") != VAL_PROTOCOL_VERSION
    ):
        return False
    if cfg.model.name != "proposed":
        return True
    if (summary.get("trained_steps") or 0) < cfg.train.max_steps:
        return False
    best_path = run / "best.pt"
    if not best_path.is_file() or summary.get("best_step") is None:
        return False
    try:
        state = torch.load(best_path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return False
    return (
        state.get("fingerprint") == fp
        and state.get("val_protocol_version") == VAL_PROTOCOL_VERSION
        and state.get("step") == summary.get("best_step")
        and "select_score" in state.get("val", {})
    )

