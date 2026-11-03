#!/usr/bin/env python3
"""Run one system of the comparison ladder on the two-talker eval set and score it.

Usage::

    # an untrained rung, with the thresholds its validation run chose
    python3 eval_t2.py --config configs/voice_only.json --summary RUN/summary.json --out runs/eval/voice_only
    # a trained model: the full system and its two diagnostic variants
    python3 eval_t2.py --run-dir RUN --out runs/eval/proposed_carry
    # thresholds chosen on the eval set itself: optimistic, only to reproduce
    # the README's first measurement
    python3 eval_t2.py --config configs/space_only.json --tune-on-eval --out runs/eval/space_only-evaltuned

Both sets (``data/eval-t2``, ``data/eval-t2-static``) are run unless ``--sets``
says otherwise.  The static set uses the same thresholds as the main set: they
are chosen once, on the main condition of validation.

Per set, under ``--out/<set>/``:

``embeddings.npz``          per case: voice, space, fused and spatial embeddings and
                            the causal speech flags (reused when the inputs match)
``<variant>/predictions.jsonl``  80 timepoints and one ``final_transcripts`` per case
``<variant>/score.json``    the ``evaluate.py --track 2`` report

Transcripts come from :mod:`transcripts` (one ASR call per run of a predicted
slot, through a cache shared by every system).  ``--asr wav2vec2`` transcribes
missing segments here; ``--asr <backend-id>`` only reads that backend's cache
and, when segments are missing, writes them to ``missing_segments.jsonl`` for a
GPU notebook to fill, then stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, "") and str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import evaluate  # noqa: E402
import spec  # noqa: E402
import transcripts as tr  # noqa: E402
from models.slots import best_assignment_accuracy, online_slots, theta_grid, tune_theta  # noqa: E402
from training.config import load_config, load_config_dict  # noqa: E402
from training.official import timepoint_records, track_for  # noqa: E402

REPO = Path(__file__).resolve().parent
SETS = ("eval-t2", "eval-t2-static")
TRACK = track_for(2)
CAP = spec.MAX_SPEAKERS_PER_TRACK[TRACK]
VARIANTS = {
    "voice_only": ("full",),
    "space_only": ("full",),
    "naive_fusion": ("full",),
    "proposed": ("full", "learned_space", "fixed_gate"),
}
LOCAL_ASR = {"wav2vec2": "facebook/wav2vec2-base-960h"}


# -- inputs --------------------------------------------------------------------
def load_system_inputs(args: argparse.Namespace) -> tuple[Any, dict[str, Any] | None, Path | None]:
    if args.run_dir:
        run = Path(args.run_dir)
        cfg = load_config_dict(json.loads((run / "config.json").read_text(encoding="utf-8"))["config"])
        summary_path = run / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
        best = run / "best.pt" if cfg.model.name == "proposed" else None
        if best is not None and summary is not None and not args.tune_on_eval:
            # summary.json is written once, by finalize; best.pt can change after it
            # (a resumed run that dropped before finalizing).  Thresholds tuned for
            # other weights would score the model silently low.
            import torch

            state = torch.load(best, map_location="cpu", weights_only=False)
            if summary.get("best_step") is None or (state.get("step"), state.get("fingerprint")) != (
                    summary["best_step"], summary.get("fingerprint")):
                raise SystemExit(f"{summary_path} was not written for this best.pt (step {state.get('step')} vs "
                                 f"{summary.get('best_step')}): rerun train.py on {run} to refresh it")
        return cfg, summary, best
    cfg = load_config(args.config)
    if cfg.model.name == "proposed":
        raise SystemExit("a trained model needs --run-dir (config.json, best.pt, summary.json)")
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8")) if args.summary else None
    if summary is not None:
        if summary.get("system") != cfg.model.name:
            raise SystemExit(f"{args.summary} belongs to {summary.get('system')!r}, not {cfg.model.name!r}")
        changed = [k for k in summary.get("overrides") or {} if k.split(".")[0] in ("model", "slots")]
        if changed:
            raise SystemExit(f"that validation run overrode {changed}; pass its run directory as --run-dir "
                             "so the same settings are used here")
    return cfg, summary, None


def read_set(set_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    gt = [json.loads(line) for line in (set_dir / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines() if line]
    meta = {}
    for line in (set_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            meta[row["case_id"]] = row
    return gt, meta


def frame_labels(row: dict[str, Any]) -> np.ndarray:
    """Talker index per grid instant from a ground-truth row, ``-1`` where unlabelled."""
    labels = np.full(spec.N_GRID_POINTS, -1, dtype=np.int64)
    for f in row[spec.F_FRAMES]:
        labels[int(round(f["eval_time_sec"] / spec.DELTA_T_SEC))] = int(f["speaker"].split("_")[-1])
    return labels


def read_session(set_dir: Path, session_id: str) -> np.ndarray:
    """``[4, S]`` float32, the int16 samples over 32768 like the training renderer."""
    import soundfile as sf

    x, sr = sf.read(set_dir / "audio" / f"{session_id}.wav", dtype="float32", always_2d=True)
    if sr != spec.SAMPLE_RATE_HZ:
        raise ValueError(f"{session_id}: {sr} Hz")
    return np.ascontiguousarray(x.T)


# -- embeddings ----------------------------------------------------------------
def embed_set(cfg: Any, set_dir: Path, gt: list[dict[str, Any]], meta: dict[str, dict[str, Any]], device: str,
              model_cache: str | None, best: Path | None, limit: int) -> dict[str, np.ndarray]:
    """Embeddings of every case, grouped by session so each WAV is read and each
    reference channel's ECAPA computed once."""
    import torch

    from models import build_system, needs_voice
    from models.common import (build_voice_encoder, causal_vad, reference_audio, reference_channel, srp_embeddings,
                               voice_embeddings)

    name = cfg.model.name
    voice = build_voice_encoder(cfg.model, device, savedir=model_cache) if needs_voice(cfg) else None
    system = None
    if name == "proposed":
        system = build_system(cfg, voice).to(device)
        state = torch.load(best, map_location=device, weights_only=False)
        system.load_state_dict(state["model"])
        system.eval()

    sessions: dict[str, list[dict[str, Any]]] = {}
    for row in gt:
        sessions.setdefault(row["session_id"], []).append(row)
    order = list(sessions)[: limit or None]
    out: dict[str, list[Any]] = {k: [] for k in ("case_id", "session_id", "mask", "ref", "speech", "labels",
                                                 "voice", "space", "z", "spatial")}
    for i, sid in enumerate(order):
        rows = sessions[sid]
        x = torch.from_numpy(read_session(set_dir, sid))
        pos = np.zeros((spec.N_CHANNELS_MAX, 2), dtype=np.float32)
        for row in rows:
            m = meta[row["case_id"]]
            pos[m["active_channels"]] = np.asarray(m["mic_positions_xy_m"], dtype=np.float32)
        masks = torch.tensor([[b == "1" for b in row["mask_bits"]] for row in rows])
        batch = {
            "audio": (x[None] * masks[:, :, None].to(x.dtype)).to(device),
            "mask": masks.to(device),
            "mic_xy": torch.from_numpy(pos)[None].expand(len(rows), -1, -1).contiguous().to(device),
        }
        refs = reference_channel(batch["mask"]).tolist()
        with torch.no_grad():
            cache: dict[str, Any] = {}
            if voice is not None:
                first = {r: refs.index(r) for r in sorted(set(refs))}
                idx = torch.tensor(list(first.values()), device=device)
                sub = {k: v[idx] for k, v in batch.items()}
                v_sub = voice_embeddings(voice, sub)
                cache["voice"] = v_sub[[list(first).index(r) for r in refs]]
            if name in ("space_only", "naive_fusion"):
                cache["space"] = srp_embeddings(batch, cfg.model.srp_temperature)
            speech = causal_vad(reference_audio(batch), cfg.slots.vad_rel_db, cfg.slots.vad_floor_dbfs)
            if system is not None:
                z = system.embed(batch, cache=cache).float()
                spatial = system.last_spatial.float()
        for b, row in enumerate(rows):
            out["case_id"].append(row["case_id"])
            out["session_id"].append(sid)
            out["mask"].append(row["mask_bits"])
            out["ref"].append(refs[b])
            out["speech"].append(speech[b].cpu().numpy())
            out["labels"].append(frame_labels(row))
            if "voice" in cache:
                v = cache["voice"][b].float()
                out["voice"].append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
            if "space" in cache:
                out["space"].append(cache["space"][b].float().cpu().numpy())
            if system is not None:
                out["z"].append(z[b].cpu().numpy())
                out["spatial"].append(spatial[b].cpu().numpy())
        if (i + 1) % 20 == 0 or i + 1 == len(order):
            print(f"  embedded {i + 1}/{len(order)} sessions", flush=True)
    return {k: np.asarray(v) for k, v in out.items() if len(v)}


def inputs_fingerprint(cfg: Any, set_dir: Path, best: Path | None, limit: int) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"model": cfg.model.__dict__, "slots": cfg.slots.__dict__, "limit": limit}, sort_keys=True,
                        default=str).encode())
    h.update((set_dir / "manifest.json").read_bytes())
    if best is not None:
        h.update(hashlib.sha256(best.read_bytes()).digest())
    return h.hexdigest()[:16]


def load_or_embed(cfg: Any, set_dir: Path, out_dir: Path, args: argparse.Namespace, best: Path | None,
                  gt: list[dict[str, Any]], meta: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    path = out_dir / "embeddings.npz"
    fp = inputs_fingerprint(cfg, set_dir, best, args.limit)
    if path.is_file():
        data = dict(np.load(path, allow_pickle=False))
        if str(data.pop("fingerprint")) == fp:
            print(f"  embeddings reused: {path}")
            return data
    if args.components_from:
        data = assemble_components(args.components_from, set_dir.name, cfg)
    else:
        data = embed_set(cfg, set_dir, gt, meta, args.device, args.model_cache, best, args.limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fingerprint=np.asarray(fp), **data)
    return data


def assemble_components(dirs: list[str], set_name: str, cfg: Any) -> dict[str, np.ndarray]:
    """Naive fusion from the voice-only and space-only runs' embeddings: the same
    frozen ECAPA and SRP-PHAT outputs, so nothing is recomputed.  The runs must
    cover the same cases with the same speech flags."""
    if cfg.model.name != "naive_fusion":
        raise SystemExit("--components-from only assembles naive fusion")
    out: dict[str, np.ndarray] = {}
    for d in dirs:
        data = dict(np.load(Path(d) / set_name / "embeddings.npz", allow_pickle=False))
        data.pop("fingerprint")
        for key, value in data.items():
            if key in out and key in ("case_id", "speech", "labels", "ref", "mask", "session_id"):
                if not np.array_equal(out[key], value):
                    raise SystemExit(f"{d}: {key} differs from the other component run")
            out.setdefault(key, value)
    missing = {"voice", "space"} - set(out)
    if missing:
        raise SystemExit(f"--components-from lacks {sorted(missing)}")
    return out


# -- thresholds ----------------------------------------------------------------
def fused(space: np.ndarray, voice: np.ndarray, w: float) -> np.ndarray:
    """``[sqrt(w) s, sqrt(1 - w) v]``: naive fusion and the fixed-gate variant."""
    return np.concatenate([math.sqrt(w) * space, math.sqrt(1.0 - w) * voice], axis=-1)


def variant_parts(variant: str, name: str) -> tuple[str, str | None]:
    """Which embedding a variant reads (or the two it fuses)."""
    if variant == "learned_space":
        return "spatial", None
    if variant == "fixed_gate":
        return "spatial", "voice"
    return {"voice_only": ("voice", None), "space_only": ("space", None), "naive_fusion": ("space", "voice"),
            "proposed": ("z", None)}[name]


def summary_settings(summary: dict[str, Any], name: str, variant: str) -> tuple[dict[int, float], dict[int, float] | None]:
    """``(theta_by_M, weight_by_M or None)`` that validation chose (two-talker cases, cap 2)."""
    diag = summary.get("diagnostics") or {}
    if variant == "learned_space":
        theta, weight = diag.get("learned_space_only_theta_official_by_M"), None
    elif variant == "fixed_gate":
        theta, weight = diag.get("fixed_gate_theta_official_by_M"), diag.get("fixed_gate_weight_by_M")
    else:
        theta = summary.get("theta_official_by_M")
        weight = summary.get("fusion_weight_by_M") if name == "naive_fusion" else None
    if theta is None:
        raise SystemExit(f"summary.json has no thresholds for {name}/{variant}: it predates them; rerun train.py on "
                         "the run directory with this commit (add --allow-code-change for a checkpoint of another commit)")
    return ({int(m): float(t) for m, t in theta.items()},
            {int(m): float(w) for m, w in weight.items()} if weight is not None else None)


def tune_on_eval(emb: dict[str, np.ndarray], key: str, fuse_with: str | None, M: np.ndarray,
                 cfg: Any) -> tuple[dict[int, float], dict[int, float] | None]:
    """Per M, the threshold (and fusion weight) with the best Spk Acc *on the eval set*."""
    thetas = theta_grid(cfg.slots.theta_grid)
    theta: dict[int, float] = {}
    weight: dict[int, float] | None = {} if fuse_with else None
    weights = np.round(np.arange(cfg.slots.weight_grid[0], cfg.slots.weight_grid[1] + cfg.slots.weight_grid[2] / 2,
                                 cfg.slots.weight_grid[2]), 6) if fuse_with else [None]
    for m in sorted(set(M.tolist())):
        idx = np.flatnonzero(M == m)
        best = (-1.0, 0.0, None)
        for w in weights:
            cases = [{"emb": emb[key][i] if w is None else fused(emb[key][i], emb[fuse_with][i], float(w)),
                      "speech": emb["speech"][i], "labels": emb["labels"][i]} for i in idx]
            r = tune_theta(cases, thetas, CAP, label_key="labels")
            best = max(best, (r["acc"], r["theta"], w), key=lambda t: t[0])
        theta[m] = best[1]
        if weight is not None:
            weight[m] = float(best[2])
    return theta, weight


# -- run -----------------------------------------------------------------------
def slots_for(emb: dict[str, np.ndarray], key: str, fuse_with: str | None, M: np.ndarray,
              theta: dict[int, float], weight: dict[int, float] | None) -> np.ndarray:
    out = np.zeros((len(M), spec.N_GRID_POINTS), dtype=np.int64)
    for i, m in enumerate(M.tolist()):
        e = emb[key][i] if fuse_with is None else fused(emb[key][i], emb[fuse_with][i], weight[m])
        out[i] = online_slots(e, emb["speech"][i], [theta[m]], CAP)[0]
    return out


def asr_cache(args: argparse.Namespace) -> tuple[tr.SegmentCache, Any]:
    backend_id = LOCAL_ASR.get(args.asr, args.asr)
    path = Path(args.asr_cache) / f"{backend_id.replace('/', '__')}.jsonl"
    backend = None
    if args.asr in LOCAL_ASR:
        from baselines import FrozenASR

        backend = FrozenASR(model_name=backend_id)
    return tr.SegmentCache(path, backend_id), backend


def run_variant(set_name: str, set_dir: Path, out_dir: Path, emb: dict[str, np.ndarray], slots: np.ndarray,
                cache: tr.SegmentCache, backend: Any, info: dict[str, Any]) -> dict[str, Any] | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = {cid: tr.case_segments(set_name, cid, sid, int(ref), s.tolist())
                for cid, sid, ref, s in zip(emb["case_id"], emb["session_id"], emb["ref"], slots)}
    flat = [seg for segs in segments.values() for seg in sorted(segs, key=lambda s: (s.session_id, s.start))]
    if backend is not None:
        n = tr.transcribe_missing(sorted(flat, key=lambda s: s.session_id), cache, backend,
                                  lambda sid: read_session(set_dir, sid))
        print(f"  transcribed {n} new segments ({cache.backend})", flush=True)
    else:
        missing = tr.export_missing(flat, cache, out_dir / "missing_segments.jsonl")
        if missing:
            print(f"  {missing} segments are not in the {cache.backend} cache: "
                  f"{out_dir / 'missing_segments.jsonl'} lists them; transcribe them and rerun", flush=True)
            return None
    records: list[dict[str, Any]] = []
    for cid, s in zip(emb["case_id"], slots):
        records.extend(timepoint_records(str(cid), s.tolist(), TRACK))
        records.append(tr.transcript_record(str(cid), tr.final_transcripts(segments[cid], cache)))
    pred_path = out_dir / "predictions.jsonl"
    pred_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    gt_path = set_dir / "ground_truth.jsonl"
    if len(emb["case_id"]) < sum(1 for _ in gt_path.open(encoding="utf-8")):
        keep = set(emb["case_id"].tolist())
        rows = [line for line in gt_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["case_id"] in keep]
        gt_path = out_dir / "ground_truth.subset.jsonl"
        gt_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    report = evaluate.evaluate_files(pred_path, gt_path, TRACK)
    # Frame-optimal Spk Acc per case (models/slots.py), the number validation selects on.
    by_M: dict[str, list[float]] = {}
    for lab, s, mask in zip(emb["labels"], slots, emb["mask"]):
        acc = best_assignment_accuracy(lab, s)
        if acc is not None:
            by_M.setdefault(str(str(mask).count("1")), []).append(acc)
    report["frame_optimal_spk_acc_by_M"] = {m: round(float(np.mean(v)), 4) for m, v in sorted(by_M.items())}
    report["eval_t2_run"] = info
    (out_dir / "score.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def table_line(report: dict[str, Any]) -> str:
    cells = []
    for m in ("4", "3", "2", "1"):
        entry = report["breakdown"]["by_n_channels"].get(m)
        if entry:
            met = entry["metrics"]
            cells.append(f"M={m} acc {met[evaluate.METRIC_SPEAKER_ACC]:.3f} cpWER {met['cpwer']:.3f}")
    return " | ".join(cells)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="a train.py run directory (config.json, best.pt, summary.json)")
    src.add_argument("--config", help="configs/<untrained rung>.json")
    parser.add_argument("--summary", help="summary.json of that config's validation run (its thresholds)")
    parser.add_argument("--tune-on-eval", action="store_true",
                        help="choose thresholds and weights on the eval set itself (optimistic; diagnostics only)")
    parser.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    parser.add_argument("--variants", nargs="+", help="subset of the system's variants (default: all)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--asr", default="wav2vec2",
                        help="'wav2vec2' (run here) or the id of a backend whose cache another machine fills")
    parser.add_argument("--asr-cache", default=str(REPO / "runs" / "asr_cache"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-cache", help="speechbrain savedir for the ECAPA checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="first N sessions only (smoke runs)")
    parser.add_argument("--components-from", nargs="+", metavar="OUT_DIR",
                        help="naive fusion: take voice and space embeddings from the --out of a voice-only and a "
                             "space-only run instead of recomputing them")
    parser.add_argument("--embed-only", action="store_true",
                        help="compute (or check) the embeddings of every set, then stop: lets several systems "
                             "embed in parallel and transcribe one after another through the shared ASR cache")
    args = parser.parse_args(argv)

    cfg, summary, best = load_system_inputs(args)
    name = cfg.model.name
    if summary is None and not args.tune_on_eval:
        raise SystemExit("thresholds come from --summary / RUN/summary.json, or pass --tune-on-eval")
    variants = args.variants or list(VARIANTS[name])
    cache, backend = asr_cache(args)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    settings: dict[str, tuple[dict[int, float], dict[int, float] | None]] = {}
    results: dict[str, Any] = {}
    for set_name in args.sets:
        set_dir = REPO / "data" / set_name
        print(f"[{set_name}] {name} ({cfg.name})", flush=True)
        gt, meta = read_set(set_dir)
        emb = load_or_embed(cfg, set_dir, Path(args.out) / set_name, args, best, gt, meta)
        if args.embed_only:
            continue
        M = np.array([str(m).count("1") for m in emb["mask"]])
        for variant in variants:
            key, fuse_with = variant_parts(variant, name)
            if variant not in settings:  # chosen once (main set first), reused for static
                settings[variant] = (tune_on_eval(emb, key, fuse_with, M, cfg) if args.tune_on_eval
                                     else summary_settings(summary, name, variant))
            theta, weight = settings[variant]
            slots = slots_for(emb, key, fuse_with, M, theta, weight)
            info = {"system": name, "config": cfg.name, "variant": variant, "set": set_name,
                    "thresholds": "eval set (optimistic)" if args.tune_on_eval else "validation",
                    "theta_by_M": theta, "weight_by_M": weight, "asr": cache.backend,
                    "n_cases": int(len(M)), "limit": args.limit,
                    "summary": {k: (summary or {}).get(k) for k in ("run", "config", "best_step", "fingerprint")}}
            report = run_variant(set_name, set_dir, Path(args.out) / set_name / variant, emb, slots, cache, backend, info)
            if report is None:
                return 2
            results[f"{set_name}/{variant}"] = report
            print(f"  {variant:13s} {table_line(report)}", flush=True)
            print(f"  {'':13s} frame-optimal Spk Acc by M {report['frame_optimal_spk_acc_by_M']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
