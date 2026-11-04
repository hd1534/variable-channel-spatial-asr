#!/usr/bin/env python3
"""Train (or, for the untrained rungs, tune) one system of the comparison ladder.

Usage::

    python3 train.py --config configs/proposed_carry.json --data-root /content/data \\
        --run-dir /content/drive/MyDrive/vcsasr/runs/proposed_carry
    python3 train.py --config configs/smoke.json --data-root runs/smoke-data --run-dir runs/smoke-train

``--data-root`` holds ``LibriSpeech/<subset>/`` and the room banks
(``rirbanks/``, built on first use).  Rerunning the same command on the same
``--run-dir`` resumes from the newest intact checkpoint listed in
``ckpt/LATEST.json``.  ``--set section.key=value`` overrides
one config field (the value is parsed as JSON when it parses, else taken as a
string), e.g. ``--set train.max_steps=200 --set model.state=reset``.

The last lines printed are a summary block meant to be pasted back as is.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, "") and str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.config import load_config  # noqa: E402

SUMMARY_KEYS = (
    "run", "config", "system", "state", "voice_encoder", "trained_steps", "best_step", "train_speakers", "val_speakers",
    "val_sessions", "val_rooms", "overrides", "commits", "gpu", "gpu_max_mem_gb", "amp_dtype", "bank_engine_version",
    "wall_sec", "synthetic_data", "theta_by_M", "theta_official_by_M", "theta_learned", "fusion_weight_by_M",
    "spk_acc", "spk_acc_n2_cap2", "spk_acc_n2_cap2_by_M", "official_n2", "spk_acc_by_M", "spk_acc_by_N", "spk_acc_after_change_by_M",
    "spk_acc_steady_by_M", "centroid_acc_by_M", "position_acc_by_M", "causal_ceiling_by_N", "gate_by_M",
    "slot_count_error_by_N", "diagnostics", "n_cases",
)


def _parse_set(items: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects section.key=value, got {item!r}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="configs/<name>.json")
    parser.add_argument("--data-root", required=True, help="directory holding LibriSpeech/ and rirbanks/")
    parser.add_argument("--run-dir", required=True, help="checkpoints, metrics and summary go here")
    parser.add_argument("--device", help="cuda / cpu (default: cuda when available)")
    parser.add_argument("--workers", type=int, help="DataLoader workers (default: data.num_workers)")
    parser.add_argument("--model-cache", help="directory speechbrain saves the ECAPA checkpoint into")
    parser.add_argument("--set", action="append", default=[], metavar="section.key=value")
    parser.add_argument("--prepare-only", action="store_true",
                        help="index the corpora, build the room banks, fetch the voice encoder, then stop")
    parser.add_argument("--bank-mirror", help="second copy of the room banks that outlives the machine (Drive)")
    parser.add_argument("--allow-code-change", action="store_true",
                        help="resume a checkpoint written by another commit (both commits are recorded)")
    parser.add_argument("--check-banks", action="store_true",
                        help="exit 3 if pyroomacoustics is needed to build a room bank this config reads "
                             "(missing locally and in --bank-mirror), else 0")
    args = parser.parse_args(argv)

    overrides = _parse_set(args.set)
    cfg = load_config(args.config, overrides)
    if args.check_banks:
        from training.engine import needed_banks
        from training.rirbank import bank_ready

        missing = {name: bank for name, bank in needed_banks(cfg).items()
                   if not bank_ready(bank, Path(args.data_root) / "rirbanks", args.bank_mirror)}
        print(f"room banks to build: {sorted(missing) or 'none'}", flush=True)
        return 3 if any(bank.engine == "pyroomacoustics" for bank in missing.values()) else 0

    from training.engine import Run  # torch is imported here, after argument errors

    started = time.time()
    run = Run(cfg, args.run_dir, args.data_root, device=args.device, workers=args.workers, model_cache=args.model_cache,
              bank_mirror=args.bank_mirror, allow_code_change=args.allow_code_change, overrides=overrides)
    if args.prepare_only:
        print(f"prepared: train bank {run.train_bank}, val bank {run.val_bank}", flush=True)
        return 0
    run.fit()
    summary = run.finalize(started)
    block = {k: summary.get(k) for k in SUMMARY_KEYS if summary.get(k) is not None}
    print("===== VC-SASR RUN SUMMARY (paste this back) =====")
    print(json.dumps(block, ensure_ascii=False))
    print("===== END =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
