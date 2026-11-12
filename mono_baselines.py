"""Run modern 1ch speaker-attributed ASR baselines on the paired T2 sets.

Each job selects the same reference microphone as the corresponding proposed
case (lowest active channel). Results are paired by case, but the baseline's
actual input channel count is always ONE. Native diarizing ASR uses its own
transcripts. Nemotron uses Qwen on the same padded slot-run segmentation as
the comparison ladder. All three initial presets observe the full recording.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

import spec
import track_b
import transcripts as tr
from modern_asr import (CODE_REVISIONS, MODEL_SPECS, QWEN_BACKEND, VIBE_TOKENIZER_REVISION, atomic_json,
                        audio_path, cache_path, file_hash, fingerprint, jsonl_rows, normalize_segments)

REPO: Path = Path(__file__).resolve().parent
PROTOCOL: int = 1


def build_jobs(data_root: Path, sets: Sequence[str], backend: str, limit: int,
               max_new_tokens: int) -> list[dict[str, Any]]:
    """Build unique session/microphone jobs from metadata, never reference labels."""
    jobs: list[dict[str, Any]] = []
    model, revision = MODEL_SPECS[backend]
    for set_name in sets:
        metas = jsonl_rows(data_root / set_name / "meta.jsonl")
        sessions = list(dict.fromkeys(row["session_id"] for row in metas))[:limit or None]
        seen: set[tuple[str, int]] = set()
        hashes: dict[str, str] = {}
        for row in metas:
            sid, channel = row["session_id"], min(row["active_channels"])
            if sid not in sessions or (sid, channel) in seen:
                continue
            seen.add((sid, channel))
            if sid not in hashes:
                hashes[sid] = file_hash(audio_path(data_root, set_name, sid))
            contract = {"protocol": PROTOCOL, "model": model, "revision": revision,
                        "adapter": CODE_REVISIONS[backend],
                        "max_new_tokens": max_new_tokens if backend != "nemotron" else None,
                        "observation_mode": "full_audio", "input_channels": 1,
                        "speaker_policy": "top2-duration-fold-v1", "audio_sha256": hashes[sid],
                        "channel": channel, "asr": QWEN_BACKEND if backend == "nemotron" else model}
            if backend == "vibevoice":
                contract["tokenizer_revision"] = VIBE_TOKENIZER_REVISION
            jobs.append({"job_id": f"{set_name}/{sid}/ch{channel}", "set": set_name,
                         "session_id": sid, "channel": channel, "backend": model,
                         "audio_sha256": hashes[sid], "contract": contract})
    return jobs


def run_command(command: list[str]) -> None:
    """Run a worker with arguments preserved as separate shell-free values."""
    print("worker:", " ".join(command), flush=True)
    subprocess.run(command, check=True,
                   env={**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"})


def attach_qwen(rows: list[dict[str, Any]], data_root: Path, out: Path,
                cache_root: Path, python: str, device: str) -> list[dict[str, Any]]:
    """Transcribe Nemotron's cap-2 slots while preserving its native time regions."""
    grouped: dict[str, list[tr.Segment]] = {}
    for row in rows:
        segments = normalize_segments(row["segments"], spec.SESSION_LENGTH_SEC)
        slots = track_b.keep_speakers(segments) if segments else {}
        frames = track_b.frame_slots(segments, slots) if segments else [0] * spec.N_GRID_POINTS
        grouped[row["job_id"]] = tr.case_segments(row["set"], row["job_id"], row["session_id"],
                                                 row["channel"], frames)
    flat = [segment for segments in grouped.values() for segment in segments]
    cache = tr.SegmentCache(cache_path(cache_root, QWEN_BACKEND), QWEN_BACKEND)
    requests = out / "qwen_requests.jsonl"
    tr.export_missing(flat, cache, requests, include_cached=True)
    run_command([python, str(REPO / "modern_asr.py"), "--device", device, "--data-root", str(data_root),
                 "transcribe", "--requests", str(requests), "--cache-dir", str(cache_root)])
    cache = tr.SegmentCache(cache.path, QWEN_BACKEND)
    return [{**row, "final_transcripts": tr.final_transcripts(grouped[row["job_id"]], cache)} for row in rows]


def run(backend: str, python: str, qwen_python: str | None, data_root: Path, out: Path,
        cache_root: Path, sets: Sequence[str], device: str, limit: int, max_new_tokens: int) -> dict[str, Any]:
    """Resume model inference, validate provenance and score exactly the requested jobs."""
    if limit < 0 or max_new_tokens < 1:
        raise ValueError("limit must be nonnegative and max_new_tokens positive")
    if backend == "nemotron" and not qwen_python:
        raise ValueError("Nemotron requires --qwen-python for its common ASR stage")
    out.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(data_root, sets, backend, limit, max_new_tokens)
    if not jobs:
        raise ValueError("no inference jobs")
    jobs_path, raw_path = out / "jobs.jsonl", out / "native_segments.jsonl"
    old = {row["job_id"]: row for row in jsonl_rows(raw_path, repair_tail=True)}
    for job in jobs:
        if job["job_id"] in old and old[job["job_id"]]["contract"] != job["contract"]:
            raise ValueError(f"{out}: existing outputs use different data/model/settings; choose another --out")
    jobs_path.write_text("".join(json.dumps(job) + "\n" for job in jobs), encoding="utf-8")
    run_command([python, str(REPO / "modern_asr.py"), "--device", device, "--data-root", str(data_root),
                 "diarize", "--backend", backend, "--jobs", str(jobs_path), "--out", str(raw_path),
                 "--max-new-tokens", str(max_new_tokens)])
    current = {row["job_id"]: row for row in jsonl_rows(raw_path)}
    rows: list[dict[str, Any]] = []
    for job in jobs:
        row = current[job["job_id"]]
        if row["contract"] != job["contract"]:
            raise ValueError(f"worker returned mismatched contract for {job['job_id']}")
        normalize_segments(row["segments"], spec.SESSION_LENGTH_SEC)
        rows.append(row)
    if backend == "nemotron":
        rows = attach_qwen(rows, data_root, out, cache_root, str(qwen_python), device)
    segments_path = out / "segments.jsonl"
    temporary = segments_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(segments_path)
    report = track_b.score_segments(segments_path, data_root, sets, out, limit=limit)
    report["baseline"] = {"name": backend, "run": out.name, "input_channels": 1, "observation_mode": "full_audio",
                          "model": MODEL_SPECS[backend][0], "revision": MODEL_SPECS[backend][1],
                          "adapter": CODE_REVISIONS[backend],
                          "latency_sec": spec.SESSION_LENGTH_SEC, "latency_kind": "audio_buffer_excluding_compute",
                          "asr": QWEN_BACKEND if backend == "nemotron" else "@".join(MODEL_SPECS[backend]),
                          "speaker_policy": "top2-duration-fold-v1", "limit": limit,
                          "contracts_sha256": fingerprint([job["contract"] for job in jobs])}
    report["runtime"] = json.loads(raw_path.with_suffix(".runtime.json").read_text())
    if backend == "nemotron":
        report["baseline"]["asr_cache"] = cache_root.name
        report["qwen_runtime"] = json.loads((cache_root / "qwen-runtime.json").read_text())
    atomic_json(out / "score.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run one modern mono baseline using separately provisioned model runtimes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=MODEL_SPECS, required=True)
    parser.add_argument("--python", required=True, help="baseline-specific isolated interpreter")
    parser.add_argument("--qwen-python", help="Qwen interpreter, required for Nemotron")
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asr-cache", type=Path, default=REPO / "runs" / "asr_cache")
    parser.add_argument("--sets", nargs="+", default=["eval-t2", "eval-t2-static"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="first N sessions per set; 0 = all")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args(argv)
    run(args.backend, args.python, args.qwen_python, args.data_root, args.out, args.asr_cache,
        args.sets, args.device, args.limit, args.max_new_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
