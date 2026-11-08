#!/usr/bin/env python3
"""Score a diarizing ASR (Track B reference row) on the two-talker eval set.

A diarizing ASR reads one channel of the whole 20 s session and returns
``[{"start_time", "end_time", "speaker_id", "text"}]``.  It neither uses space
nor runs causally, so it is reported as a reference (``observation_mode =
"full_audio"``, outside the official causal table) and its numbers do not move
with M except through which microphone it hears (the lowest active one, like
every other row).

Input: JSONL from ``notebooks/track_b_vibevoice.ipynb``, one row per
``(set, session_id, channel)`` with a ``segments`` list.  Usage::

    python3 track_b.py --segments vibevoice_segments.jsonl --out runs/eval/track_b_vibevoice
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, "") and str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate  # noqa: E402
import spec  # noqa: E402

REPO = Path(__file__).resolve().parent
TRACK = 2
CAP = spec.MAX_SPEAKERS_PER_TRACK[TRACK]
HOP = spec.DELTA_T_SEC
MODE = "full_audio"


NON_SPEECH = re.compile(r"\[[^\]]*\]")
"""Bracketed tags such as ``[Silence]`` or ``[Noise]``: not words of any talker."""


def clean_segments(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Segments a talker actually spoke.  The model's parser copies only the keys
    it emitted and checks no types, so a segment can lack a time or a speaker;
    such a segment, or one holding only non-speech tags, is dropped rather than
    let a ``None`` speaker take one of the two slots.

    >>> clean_segments([{"speaker_id": None, "start_time": 0, "end_time": 6, "text": "[Silence]"},
    ...                 {"speaker_id": 0, "start_time": 6, "end_time": 9, "text": "hello [Noise] there"},
    ...                 {"speaker_id": 1, "start_time": None, "end_time": 9, "text": "x"}])
    [{'speaker_id': 0, 'start_time': 6.0, 'end_time': 9.0, 'text': 'hello there'}]
    """
    out = []
    for seg in segments:
        try:
            start, end = float(seg["start_time"]), float(seg["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        raw = str(seg.get("text") or "")
        text = " ".join(NON_SPEECH.sub(" ", raw).split())
        if seg.get("speaker_id") is None or end <= start or (raw.strip() and not text):
            continue  # no talker, no time span, or only non-speech tags
        out.append({"speaker_id": seg["speaker_id"], "start_time": start, "end_time": end, "text": text})
    return out


def keep_speakers(segments: Sequence[Mapping[str, Any]], cap: int = CAP) -> dict[Any, int]:
    """Speaker id -> slot.  The ``cap`` longest-talking speakers keep a slot each,
    numbered by first appearance; any further speaker is folded into the
    longest-talking slot (the contract allows two labels per case).

    >>> keep_speakers([{"speaker_id": 3, "start_time": 0, "end_time": 5},
    ...                {"speaker_id": 1, "start_time": 5, "end_time": 6},
    ...                {"speaker_id": 7, "start_time": 6, "end_time": 9}])
    {3: 0, 7: 1, 1: 0}
    """
    talk: dict[Any, float] = {}
    first: dict[Any, float] = {}
    for seg in segments:
        sid = seg["speaker_id"]
        talk[sid] = talk.get(sid, 0.0) + max(0.0, float(seg["end_time"]) - float(seg["start_time"]))
        first.setdefault(sid, float(seg["start_time"]))
    kept = sorted(talk, key=lambda s: -talk[s])[:cap]
    slots = {sid: i for i, sid in enumerate(sorted(kept, key=lambda s: first[s]))}
    for sid in talk:
        slots.setdefault(sid, slots[kept[0]])
    return slots


def frame_slots(segments: Sequence[Mapping[str, Any]], slots: Mapping[Any, int]) -> list[int]:
    """Slot per grid instant: the speaker covering most of ``[t_k, t_k + 0.25)``,
    the official label window (allowed because this system hears the whole
    session); silent instants repeat the previous slot, or the next one before
    the first speech.

    >>> segs = [{"speaker_id": "a", "start_time": 0.3, "end_time": 1.0},
    ...         {"speaker_id": "b", "start_time": 1.1, "end_time": 2.0}]
    >>> frame_slots(segs, {"a": 0, "b": 1})[:9]
    [0, 0, 0, 0, 1, 1, 1, 1, 1]
    """
    out: list[int | None] = []
    for k in range(spec.N_GRID_POINTS):
        lo, hi = k * HOP, (k + 1) * HOP
        cover: dict[int, float] = {}
        for seg in segments:
            ov = min(hi, float(seg["end_time"])) - max(lo, float(seg["start_time"]))
            if ov > 0:
                slot = slots[seg["speaker_id"]]
                cover[slot] = cover.get(slot, 0.0) + ov
        out.append(max(cover, key=lambda s: (cover[s], -s)) if cover else None)
    first = next((s for s in out if s is not None), 0)
    filled, last = [], first
    for s in out:
        last = s if s is not None else last
        filled.append(last)
    return filled


def case_records(case_id: str, segments: Sequence[Mapping[str, Any]],
                 final_transcripts: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Convert native regions and optional common-ASR text to full-audio records."""
    segments = clean_segments(segments)
    slots = keep_speakers(segments) if segments else {}
    frames = frame_slots(segments, slots) if segments else [0] * spec.N_GRID_POINTS
    records = [{"record_type": spec.RECORD_TYPE_TIMEPOINT, "case_id": case_id, "eval_time_sec": k * HOP,
                "pred_speaker": spec.speaker_slot(s), "observation_mode": MODE} for k, s in enumerate(frames)]
    texts: dict[str, list[str]] = {}
    for seg in sorted(segments, key=lambda s: s["start_time"]):
        if seg["text"]:
            texts.setdefault(spec.speaker_slot(slots[seg["speaker_id"]]), []).append(seg["text"])
    transcripts = {k: " ".join(v) for k, v in texts.items()} if final_transcripts is None else dict(final_transcripts)
    if any(key not in {spec.speaker_slot(i) for i in range(CAP)} or not isinstance(value, str)
           for key, value in transcripts.items()):
        raise ValueError("invalid cap-2 final_transcripts")
    records.append({"record_type": spec.RECORD_TYPE_FINAL_TRANSCRIPTS, "case_id": case_id,
                    "final_transcripts": transcripts, "observation_mode": MODE})
    return records


def score(records: list[dict[str, Any]], gt_rows: list[str]) -> dict[str, Any]:
    """Use the authoritative scorer with separate full-audio aggregation."""
    with tempfile.TemporaryDirectory() as tmp:
        p, g = Path(tmp) / "p.jsonl", Path(tmp) / "g.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        g.write_text("\n".join(gt_rows) + "\n", encoding="utf-8")
        return evaluate.evaluate_files(p, g, TRACK)["scopes"][MODE]


def score_segments(segments_path: Path, data_root: Path, sets: Sequence[str], out: Path,
                   *, limit: int = 0) -> dict[str, Any]:
    """Score mono results on paired cases, labelling M as the comparison condition.

    ``by_M`` is retained for existing consumers; it groups the proposed case's
    available microphones, never the number the mono baseline actually used.
    """
    raw: dict[tuple[str, str, int], dict[str, Any]] = {}
    backends: set[str | None] = set()
    contracts: set[str] = set()
    runtimes: set[str] = set()
    for line in segments_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (row["set"], row["session_id"], int(row["channel"]))
            if key in raw:
                raise ValueError(f"duplicate mono result: {key}")
            raw[key] = row
            backends.add(row.get("backend"))
            contract = row.get("contract")
            identity = {k: v for k, v in contract.items() if k not in ("audio_sha256", "channel")} if contract else None
            contracts.add(json.dumps(identity, sort_keys=True))
            runtimes.add(json.dumps(row.get("runtime"), sort_keys=True))
    if len(backends) != 1:
        raise ValueError("expected results from exactly one baseline backend")
    if len(contracts) != 1 or len(runtimes) != 1:
        raise ValueError("inconsistent inference contracts or runtimes in mono results")
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"backend": next(iter(backends)), "observation_mode": MODE, "input_channels": 1,
                               "inference_contract": json.loads(next(iter(contracts))),
                               "runtime": json.loads(next(iter(runtimes))),
                               "grouping": "M is the paired proposed condition; baseline input is always 1ch"}
    for set_name in sets:
        set_dir = data_root / set_name
        meta = {json.loads(l)["case_id"]: json.loads(l) for l in (set_dir / "meta.jsonl").read_text().splitlines() if l}
        gt_lines = [l for l in (set_dir / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines() if l]
        if limit:
            sessions = list(dict.fromkeys(json.loads(line)["session_id"] for line in gt_lines))[:limit]
            gt_lines = [line for line in gt_lines if json.loads(line)["session_id"] in sessions]
        if not gt_lines:
            raise ValueError(f"{set_name}: no ground-truth cases selected")
        records, missing, by_M = [], [], {}
        for line in gt_lines:
            row = json.loads(line)
            m = meta[row["case_id"]]
            key = (set_name, row["session_id"], min(m["active_channels"]))
            if key not in raw:
                missing.append(key)
                continue
            recs = case_records(row["case_id"], raw[key]["segments"], raw[key].get("final_transcripts"))
            records.extend(recs)
            by_M.setdefault(str(len(m["active_channels"])), ([], []))
            by_M[str(len(m["active_channels"]))][0].extend(recs)
            by_M[str(len(m["active_channels"]))][1].append(line)
        if missing:
            raise SystemExit(f"{set_name}: {len(missing)} cases have no segments, e.g. {missing[:3]}")
        (out / f"{set_name}.predictions.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        summary[set_name] = {"all": score(records, gt_lines)["metrics"],
                             "by_M": {M: score(r, g)["metrics"] for M, (r, g) in sorted(by_M.items())}}
        summary[set_name]["by_comparison_M"] = summary[set_name]["by_M"]
        cells = " | ".join(f"M={M} acc {v[evaluate.METRIC_SPEAKER_ACC]:.3f} cpWER {v['cpwer']:.3f}"
                           for M, v in sorted(summary[set_name]["by_M"].items(), reverse=True))
        print(f"[{set_name}] mono input=1ch, paired comparison: {cells}")
    (out / "score.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Score existing legacy or modern diarizing-ASR segment files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", default=["eval-t2", "eval-t2-static"])
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    score_segments(args.segments, args.data_root, args.sets, args.out, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
