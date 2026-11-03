#!/usr/bin/env python3
"""Score a diarizing ASR (Track B reference row) on the two-talker eval set.

A diarizing ASR -- VibeVoice-ASR here, Qwen-Audio-3.1-ASR through its API would
fit the same format -- reads one channel of the whole 20 s session and returns
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


def case_records(case_id: str, segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    segments = clean_segments(segments)
    slots = keep_speakers(segments) if segments else {}
    frames = frame_slots(segments, slots) if segments else [0] * spec.N_GRID_POINTS
    records = [{"record_type": spec.RECORD_TYPE_TIMEPOINT, "case_id": case_id, "eval_time_sec": k * HOP,
                "pred_speaker": spec.speaker_slot(s), "observation_mode": MODE} for k, s in enumerate(frames)]
    texts: dict[str, list[str]] = {}
    for seg in sorted(segments, key=lambda s: s["start_time"]):
        if seg["text"]:
            texts.setdefault(spec.speaker_slot(slots[seg["speaker_id"]]), []).append(seg["text"])
    records.append({"record_type": spec.RECORD_TYPE_FINAL_TRANSCRIPTS, "case_id": case_id,
                    "final_transcripts": {k: " ".join(v) for k, v in texts.items()}, "observation_mode": MODE})
    return records


def score(records: list[dict[str, Any]], gt_rows: list[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        p, g = Path(tmp) / "p.jsonl", Path(tmp) / "g.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        g.write_text("\n".join(gt_rows) + "\n", encoding="utf-8")
        return evaluate.evaluate_files(p, g, TRACK)["scopes"][MODE]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--segments", required=True, help="JSONL written by notebooks/track_b_vibevoice.ipynb")
    parser.add_argument("--sets", nargs="+", default=["eval-t2", "eval-t2-static"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    raw: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    backend = None
    for line in Path(args.segments).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            raw[(row["set"], row["session_id"], int(row["channel"]))] = row["segments"]
            backend = row.get("backend", backend)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"backend": backend, "observation_mode": MODE}
    for set_name in args.sets:
        set_dir = REPO / "data" / set_name
        meta = {json.loads(l)["case_id"]: json.loads(l) for l in (set_dir / "meta.jsonl").read_text().splitlines() if l}
        gt_lines = [l for l in (set_dir / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines() if l]
        records, missing, by_M = [], [], {}
        for line in gt_lines:
            row = json.loads(line)
            m = meta[row["case_id"]]
            key = (set_name, row["session_id"], min(m["active_channels"]))
            if key not in raw:
                missing.append(key)
                continue
            recs = case_records(row["case_id"], raw[key])
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
        cells = " | ".join(f"M={M} acc {v[evaluate.METRIC_SPEAKER_ACC]:.3f} cpWER {v['cpwer']:.3f}"
                           for M, v in sorted(summary[set_name]["by_M"].items(), reverse=True))
        print(f"[{set_name}] {cells}")
    (out / "score.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
