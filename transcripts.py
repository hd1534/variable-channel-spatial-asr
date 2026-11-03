"""``final_transcripts`` from predicted speaker slots and a frozen ASR.

Segmentation follows :data:`spec.ASR_MIN_SEGMENT_SEC`: a segment is a run of
one predicted slot on the 0.25 s grid, ``[k0 * 0.25, k1 * 0.25)``, padded
symmetrically to 1.0 s when shorter (clipped to the session); the ASR runs once
per segment and its text goes to that slot.  Segments are *not* taken from an energy VAD: on ``eval-t2`` the
repo's ``energy_vad_segments`` returns one 20 s segment in 133 of the 134
sessions at SNR 5/10 dB, which hands every word to one slot whatever the
attribution was (measured 2026-11-03 with ground-truth slots: cpWER 0.984 at
SNR 5 against 0.903 with slot runs and 0.845 with the true turns).

The recogniser is a controlled variable shared by every row, so it is called
through a segment cache keyed by ``(set, session, channel, start, end)``: the
same segment is transcribed once whichever system asked for it, and a backend
that only runs elsewhere (Qwen3-ASR on a Colab GPU) can fill the cache from an
exported list of the segments still missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import spec

HOP = spec.DELTA_T_SEC
N_GRID = spec.N_GRID_POINTS


def slot_runs(slots: Sequence[int], min_sec: float = spec.ASR_MIN_SEGMENT_SEC,
              length_sec: float = spec.SESSION_LENGTH_SEC) -> list[tuple[int, float, float]]:
    """``[(slot, start_sec, end_sec)]``: maximal runs of one slot, short ones padded.

    >>> slot_runs([0] * 40 + [1] * 40)
    [(0, 0.0, 10.0), (1, 10.0, 20.0)]
    >>> slot_runs([0] * 40 + [1] * 2 + [0] * 38)  # a 0.5 s flicker is padded to 1.0 s
    [(0, 0.0, 10.0), (1, 9.75, 10.75), (0, 10.5, 20.0)]
    """
    if len(slots) != N_GRID:
        raise ValueError(f"expected {N_GRID} slots, got {len(slots)}")
    out: list[tuple[int, float, float]] = []
    k0 = 0
    for k in range(1, N_GRID + 1):
        if k == N_GRID or slots[k] != slots[k0]:
            start, end = k0 * HOP, k * HOP
            if end - start < min_sec:
                pad = (min_sec - (end - start)) / 2.0
                start, end = max(0.0, start - pad), min(length_sec, end + pad)
            out.append((int(slots[k0]), round(start, 6), round(end, 6)))
            k0 = k
    return out


def segment_key(set_name: str, session_id: str, channel: int, start: float, end: float) -> str:
    """>>> segment_key("eval-t2", "t2_0003", 2, 1.0, 3.25)
    'eval-t2/t2_0003/ch2/1.000-3.250'
    """
    return f"{set_name}/{session_id}/ch{channel}/{start:.3f}-{end:.3f}"


class SegmentCache:
    """Append-only JSONL ``{"key", "backend", "text"}``; one file per backend."""

    def __init__(self, path: str | Path, backend: str) -> None:
        self.path = Path(path)
        self.backend = backend
        self.text: dict[str, str] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row["backend"] != backend:
                        raise ValueError(f"{self.path} holds {row['backend']!r} transcripts, not {backend!r}")
                    self.text[row["key"]] = row["text"]

    def __contains__(self, key: str) -> bool:
        return key in self.text

    def add(self, key: str, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "backend": self.backend, "text": text}, ensure_ascii=False) + "\n")
        self.text[key] = text


@dataclass(frozen=True)
class Segment:
    case_id: str
    slot: int
    key: str
    session_id: str
    channel: int
    start: float
    end: float


def case_segments(set_name: str, case_id: str, session_id: str, channel: int, slots: Sequence[int]) -> list[Segment]:
    return [Segment(case_id, slot, segment_key(set_name, session_id, channel, s, e), session_id, channel, s, e)
            for slot, s, e in slot_runs(slots)]


def transcribe_missing(segments: Iterable[Segment], cache: SegmentCache, backend: Any, read_session) -> int:
    """Run ``backend.transcribe(mono, sr)`` on every segment the cache lacks; returns how many."""
    done = 0
    audio: dict[str, Any] = {}
    for seg in segments:
        if seg.key in cache:
            continue
        if seg.session_id not in audio:
            audio.clear()  # segments arrive grouped by session; keep one file in memory
            audio[seg.session_id] = read_session(seg.session_id)
        x = audio[seg.session_id][seg.channel]
        lo, hi = int(round(seg.start * spec.SAMPLE_RATE_HZ)), int(round(seg.end * spec.SAMPLE_RATE_HZ))
        cache.add(seg.key, backend.transcribe(x[lo:hi], spec.SAMPLE_RATE_HZ))
        done += 1
    return done


def export_missing(segments: Iterable[Segment], cache: SegmentCache, path: str | Path) -> int:
    """Write the segments the cache lacks as JSONL, for a backend that runs elsewhere."""
    seen: set[str] = set()
    rows = []
    for seg in segments:
        if seg.key in cache or seg.key in seen:
            continue
        seen.add(seg.key)
        rows.append({"key": seg.key, "session_id": seg.session_id, "channel": seg.channel,
                     "start_sec": seg.start, "end_sec": seg.end})
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def final_transcripts(segments: Sequence[Segment], cache: SegmentCache) -> dict[str, str]:
    """Slot label -> concatenated text, in time order.  Every segment must be cached."""
    parts: dict[str, list[str]] = {}
    for seg in sorted(segments, key=lambda s: s.start):
        text = cache.text[seg.key].strip()
        if text:
            parts.setdefault(spec.speaker_slot(seg.slot), []).append(text)
    return {label: " ".join(texts) for label, texts in parts.items()}


def transcript_record(case_id: str, transcripts: Mapping[str, str]) -> dict[str, Any]:
    return {"record_type": spec.RECORD_TYPE_FINAL_TRANSCRIPTS, "case_id": case_id,
            "final_transcripts": dict(transcripts), "observation_mode": "causal"}
