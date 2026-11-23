"""Speaker-tagged text for Track C, and the grid slots of its frame head.

The model writes one transcript for the whole session with a tag before every
turn: ``[S1] so i went there [S2] did you see him [S1] no``.  Tags number the
talkers by first appearance, the same rule as the frame head's slots, so the
two outputs agree without any matching.  Words are lower case (the scorer
case-folds and strips punctuation anyway).
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import spec

TAGS: tuple[str, ...] = ("[S1]", "[S2]")
TAG_RE = re.compile(r"\[S(\d+)\]")
CAP = spec.MAX_SPEAKERS_PER_TRACK[2]


def normalise(text: str) -> str:
    """LibriSpeech transcript words, lower case, single spaces.

    >>> normalise("  HE SAID  DON'T ")
    "he said don't"
    """
    return " ".join(text.lower().split())


def slot_order(turn_speakers: Sequence[int]) -> dict[int, int]:
    """Talker -> slot by first appearance.

    >>> slot_order([1, 0, 1])
    {1: 0, 0: 1}
    """
    order: dict[int, int] = {}
    for talker in turn_speakers:
        order.setdefault(talker, len(order))
    return order


def target_text(turns: Sequence[Any]) -> str:
    """The training target for a session's turns (objects with ``speaker`` and ``text``).

    >>> from types import SimpleNamespace as T
    >>> target_text([T(speaker=1, text="SO I WENT"), T(speaker=0, text="DID YOU"), T(speaker=1, text="NO")])
    '[S1] so i went [S2] did you [S1] no'
    """
    order = slot_order([t.speaker for t in turns])
    if len(order) > CAP:
        raise ValueError(f"{len(order)} talkers, the contract allows {CAP}")
    return " ".join(f"{TAGS[order[t.speaker]]} {normalise(t.text)}" for t in turns if normalise(t.text))


def parse_tagged(text: str) -> dict[str, str]:
    """Final transcripts per slot from model output; words before any tag go to the first slot,
    tags beyond the cap fold into the last slot.

    >>> parse_tagged("[S1] so i went [S2] did you [S1] no")
    {'spk_0': 'so i went no', 'spk_1': 'did you'}
    >>> parse_tagged("hello [S3] there")
    {'spk_0': 'hello', 'spk_1': 'there'}
    """
    pieces: dict[int, list[str]] = {}
    slot = 0
    position = 0
    for match in TAG_RE.finditer(text):
        chunk = text[position:match.start()].split()
        if chunk:
            pieces.setdefault(slot, []).extend(chunk)
        slot = min(max(int(match.group(1)) - 1, 0), CAP - 1)
        position = match.end()
    tail = text[position:].split()
    if tail:
        pieces.setdefault(slot, []).extend(tail)
    return {spec.speaker_slot(s): " ".join(w) for s, w in sorted(pieces.items())}


def slot_labels(labels_eval: Sequence[int], turn_speakers: Sequence[int]) -> list[int]:
    """Grid labels in slot numbering (``-1`` silence) from talker labels.

    >>> slot_labels([-1, 1, 1, 0, -1], [1, 0])
    [-1, 0, 0, 1, -1]
    """
    order = slot_order(turn_speakers)
    return [order.get(int(x), -1) if int(x) >= 0 else -1 for x in labels_eval]


def grid_slots(pred: Sequence[int]) -> list[int]:
    """Frame-head classes per instant (0 silence, 1.. slots) to a slot per instant: silence repeats the
    previous slot, or takes the first one before any speech (like :func:`track_b.frame_slots`).

    >>> grid_slots([0, 2, 2, 0, 1, 0])
    [1, 1, 1, 1, 0, 0]
    """
    slots = [int(p) - 1 if int(p) > 0 else None for p in pred]
    first = next((s for s in slots if s is not None), 0)
    out, last = [], first
    for s in slots:
        last = s if s is not None else last
        out.append(last)
    return out


def records(case_id: str, slots: Sequence[int], transcripts: Mapping[str, str], mode: str = "full_audio") -> list[dict]:
    """Prediction records for :mod:`evaluate` (``timepoint`` x 80 + ``final_transcripts``)."""
    rows = [{"record_type": spec.RECORD_TYPE_TIMEPOINT, "case_id": case_id, "eval_time_sec": k * spec.DELTA_T_SEC,
             "pred_speaker": spec.speaker_slot(int(s)), "observation_mode": mode} for k, s in enumerate(slots)]
    rows.append({"record_type": spec.RECORD_TYPE_FINAL_TRANSCRIPTS, "case_id": case_id,
                 "final_transcripts": dict(transcripts), "observation_mode": mode})
    return rows
