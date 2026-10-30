"""Speaker predictions and ground truth in the official format, scored by ``evaluate.py``.

Every rule below exists because breaking it either voids the *whole* file or
silently costs labelled instants (measured against ``evaluate.py`` 0.2.0):

* exactly 80 ``timepoint`` records per case, one per ``T_all`` instant, unique
  (a duplicate ``(case_id, t)`` rejects the file: R-DUP-02);
* ``pred_speaker`` is always a real slot label ``spec.speaker_slot(i)`` numbered
  in slot-creation order -- ``spk_0`` before any slot exists, the nearest slot
  at silent instants -- never ``None``, ``""`` or a sentinel (a null is scored
  wrong, an empty string is R-PARSE);
* the union of frame labels and transcript keys never exceeds
  ``spec.MAX_SPEAKERS_PER_TRACK[track]`` (one stray label on an unscored instant
  rejects the file: R-SPK-01), so it is asserted here before anything is written;
* no ``spec_version`` field and no spatial fields: speaker-only predictions are
  valid, the speaker metrics are computed, and the spatial point metrics of
  such a file are penalty values that must not be read.

Without ``final_transcripts`` every speaker permutation ties on cpWER, so the
official Spk Acc uses the frame-optimal permutation -- per case the same number
validation computes, and averaged over cases the same way.  With transcripts the
permutation is chosen by cpWER first.

Ground-truth talkers are named ``spk_0, spk_1, ...`` in order of first
appearance, the order the slots open in, so the speaker-agnostic WER (which
concatenates slots by label) does not depend on naming.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import evaluate
import spec


def track_for(n_speakers: int) -> int:
    """The smallest contract track that admits ``n_speakers`` talkers per case.

    >>> track_for(1), track_for(2)
    (1, 2)
    >>> track_for(3)
    Traceback (most recent call last):
        ...
    ValueError: the scoring contract (spec 0.2.0) has no track for 3 talkers per case
    """
    for track, cap in sorted(spec.MAX_SPEAKERS_PER_TRACK.items()):
        if n_speakers <= cap:
            return track
    raise ValueError(f"the scoring contract (spec {spec.SPEC_VERSION}) has no track for {n_speakers} talkers per case")


def timepoint_records(case_id: str, slots: Sequence[int], track: int) -> list[dict[str, Any]]:
    """80 ``timepoint`` records from online slot indices (creation order).

    >>> recs = timepoint_records("val_0000__m1100", [0] * 79 + [1], 2)
    >>> len(recs), recs[0]["pred_speaker"], recs[-1]["eval_time_sec"], recs[-1]["pred_speaker"]
    (80, 'spk_0', 19.75, 'spk_1')
    >>> timepoint_records("val_0000__m1100", [0, 1, 2] + [0] * 77, 2)
    Traceback (most recent call last):
        ...
    ValueError: val_0000__m1100: 3 slots but track 2 allows 2
    """
    if len(slots) != spec.N_GRID_POINTS:
        raise ValueError(f"{case_id}: expected {spec.N_GRID_POINTS} instants, got {len(slots)}")
    labels = [spec.speaker_slot(int(s)) for s in slots]
    cap = spec.MAX_SPEAKERS_PER_TRACK[track]
    if len(set(labels)) > cap:
        raise ValueError(f"{case_id}: {len(set(labels))} slots but track {track} allows {cap}")
    return [
        {"record_type": spec.RECORD_TYPE_TIMEPOINT, "case_id": case_id, "eval_time_sec": k * spec.DELTA_T_SEC,
         "pred_speaker": label}
        for k, label in enumerate(labels)
    ]


def truth_row(case_id: str, labels_eval: Sequence[int], transcripts: Mapping[int, str] | None = None) -> dict[str, Any]:
    """One ground-truth row: the labelled instants, talkers renamed by first appearance.

    >>> row = truth_row("val_0001__m1111", [-1, 1, 1, -1, 0] + [-1] * 75)
    >>> [(f["eval_time_sec"], f["speaker"]) for f in row["frames"]]
    [(0.25, 'spk_0'), (0.5, 'spk_0'), (1.0, 'spk_1')]
    """
    order: list[int] = []
    for v in labels_eval:
        if v >= 0 and v not in order:
            order.append(int(v))
    name = {talker: spec.speaker_slot(i) for i, talker in enumerate(order)}
    row: dict[str, Any] = {
        "case_id": case_id,
        spec.F_FRAMES: [
            {"eval_time_sec": k * spec.DELTA_T_SEC, "speaker": name[int(v)]}
            for k, v in enumerate(labels_eval) if v >= 0
        ],
    }
    if transcripts is not None:
        row[spec.F_TRANSCRIPTS_REFERENCE] = {name[t]: text for t, text in transcripts.items() if t in name}
    return row


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_cases(prediction_records: Sequence[Mapping[str, Any]], truth_rows: Sequence[Mapping[str, Any]],
                track: int, keep_dir: str | Path | None = None) -> dict[str, Any]:
    """Run ``evaluate.evaluate_files`` on the given records; optionally keep the files.

    >>> preds = timepoint_records("v_0__m1100", [0] * 40 + [1] * 40, 2)
    >>> truth = [truth_row("v_0__m1100", [1] * 40 + [0] * 40)]
    >>> score_cases(preds, truth, 2)["metrics"]["speaker_attribution_accuracy"]
    1.0
    """
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(keep_dir) if keep_dir else Path(tmp)
        base.mkdir(parents=True, exist_ok=True)
        pred_path, gt_path = base / "predictions.jsonl", base / "ground_truth.jsonl"
        write_jsonl(pred_path, prediction_records)
        write_jsonl(gt_path, truth_rows)
        return evaluate.evaluate_files(pred_path, gt_path, track)


def official_speaker_check(cases: Sequence[Mapping[str, Any]], slot_fn, keep_dir: str | Path | None = None) -> dict[str, Any] | None:
    """Score the two-talker validation cases with ``evaluate.py`` itself.

    ``slot_fn(case) -> [80] slot indices`` must already respect the track's cap.
    Returns the official Spk Acc overall and per mask, or ``None`` without
    two-talker cases.
    """
    two = [c for c in cases if c["N"] == 2]
    if not two:
        return None
    track = track_for(2)
    preds: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    for c in two:
        case_id = spec.make_case_id(f"val_{c['index']:04d}", c["mask"])
        preds.extend(timepoint_records(case_id, slot_fn(c), track))
        truth.append(truth_row(case_id, np.asarray(c["labels_eval"]).tolist()))
    report = score_cases(preds, truth, track, keep_dir=keep_dir)
    by_mask = {
        mask: entry["metrics"].get(evaluate.METRIC_SPEAKER_ACC)
        for mask, entry in report.get("breakdown", {}).get("by_mask", {}).items()
    }
    return {
        "track": track,
        "n_cases": len(two),
        "speaker_attribution_accuracy": report["metrics"][evaluate.METRIC_SPEAKER_ACC],
        "by_mask": by_mask,
    }
