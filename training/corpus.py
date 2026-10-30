"""Speech sources for training and validation sessions.

Training voices come from LibriSpeech train-clean-100 and validation voices from
the dev-clean speakers *outside* the eval split, so no voice the official
multi-speaker set is built from is ever trained or tuned on.  The split is
recomputed with the very function and seed the eval generator uses
(:func:`simulate.assign_splits`, ``make_eval_dev.py --seed 20261029``), so the
exclusion cannot drift from the committed set.

``synthetic`` corpora stand in for LibriSpeech in smoke runs: every "speaker"
gets placeholder audio from :func:`simulate.synthesise_utterance`, which has no
consistent voice, so nothing about speaker identity can be learned from them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import simulate
import spec
from training.config import EVAL_SPLIT_SEED

SAMPLE_RATE_HZ = spec.SAMPLE_RATE_HZ


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    speaker: str
    path: str | None
    transcript: str


class SpeechCorpus:
    """``{speaker: [Utterance]}`` plus a reader.

    Audio is decoded on demand (a train-clean-100 utterance is ~13 s, ~10 ms to
    decode), so a DataLoader worker never holds the corpus in memory.
    """

    def __init__(self, by_speaker: dict[str, list[Utterance]], synthetic: bool = False,
                 sex: dict[str, str] | None = None) -> None:
        self.by_speaker = {s: u for s, u in sorted(by_speaker.items()) if u}
        self.speakers = tuple(self.by_speaker)
        self.synthetic = synthetic
        self.sex = dict(sex or {})
        """``{speaker: "M" | "F"}`` from LibriSpeech's SPEAKERS.TXT, for the per-case analysis."""
        if len(self.speakers) < 2:
            raise ValueError(f"a corpus needs at least two speakers, got {len(self.speakers)}")

    def __len__(self) -> int:
        return len(self.speakers)

    def n_samples(self, utt: Utterance) -> int:
        """Length without decoding (FLAC header)."""
        if self.synthetic:
            return len(self.read(utt))
        import soundfile as sf

        return int(sf.info(utt.path).frames)

    def read(self, utt: Utterance, start: int = 0, stop: int | None = None):  # noqa: ANN201 - numpy array
        """Mono float32 audio at 16 kHz, samples ``[start, stop)``.

        A crop decodes only its own frames (``soundfile`` seeks), which is
        bit-identical to slicing the full decode and about half the cost.
        """
        import numpy as np

        if self.synthetic:
            rng = random.Random(f"synthetic:{utt.utt_id}")
            signal, _, _ = simulate.synthesise_utterance(rng, SAMPLE_RATE_HZ, duration_sec=rng.uniform(4.0, 9.0))
            return np.asarray(signal, dtype=np.float32)[start:stop]
        import soundfile as sf

        audio, rate = sf.read(utt.path, start=start, stop=stop, dtype="float32", always_2d=False)
        if int(rate) != SAMPLE_RATE_HZ:
            raise ValueError(f"{utt.path} is {rate} Hz, expected {SAMPLE_RATE_HZ}; resample the corpus offline")
        return audio if audio.ndim == 1 else audio.mean(axis=1)


def librispeech_corpus(root: str | Path, exclude_eval_split: bool = False) -> SpeechCorpus:
    """Index ``<root>/<speaker>/<chapter>/*.flac``.

    With ``exclude_eval_split`` the eval-split speakers of the committed
    generator are dropped (use it for dev-clean, never needed for train-clean).
    """
    import make_eval_dev  # local import: pulls nothing heavy, but keeps this module light

    index = make_eval_dev.scan_librispeech(root)
    if not index:
        raise FileNotFoundError(
            f"no LibriSpeech utterances under {root} (expected <root>/<speaker>/<chapter>/*.flac "
            "next to *.trans.txt)"
        )
    if exclude_eval_split:
        split_of = simulate.assign_splits(sorted(index), seed=EVAL_SPLIT_SEED)
        held_out = {s for s, split in split_of.items() if split == "eval"}
        committed = committed_eval_speakers()
        if committed and held_out != committed:
            # The split depends on every speaker under the root: an incomplete tree (an
            # interrupted extraction, say) moves eval voices into the validation pool.
            raise ValueError(
                f"{root} does not hold the full dev-clean speaker set: its eval split would be {sorted(held_out)}, "
                f"but the committed eval set uses {sorted(committed)}. Re-extract the corpus."
            )
        index = {s: rows for s, rows in index.items() if s not in held_out | committed}
    by_speaker = {
        speaker: [Utterance(utt_id, speaker, str(path), text) for utt_id, path, text in rows]
        for speaker, rows in index.items()
    }
    return SpeechCorpus(by_speaker, sex=_read_sex(Path(root).parent / "SPEAKERS.TXT"))


def _read_sex(path: Path) -> dict[str, str]:
    """``{speaker: "M" | "F"}`` from LibriSpeech's ``SPEAKERS.TXT`` (empty if absent)."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(";"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[1] in ("M", "F"):
            out[parts[0]] = parts[1]
    return out


def committed_eval_speakers() -> set[str]:
    """Talkers of the committed evaluation set (``data/eval-dev``), read from its ground truth.

    >>> sorted(committed_eval_speakers())
    ['2803', '3170', '3853', '6241', '6295', '777']
    """
    import json

    path = Path(__file__).resolve().parent.parent / "data" / "eval-dev" / "ground_truth.jsonl"
    if not path.is_file():
        return set()
    speakers: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            speaker = json.loads(line).get("speaker_id")
            if speaker:
                speakers.add(str(speaker))
    return speakers


TEST_CLEAN_SPEAKERS = frozenset({
    "1089", "1188", "121", "1221", "1284", "1320", "1580", "1995", "2094", "2300",
    "237", "260", "2830", "2961", "3570", "3575", "3729", "4077", "4446", "4507",
    "4970", "4992", "5105", "5142", "5639", "5683", "61", "672", "6829", "6930",
    "7021", "7127", "7176", "7729", "8224", "8230", "8455", "8463", "8555", "908",
})
"""The 40 talkers of LibriSpeech ``test-clean``: the two-talker eval set's voices (``make_eval_t2.py``)."""


def eval_t2_speakers() -> set[str]:
    """Talkers of the two-talker eval set: all of ``test-clean``, plus any listed in ``data/eval-t2/manifest.json``.

    The constant keeps the guard active before the set is generated or wherever it is stored.

    >>> len(eval_t2_speakers()) >= 40
    True
    """
    import json

    speakers = set(TEST_CLEAN_SPEAKERS)
    path = Path(__file__).resolve().parent.parent / "data" / "eval-t2" / "manifest.json"
    try:
        speakers |= set(json.loads(path.read_text(encoding="utf-8"))["speakers"]["ids"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return speakers


def synthetic_corpus(n_speakers: int = 8, utts_per_speaker: int = 6, tag: str = "train") -> SpeechCorpus:
    """Placeholder corpus for smoke runs; ``tag`` keeps train and val disjoint.

    >>> c = synthetic_corpus(3, 2, tag="val")
    >>> len(c), c.speakers[0], len(c.by_speaker[c.speakers[0]])
    (3, 'syn_val_00', 2)
    """
    by_speaker = {}
    for s in range(n_speakers):
        speaker = f"syn_{tag}_{s:02d}"
        by_speaker[speaker] = [
            Utterance(f"{speaker}_{u:02d}", speaker, None, "<synthetic>") for u in range(utts_per_speaker)
        ]
    return SpeechCorpus(by_speaker, synthetic=True)


def build_corpora(data_cfg: Any, data_root: str | Path, need_train: bool = True) -> tuple[SpeechCorpus | None, SpeechCorpus]:
    """Return ``(train, val)`` for a :class:`training.config.DataConfig`.

    ``need_train=False`` (the untrained rungs) skips the training corpus, so
    they run without train-clean-100 on disk; the validation speakers are the
    same either way, so every system is compared on the same sessions.
    """
    if data_cfg.synthetic:
        return synthetic_corpus(tag="train"), synthetic_corpus(tag="val")
    root = Path(data_root)
    train_root, val_root = root / data_cfg.train_root, root / data_cfg.val_root
    if train_root.resolve() == val_root.resolve():
        # One corpus for both (e.g. a dev-clean-only quick run): its non-eval
        # speakers are split 70/30 so training and validation voices stay disjoint.
        pool = librispeech_corpus(val_root, exclude_eval_split=True)
        speakers = list(pool.speakers)
        random.Random(EVAL_SPLIT_SEED + 1).shuffle(speakers)
        cut = max(2, int(round(0.7 * len(speakers))))
        train = SpeechCorpus({s: pool.by_speaker[s] for s in speakers[:cut]}, sex=pool.sex)
        val = SpeechCorpus({s: pool.by_speaker[s] for s in speakers[cut:]}, sex=pool.sex)
    else:
        train = librispeech_corpus(train_root) if need_train else None
        val = librispeech_corpus(val_root, exclude_eval_split=True)
    overlap = set(train.speakers if train else ()) & set(val.speakers)
    if overlap:
        raise ValueError(f"train and validation corpora share speakers {sorted(overlap)[:5]}")
    leaked = (set(train.speakers if train else ()) | set(val.speakers)) & eval_t2_speakers()
    if leaked:
        raise ValueError(f"speakers {sorted(leaked)[:5]} belong to the two-talker eval set (data/eval-t2)")
    return (train if need_train else None), val
