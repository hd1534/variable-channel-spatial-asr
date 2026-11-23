#!/usr/bin/env python3
"""Word times for LibriSpeech utterances, once, with Qwen3-ForcedAligner-0.6B (Qwen venv).

Track C's training sessions cut turns between words (:mod:`trackc.scenes`),
which needs the time of every word.  The aligner is the official Qwen one
(Apache-2.0, pinned below), run on each utterance with its LibriSpeech
transcript.  An utterance is kept only when the aligner returns one time per
transcript word, so a turn's text is always the transcript's own words.

Output: one JSON line per utterance, appended as it goes, so a rerun after a
Colab disconnect continues where it stopped::

    {"utt_id": "103-1240-0000", "words": [["CHAPTER", 0.48, 0.96], ...]}

Usage::

    python3 -m trackc.align --corpus /content/data/LibriSpeech/train-clean-100 --out ALIGN/train-clean-100.jsonl
    python3 -m trackc.align --corpus /content/data/LibriSpeech/dev-clean --exclude-eval-split --out ALIGN/dev-clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, "") and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spec  # noqa: E402

ALIGNER_MODEL: str = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNER_REVISION: str = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
LANGUAGE: str = "English"


def done_ids(path: Path) -> set[str]:
    """Utterances already in the index; a line cut by a crash is dropped (and redone)."""
    if not path.is_file():
        return set()
    out, good = set(), []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.add(json.loads(line)["utt_id"])
            good.append(line)
        except (ValueError, KeyError):
            break
    path.write_text("".join(g + "\n" for g in good), encoding="utf-8")
    return out


def match_words(transcript: str, items: Sequence[Any]) -> list[list[Any]] | None:
    """Pair each transcript word with the aligner's time, or ``None`` when the counts differ.

    >>> from types import SimpleNamespace as I
    >>> match_words("HE SAID", [I(text="HE", start_time=0.1, end_time=0.3), I(text="SAID", start_time=0.4, end_time=0.7)])
    [['HE', 0.1, 0.3], ['SAID', 0.4, 0.7]]
    >>> match_words("DON'T GO", [I(text="DONT", start_time=0.1, end_time=0.3)]) is None
    True
    """
    words = transcript.split()
    if len(words) != len(items):
        return None
    return [[w, round(float(i.start_time), 3), round(float(i.end_time), 3)] for w, i in zip(words, items)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, required=True, help="a LibriSpeech subset directory")
    parser.add_argument("--exclude-eval-split", action="store_true", help="dev-clean: leave out the eval speakers")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="first N utterances only (smoke runs)")
    args = parser.parse_args(argv)

    import torch
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

    from modern_asr import model_snapshot
    from training.corpus import librispeech_corpus

    corpus = librispeech_corpus(args.corpus, exclude_eval_split=args.exclude_eval_split)
    utts = sorted((u for us in corpus.by_speaker.values() for u in us), key=lambda u: u.utt_id)
    utts = utts[: args.limit or None]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    finished = done_ids(args.out)
    todo = [u for u in utts if u.utt_id not in finished]
    todo.sort(key=corpus.n_samples)  # similar lengths per batch: less padding
    print(f"[align] {len(utts)} utterances, {len(finished)} already aligned, {len(todo)} to go", flush=True)
    if not todo:
        return 0
    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.get_device_capability(args.device)[0] >= 8 \
        else (torch.float16 if args.device.startswith("cuda") else torch.float32)
    aligner = Qwen3ForcedAligner.from_pretrained(model_snapshot(ALIGNER_MODEL, ALIGNER_REVISION), dtype=dtype,
                                                 device_map=args.device)
    kept = skipped = 0
    started = time.time()
    with args.out.open("a", encoding="utf-8") as fh:
        for i in range(0, len(todo), args.batch):
            batch = todo[i : i + args.batch]
            audio = [(corpus.read(u), spec.SAMPLE_RATE_HZ) for u in batch]
            with torch.inference_mode():
                results = aligner.align(audio=audio, text=[u.transcript for u in batch], language=LANGUAGE)
            for u, result in zip(batch, results):
                words = match_words(u.transcript, list(result))
                if words is None:
                    skipped += 1
                    continue
                fh.write(json.dumps({"utt_id": u.utt_id, "words": words}) + "\n")
                kept += 1
            fh.flush()
            done = i + len(batch)
            if done % (args.batch * 50) < args.batch or done == len(todo):
                rate = done / max(time.time() - started, 1e-6)
                print(f"[align] {done}/{len(todo)} ({rate:.1f} utt/s), kept {kept}, word count mismatch {skipped}",
                      flush=True)
    print(f"[align] done: kept {kept}, skipped {skipped} (aligner word count differed from the transcript)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
