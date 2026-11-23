"""Word-aligned two-talker sessions for Track C.

The other systems' training sessions cut utterances at arbitrary samples, so a
turn has no transcript.  Here every turn is a run of whole words of one
utterance, cut in a gap between words (:class:`AlignedReader`), so each turn
comes with exactly the words it contains.  Everything else -- rooms, displaced
turn positions, co-location, noise, masks, the label grids -- is the shared
generator of :mod:`training.scenes`, in the eval-like layout (two talkers, one
near and one far, 2-4 long turns).

The word times come from an index built once by :mod:`trackc.align`:
``{"utt_id": ..., "words": [[word, start_sec, end_sec], ...]}`` per line.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import spec
from training.corpus import SpeechCorpus, Utterance
from training.scenes import FADE_SEC, Session, draw_session, sample_mask, train_recipe, val_recipe
from trackc import text as tt

SR = spec.SAMPLE_RATE_HZ
MIN_CUT_GAP_SEC: float = 0.08
"""A turn may only end in a gap between words at least this long (the aligner's time step),
so a cut never clips a word."""

Words = list[tuple[str, float, float]]


def load_alignments(paths: Iterable[str | Path]) -> dict[str, Words]:
    """``{utt_id: [(word, start, end)]}`` from one or more index files (later files win)."""
    out: dict[str, Words] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["utt_id"]] = [(str(w), float(a), float(b)) for w, a, b in row["words"]]
    return out


def aligned_corpus(corpus: SpeechCorpus, alignments: Mapping[str, Words]) -> SpeechCorpus:
    """The corpus restricted to utterances with word times (speakers left with none are dropped)."""
    by_speaker = {s: [u for u in utts if alignments.get(u.utt_id)] for s, utts in corpus.by_speaker.items()}
    return SpeechCorpus({s: u for s, u in by_speaker.items() if u}, synthetic=corpus.synthetic, sex=corpus.sex)


class AlignedReader:
    """Consecutive runs of whole words of one utterance per talker.

    ``take(seconds)`` returns the longest run of whole words that fits in
    ``seconds`` (one word at least, then to the next legal cut), cut in the
    middle of a gap of at least :data:`MIN_CUT_GAP_SEC`.  ``last_text`` holds
    the words of the piece just returned and ``taken`` the length of every piece.
    Like the shared reader, a talker's next turn continues where the last one
    stopped; a new utterance (from a random legal start) begins only when less
    than half a turn is left.

    Calling an instance with ``(rng, corpus, utterances)`` makes the reader of
    one talker, the factory form :func:`training.scenes.draw_session` expects.
    """

    def __init__(self, alignments: Mapping[str, Words]):
        self.alignments = alignments

    def __call__(self, rng: random.Random, corpus: SpeechCorpus, utts: Sequence[Utterance]) -> "AlignedReader":
        reader = AlignedReader(self.alignments)
        reader.rng, reader.corpus = rng, corpus
        reader.utts = [u for u in utts if self.alignments.get(u.utt_id)]
        if not reader.utts:
            raise ValueError("a talker without word-aligned utterances")
        reader.utt, reader.words, reader.cuts, reader.next_word = None, [], [], 0
        reader.last_text, reader.taken = "", []
        return reader

    @staticmethod
    def cut_times(words: Words, total_sec: float) -> list[float | None]:
        """Cut time before word ``i`` for ``i = 0..n`` (``None`` where the gap is too short).

        >>> AlignedReader.cut_times([("a", 0.2, 0.5), ("b", 0.52, 0.9), ("c", 1.2, 1.5)], 2.0)
        [0.0, None, 1.05, 2.0]
        """
        cuts: list[float | None] = [0.0]
        for (_, _, end), (_, start, _) in zip(words, words[1:]):
            cuts.append(round((end + start) / 2, 6) if start - end >= MIN_CUT_GAP_SEC else None)
        cuts.append(total_sec)
        return cuts

    def _left(self) -> float:
        if self.utt is None or self.next_word >= len(self.words):
            return 0.0
        return self.cuts[-1] - self.cuts[self.next_word]

    def _new_utterance(self, seconds: float) -> None:
        self.utt = self.rng.choice(self.utts)
        self.words = self.alignments[self.utt.utt_id]
        self.cuts = self.cut_times(self.words, self.corpus.n_samples(self.utt) / SR)
        starts = [i for i in range(len(self.words)) if self.cuts[i] is not None]
        fits = [i for i in starts if self.cuts[-1] - self.cuts[i] >= seconds] or starts[:1]
        self.next_word = self.rng.choice(fits)

    def take(self, seconds: float):  # noqa: ANN201 - (utterance, numpy array)
        if self._left() < seconds / 2:
            self._new_utterance(seconds)
        first, cuts = self.next_word, self.cuts
        begin = cuts[first]
        legal = [j for j in range(first + 1, len(self.words) + 1) if cuts[j] is not None]
        fitting = [j for j in legal if cuts[j] - begin <= seconds]
        last = fitting[-1] if fitting else legal[0]
        self.next_word = last
        x = np.array(self.corpus.read(self.utt, int(round(begin * SR)), int(round(cuts[last] * SR))), dtype=np.float64)
        fade = min(int(FADE_SEC * SR), len(x) // 2)
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade)
            x[:fade] *= ramp
            x[-fade:] *= ramp[::-1]
        self.last_text = " ".join(w for w, _, _ in self.words[first:last])
        self.taken.append(len(x))
        return self.utt, x


def text_recipe(cfg: Any, kind: str | None = None):  # noqa: ANN201
    """Training (``kind=None``) or validation sessions, always in the two-talker eval layout."""
    if kind is None:
        return replace(train_recipe(cfg), n_speakers=(2, 2), eval_layout_prob=1.0)
    return replace(val_recipe(cfg, kind), n_speakers=(2, 2), eval_layout_prob=1.0)


def draw_text_session(rng: random.Random, corpus: SpeechCorpus, bank: Any, cfg: Any, recipe: Any,
                      reader: AlignedReader, attempts: int = 20) -> Session:
    """A two-talker session whose every turn kept all of its words.

    A turn the session end cut short would carry words that are not in the
    audio, so such a session is drawn again from the same random stream.
    """
    for _ in range(attempts):
        readers: list[AlignedReader] = []

        def make(r: random.Random, c: SpeechCorpus, u: Sequence[Utterance]) -> AlignedReader:
            readers.append(reader(r, c, u))
            return readers[-1]

        session = draw_session(rng, corpus, bank, cfg, recipe, make_reader=make)
        seen = [0] * len(readers)
        whole = True
        for turn in session.turns:
            k = seen[turn.speaker]
            seen[turn.speaker] += 1
            placed = int(round((turn.end_sec - turn.start_sec) * SR))
            if placed < readers[turn.speaker].taken[k]:
                whole = False
        if whole and len({t.speaker for t in session.turns}) == 2:
            return session
    raise RuntimeError("could not draw a two-talker session whose turns keep all their words")


def item(session: Session, bank: Any, mask_bits: str, dry_dtype: Any = np.float16) -> dict[str, Any]:
    """The renderer inputs of :func:`training.data._item` plus the tagged target and slot labels."""
    from training.data import _item

    out = _item(session, bank, mask_bits, dry_dtype=dry_dtype)
    speakers = [t.speaker for t in session.turns]
    out["text"] = tt.target_text(session.turns)
    out["slot_labels"] = np.asarray(tt.slot_labels(session.labels_eval, speakers), dtype=np.int64)
    return out


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """:func:`training.data.collate` with the strings kept as lists."""
    from training.data import collate as tensor_collate

    strings = {k: [it[k] for it in items] for k in ("text", "meta") if k in items[0]}
    out = tensor_collate([{k: v for k, v in it.items() if k not in strings} for it in items])
    out.update(strings)
    return out


class TextTrainStream:
    """Endless training items (an ``IterableDataset`` once torch is imported)."""

    def __new__(cls, *args: Any, **kwargs: Any):  # noqa: ANN204
        from torch.utils.data import IterableDataset, get_worker_info

        from training.rirbank import RirBank

        class _Stream(IterableDataset):
            def __init__(self, data_cfg: Any, corpus: SpeechCorpus, bank_path: str, alignments: Mapping[str, Words],
                         seed: int, start_step: int = 0) -> None:
                super().__init__()
                self.cfg, self.corpus, self.bank_path = data_cfg, corpus, bank_path
                self.reader = AlignedReader(alignments)
                self.seed, self.start_step = seed, start_step
                self.recipe = text_recipe(data_cfg)

            def __iter__(self):  # noqa: ANN204
                info = get_worker_info()
                rng = random.Random(f"trackc-train:{self.seed}:{self.start_step}:{info.id if info else 0}")
                bank = RirBank(self.bank_path)
                while True:
                    session = draw_text_session(rng, self.corpus, bank, self.cfg, self.recipe, self.reader)
                    yield item(session, bank, sample_mask(rng, self.cfg.m_weights))

        return _Stream(*args, **kwargs)


class TextValSessions:
    """Fixed validation sessions, a pure function of ``(val_seed, index)`` (a ``Dataset``)."""

    def __new__(cls, *args: Any, **kwargs: Any):  # noqa: ANN204
        from torch.utils.data import Dataset

        from training.rirbank import RirBank

        class _Val(Dataset):
            def __init__(self, data_cfg: Any, corpus: SpeechCorpus, bank_path: str, alignments: Mapping[str, Words],
                         n_sessions: int, kind: str = "main") -> None:
                self.cfg, self.corpus, self.bank_path, self.kind = data_cfg, corpus, bank_path, kind
                self.reader = AlignedReader(alignments)
                self.recipe = text_recipe(data_cfg, kind)
                self.n = n_sessions
                self._bank = None

            def __len__(self) -> int:
                return self.n

            def __getitem__(self, index: int) -> dict[str, Any]:
                if self._bank is None:
                    self._bank = RirBank(self.bank_path)
                rng = random.Random(f"trackc-val:{self.cfg.val_seed}:{index}")
                session = draw_text_session(rng, self.corpus, self._bank, self.cfg, self.recipe, self.reader)
                out = item(session, self._bank, "1" * spec.N_CHANNELS_MAX, dry_dtype=np.float32)
                out["index"] = index
                talker_text: dict[int, list[str]] = {}
                for t in session.turns:
                    talker_text.setdefault(t.speaker, []).append(tt.normalise(t.text))
                out["meta"] = {"turns": [(t.speaker, round(t.start_sec, 3), round(t.end_sec, 3)) for t in session.turns],
                               "speaker_ids": session.speaker_ids,
                               "talker_text": {k: " ".join(v) for k, v in talker_text.items()}}
                return out

        return _Val(*args, **kwargs)

