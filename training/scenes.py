"""Multi-talker turn-taking sessions, drawn on the fly from a room bank.

A session follows the same rules as the evaluation sessions: 20 s, 4 channels
at 16 kHz, talkers at their positions in one room taking turns with a silence
between turns (so at most one talker is active per grid instant), one common
gain for all channels, a capped per-channel mismatch, a single anti-clip scalar
and 16-bit quantisation (:func:`simulate.render_session`,
:func:`simulate.write_wav_pcm16`).

What a session looks like is set by a :class:`Recipe`.  Training mixes:

* two layouts -- *eval-like* (two talkers, one near and one far, 2-4 long turns,
  as in the planned two-talker eval) and *dense* (2-4 talkers, short turns);
  in both the turn order is random, so a pause never tells who speaks next;
* *displaced turns* -- each turn comes from one of a few copies of the talker's
  position (each 0.1-0.3 m from the first), and a talker who speaks again moves
  to another copy, so the exact reflection pattern of a position cannot
  identify a talker;
* *co-located talkers* -- now and then a later talker speaks from an earlier
  talker's position, so "same position" does not imply "same talker";
* *diffuse noise* at a random SNR (applied on the device, :mod:`training.render`).

Validation pins the conditions of the eval's main set (eval-like layout,
displaced turns, diffuse noise at 5/10/20 dB; :func:`val_recipe`); the static
and co-located variants of the same sessions are diagnostics.

Three label grids are produced over ``T_all`` (80 instants):

``labels``       the talker heard in the *causal* window ``[t_k - 0.25, t_k)``,
                 i.e. what the chunk ending at ``t_k`` actually contains.  The
                 speaker loss is trained on it.
``pos_labels``   the *position* (bank source) of that talker -- shared by
                 co-located talkers.  The spatial branch is trained on it.
``labels_eval``  the official rule (:func:`simulate.activity_grid`, spec
                 ``T_EVAL_RULE``): the window ``[t_k, t_k + 0.25)`` *after*
                 ``t_k``.  Validation accuracy is scored on it.

Activity is read from each talker's dry signal, gated at
``spec.VAD_REL_THRESHOLD_DB`` below that talker's ``spec.VAD_REF_PERCENTILE``
level, so a reverberant tail never extends a talker's labelled span.  ``-1``
marks silence.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import spec
from simulate import CHANNEL_GAIN_MISMATCH_DB, PEAK_CEILING, SESSION_GAIN_DBFS_RANGE
from training.corpus import SpeechCorpus, Utterance
from training.rirbank import SOURCES_PER_ROOM, RirBank

SR = spec.SAMPLE_RATE_HZ
SESSION_SAMPLES = int(round(spec.SESSION_LENGTH_SEC * SR))
GRID = int(round(spec.DELTA_T_SEC * SR))
N_GRID = spec.N_GRID_POINTS
DRY_RMS = 0.05
"""Common dry level each turn is normalised to before the talker's own gain."""
FADE_SEC = 0.01
"""Raised-ramp length at both ends of a cut, so a cut never clicks."""
TAIL_SEC = 0.25
"""The last turn must start this long before the session ends."""
NEAR_FAR_SPLIT_M = spec.DISTANCE_BIN_EDGES_M[spec.K_DISTANCE // 2]
"""1.9 m: the eval generator puts one talker in the near distance bins and one in the far bins."""
EVAL_TURN_COUNT_WEIGHTS = {2: 1, 3: 1, 4: 1}
"""Turns per session, as evenly spread as in the two-talker eval set (``make_eval_t2.TURN_COUNTS``)."""


@dataclass(frozen=True)
class Recipe:
    """How sessions are drawn."""

    n_speakers: tuple[int, int]
    eval_layout_prob: float
    colocate_prob: float
    jitter: bool
    noise_snr_db: tuple[float, float] | None
    all_colocated: bool = False
    snr_levels_db: tuple[float, ...] | None = None
    """If set, the SNR is one of these (as in the eval set) instead of uniform in ``noise_snr_db``."""


def train_recipe(cfg: Any) -> Recipe:
    snr = tuple(cfg.noise_snr_db) if cfg.noise_snr_db else None
    return Recipe(tuple(cfg.n_speakers), cfg.eval_layout_prob, cfg.colocate_prob, cfg.bank_variants > 1, snr)


def val_recipe(cfg: Any, kind: str = "main") -> Recipe:
    """The conditions of the two-talker eval set (``make_eval_t2.py``), on validation voices.

    ``main``: displaced turns and diffuse noise at one of ``val_snr_levels_db`` --
    what thresholds and checkpoints are chosen on, like the eval's main set;
    ``static``: the same sessions with talkers at rest and no noise (the eval's
    static set); ``colocated``: the main sessions with every talker at the first
    talker's position.
    """
    levels = tuple(cfg.val_snr_levels_db)
    main = Recipe(tuple(cfg.val_n_speakers), cfg.val_eval_layout_prob, 0.0, cfg.bank_variants > 1,
                  (min(levels), max(levels)), snr_levels_db=levels)
    if kind == "main":
        return main
    if kind == "static":
        return replace(main, jitter=False, noise_snr_db=None, snr_levels_db=None)
    if kind == "colocated":
        return replace(main, all_colocated=True)
    raise ValueError(f"unknown validation kind {kind!r}")


@dataclass
class Turn:
    speaker: int
    start_sec: float
    end_sec: float
    utt_id: str
    variant: int


@dataclass
class Session:
    """One drawn session *before* the room is applied.

    The convolution is done by :func:`render_numpy` (reference) or by
    :func:`training.render.render_batch` on the training device (fast path).
    """

    dry: Any
    """float32 ``[N, SESSION_SAMPLES]``: each talker's turns, placed, before the room."""
    variant_map: Any
    """int8 ``[N, SESSION_SAMPLES]``: which displaced copy of its position each sample of a talker is rendered from."""
    room_index: int
    sources: list[int]
    """Bank position of each talker (two talkers share one when co-located)."""
    target_rms: float
    channel_gain: Any
    snr_db: float
    """Diffuse-noise SNR, ``nan`` for none."""
    noise_seed: int
    labels: Any
    pos_labels: Any
    labels_eval: Any
    mic_xy: Any
    """float32 ``[4, 2]`` in the array frame (origin = array reference point)."""
    n_speakers: int
    speaker_ids: list[str]
    source_azimuth_deg: list[float]
    source_distance_m: list[float]
    rt60_requested_s: float
    layout: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def n_changes(self) -> int:
        return sum(1 for a, b in zip(self.turns, self.turns[1:]) if a.speaker != b.speaker)


def _fit_durations(rng: random.Random, n_turns: int, turn_sec: Sequence[float], cfg: Any):  # noqa: ANN202
    """``(lead_in, durations, gaps)`` fitting 20 s, or ``None`` if minimum-length turns do not fit.

    When the drawn turns overrun the session only the part above the minimum is
    shrunk, so every turn stays at least ``turn_sec[0]`` long.

    >>> from types import SimpleNamespace
    >>> cfg = SimpleNamespace(lead_in_sec=(1.0, 1.0), gap_sec=(0.5, 0.5))
    >>> lead, d, g = _fit_durations(random.Random(0), 8, (1.0, 4.0), cfg)
    >>> round(lead + sum(d) + sum(g) + TAIL_SEC, 6) <= spec.SESSION_LENGTH_SEC, min(d) >= 1.0
    (True, True)
    >>> _fit_durations(random.Random(0), 13, (1.0, 4.0), cfg) is None
    True
    """
    lead = rng.uniform(*cfg.lead_in_sec)
    gaps = [rng.uniform(*cfg.gap_sec) for _ in range(n_turns - 1)]
    budget = spec.SESSION_LENGTH_SEC - lead - TAIL_SEC - sum(gaps)
    floor = turn_sec[0]
    if budget < n_turns * floor:
        return None
    durations = [rng.uniform(*turn_sec) for _ in range(n_turns)]
    total = sum(durations)
    if total > budget:
        shrink = (budget - n_turns * floor) / (total - n_turns * floor)
        durations = [floor + (d - floor) * shrink for d in durations]
    return lead, durations, gaps


def _dense_order(rng: random.Random, n_speakers: int, turns_each: int) -> list[int]:
    """Every talker ``turns_each`` times, in random order -- the same talker may go twice in a row.

    >>> sorted(_dense_order(random.Random(0), 3, 2))
    [0, 0, 1, 1, 2, 2]
    """
    order = [talker for talker in range(n_speakers) for _ in range(turns_each)]
    rng.shuffle(order)
    return order


def _eval_order(rng: random.Random) -> list[int]:
    """2-4 turns of two talkers in random order, each talker at least once.

    >>> orders = [_eval_order(random.Random(i)) for i in range(200)]
    >>> sorted({len(o) for o in orders}), all(set(o) == {0, 1} for o in orders)
    ([2, 3, 4], True)
    """
    counts, weights = zip(*EVAL_TURN_COUNT_WEIGHTS.items())
    n = rng.choices(counts, weights=weights)[0]
    while True:
        order = [rng.randrange(2) for _ in range(n)]
        if set(order) == {0, 1}:
            return order


class _TalkerReader:
    """Consecutive pieces of one utterance per talker, like the eval's pause-split halves.

    A talker's next turn continues where its last one stopped; only when the
    utterance runs out does it move on to another one.
    """

    def __init__(self, rng: random.Random, corpus: SpeechCorpus, utts: Sequence[Utterance]) -> None:
        self.rng, self.corpus, self.utts = rng, corpus, list(utts)
        self.utt: Utterance | None = None
        self.pos = self.total = 0

    def take(self, seconds: float):  # noqa: ANN201 - (utterance, numpy array)
        import numpy as np

        n = int(round(seconds * SR))
        if self.utt is None or self.total - self.pos < n // 2:
            self.utt = self.rng.choice(self.utts)
            self.total = self.corpus.n_samples(self.utt)
            self.pos = self.rng.randrange(0, self.total - n + 1) if self.total > n else 0
        start, stop = self.pos, min(self.total, self.pos + n)
        self.pos = stop
        x = np.array(self.corpus.read(self.utt, start, stop), dtype=np.float64)
        fade = min(int(FADE_SEC * SR), len(x) // 2)
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade)
            x[:fade] *= ramp
            x[-fade:] *= ramp[::-1]
        return self.utt, x


def _active_rms(x):  # noqa: ANN001, ANN202
    """RMS over the speech-dense part, so a piece with a pause is not boosted."""
    import numpy as np

    frame = 400
    n = len(x) // frame
    if n == 0:
        return float(np.sqrt(np.mean(x**2)) + 1e-12)
    e = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-12
    keep = e >= np.percentile(e, 90) * 10 ** (-30 / 20)
    return float(np.sqrt(np.mean(e[keep] ** 2)))


def _window_rms(placed, offsets: Sequence[int]):  # noqa: ANN001, ANN202
    import numpy as np

    out = np.zeros(len(offsets))
    for k, a in enumerate(offsets):
        lo, hi = max(0, a), min(len(placed), a + GRID)
        if hi > lo:
            out[k] = math.sqrt(float(np.mean(placed[lo:hi] ** 2)))
    return out


def grid_labels(dry_by_speaker) -> tuple[Any, Any]:  # noqa: ANN001
    """``(labels, labels_eval)`` from ``[N, SESSION_SAMPLES]`` placed dry talkers.

    >>> import numpy as np
    >>> dry = np.zeros((2, SESSION_SAMPLES)); dry[0, 16000:48000] = 0.1; dry[1, 64000:96000] = 0.1
    >>> causal, official = grid_labels(dry)
    >>> causal[[4, 5, 12, 16, 17]].tolist()
    [-1, 0, 0, -1, 1]
    >>> official[[3, 4, 11, 12, 16]].tolist()
    [-1, 0, 0, -1, 1]
    """
    import numpy as np

    n = dry_by_speaker.shape[0]
    causal_rms = np.stack([_window_rms(dry_by_speaker[j], [k * GRID - GRID for k in range(N_GRID)]) for j in range(n)])
    causal_rms[:, 0] = 0.0
    eval_rms = np.stack([_window_rms(dry_by_speaker[j], [k * GRID for k in range(N_GRID)]) for j in range(n)])
    out = []
    for rms in (causal_rms, eval_rms):
        active = np.zeros_like(rms, dtype=bool)
        for j in range(n):
            nonzero = eval_rms[j][eval_rms[j] > 0]
            if nonzero.size:
                ref = float(np.percentile(nonzero, spec.VAD_REF_PERCENTILE))
                active[j] = rms[j] > ref * 10 ** (spec.VAD_REL_THRESHOLD_DB / 20)
        lab = np.where(active.any(0), np.argmax(np.where(active, rms, -1.0), axis=0), -1)
        out.append(lab.astype(np.int64))
    return out[0], out[1]


def _has_near_and_far(bank: RirBank, room: int) -> bool:
    dist = [s["distance_m"] for s in bank.rooms[room]["sources"]]
    return any(d < NEAR_FAR_SPLIT_M for d in dist) and any(d >= NEAR_FAR_SPLIT_M for d in dist)


def _assign_sources(rng: random.Random, bank: RirBank, room: int, n: int, eval_layout: bool) -> list[int]:
    """Distinct bank positions; for the eval layout one near (< 1.9 m) and one far when the room has both."""
    if eval_layout:
        dist = [s["distance_m"] for s in bank.rooms[room]["sources"]]
        near = [i for i, d in enumerate(dist) if d < NEAR_FAR_SPLIT_M]
        far = [i for i, d in enumerate(dist) if d >= NEAR_FAR_SPLIT_M]
        if near and far:
            pair = [rng.choice(near), rng.choice(far)]
            rng.shuffle(pair)
            return pair
    return rng.sample(range(SOURCES_PER_ROOM), n)


def draw_session(rng: random.Random, corpus: SpeechCorpus, bank: RirBank, cfg: Any, recipe: Recipe) -> Session:
    """Draw one session.  ``rng`` fully determines it (given corpus, bank and recipe).

    Jitter variants, noise and co-location come from their own random stream,
    so the main, robust and co-located validation sets are the *same* sessions
    under different conditions (a paired comparison).
    """
    import numpy as np

    aux = random.Random(rng.getrandbits(64))
    room = rng.randrange(len(bank))
    eval_layout = rng.random() < recipe.eval_layout_prob
    for _ in range(50):  # the eval layout wants one near and one far talker; most rooms have both
        if not eval_layout or _has_near_and_far(bank, room):
            break
        room = rng.randrange(len(bank))
    if eval_layout:
        n, order = 2, _eval_order(rng)
        plan = _fit_durations(rng, len(order), cfg.eval_turn_sec, cfg)
    else:
        n = rng.randint(*recipe.n_speakers)
        turns_each = rng.randint(*cfg.turns_per_speaker)
        while True:
            order = _dense_order(rng, n, turns_each)
            plan = _fit_durations(rng, len(order), cfg.turn_sec, cfg)
            if plan is not None or turns_each == 1:
                break
            turns_each -= 1
    if plan is None:
        raise ValueError(f"{n} talkers do not fit {spec.SESSION_LENGTH_SEC} s with the configured turn lengths")
    lead, durations, gaps = plan
    sources = _assign_sources(rng, bank, room, n, eval_layout)
    speakers = rng.sample(list(corpus.speakers), n)
    gain = [10 ** (rng.uniform(-cfg.speaker_gain_db, cfg.speaker_gain_db) / 20) for _ in range(n)]
    target_rms = 10 ** (rng.uniform(*SESSION_GAIN_DBFS_RANGE) / 20)
    channel_gain = np.asarray([10 ** (rng.uniform(-CHANNEL_GAIN_MISMATCH_DB, CHANNEL_GAIN_MISMATCH_DB) / 20)
                               for _ in range(spec.N_CHANNELS_MAX)], dtype=np.float32)

    if n >= 2 and aux.random() < recipe.colocate_prob:
        later = aux.randrange(1, n)
        sources[later] = sources[aux.randrange(0, later)]
    if recipe.all_colocated:
        sources = [sources[0]] * n

    readers = [_TalkerReader(rng, corpus, corpus.by_speaker[s]) for s in speakers]
    dry = np.zeros((n, SESSION_SAMPLES), dtype=np.float64)
    variant_map = np.zeros((n, SESSION_SAMPLES), dtype=np.int8)
    turns: list[Turn] = []
    last_variant: dict[int, int] = {}
    t = lead
    for i, talker in enumerate(order):
        utt, seg = readers[talker].take(durations[i])
        seg = seg / _active_rms(seg) * DRY_RMS * gain[talker]
        variant = 0
        if recipe.jitter:  # a talker who speaks again moves to another copy, as in make_eval_t2._turn_copies
            options = [v for v in range(bank.variants) if v != last_variant.get(talker)] or [0]
            variant = last_variant[talker] = aux.choice(options)
        a = int(round(t * SR))
        b = min(SESSION_SAMPLES, a + len(seg))
        if b <= a:
            break
        dry[talker, a:b] += seg[: b - a]
        variant_map[talker, a:b] = variant
        turns.append(Turn(talker, a / SR, b / SR, utt.utt_id, variant))
        t = b / SR + (gaps[i] if i < len(gaps) else 0.0)
        if t >= spec.SESSION_LENGTH_SEC - TAIL_SEC:
            break
    snr = float("nan")
    if recipe.snr_levels_db:
        snr = float(aux.choice(recipe.snr_levels_db))
    elif recipe.noise_snr_db:
        snr = aux.uniform(*recipe.noise_snr_db)
    noise_seed = aux.getrandbits(31)

    labels, labels_eval = grid_labels(dry)
    src = np.asarray(sources)
    pos_labels = np.where(labels >= 0, src[np.clip(labels, 0, None)], -1).astype(np.int64)
    meta = bank.rooms[room]
    return Session(
        dry=dry.astype(np.float32),
        variant_map=variant_map,
        room_index=room,
        sources=list(sources),
        target_rms=float(target_rms),
        channel_gain=channel_gain,
        snr_db=snr,
        noise_seed=int(noise_seed),
        labels=labels,
        pos_labels=pos_labels,
        labels_eval=labels_eval,
        mic_xy=np.asarray(meta["mic_xy_m"], dtype=np.float32),
        n_speakers=n,
        speaker_ids=speakers,
        source_azimuth_deg=[meta["sources"][s]["azimuth_deg"] for s in sources],
        source_distance_m=[meta["sources"][s]["distance_m"] for s in sources],
        rt60_requested_s=meta["rt60_requested_s"],
        layout="eval" if eval_layout else "dense",
        turns=turns,
    )


NOISE_N_FFT = 512
NOISE_HOP = 256
PINK_KNEE_HZ = 100.0


def diffuse_noise_numpy(mic_xy, n_samples: int, seed: int):  # noqa: ANN001, ANN201 - numpy arrays
    """Spherically isotropic pink noise for every microphone, ``[C, n_samples]``.

    Per STFT bin, white noise is mixed through the Cholesky factor of the
    diffuse-field coherence ``sinc(2 f d_ij / c)`` and shaped pink, then
    overlap-added.  :func:`training.render.diffuse_noise` is the same
    construction on the training device; levels are always rescaled to an SNR,
    so only the spatial and spectral shape matters.
    """
    import numpy as np

    xy = np.asarray(mic_xy, dtype=np.float64)
    n_bins = NOISE_N_FFT // 2 + 1
    n_frames = n_samples // NOISE_HOP + 2
    freqs = np.arange(n_bins) * SR / NOISE_N_FFT
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    coherence = np.sinc(2.0 * freqs[:, None, None] * dist[None] / spec.SPEED_OF_SOUND_M_S) + 1e-4 * np.eye(len(xy))
    mixing = np.linalg.cholesky(coherence)  # [F, C, C]
    rng = np.random.default_rng(seed)
    white = (rng.standard_normal((n_bins, n_frames, len(xy))) + 1j * rng.standard_normal((n_bins, n_frames, len(xy)))) / np.sqrt(2.0)
    spectra = np.einsum("fij,ftj->fti", mixing, white)
    spectra *= (np.maximum(freqs, PINK_KNEE_HZ) / PINK_KNEE_HZ)[:, None, None] ** -0.5
    window = np.hanning(NOISE_N_FFT + 1)[:-1]
    frames = np.fft.irfft(spectra, NOISE_N_FFT, axis=0) * window[:, None, None]  # [N_FFT, T, C]
    out = np.zeros((len(xy), (n_frames - 1) * NOISE_HOP + NOISE_N_FFT))
    for t in range(n_frames):
        out[:, t * NOISE_HOP : t * NOISE_HOP + NOISE_N_FFT] += frames[:, t, :].T
    return out[:, NOISE_N_FFT // 2 : NOISE_N_FFT // 2 + n_samples]


def render_arrays(dry, variant_map, rirs, target_rms: float, channel_gain, snr_db: float = float("nan"),  # noqa: ANN001
                  noise=None, quantize: bool = True):  # noqa: ANN001, ANN201 - numpy array
    """``float32 [4, S]`` from dry talkers ``[N, S]``, their variant map and responses ``[N][K] -> [4, L]``.

    Same order as :func:`simulate.render_session`: convolve each sample of a
    talker with the response of the copy of its position it was spoken from,
    add diffuse ``noise`` at ``snr_db`` (relative to the reverberant speech over
    all channels), scale the session to its target level, apply the
    per-channel mismatch, one anti-clip scalar for all channels, then 16-bit
    PCM quantisation as :func:`simulate.write_wav_pcm16` writes it (and a WAV
    reader reads it back: ``int16 / 32768``).  ``quantize=False`` stops before
    that last step, for a caller that writes the WAV itself.
    """
    import numpy as np

    n_samples = dry.shape[1]
    longest = max(len(rirs[j][v][m]) for j in range(len(rirs)) for v in range(len(rirs[j])) for m in range(len(rirs[j][v])))
    size = 1 << (n_samples + longest - 2).bit_length()
    spectrum = 0.0
    for j in range(dry.shape[0]):
        for v in range(len(rirs[j])):
            part = np.asarray(dry[j], dtype=np.float64) * (variant_map[j] == v)
            if part.any():
                h = np.stack([np.fft.rfft(np.asarray(r, dtype=np.float64), size) for r in rirs[j][v]])
                spectrum = spectrum + h * np.fft.rfft(part, size)[None, :]
    wet = np.fft.irfft(spectrum, size, axis=-1)[:, :n_samples]
    if noise is not None and snr_db == snr_db:
        scale = math.sqrt(float(np.mean(wet**2)) / max(float(np.mean(noise**2)), 1e-20) / 10 ** (snr_db / 10))
        wet = wet + noise[:, :n_samples] * scale
    wet *= target_rms / max(float(np.sqrt(np.mean(wet**2))), 1e-12)
    wet *= np.asarray(channel_gain, dtype=np.float64)[:, None]
    peak = float(np.abs(wet).max())
    if peak > PEAK_CEILING:
        wet *= PEAK_CEILING / peak
    if not quantize:
        return wet
    return (np.round(np.clip(wet, -1.0, 1.0) * 32767.0) / 32768.0).astype(np.float32)


def render_numpy(session: Session, bank: RirBank):  # noqa: ANN201 - numpy array
    """Reference renderer for a training session, without noise: ``float32 [4, SESSION_SAMPLES]``.

    :func:`training.render.render_batch` does the same on the training device
    and is tested against this.
    """
    import numpy as np

    rirs = bank.rirs(session.room_index, session.sources).astype(np.float64)  # [N, K, 4, L]
    return render_arrays(session.dry, session.variant_map, rirs, session.target_rms, session.channel_gain)


def mix_session(rng: random.Random, corpus: SpeechCorpus, bank: RirBank, cfg: Any, recipe: Recipe) -> tuple[Session, Any]:
    """:func:`draw_session` plus :func:`render_numpy` (no noise): ``(session, audio [4, S])``."""
    session = draw_session(rng, corpus, bank, cfg, recipe)
    return session, render_numpy(session, bank)


def sample_mask(rng: random.Random, m_weights: Sequence[float]) -> str:
    """A channel subset: M by ``m_weights``, then a uniform subset of that size.

    >>> masks = {sample_mask(random.Random(i), (0, 1, 0, 0)) for i in range(200)}
    >>> sorted(masks)
    ['0011', '0101', '0110', '1001', '1010', '1100']
    """
    m = rng.choices(range(1, spec.N_CHANNELS_MAX + 1), weights=m_weights)[0]
    on = set(rng.sample(range(spec.N_CHANNELS_MAX), m))
    return "".join("1" if c in on else "0" for c in range(spec.N_CHANNELS_MAX))
