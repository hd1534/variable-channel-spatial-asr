#!/usr/bin/env python3
"""Build the two-talker evaluation set of the main evaluation (본 평가).

Usage::

    python3 make_eval_t2.py --out data/eval-t2 --librispeech-root <LibriSpeech test-clean root>
    python3 make_eval_t2.py --verify-only                      # plan only: no numpy, nothing written
    python3 make_eval_t2.py --out /tmp/t2 --n-sessions 6 --synthetic --rir-engine fallback_ism

Every session is written twice, as a *paired* set:

``--out`` (default ``data/eval-t2``)       the **main** condition: each turn is spoken
                                            from one of three copies of the talker's
                                            position (the first, and two moved
                                            0.1-0.3 m from it), and spatially
                                            diffuse pink noise is added at 5, 10 or
                                            20 dB SNR;
``--static-out`` (default ``<out>-static``)  the same sessions with the talkers at
                                            rest and no noise.

Why the main condition moves and is noisy: with static talkers in a noise-free
simulated room, the exact reflection pattern of a position identifies a talker
(even mirror-image positions separate), and untrained spatial features already
reach the causal oracle -- on this set's static condition naive fusion scores
0.935/0.954/0.964 at M = 2/3/4 against an oracle of 0.948, voice-only 0.89.  A
learned model then has nothing left to win and the comparison the benchmark
exists for is decided by noise.  On the main condition the untrained rungs sit
7-12 points below the oracle (naive fusion 0.833/0.860/0.878, voice-only
0.833).  The static set keeps the idealised view, on the same sessions.

What a session holds:

* two talkers from LibriSpeech **test-clean** (40 speakers, 20 F / 20 M), used
  evenly; none of them is in train-clean-100 or dev-clean, so no training or
  validation voice appears here;
* one room per session (no room is reused), sized around a fresh asymmetric
  array (:func:`simulate.make_mic_array`) and the two talkers, one near
  (0.5-1.9 m) and one far (1.9-5.0 m); azimuth bins, near and far distance bins,
  the low/high RT60 band, the SNR level and the number of turns are each spread
  evenly over the sessions (:func:`simulate.even_bin_schedule`);
* 2, 3 or 4 turns in random order -- the same talker may speak twice in a row,
  so a pause never says who comes next; a talker who speaks again does so from
  another position copy; each talker speaks one whole utterance, cut at pauses
  into as many pieces as it has turns, so its reference transcript is exact,
  and no utterance appears in two sessions;
* labels by the official rule (spec ``T_EVAL_RULE``) on each talker's dry
  signal; talkers are named ``spk_0`` / ``spk_1`` in order of their first turn.

The acoustics come from the same code the training sessions use
(:mod:`training.scenes`), with untruncated room responses.  Only
``pyroomacoustics`` output on LibriSpeech is official.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, "") and str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import simulate  # noqa: E402
import spec  # noqa: E402

TRACK = 2
N_TALKERS = spec.MAX_SPEAKERS_PER_TRACK[TRACK]
GENERATOR = "make_eval_t2"
GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 20261031
DEFAULT_N_SESSIONS = 200
VARIANTS = 3
"""The position and two displaced copies of it, one picked per turn (main condition)."""
JITTER_M = (0.10, 0.30)
JITTER_WALL_MARGIN_M = 0.2
SNR_LEVELS_DB = (5.0, 10.0, 20.0)
TURN_COUNTS = (2, 3, 4)
LEAD_IN_RANGE_SEC = (0.25, 1.50)
TURN_GAP_RANGE_SEC = (0.30, 0.80)
TAIL_SEC = 1.0
"""Silence left after the last turn so its reverberant tail stays inside the session."""
MIN_PIECE_SEC = 1.0
MIN_PAUSE_SEC = 0.18
"""Shortest silence accepted as a cut between two turns, so a cut lands between words."""
PAUSE_RULES = ((30.0, MIN_PAUSE_SEC), (25.0, MIN_PAUSE_SEC), (20.0, 0.12))
"""``(dB below the 90th-percentile frame level, shortest run in s)``, strictest first.
Some talkers rarely pause 30 dB deep; only when none of their utterances can be
cut under a rule is the next one tried (the rule used is recorded per talker)."""
SPEAKER_GAIN_DB = 3.0
ABSORPTION_LEVELS = ("low_rt60", "high_rt60")
RT60_RANGES_SEC = {"low_rt60": simulate.RT60_RANGE_LOW_S, "high_rt60": simulate.RT60_RANGE_HIGH_S}
MAX_SOURCE_CANDIDATES = 12
"""Cuttable utterances collected per talker under each pause rule before pairing them by length."""


# ---------------------------------------------------------------------------
# Plan layer (standard library only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TalkerPlan:
    speaker_id: str
    distance_level: str
    base: simulate.SourcePlacement
    variants_xy: tuple[tuple[float, float], ...]
    """``variants_xy[0]`` is the position itself."""
    gain_db: float


@dataclass(frozen=True)
class SessionPlan:
    index: int
    session_id: str
    geometry: simulate.MicArrayGeometry
    dims_xyz_m: tuple[float, float, float]
    array_center_xy_m: tuple[float, float]
    talkers: tuple[TalkerPlan, TalkerPlan]
    absorption_level: str
    rt60_requested_s: float
    snr_db: float
    order: tuple[int, ...]
    """Talker of each turn."""
    variant_of_turn: tuple[int, ...]
    lead_in_sec: float
    gaps_sec: tuple[float, ...]
    common_gain_dbfs: float
    channel_gain_db: tuple[float, ...]
    reports: dict[str, Any] = field(default_factory=dict, compare=False)

    def room_xyz(self, xy: tuple[float, float]) -> tuple[float, float, float]:
        return (self.array_center_xy_m[0] + xy[0], self.array_center_xy_m[1] + xy[1], spec.PLANE_HEIGHT_M)

    def mic_room_xyz(self) -> list[tuple[float, float, float]]:
        return [self.room_xyz(p) for p in self.geometry.positions_xy]


def _balanced(rng: random.Random, values: Sequence[Any], n: int) -> list[Any]:
    """``n`` items cycling through ``values``, shuffled: every value appears n/len or n/len + 1 times.

    >>> sorted(_balanced(random.Random(0), ("a", "b"), 5))
    ['a', 'a', 'a', 'b', 'b']
    """
    out = [values[i % len(values)] for i in range(n)]
    rng.shuffle(out)
    return out


def _placement(key: str, level: str, x: float, y: float) -> simulate.SourcePlacement:
    az = math.degrees(math.atan2(y, x))
    az = az - 360.0 if az >= 180.0 else az
    dist = math.hypot(x, y)
    return simulate.SourcePlacement(key, level, az, dist, spec.azimuth_bin_index(az), spec.distance_bin_index(dist))


def _displaced(rng: random.Random, base: simulate.SourcePlacement, dims: Sequence[float],
               center: Sequence[float]) -> tuple[float, float]:
    """A copy of ``base`` 0.1-0.3 m away, inside the evaluated distance range and away from the walls."""
    bx, by = base.xy
    lo, hi = spec.DISTANCE_RANGE_M
    for attempt in range(400):
        r = rng.uniform(*JITTER_M) * (0.5 if attempt >= 200 else 1.0)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        x, y = bx + r * math.cos(angle), by + r * math.sin(angle)
        rx, ry = center[0] + x, center[1] + y
        if (lo <= math.hypot(x, y) <= hi
                and JITTER_WALL_MARGIN_M <= rx <= dims[0] - JITTER_WALL_MARGIN_M
                and JITTER_WALL_MARGIN_M <= ry <= dims[1] - JITTER_WALL_MARGIN_M):
            return x, y
    return bx, by


def _turn_order(rng: random.Random, n_turns: int) -> tuple[int, ...]:
    """``n_turns`` turns of two talkers in random order, each talker at least once.

    >>> orders = {_turn_order(random.Random(i), 3) for i in range(200)}
    >>> all(set(o) == {0, 1} for o in orders), (0, 0, 1) in orders
    (True, True)
    """
    while True:
        order = tuple(rng.randrange(N_TALKERS) for _ in range(n_turns))
        if set(order) == set(range(N_TALKERS)):
            return order


def _speaker_pairs(rng: random.Random, speakers: Sequence[str], n: int) -> list[tuple[str, str]]:
    """``n`` pairs of distinct talkers drawn from a reshuffled deck, so everyone is used about equally.

    >>> pairs = _speaker_pairs(random.Random(1), ["a", "b", "c", "d"], 6)
    >>> all(a != b for a, b in pairs), sorted({s for p in pairs for s in p})
    (True, ['a', 'b', 'c', 'd'])
    """
    deck: list[str] = []
    out = []
    while len(out) < n:
        if len(deck) < 2:
            fresh = list(speakers)
            rng.shuffle(fresh)
            deck = deck + fresh
        a = deck.pop(0)
        j = next(i for i, s in enumerate(deck) if s != a)
        b = deck.pop(j)
        out.append((a, b))
    return out


def build_plan(n_sessions: int, seed: int, speakers: Sequence[str]) -> list[SessionPlan]:
    """Every session of the set, from the seed alone.  Pure standard library.

    >>> plan = build_plan(6, 7, [f"s{i}" for i in range(6)])
    >>> len(plan), sorted({p.talkers[0].distance_level for p in plan} | {p.talkers[1].distance_level for p in plan})
    (6, ['far', 'near'])
    >>> sorted(len(p.order) for p in plan)
    [2, 2, 3, 3, 4, 4]
    """
    if n_sessions <= 0:
        raise ValueError(f"n_sessions must be > 0, got {n_sessions}")
    if len(set(speakers)) < N_TALKERS:
        raise ValueError("need at least two distinct speakers")
    rng = random.Random(f"{seed}:plan")
    k_near = spec.K_DISTANCE // 2
    azimuth_bins = list(simulate.even_bin_schedule(2 * n_sessions, spec.K_AZIMUTH))
    rng.shuffle(azimuth_bins)
    near_bins = list(simulate.even_bin_schedule(n_sessions, k_near))
    far_bins = list(simulate.even_bin_schedule(n_sessions, spec.K_DISTANCE - k_near, bin_offset=k_near))
    rng.shuffle(near_bins)
    rng.shuffle(far_bins)
    absorption = _balanced(rng, ABSORPTION_LEVELS, n_sessions)
    snr = _balanced(rng, SNR_LEVELS_DB, n_sessions)
    turns = _balanced(rng, TURN_COUNTS, n_sessions)
    pairs = _speaker_pairs(rng, sorted(set(speakers)), n_sessions)

    plans = []
    for i in range(n_sessions):
        srng = random.Random(f"{seed}:session:{i}")
        geom = simulate.make_mic_array(srng)
        near = _placement("near", "near", *_polar(simulate.sample_azimuth_in_bin(srng, azimuth_bins[2 * i]),
                                                  simulate.sample_distance_in_bin(srng, near_bins[i])))
        far = _placement("far", "far", *_polar(simulate.sample_azimuth_in_bin(srng, azimuth_bins[2 * i + 1]),
                                               simulate.sample_distance_in_bin(srng, far_bins[i])))
        dims, center = simulate._fit_room_to_block(srng, geom, {"near": near, "far": far})
        near_first = srng.random() < 0.5
        bases = (near, far) if near_first else (far, near)
        talkers = []
        for t, base in enumerate(bases):
            copies = [base.xy] + [_displaced(srng, base, dims, center) for _ in range(VARIANTS - 1)]
            talkers.append(TalkerPlan(
                speaker_id=pairs[i][t],
                distance_level=base.distance_level,
                base=base,
                variants_xy=tuple(copies),
                gain_db=srng.uniform(-SPEAKER_GAIN_DB, SPEAKER_GAIN_DB),
            ))
        order = _turn_order(srng, turns[i])
        plans.append(SessionPlan(
            index=i,
            session_id=f"t2_{i:04d}",
            geometry=geom,
            dims_xyz_m=tuple(dims),
            array_center_xy_m=tuple(center),
            talkers=(talkers[0], talkers[1]),
            absorption_level=absorption[i],
            rt60_requested_s=srng.uniform(*RT60_RANGES_SEC[absorption[i]]),
            snr_db=snr[i],
            order=order,
            variant_of_turn=_turn_copies(srng, order),
            lead_in_sec=srng.uniform(*LEAD_IN_RANGE_SEC),
            gaps_sec=tuple(srng.uniform(*TURN_GAP_RANGE_SEC) for _ in range(len(order) - 1)),
            common_gain_dbfs=srng.uniform(*simulate.SESSION_GAIN_DBFS_RANGE),
            channel_gain_db=tuple(srng.uniform(-simulate.CHANNEL_GAIN_MISMATCH_DB, simulate.CHANNEL_GAIN_MISMATCH_DB)
                                  for _ in range(spec.N_CHANNELS_MAX)),
        ))
    return plans


def _polar(azimuth_deg: float, distance_m: float) -> tuple[float, float]:
    rad = math.radians(azimuth_deg)
    return distance_m * math.cos(rad), distance_m * math.sin(rad)


def verify_plan(plans: Sequence[SessionPlan]) -> dict[str, Any]:
    """Balance and geometry checks; raises ``AssertionError`` on a violation."""
    from collections import Counter

    for p in plans:
        simulate.verify_array_geometry(p.geometry)
        for t in p.talkers:
            for x, y in t.variants_xy:
                d = math.hypot(x, y)
                assert spec.DISTANCE_RANGE_M[0] - 1e-9 <= d <= spec.DISTANCE_RANGE_M[1] + 1e-9, f"{p.session_id}: distance {d}"
                rx, ry, _ = p.room_xyz((x, y))
                assert 0.0 < rx < p.dims_xyz_m[0] and 0.0 < ry < p.dims_xyz_m[1], f"{p.session_id}: talker outside the room"
        assert {t.distance_level for t in p.talkers} == {"near", "far"}
        assert p.talkers[0].speaker_id != p.talkers[1].speaker_id
        assert set(p.order) == {0, 1}
        for t in (0, 1):
            mine = [v for v, who in zip(p.variant_of_turn, p.order) if who == t]
            assert all(a != b for a, b in zip(mine, mine[1:])), f"{p.session_id}: talker {t} returns to the same copy"
    count = lambda values: dict(sorted(Counter(values).items()))  # noqa: E731
    speakers = Counter(t.speaker_id for p in plans for t in p.talkers)
    report = {
        "n_sessions": len(plans),
        "azimuth_bins": count(t.base.azimuth_bin for p in plans for t in p.talkers),
        "distance_bins": count(t.base.distance_bin for p in plans for t in p.talkers),
        "absorption_levels": count(p.absorption_level for p in plans),
        "snr_db": count(p.snr_db for p in plans),
        "turns": count(len(p.order) for p in plans),
        "talker_changes": count(sum(a != b for a, b in zip(p.order, p.order[1:])) for p in plans),
        "sessions_per_speaker": {"min": min(speakers.values()), "max": max(speakers.values()), "n_speakers": len(speakers)},
    }
    for key in ("azimuth_bins", "distance_bins", "absorption_levels", "snr_db", "turns"):
        values = list(report[key].values())
        assert max(values) - min(values) <= 1, f"{key} is not balanced: {report[key]}"
    assert report["sessions_per_speaker"]["max"] - report["sessions_per_speaker"]["min"] <= 2
    return report


def _turn_copies(rng: random.Random, order: Sequence[int]) -> tuple[int, ...]:
    """Position copy of each turn; a talker who speaks again moves to another copy.

    >>> _turn_copies(random.Random(0), (0, 1, 0, 0))[2:] != (0, 0)
    True
    >>> all(a != b for a, b in [_turn_copies(random.Random(s), (0, 0))  for s in range(20)])
    True
    """
    last: dict[int, int] = {}
    copies = []
    for talker in order:
        options = [v for v in range(VARIANTS) if v != last.get(talker)]
        last[talker] = rng.choice(options)
        copies.append(last[talker])
    return tuple(copies)


# ---------------------------------------------------------------------------
# Source layer: whole utterances cut at pauses
# ---------------------------------------------------------------------------


def find_pause_cuts(signal: Any, n_pieces: int, sample_rate_hz: int = spec.SAMPLE_RATE_HZ,
                    rule: tuple[float, float] = PAUSE_RULES[0]) -> list[int] | None:
    """Cut points (samples) splitting ``signal`` into ``n_pieces`` at pauses, or ``None``.

    A pause is a run of 10 ms frames at least ``rule[0]`` dB below the
    utterance's 90th percentile frame level, lasting ``rule[1]`` s or more; the
    cut is its middle.  Cuts are chosen closest to equal-length pieces, each piece at least
    ``MIN_PIECE_SEC`` long.

    >>> import numpy as np
    >>> x = np.concatenate([np.ones(32000), np.zeros(4000), np.ones(32000)])
    >>> find_pause_cuts(x, 2)
    [33920]
    >>> find_pause_cuts(np.ones(64000), 2) is None
    True
    """
    import numpy as np

    if n_pieces <= 1:
        return []
    hop = int(0.010 * sample_rate_hz)
    n = len(signal) // hop
    if n == 0:
        return None
    frames = np.asarray(signal[: n * hop], dtype=np.float64).reshape(n, hop)
    level = 10.0 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    quiet = level < np.percentile(level, 90) - rule[0]
    pauses = []
    start = None
    for i, q in enumerate(list(quiet) + [False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * hop >= rule[1] * sample_rate_hz:
                pauses.append(((start + i) // 2) * hop)
            start = None
    min_piece = int(MIN_PIECE_SEC * sample_rate_hz)
    cuts: list[int] = []
    previous = 0
    for k in range(1, n_pieces):
        target = len(signal) * k / n_pieces
        remaining = n_pieces - k
        options = [c for c in pauses if c - previous >= min_piece and len(signal) - c >= remaining * min_piece and c > previous]
        if not options:
            return None
        best = min(options, key=lambda c: abs(c - target))
        cuts.append(best)
        previous = best
    return cuts


@dataclass
class Source:
    speaker_id: str
    utt_id: str
    relpath: str | None
    transcript: str
    signal: Any
    cuts: list[int]
    pause_rule: tuple[float, float] = PAUSE_RULES[0]
    path: Path | None = None

    def pieces(self) -> list[Any]:
        bounds = [0, *self.cuts, len(self.signal)]
        return [self.signal[a:b] for a, b in zip(bounds, bounds[1:])]


def choose_sources(plan: SessionPlan, index: dict[str, list[tuple[str, Path, str]]], seed: int,
                   reader, duration, used: set[str] | frozenset[str] = frozenset()) -> tuple[Source, Source]:  # noqa: ANN001
    """One whole utterance per talker that fits the session and can be cut into its turns.

    ``reader(path) -> mono float array``, ``duration(path) -> seconds``.  The
    talkers' utterances must fit ``20 s - lead-in - gaps - TAIL_SEC`` together.
    Utterances in ``used`` (earlier sessions') are skipped.  Among the pairs that
    fit under the strictest pause rule that yields one, a pair is drawn at random,
    so neither talker is favoured with the longer utterance.
    """
    rng = random.Random(f"{seed}:source:{plan.index}")
    turns_of = [plan.order.count(t) for t in range(N_TALKERS)]
    budget = spec.SESSION_LENGTH_SEC - plan.lead_in_sec - sum(plan.gaps_sec) - TAIL_SEC
    rows_of = []
    for t, talker in enumerate(plan.talkers):
        rows = list(index[talker.speaker_id])
        rng.shuffle(rows)
        rows_of.append([r for r in rows if r[0] not in used
                        and turns_of[t] * MIN_PIECE_SEC <= duration(r[1]) <= budget - N_TALKERS * MIN_PIECE_SEC])
    signals: dict[str, Any] = {}
    usable: list[list[Source]] = [[], []]
    for rule in PAUSE_RULES:  # strictest first; relax only while no pair fits
        for t, talker in enumerate(plan.talkers):
            have = {s.utt_id for s in usable[t]}
            added = 0
            for utt, path, text in rows_of[t]:
                if added >= MAX_SOURCE_CANDIDATES:
                    break
                if utt in have:
                    continue
                if utt not in signals:
                    signals[utt] = reader(path)
                cuts = find_pause_cuts(signals[utt], turns_of[t], rule=rule)
                if cuts is not None:
                    usable[t].append(Source(talker.speaker_id, utt, "/".join(Path(path).parts[-3:]), text,
                                            signals[utt], cuts, rule, Path(path)))
                    added += 1
        fits = [(a, b) for a in usable[0] for b in usable[1]
                if (len(a.signal) + len(b.signal)) / spec.SAMPLE_RATE_HZ <= budget]
        if fits:
            return rng.choice(fits)
    raise RuntimeError(f"{plan.session_id}: no utterance pair fits {budget:.1f} s with {turns_of} turns "
                       f"({len(usable[0])} and {len(usable[1])} cuttable utterances)")


# ---------------------------------------------------------------------------
# Render layer
# ---------------------------------------------------------------------------


def _room_responses(plan: SessionPlan, engine: str) -> tuple[list[list[list[Any]]], list[float]]:
    """``rirs[talker][copy][mic]`` (untruncated) and the measured RT60 of each talker's position."""
    import numpy as np

    sources = {}
    for t, talker in enumerate(plan.talkers):
        for v, xy in enumerate(talker.variants_xy):
            key = f"t{t}v{v}"
            sources[key] = _placement(key, key, *xy)
    block = simulate.RoomBlock(f"room_{plan.session_id}", plan.dims_xyz_m, plan.array_center_xy_m, plan.geometry,
                               sources, {"requested": plan.rt60_requested_s})
    keys = [f"t{t}v{v}" for t in range(N_TALKERS) for v in range(VARIANTS)]
    if engine == "pyroomacoustics":
        from training.rirbank import _rirs_pyroomacoustics_multi

        flat = _rirs_pyroomacoustics_multi(block, plan.rt60_requested_s, keys)
    elif engine == "fallback_ism":
        flat = [simulate._rirs_fallback_ism(block, key, plan.rt60_requested_s, spec.SAMPLE_RATE_HZ)[0] for key in keys]
    else:
        raise ValueError(f"unknown RIR engine {engine!r}")
    rirs = [[flat[t * VARIANTS + v] for v in range(VARIANTS)] for t in range(N_TALKERS)]
    rt60 = [float(np.median([simulate._schroeder_rt60(np.asarray(r), spec.SAMPLE_RATE_HZ) for r in rirs[t][0]]))
            for t in range(N_TALKERS)]
    return rirs, rt60


def _layout(plan: SessionPlan, sources: Sequence[Source]):  # noqa: ANN202
    """Dry tracks ``[2, S]``, each talker's turn spans, and the turns in time order."""
    import numpy as np

    from training.scenes import DRY_RMS, _active_rms

    sr = spec.SAMPLE_RATE_HZ
    session_samples = int(round(spec.SESSION_LENGTH_SEC * sr))
    grid = int(round(spec.DELTA_T_SEC * sr))
    dry = np.zeros((N_TALKERS, session_samples))
    pieces = [list(s.pieces()) for s in sources]
    scale = [DRY_RMS * 10 ** (plan.talkers[t].gain_db / 20) / _active_rms(np.asarray(sources[t].signal)) for t in range(N_TALKERS)]
    next_piece = [0, 0]
    turns = []
    cursor = simulate._dither_off_grid(int(round(plan.lead_in_sec * sr)), grid, session_samples)
    for i, talker in enumerate(plan.order):
        if i > 0:
            cursor = simulate._dither_off_grid(cursor + int(round(plan.gaps_sec[i - 1] * sr)), grid, session_samples)
        piece = np.asarray(pieces[talker][next_piece[talker]], dtype=np.float64) * scale[talker]
        next_piece[talker] += 1
        end = cursor + len(piece)
        if end > session_samples:
            raise RuntimeError(f"{plan.session_id}: turns overflow the session")
        dry[talker, cursor:end] = piece
        turns.append({"talker": talker, "start_sample": cursor, "end_sample": end, "variant": plan.variant_of_turn[i]})
        cursor = end
    return dry, turns


def render_session(plan: SessionPlan, sources: Sequence[Source], engine: str, seed: int) -> dict[str, Any]:
    """Audio of both conditions, labels and per-session facts."""
    import numpy as np

    from training.scenes import diffuse_noise_numpy, grid_labels, render_arrays

    rirs, rt60 = _room_responses(plan, engine)
    dry, turns = _layout(plan, sources)
    gain = 10 ** (plan.common_gain_dbfs / 20)
    channel_gain = [10 ** (g / 20) for g in plan.channel_gain_db]
    variant_main = np.zeros(dry.shape, dtype=np.int8)
    for turn in turns:
        variant_main[turn["talker"], turn["start_sample"]:turn["end_sample"]] = turn["variant"]
    noise = diffuse_noise_numpy(plan.geometry.positions_xy, dry.shape[1], seed=_noise_seed(plan, seed))
    audio = {
        "main": render_arrays(dry, variant_main, rirs, gain, channel_gain, plan.snr_db, noise, quantize=False),
        "static": render_arrays(dry, np.zeros_like(variant_main), rirs, gain, channel_gain, quantize=False),
    }
    labels_causal, labels_eval = grid_labels(dry)
    return {"audio": audio, "labels_eval": labels_eval, "labels_causal": labels_causal, "turns": turns, "rt60": rt60}


def _noise_seed(plan: SessionPlan, seed: int) -> int:
    return random.Random(f"{seed}:noise:{plan.session_id}:{plan.index}").getrandbits(31)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _names(plan: SessionPlan) -> dict[int, str]:
    """``spk_0`` is whoever speaks first (the order online slots open in)."""
    first = []
    for t in plan.order:
        if t not in first:
            first.append(t)
    return {t: spec.speaker_slot(i) for i, t in enumerate(first)}


def _position_at(plan: SessionPlan, turns: Sequence[dict[str, Any]], talker: int, k: int, condition: str):  # noqa: ANN202
    """Where ``talker`` speaks during grid window ``k`` (the copy of its turn there)."""
    if condition == "static":
        return plan.talkers[talker].variants_xy[0]
    grid = int(round(spec.DELTA_T_SEC * spec.SAMPLE_RATE_HZ))
    a, b = k * grid, (k + 1) * grid
    best = max((t for t in turns if t["talker"] == talker),
               key=lambda t: min(b, t["end_sample"]) - max(a, t["start_sample"]))
    return plan.talkers[talker].variants_xy[best["variant"]]


def meta_record(plan: SessionPlan, mask_bits: str, audio_rel: str, n_samples: int, condition: str) -> dict[str, Any]:
    """One ``meta.jsonl`` row: inputs only, in the format of ``data/eval-dev``."""
    channels = spec.mask_to_channels(mask_bits)
    sub = plan.geometry.for_mask(mask_bits)
    return {
        "case_id": spec.make_case_id(plan.session_id, mask_bits),
        "session_id": plan.session_id,
        "mask_bits": mask_bits,
        "track": TRACK,
        "condition": condition,
        "n_channels": len(channels),
        "active_channels": list(channels),
        "audio": {"path": audio_rel, "channels_in_file": spec.N_CHANNELS_MAX, "read_channels": list(channels),
                  "sample_rate_hz": spec.SAMPLE_RATE_HZ, "n_samples": int(n_samples), "encoding": "pcm_s16le"},
        "session_length_sec": spec.SESSION_LENGTH_SEC,
        "mic_positions_xy_m": sub.as_json(),
        "array_reference_point_xy_m": list(sub.reference_point_xy),
        "plane_height_m": spec.PLANE_HEIGHT_M,
        "speed_of_sound_m_s": spec.SPEED_OF_SOUND_M_S,
        "max_pair_spacing_m": sub.max_pair_spacing_m(),
        "time_origin": spec.TIME_ORIGIN,
        "delta_t_sec": spec.DELTA_T_SEC,
        "n_grid_points": spec.N_GRID_POINTS,
        "official_scopes": list(spec.OFFICIAL_SCOPES),
        "max_speakers": N_TALKERS,
    }


def truth_record(plan: SessionPlan, mask_bits: str, sources: Sequence[Source], rendered: dict[str, Any],
                 condition: str, sex: dict[str, str]) -> dict[str, Any]:
    """One ``ground_truth.jsonl`` row: per-instant talker and position, both reference transcripts."""
    names = _names(plan)
    frames = []
    for k, talker in enumerate(rendered["labels_eval"]):
        if talker < 0:
            continue
        x, y = _position_at(plan, rendered["turns"], int(talker), k, condition)
        p = _placement("p", "p", x, y)
        frames.append({"eval_time_sec": k * spec.DELTA_T_SEC, "speaker": names[int(talker)],
                       "azimuth_deg": p.azimuth_deg, "distance_m": p.distance_m,
                       "azimuth_bin": p.azimuth_bin, "distance_bin": p.distance_bin})
    sr = spec.SAMPLE_RATE_HZ
    return {
        "case_id": spec.make_case_id(plan.session_id, mask_bits),
        "session_id": plan.session_id,
        "mask_bits": mask_bits,
        "track": TRACK,
        "condition": condition,
        "final_transcripts_reference": {names[t]: sources[t].transcript for t in range(N_TALKERS)},
        "talkers": {
            names[t]: {
                "speaker_id": plan.talkers[t].speaker_id,
                "sex": sex.get(plan.talkers[t].speaker_id),
                "utt_id": sources[t].utt_id,
                "source_path": sources[t].relpath,
                "distance_level": plan.talkers[t].distance_level,
                "azimuth_deg": plan.talkers[t].base.azimuth_deg,
                "distance_m": plan.talkers[t].base.distance_m,
                "gain_db": plan.talkers[t].gain_db,
                "measured_rt60_s": rendered["rt60"][t],
                "pause_rule_db_s": list(sources[t].pause_rule) if len(sources[t].cuts) else None,
            }
            for t in range(N_TALKERS)
        },
        "turns": [
            {"speaker": names[t["talker"]], "start_sec": t["start_sample"] / sr, "end_sec": t["end_sample"] / sr,
             "position_copy": t["variant"] if condition == "main" else 0}
            for t in rendered["turns"]
        ],
        "n_turns": len(plan.order),
        "n_talker_changes": sum(a != b for a, b in zip(plan.order, plan.order[1:])),
        "room": {"dims_xyz_m": list(plan.dims_xyz_m), "array_center_xy_m": list(plan.array_center_xy_m),
                 "absorption_level": plan.absorption_level, "rt60_requested_s": plan.rt60_requested_s},
        "snr_db": plan.snr_db if condition == "main" else None,
        "gain": {"common_gain_dbfs_target": plan.common_gain_dbfs, "channel_gain_db": list(plan.channel_gain_db)},
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _worker_init() -> None:
    try:
        import pyroomacoustics as pra

        pra.constants.set("num_threads", 1)
    except ImportError:  # pragma: no cover - fallback engine
        pass


def _readers(synthetic: bool):  # noqa: ANN202
    """``(reader, duration)`` for the speech source."""
    if synthetic:
        return _synthetic_reader, lambda path: SYNTHETIC_UTTERANCE_SEC
    import soundfile as sf

    return _librispeech_reader, lambda path: sf.info(str(path)).duration


def pick_all_sources(plans: Sequence[SessionPlan], index: dict[str, list[tuple[str, Path, str]]], seed: int,
                     synthetic: bool) -> list[tuple[Source, Source]]:
    """Sources of every session, in session order, never reusing an utterance across sessions.

    Runs in one process so that the exclusion is deterministic; the signals are
    dropped (workers read them again) to keep the jobs small.
    """
    from dataclasses import replace

    reader, duration = _readers(synthetic)
    used: set[str] = set()
    picked = []
    for plan in plans:
        pair = choose_sources(plan, index, seed, reader, duration, used)
        used.update(s.utt_id for s in pair)
        picked.append(tuple(replace(s, signal=None) for s in pair))
    return picked


def _session_job(args: tuple[Any, ...]) -> dict[str, Any]:
    plan, picked, seed, engine, synthetic, outs, sex = args
    import numpy as np
    from dataclasses import replace

    reader, _ = _readers(synthetic)
    sources = [replace(s, signal=reader(s.path)) for s in picked]
    rendered = render_session(plan, sources, engine, seed)
    rows: dict[str, Any] = {}
    for condition, out in outs.items():
        audio_rel = f"audio/{plan.session_id}.wav"
        audio = rendered["audio"][condition]
        simulate.write_wav_pcm16(Path(out) / audio_rel, np.ascontiguousarray(audio.T))
        rows[condition] = {
            "meta": [meta_record(plan, m, audio_rel, audio.shape[1], condition) for m in spec.EVAL_MASKS],
            "truth": [truth_record(plan, m, sources, rendered, condition, sex) for m in spec.EVAL_MASKS],
        }
    labelled = int((rendered["labels_eval"] >= 0).sum())
    return {"index": plan.index, "rows": rows, "labelled": labelled,
            "rt60": rendered["rt60"], "speech_sec": sum(len(s.signal) for s in sources) / spec.SAMPLE_RATE_HZ}


def _librispeech_reader(path: Path):  # noqa: ANN202
    return simulate.load_librispeech_utterance(path)


SYNTHETIC_UTTERANCE_SEC = 7.0


def _synthetic_reader(path: Path):  # noqa: ANN202
    signal, _, _ = simulate.synthesise_utterance(random.Random(str(path)), spec.SAMPLE_RATE_HZ,
                                                 duration_sec=SYNTHETIC_UTTERANCE_SEC)
    return signal


def _synthetic_index(n_speakers: int = 8, per_speaker: int = 64) -> dict[str, list[tuple[str, Path, str]]]:
    """Placeholder corpus; ``per_speaker`` covers the default 200 sessions without reusing an utterance."""
    return {f"syn{s:02d}": [(f"syn{s:02d}-{u}", Path(f"syn{s:02d}/{u}"), "synthetic placeholder") for u in range(per_speaker)]
            for s in range(n_speakers)}


def generate(out: Path, static_out: Path, n_sessions: int, seed: int, librispeech_root: Path | None, engine: str,
             synthetic: bool, workers: int, overwrite_official: bool = False, log=print) -> dict[str, Any]:  # noqa: ANN001
    import make_eval_dev
    from training.corpus import TEST_CLEAN_SPEAKERS, _read_sex, committed_eval_speakers

    for target in (out, static_out):
        if not overwrite_official:
            make_eval_dev.refuse_to_replace_official(target)
    if synthetic:
        index, sex, subset = _synthetic_index(), {}, "synthetic"
    else:
        if librispeech_root is None:
            raise SystemExit("--librispeech-root <LibriSpeech test-clean root> is required (or --synthetic)")
        index = make_eval_dev.scan_librispeech(librispeech_root)
        if not index:
            raise SystemExit(f"no LibriSpeech utterances under {librispeech_root}")
        sex = _read_sex(Path(librispeech_root).parent / "SPEAKERS.TXT")
        subset = Path(librispeech_root).name
        leaked = set(index) & committed_eval_speakers()
        if leaked:
            raise SystemExit(f"speakers {sorted(leaked)} are the pre-measurement eval talkers; use test-clean")
    plans = build_plan(n_sessions, seed, sorted(index))
    report = verify_plan(plans)
    outs = {"main": out, "static": static_out}
    for target in outs.values():
        (Path(target) / "audio").mkdir(parents=True, exist_ok=True)

    log("  choosing utterances")
    picked = pick_all_sources(plans, index, seed, synthetic)
    jobs = [(p, pair, seed, engine, synthetic, {k: str(v) for k, v in outs.items()}, sex)
            for p, pair in zip(plans, picked)]
    results = []
    if workers <= 1:
        _worker_init()  # the pool's per-worker setting; pyroomacoustics' sums depend on its thread count
        for job in jobs:
            results.append(_session_job(job))
            log(f"  {job[0].session_id} done")
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers, initializer=_worker_init) as pool:
            for result in pool.imap_unordered(_session_job, jobs):
                results.append(result)
                if len(results) % 10 == 0 or len(results) == len(jobs):
                    log(f"  {len(results)}/{len(jobs)} sessions")
    results.sort(key=lambda r: r["index"])

    manifests = {}
    for condition, target in outs.items():
        target = Path(target)
        meta = [row for r in results for row in r["rows"][condition]["meta"]]
        truth = [row for r in results for row in r["rows"][condition]["truth"]]
        make_eval_dev.write_jsonl(target / "meta.jsonl", meta)
        make_eval_dev.write_jsonl(target / "ground_truth.jsonl", truth)
        rt60 = [v for r in results for v in r["rt60"] if v == v]
        _write_attribution(target, subset, condition, len(plans), seed, engine, rt60)
        files = {}
        for rel in [f"audio/{p.session_id}.wav" for p in plans] + ["meta.jsonl", "ground_truth.jsonl", "ATTRIBUTION.md"]:
            files[rel] = {"sha256": simulate.sha256_file(target / rel), "bytes": (target / rel).stat().st_size}
        blockers = []
        if engine != "pyroomacoustics":
            blockers.append(f"rir engine {engine} is not official")
        if synthetic:
            blockers.append("synthetic speech")
        if not synthetic and (subset != "test-clean" or set(index) != TEST_CLEAN_SPEAKERS):
            blockers.append(f"corpus {subset} is not the 40 test-clean speakers")
        if (n_sessions, seed) != (DEFAULT_N_SESSIONS, DEFAULT_SEED):
            blockers.append(f"{n_sessions} sessions with seed {seed}, the official set is "
                            f"{DEFAULT_N_SESSIONS} with seed {DEFAULT_SEED}")
        if engine == "pyroomacoustics":
            import pyroomacoustics

            if pyroomacoustics.__version__ != "0.8.5":
                blockers.append(f"pyroomacoustics {pyroomacoustics.__version__}, the official sets use 0.8.5")
        manifest = {
            "spec_version": spec.SPEC_VERSION,
            "generator": GENERATOR,
            "generator_version": GENERATOR_VERSION,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "official": not blockers,
            "official_blockers": blockers,
            "track": TRACK,
            "condition": condition,
            "paired_with": (static_out if condition == "main" else out).name,
            "seed": seed,
            "n_sessions": len(plans),
            "n_cases": len(meta),
            "eval_masks": list(spec.EVAL_MASKS),
            "rir_engine": engine,
            "corpus": f"LibriSpeech/{subset}" if not synthetic else "synthetic",
            "speakers": {"n": len(index), "ids": sorted(index),
                         "sex": dict(sorted(__import__("collections").Counter(sex.get(s) for s in index).items(), key=str))},
            "balance": report,
            "conditions": {
                "main": {"position_copies": VARIANTS, "jitter_m": list(JITTER_M), "snr_levels_db": list(SNR_LEVELS_DB),
                         "noise": "spherically isotropic, pink"},
                "static": {"position_copies": 1, "noise": None},
            },
            "rt60_measured_s": {"min": min(rt60), "median": statistics.median(rt60), "max": max(rt60)} if rt60 else None,
            "labelled_instants_per_session": {"min": min(r["labelled"] for r in results),
                                              "mean": statistics.mean(r["labelled"] for r in results)},
            "files": files,
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
                                              encoding="utf-8")
        manifests[condition] = manifest
    return manifests


TEST_CLEAN_SHA256 = "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23"
"""SHA-256 of ``test-clean.tar.gz`` from https://www.openslr.org/12 (md5 32fa31d27d2e1cad72775fee3f4849a9)."""


def _write_attribution(target: Path, subset: str, condition: str, n_sessions: int, seed: int, engine: str,
                       rt60: Sequence[float]) -> None:
    """The CC BY 4.0 notice next to the derived audio, in the form of ``data/eval-dev/ATTRIBUTION.md``."""
    if condition == "main":
        steps = (f"3. 화자마다 위치 사본 {VARIANTS}개(첫 위치와 그것을 {JITTER_M[0]:.1f}~{JITTER_M[1]:.1f} m 옮긴 사본)의 "
                 "RIR을 만들고, 턴마다 그중 하나를 썼다.\n"
                 "4. 시뮬레이션한 확산 잡음(구면 등방, 핑크)을 세션마다 SNR "
                 + "·".join(f"{v:g}" for v in SNR_LEVELS_DB) + " dB 중 하나로 더했다.\n")
    else:
        steps = "3. 화자는 첫 위치에 머물고 잡음은 더하지 않았다(주 조건과 같은 세션, 같은 음성, 같은 방).\n"
    rt = f", 실측 `RT60` 범위 {min(rt60):.2f}~{max(rt60):.2f}초" if rt60 else ""
    (target / "ATTRIBUTION.md").write_text(
        f"# 본 평가셋({condition}) 출처 및 라이선스 고지\n\n"
        "이 디렉터리에 수록된 다채널 오디오는 **LibriSpeech ASR 코퍼스의 파생 저작물**이다. 원저작물이 "
        "**CC BY 4.0** 라이선스로 배포되므로 파생 오디오 재배포 시에도 원저작물 출처와 변경 내역을 함께 명시해야 "
        "하며, 본 문서는 해당 출처 표기 의무를 이행한다.\n\n"
        "## 1. 원저작물 정보\n\n"
        "| 항목 | 내용 |\n|---|---|\n"
        f"| 데이터셋명 | LibriSpeech ASR corpus (`{subset}` 분할) |\n"
        "| 저자 | Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur |\n"
        "| 공식 배포처 | https://www.openslr.org/12 |\n"
        "| 라이선스 | [Creative Commons Attribution 4.0 International (CC BY 4.0)]"
        "(https://creativecommons.org/licenses/by/4.0/) |\n"
        "| 참고문헌 | V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “LibriSpeech: An ASR Corpus Based on Public "
        "Domain Audio Books,” in *Proc. IEEE ICASSP*, 2015, pp. 5206–5210. |\n\n"
        + (f"합성에 사용한 원본 아카이브(`test-clean.tar.gz`)의 SHA-256 해시는 다음과 같다.\n\n"
           f"```text\n{TEST_CLEAN_SHA256}\n```\n\n" if subset == "test-clean" else "")
        + "## 2. 원본에 가한 변환 내역\n\n"
        "원본 단채널 녹음을 그대로 재배포하지 않고 아래 음향 합성 절차를 거쳐 생성한 결과물이다.\n\n"
        f"1. `{subset}` 화자의 발화를 세션마다 화자당 하나씩 골라 쉼 지점에서 턴으로 나눴다. 발화 내용은 "
        "자르거나 바꾸지 않았다. 이 화자들은 학습·검증에 쓰지 않는다.\n"
        f"2. 각 턴을 `{engine}` 이미지 소스법(ISM)으로 시뮬레이션한 **4채널 실내 임펄스 응답(RIR)**과 "
        f"컨볼루션했다(가상 방 {n_sessions}개, 세션마다 새 방{rt}).\n"
        + steps
        + f"{5 if condition == 'main' else 4}. 세션마다 무작위 공통 이득과 채널별 미세 이득 차를 적용하고, 세션 길이를 "
        f"{spec.SESSION_LENGTH_SEC:.1f}초로 맞춰 `16 kHz · 4채널 · 16-bit PCM` WAV 형식으로 직렬화했다.\n\n"
        f"전체 합성 코드는 `make_eval_t2.py`와 `training/scenes.py`에 공개되어 있으며, 고정 시드 `{seed}`로 "
        "다시 만들 수 있다.\n\n"
        "## 3. 파생 저작물의 라이선스 및 재배포 조건\n\n"
        "CC BY 4.0은 동일조건변경허락(ShareAlike) 조항을 포함하지 않으나, **파생 저작물을 재배포할 때 원저작물 "
        "출처 표기와 변경 내역 고지 의무는 그대로 유지된다.** 따라서 본 디렉터리의 오디오 파일을 외부로 복사하거나 "
        "재배포할 때는 본 `ATTRIBUTION.md` 파일을 반드시 함께 포함해야 한다.\n\n"
        "## 4. 비음성 구성 요소\n\n"
        "- `ground_truth.jsonl`, `meta.jsonl`, `manifest.json`에 수록된 라벨 및 메타데이터는 본 벤치마크 "
        "저장소에서 자체 생성한 산출물이다. 참조 전사는 LibriSpeech 원문 전사를 그대로 옮긴 것이다.\n"
        "- 실내 임펄스 응답 합성에 사용한 `pyroomacoustics`는 MIT 라이선스로 배포되며, 데이터 생성 시에만 "
        "호출되었고 라이브러리 코드는 본 디렉터리에 재배포되지 않는다.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="data/eval-t2", help="main condition (moving talkers, diffuse noise)")
    parser.add_argument("--static-out", help="static condition, same sessions (default: <out>-static)")
    parser.add_argument("--librispeech-root", help="LibriSpeech test-clean root (<root>/<speaker>/<chapter>/*.flac)")
    parser.add_argument("--n-sessions", type=int, default=DEFAULT_N_SESSIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rir-engine", choices=simulate.RIR_ENGINES, default="pyroomacoustics")
    parser.add_argument("--synthetic", action="store_true", help="placeholder speech; never official")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--overwrite-official", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="check the plan only; writes nothing")
    args = parser.parse_args(argv)
    out = Path(args.out)
    static_out = Path(args.static_out) if args.static_out else out.with_name(out.name + "-static")
    if args.verify_only:
        speakers = [f"spk{i:02d}" for i in range(40)]
        report = verify_plan(build_plan(args.n_sessions, args.seed, speakers))
        print(json.dumps(report, indent=1))
        return 0
    try:
        manifests = generate(out, static_out, args.n_sessions, args.seed,
                             Path(args.librispeech_root) if args.librispeech_root else None,
                             args.rir_engine, args.synthetic, args.workers, args.overwrite_official)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for condition, m in manifests.items():
        print(f"{condition}: {m['n_sessions']} sessions, {m['n_cases']} cases, official={m['official']} "
              f"{m['official_blockers'] or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
