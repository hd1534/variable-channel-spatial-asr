"""Shared machinery for the controlled baselines of the VCS-ASR benchmark.

Everything in this package that touches the evaluation contract reads its
numbers from :mod:`spec`.  Nothing here re-states a frozen constant.

The package exists mainly so that *Controlled Baseline 1* and *Controlled
Baseline 2* can share one literal code path for the pieces that must be held
constant between them -- the frozen ASR backend, the inference time grid, the
causal windowing rule, the distribution normalisation and the output record
builders.  If the ASR were wired up twice, "same ASR" would be a claim instead
of a fact.

Design notes
------------
*Lazy imports.*  ``numpy`` and the ASR stack are imported inside the functions
that need them, never at module import time, so ``python3 -m py_compile`` and
stdlib-only unit tests never require them.

*Causality.*  Audio is only ever sliced through
:meth:`AudioSession.window`, which raises if a caller asks for a sample past
the observation boundary implied by the scope.  That makes the causality claim
checkable in one place instead of being spread over both baselines.

Contract items this package has to define because :mod:`spec` does not freeze
them are collected in :data:`LOCAL_CONVENTIONS`; each one is reported upstream
as a requested spec addition.

>>> import spec
>>> len(inference_grid_sec()) * spec.DELTA_T_SEC == spec.SESSION_LENGTH_SEC
True
>>> inference_grid_sec()[0], inference_grid_sec()[-1]
(0.0, 19.75)
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Literal, Mapping, Sequence, TYPE_CHECKING

import spec

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
else:  # pragma: no cover - runtime placeholder so annotations stay cheap
    FloatArray = Any

__all__ = [
    "LOCAL_CONVENTIONS",
    "BASELINE_IMPL_VERSION",
    "NotACaseListError",
    "RECORD_TYPE_TIMESTEP",
    "RECORD_TYPE_FINAL",
    "SPEAKER_SLOT_FORMAT",
    "AudioSession",
    "CaseSpec",
    "DistanceModel",
    "FrozenASR",
    "add_common_cli_args",
    "azimuth_centers",
    "build_final_record",
    "build_timestep_record",
    "causal_window_bounds",
    "check_fit_targets",
    "default_mic_positions_m",
    "discover_cases",
    "distance_centers",
    "energy_vad_segments",
    "MIN_DISTINCT_DISTANCE_LABELS",
    "MIN_SIGMA_LOG",
    "fit_ridge",
    "frame_rms",
    "warn_on_unphysical_level_coefficient",
    "hardware_tag",
    "inference_grid_sec",
    "kernel_distribution",
    "load_audio",
    "load_distance_model",
    "map_point",
    "normalise_distribution",
    "pair_spacings_m",
    "resolve_scope",
    "rms",
    "speaker_slot",
    "uniform_distribution",
    "write_predictions",
]


BASELINE_IMPL_VERSION: Final[str] = "0.1.0"
"""Version of this baseline implementation, stamped into every record."""


# ---------------------------------------------------------------------------
# Conventions this package still pins itself (see LOCAL_CONVENTIONS below)
# ---------------------------------------------------------------------------

LOCAL_CONVENTIONS: Final[dict[str, str]] = {
    "eval_dir_layout": (
        "No manifest format is frozen. discover_cases() accepts a JSONL/JSON/CSV "
        "case list under several candidate names with tolerant key aliases, or "
        "falls back to globbing audio files and expanding spec.EVAL_MASKS. The "
        "generator's per-case file is meta.jsonl and its manifest.json is a "
        "dataset summary, so auto-discovery tries meta.jsonl first and skips a "
        "candidate that does not parse as a case list. This is how the dataset "
        "is read, not what a submission must contain, so it stays local."
    ),
    "case_row_audio_and_geometry": (
        "A case row's 'audio' may be a path string or an object with 'path' and "
        "'sample_rate_hz'; its microphone coordinates may be the full array in "
        "absolute channel order or only the mask's active channels in subset "
        "order (the generator's 'mic_positions_xy_m'). Subset coordinates are "
        "scattered onto absolute channel indices, because every consumer looks a "
        "microphone up by absolute index. Dataset-side tolerance, not contract."
    ),
    "windowed_lookahead": (
        "spec.WINDOWED_REQUIRES_LOOKAHEAD says a windowed record must report W; "
        "this implementation exposes --lookahead-sec (default: one "
        "spec.CHUNK_HOP_SEC) and writes the value into the record metadata. The "
        "default is this package's choice, the requirement is the contract's."
    ),
    "prob_rounding": (
        "Emitted probabilities are rounded to DEFAULT_PROB_DECIMALS and then "
        "repaired to sum exactly, because spec.NORMALISE_SUBMITTED_PROBS is False "
        "and the evaluator rejects rather than rescales. The decimal count is a "
        "local margin against spec.PROB_SUM_TOLERANCE, not a contract value."
    ),
}
"""Conventions still fixed inside this package.

Everything score-bearing that used to live here is now in ``spec.py``: the
record schema and field names, the speaker slot format, the azimuth sign and
zero reference, the inference grid endpoint, the causal window alignment, the
RTF definition and the default array geometry.  What is left describes how this
package *reads the distributed dataset* and what it does with its own freedom,
neither of which a submission has to agree with.
"""


RECORD_TYPE_TIMEPOINT: Final[str] = spec.RECORD_TYPE_TIMEPOINT
RECORD_TYPE_FINAL_TRANSCRIPTS: Final[str] = spec.RECORD_TYPE_FINAL_TRANSCRIPTS
"""The canonical ``record_type`` values, from the contract."""

RECORD_TYPE_TIMESTEP: Final[str] = RECORD_TYPE_TIMEPOINT
RECORD_TYPE_FINAL: Final[str] = RECORD_TYPE_FINAL_TRANSCRIPTS
"""Former names of the two constants above, kept so that existing imports
resolve.  They now hold the canonical values: this package wrote ``timestep``
and ``final_transcript`` while ``spec.py`` owned no record schema, and the
evaluator carried an alias table to read them.  It still does, so already
generated submissions keep scoring, but nothing produces those spellings now.
"""

SPEAKER_SLOT_FORMAT: Final[str] = spec.SPEAKER_SLOT_FORMAT

DEFAULT_TRACK: Final[int] = min(spec.MAX_SPEAKERS_PER_TRACK)
"""Smallest track number the contract knows about (Track 1 is this round's scope)."""

DEFAULT_PROB_DECIMALS: Final[int] = 6
"""Rounding applied to emitted probabilities before the exact-sum repair.

Chosen so the worst-case rounding residual, ``K_AZIMUTH * 5e-7``, is three
orders of magnitude inside :data:`spec.PROB_SUM_TOLERANCE`.
"""

MIN_SIGMA_LOG: Final[float] = 0.05
"""Floor on the fitted residual spread of a distance regression, in log meters.

A distance estimator built from received level and a reverberation proxy is not
credible to better than a few percent relative error, and a fit on a small or
degenerate training set can report a residual spread near zero.  Emitting that
spread would produce a nearly one-hot distance distribution, which scores the
full :data:`spec.PENALTY_NLL` on every bin it gets wrong -- strictly worse than
answering honestly wide.  5 % relative is the floor applied instead.
"""

MIN_DISTINCT_DISTANCE_LABELS: Final[int] = 3
"""Fewest distinct distance labels a usable fit needs.

With fewer, least squares returns the mean of the targets with zero residual
and the regression looks perfect while carrying no information at all.
"""

SYNTHETIC_ASR_WARNING: Final[str] = (
    "WARNING: --synthetic-asr is on. Transcripts are placeholders, NOT speech "
    "recognition output. Any WER/cpWER computed from this file is meaningless "
    "and must not be reported. Records are stamped wer_reportable=false."
)


def speaker_slot(index: int = 0) -> str:
    """Return the anonymous speaker slot label for ``index``.

    Delegates to :func:`spec.speaker_slot`; the default argument is the only
    thing this wrapper adds, since every Track 1 call site wants slot 0.

    >>> speaker_slot(), speaker_slot(1)
    ('spk_0', 'spk_1')
    """
    return spec.speaker_slot(index)


def hardware_tag() -> str:
    """Return a coarse machine tag for the ``hardware`` metadata field.

    >>> isinstance(hardware_tag(), str) and len(hardware_tag()) > 0
    True
    """
    return f"{platform.system()}-{platform.machine()}-py{platform.python_version()}"


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------


def inference_grid_sec(
    session_length_sec: float = spec.SESSION_LENGTH_SEC,
    delta_t_sec: float = spec.DELTA_T_SEC,
) -> tuple[float, ...]:
    """Return ``T_all``, the uniform inference grid, in seconds.

    The proposal defines ``T_all = {k * dt | 0 <= k * dt < T_session}``, so the
    session end point is excluded.  ``dt`` is binary-exact, hence so is every
    grid value and it is safe as a dictionary key.

    >>> g = inference_grid_sec()
    >>> len(g)
    80
    >>> g[:3]
    (0.0, 0.25, 0.5)
    >>> max(g) < spec.SESSION_LENGTH_SEC
    True
    >>> inference_grid_sec(1.0, 0.25)
    (0.0, 0.25, 0.5, 0.75)
    """
    if delta_t_sec <= 0.0:
        raise ValueError(f"delta_t_sec must be positive, got {delta_t_sec}")
    upper = int(math.ceil(session_length_sec / delta_t_sec)) + 1
    return tuple(k * delta_t_sec for k in range(upper) if k * delta_t_sec < session_length_sec)


def resolve_scope(scope: str) -> str:
    """Validate an observation scope against the contract and return it.

    Emits a note on stderr when the scope is accepted but not official.

    >>> resolve_scope("causal")
    'causal'
    >>> resolve_scope("nope")
    Traceback (most recent call last):
        ...
    ValueError: scope must be one of ('causal', 'windowed', 'full_audio'), got 'nope'
    """
    if scope not in spec.OBSERVATION_MODES:
        raise ValueError(f"scope must be one of {spec.OBSERVATION_MODES}, got {scope!r}")
    if scope not in spec.OFFICIAL_SCOPES:
        print(
            f"note: scope {scope!r} is accepted but not in spec.OFFICIAL_SCOPES "
            f"{spec.OFFICIAL_SCOPES}; it is reported in a separate table.",
            file=sys.stderr,
        )
    return scope


def causal_window_bounds(
    eval_time_sec: float,
    scope: str,
    session_length_sec: float = spec.SESSION_LENGTH_SEC,
    chunk_length_sec: float = spec.CHUNK_LENGTH_SEC,
    lookahead_sec: float = spec.CHUNK_HOP_SEC,
) -> tuple[float, float]:
    """Return the ``(start_sec, end_sec)`` the model may read at ``eval_time_sec``.

    ``causal`` ends exactly at ``eval_time_sec``: no future sample is visible.
    ``windowed`` adds ``lookahead_sec`` of lookahead (unofficial, see
    :data:`spec.OFFICIAL_SCOPES`).  ``full_audio`` sees the whole session.

    >>> causal_window_bounds(5.0, "causal")
    (4.0, 5.0)
    >>> causal_window_bounds(0.25, "causal")
    (0.0, 0.25)
    >>> causal_window_bounds(5.0, "windowed", lookahead_sec=0.25)
    (4.0, 5.25)
    >>> causal_window_bounds(5.0, "full_audio")
    (0.0, 20.0)
    """
    if scope == "full_audio":
        return 0.0, session_length_sec
    start = max(0.0, eval_time_sec - chunk_length_sec)
    end = eval_time_sec
    if scope == "windowed":
        end = min(session_length_sec, eval_time_sec + lookahead_sec)
    elif scope != "causal":
        raise ValueError(f"unknown scope {scope!r}")
    return start, end


# ---------------------------------------------------------------------------
# Array geometry
# ---------------------------------------------------------------------------


def default_mic_positions_m(
    n_channels: int = spec.N_CHANNELS_MAX,
    diameter_m: float = spec.MAX_PAIR_SPACING_M,
) -> tuple[tuple[float, float], ...]:
    """Return fallback mic coordinates: a uniform circular array in the z plane.

    Mic ``i`` sits at angle ``360 * i / n_channels`` degrees on a circle of
    radius ``diameter_m / 2``, so for an even ``n_channels`` the longest pair
    spacing is exactly ``diameter_m`` -- which is what
    :data:`spec.MAX_PAIR_SPACING_M` documents.  Only used when the eval
    metadata provides no coordinates; the real geometry must come from the data.

    >>> p = default_mic_positions_m()
    >>> len(p)
    4
    >>> [round(v, 4) for v in p[0]]
    [0.12, 0.0]
    >>> [round(v, 4) for v in p[1]]
    [0.0, 0.12]
    """
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}")
    radius = diameter_m / 2.0
    out: list[tuple[float, float]] = []
    for i in range(n_channels):
        angle = 2.0 * math.pi * i / n_channels
        out.append((radius * math.cos(angle), radius * math.sin(angle)))
    return tuple(out)


def pair_spacings_m(
    mic_positions_m: Sequence[Sequence[float]],
    mask_bits: str,
) -> dict[tuple[int, int], float]:
    """Return the spacing of every active microphone pair, in meters.

    Pairs are keyed by absolute channel index, never by position inside the
    subset, so the geometry stays attached to the right channel when channels
    are dropped (the coordinate-invariance rule).

    >>> p = default_mic_positions_m()
    >>> round(max(pair_spacings_m(p, "1100").values()), 4)   # adjacent pair
    0.1697
    >>> round(max(pair_spacings_m(p, "1010").values()), 4)   # opposite pair
    0.24
    >>> pair_spacings_m(p, "1000")
    {}
    """
    channels = spec.mask_to_channels(mask_bits)
    out: dict[tuple[int, int], float] = {}
    for a_idx, i in enumerate(channels):
        for j in channels[a_idx + 1 :]:
            dx = mic_positions_m[i][0] - mic_positions_m[j][0]
            dy = mic_positions_m[i][1] - mic_positions_m[j][1]
            out[(i, j)] = math.hypot(dx, dy)
    return out


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def azimuth_centers() -> tuple[float, ...]:
    """Return the frozen azimuth bin centers in degrees.

    >>> c = azimuth_centers()
    >>> len(c), c[0], c[23]
    (36, -175.0, 55.0)
    """
    return spec.AZIMUTH_BIN_CENTERS_DEG


def distance_centers() -> tuple[float, ...]:
    """Return the frozen (geometric) distance bin centers in meters.

    >>> len(distance_centers()) == spec.K_DISTANCE
    True
    """
    return spec.DISTANCE_BIN_CENTERS_M


def normalise_distribution(
    values: Sequence[float],
    decimals: int | None = DEFAULT_PROB_DECIMALS,
) -> list[float]:
    """Return a probability vector that sums to 1 inside the frozen tolerance.

    Negative and non-finite entries are clipped to zero and an all-zero vector
    degrades to the uniform distribution.  Rounding to ``decimals`` uses the
    largest-remainder rule on the integer grid ``10 ** decimals``, so the
    rounded entries sum to exactly one unit of that grid *and* the ordering of
    the input is preserved -- a naive "round, then fold the residual into the
    largest entry" repair silently moves the MAP bin of a flat vector, which
    would make the submitted point prediction disagree with the submitted
    distribution.  The evaluator does not repair distributions
    (:data:`spec.NORMALISE_SUBMITTED_PROBS` is ``False``), so this has to be
    right before the record is written.

    >>> p = normalise_distribution([1.0, 1.0, 1.0])
    >>> abs(math.fsum(p) - 1.0) < spec.PROB_SUM_TOLERANCE
    True
    >>> normalise_distribution([0.0, 0.0])
    [0.5, 0.5]
    >>> normalise_distribution([-5.0, 1.0, 3.0])
    [0.0, 0.25, 0.75]

    An exactly flat vector is returned unrounded, so a uniform answer stays
    bit-for-bit uniform instead of acquiring a spurious MAP bin on the
    rounding grid:

    >>> u = normalise_distribution([1.0] * spec.K_AZIMUTH)
    >>> len(set(u))
    1
    >>> u.index(max(u))  # a flat vector keeps its lowest-index MAP bin
    0
    >>> normalise_distribution([1.0, 2.0, 3.0]) == sorted(normalise_distribution([1.0, 2.0, 3.0]))
    True
    """
    n = len(values)
    if n == 0:
        raise ValueError("cannot normalise an empty distribution")
    clipped = [float(v) if v > 0.0 and math.isfinite(v) else 0.0 for v in values]
    total = math.fsum(clipped)
    if total <= 0.0:
        clipped = [1.0] * n
        total = float(n)
    probs = [v / total for v in clipped]
    if decimals is not None and len(set(probs)) > 1:
        if decimals < 1:
            raise ValueError(f"decimals must be >= 1, got {decimals}")
        scale = 10**decimals
        scaled = [p * scale for p in probs]
        units = [int(math.floor(s)) for s in scaled]
        deficit = scale - sum(units)
        if deficit > 0:
            # Largest remainder first, ties toward the lower index.
            order = sorted(range(n), key=lambda i: (-(scaled[i] - units[i]), i))
            for i in order[:deficit]:
                units[i] += 1
        probs = [u / scale for u in units]
    err = abs(math.fsum(probs) - 1.0)
    if err >= spec.PROB_SUM_TOLERANCE:  # pragma: no cover - guards the contract
        raise AssertionError(f"normalisation failed: |sum - 1| = {err:g}")
    return probs


def uniform_distribution(k: int) -> list[float]:
    """Return the uniform distribution over ``k`` bins.

    This is the honest answer for azimuth whenever ``M = 1``: a single
    microphone carries no interchannel phase or level difference, so direction
    is not merely hard to estimate, it is unobservable.

    >>> u = uniform_distribution(spec.K_AZIMUTH)
    >>> len(u) == spec.K_AZIMUTH
    True
    >>> abs(math.fsum(u) - 1.0) < spec.PROB_SUM_TOLERANCE
    True
    """
    return normalise_distribution([1.0] * k)


def kernel_distribution(
    centers: Sequence[float],
    point: float,
    sigma: float,
    circular: bool = False,
    log_domain: bool = False,
    decimals: int | None = DEFAULT_PROB_DECIMALS,
) -> list[float]:
    """Spread a point estimate over bins with a Gaussian kernel.

    ``circular`` measures the center-to-point distance with
    :func:`spec.circular_error_deg` (azimuth); ``log_domain`` measures it
    between natural logarithms (distance, whose bins are log spaced, so a
    constant ``sigma`` in log space is a constant relative error).

    >>> p = kernel_distribution(azimuth_centers(), 55.0, 10.0, circular=True)
    >>> p.index(max(p)) == spec.azimuth_bin_index(55.0)
    True
    >>> abs(math.fsum(p) - 1.0) < spec.PROB_SUM_TOLERANCE
    True
    >>> q = kernel_distribution(distance_centers(), 2.0, 0.3, log_domain=True)
    >>> q.index(max(q)) == spec.distance_bin_index(2.0)
    True
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    weights: list[float] = []
    for center in centers:
        if circular:
            delta = spec.circular_error_deg(point, center)
        elif log_domain:
            delta = math.log(max(point, spec.NLL_EPSILON)) - math.log(
                max(center, spec.NLL_EPSILON)
            )
        else:
            delta = point - center
        weights.append(math.exp(-0.5 * (delta / sigma) ** 2))
    return normalise_distribution(weights, decimals=decimals)


def map_point(
    probs: Sequence[float],
    centers: Sequence[float],
) -> float:
    """Return the center of the highest-probability bin (ties toward low index).

    The evaluator falls back to this when a record omits an explicit point, but
    the baselines submit the point explicitly, so this is also used to keep the
    submitted point and the submitted distribution consistent by construction.

    >>> map_point([0.1, 0.9], (1.0, 2.0))
    2.0
    >>> map_point([0.5, 0.5], (1.0, 2.0))  # tie -> lower index
    1.0
    """
    if len(probs) != len(centers):
        raise ValueError(f"length mismatch: {len(probs)} probs vs {len(centers)} centers")
    best = max(range(len(probs)), key=lambda i: (probs[i], -i))
    return float(centers[best])


# ---------------------------------------------------------------------------
# Numeric helpers (numpy, lazily imported by callers)
# ---------------------------------------------------------------------------


def rms(signal: FloatArray) -> float:
    """Return the root-mean-square level of ``signal``.

    >>> import numpy as np
    >>> round(rms(np.array([3.0, -3.0])), 6)
    3.0
    >>> rms(np.zeros(0))
    0.0
    """
    import numpy as np

    arr = np.asarray(signal, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def frame_rms(
    signal: FloatArray,
    sample_rate_hz: int,
    frame_sec: float,
    hop_sec: float,
) -> FloatArray:
    """Return the per-frame RMS envelope of a mono signal.

    >>> import numpy as np
    >>> env = frame_rms(np.ones(1000), 1000, 0.1, 0.1)
    >>> env.shape, round(float(env[0]), 6)
    ((10,), 1.0)
    """
    import numpy as np

    arr = np.asarray(signal, dtype=np.float64).ravel()
    frame = max(1, int(round(frame_sec * sample_rate_hz)))
    hop = max(1, int(round(hop_sec * sample_rate_hz)))
    if arr.size < frame:
        return np.array([rms(arr)], dtype=np.float64)
    starts = np.arange(0, arr.size - frame + 1, hop)
    out = np.empty(starts.size, dtype=np.float64)
    for i, start in enumerate(starts):
        block = arr[start : start + frame]
        out[i] = np.sqrt(np.mean(block * block))
    return out


def energy_vad_segments(
    signal: FloatArray,
    sample_rate_hz: int,
    min_segment_sec: float = spec.ASR_MIN_SEGMENT_SEC,
    frame_sec: float = 0.03,
    hop_sec: float = 0.01,
    threshold_db: float = 20.0,
) -> list[tuple[float, float]]:
    """Return ``[(start_sec, end_sec)]`` speech segments from an energy VAD.

    A frame counts as speech when its RMS is within ``threshold_db`` of the
    95th percentile frame RMS of the signal.  Adjacent speech frames are merged
    and any segment shorter than ``min_segment_sec``
    (:data:`spec.ASR_MIN_SEGMENT_SEC`) is padded symmetrically up to that
    length, because the contract hands the ASR nothing shorter.

    This is deliberately crude: it is a decode-time segmenter for the frozen
    ASR, not a scored component, and it is identical for both baselines so it
    cannot explain a difference between them.  It is *not* oracle VAD -- it
    never sees a label.

    >>> import numpy as np
    >>> sig = np.concatenate([np.zeros(16000), np.ones(32000), np.zeros(16000)])
    >>> seg = energy_vad_segments(sig, 16000)
    >>> len(seg)
    1
    >>> round(seg[0][0], 1), round(seg[0][1], 1)
    (1.0, 3.0)
    """
    import numpy as np

    arr = np.asarray(signal, dtype=np.float64).ravel()
    duration = arr.size / float(sample_rate_hz)
    if duration <= 0.0:
        return []
    env = frame_rms(arr, sample_rate_hz, frame_sec, hop_sec)
    if env.size == 0 or float(np.max(env)) <= 0.0:
        return []
    reference = float(np.percentile(env, 95.0))
    if reference <= 0.0:
        return []
    floor = reference * (10.0 ** (-threshold_db / 20.0))
    active = env >= floor
    hop = max(1, int(round(hop_sec * sample_rate_hz)))
    frame = max(1, int(round(frame_sec * sample_rate_hz)))
    segments: list[tuple[float, float]] = []
    start_idx: int | None = None
    for i, flag in enumerate(active.tolist() + [False]):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            s = start_idx * hop / sample_rate_hz
            e = ((i - 1) * hop + frame) / sample_rate_hz
            segments.append((s, min(e, duration)))
            start_idx = None
    merged: list[tuple[float, float]] = []
    for s, e in segments:
        if e - s < min_segment_sec:
            pad = (min_segment_sec - (e - s)) / 2.0
            s = max(0.0, s - pad)
            e = min(duration, e + pad)
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def check_fit_targets(targets: Sequence[float]) -> None:
    """Reject a target vector too degenerate to fit a regression on.

    >>> check_fit_targets([1.0, 2.0, 3.0]) is None
    True
    >>> check_fit_targets([2.0, 2.0, 2.0, 2.0])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: need at least 3 distinct distance labels to fit, got 1. ...
    """
    distinct = len({round(float(t), 9) for t in targets})
    if distinct < MIN_DISTINCT_DISTANCE_LABELS:
        raise ValueError(
            f"need at least {MIN_DISTINCT_DISTANCE_LABELS} distinct distance "
            f"labels to fit, got {distinct}. With fewer, the fit returns the "
            "mean of the targets with zero residual, which looks like a perfect "
            "regression but carries no information. Supply a training split "
            "that spans the distance range, or use --distance-model with a prior."
        )


LEVEL_FEATURE_NAMES: Final[tuple[str, ...]] = ("log10_rms",)
"""Fitted features whose coefficient must be negative to be physical.

Received level falls as ``1 / r`` in the free field, so ``log level`` has to
decrease as ``log r`` increases.  A fit that comes out positive has not learnt
distance from level; it has learnt whatever else moved the level.
"""


def warn_on_unphysical_level_coefficient(
    coefficients: Sequence[float],
    feature_names: Sequence[str],
) -> list[str]:
    """Warn on stderr when a level coefficient has the wrong sign.

    Returns the list of offending feature names, so a caller can record them.

    The eval-dev generator normalises every session to a *random* loudness
    target, which removes the ``1 / r`` level cue from the data altogether.  A
    level regression fitted on such a set produces a positive coefficient from
    noise and then looks like a working distance estimator, so the condition is
    reported loudly instead of being left in the coefficients for a reader to
    notice.

    >>> warn_on_unphysical_level_coefficient((-0.8, 1.2), ("log10_rms",))
    []
    >>> warn_on_unphysical_level_coefficient((0.0, 1.2), ("unrelated",))
    []
    """
    offenders = [
        name
        for name, coefficient in zip(feature_names, coefficients)
        if name in LEVEL_FEATURE_NAMES and coefficient >= 0.0
    ]
    for name in offenders:
        index = list(feature_names).index(name)
        print(
            f"WARNING: fitted coefficient for {name!r} is {coefficients[index]:+.4f}, "
            "but received level must FALL with distance, so a physical fit is "
            "negative. The level cue is absent or swamped in this training set "
            "(the eval-dev generator normalises each session to a random "
            "loudness target), so this model does not estimate distance from "
            "level. Report the uniform fallback instead of this fit, or fit a "
            "gain-invariant feature.",
            file=sys.stderr,
        )
    return offenders


def fit_ridge(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    alpha: float = 1.0,
) -> tuple[list[float], float]:
    """Fit ridge regression with an unpenalised intercept.

    ``features`` rows must *not* already contain a constant column; the
    intercept is appended here so that it can be excluded from the penalty.
    Returns ``(coefficients, residual_std)`` with the intercept last.

    >>> coef, sd = fit_ridge([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0], alpha=0.0)
    >>> [round(c, 6) for c in coef]
    [2.0, 1.0]
    >>> round(sd, 6)
    0.0
    """
    import numpy as np

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).ravel()
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError(f"feature/target shape mismatch: {x.shape} vs {y.shape}")
    if x.shape[0] == 0:
        raise ValueError("cannot fit on zero samples")
    design = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0  # never penalise the intercept
    gram = design.T @ design + penalty
    coef = np.linalg.solve(gram, design.T @ y)
    residual = y - design @ coef
    dof = max(1, y.size - design.shape[1])
    return [float(c) for c in coef], float(np.sqrt(float(residual @ residual) / dof))


# ---------------------------------------------------------------------------
# Audio access with an enforced observation boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioSession:
    """Multichannel session audio with a checked observation boundary.

    ``samples`` is ``(n_channels, n_samples)`` float64 in ``[-1, 1]``.  Callers
    never index it directly; they call :meth:`window`, which refuses to hand
    back a sample past the requested boundary.  That is the one place where the
    causality claim of this submission is enforced.
    """

    samples: FloatArray
    sample_rate_hz: int
    path: str = ""

    @property
    def n_channels(self) -> int:
        """Number of channels stored in this session."""
        return int(self.samples.shape[0])

    @property
    def duration_sec(self) -> float:
        """Session duration in seconds, from the sample count."""
        return float(self.samples.shape[1]) / float(self.sample_rate_hz)

    def sample_index(self, time_sec: float) -> int:
        """Return the sample index for ``time_sec``, clamped to the file.

        >>> import numpy as np
        >>> s = AudioSession(np.zeros((1, 16000)), 16000)
        >>> s.sample_index(0.5), s.sample_index(99.0)
        (8000, 16000)
        """
        idx = int(round(time_sec * self.sample_rate_hz))
        return max(0, min(idx, int(self.samples.shape[1])))

    def window(
        self,
        start_sec: float,
        end_sec: float,
        channels: Sequence[int] | None = None,
    ) -> FloatArray:
        """Return ``samples[channels, start:end]`` for the half-open interval.

        Raises if ``end_sec < start_sec``, or if a requested channel is not in
        the file.  No sample at or after ``round(end_sec * sr)`` is returned, so
        a caller in ``causal`` scope cannot see the future even by accident.

        >>> import numpy as np
        >>> s = AudioSession(np.arange(8, dtype=float).reshape(2, 4), 4)
        >>> s.window(0.0, 0.5, (0,)).tolist()
        [[0.0, 1.0]]
        >>> s.window(0.25, 1.0, (1,)).tolist()
        [[5.0, 6.0, 7.0]]
        >>> s.window(0.5, 0.25)
        Traceback (most recent call last):
            ...
        ValueError: end_sec 0.25 precedes start_sec 0.5
        """
        if end_sec < start_sec:
            raise ValueError(f"end_sec {end_sec} precedes start_sec {start_sec}")
        if channels is None:
            channels = tuple(range(self.n_channels))
        for ch in channels:
            if not 0 <= ch < self.n_channels:
                raise ValueError(
                    f"channel {ch} not in {self.path or 'session'} "
                    f"with {self.n_channels} channels"
                )
        start = self.sample_index(start_sec)
        end = self.sample_index(end_sec)
        return self.samples[list(channels), start:end]


def load_audio(path: str | os.PathLike[str], backend: str = "auto") -> AudioSession:
    """Load a multichannel audio file into an :class:`AudioSession`.

    ``wave`` from the stdlib handles PCM WAV without any third-party package,
    which keeps the pipeline runnable in a bare environment.  Anything else
    (notably the FLAC distribution format) needs ``soundfile``, which is
    imported lazily and reported clearly when missing rather than installed.
    """
    import numpy as np

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"audio file not found: {file_path}")
    use_wave = backend == "wave" or (backend == "auto" and file_path.suffix.lower() == ".wav")
    if use_wave:
        try:
            with wave.open(str(file_path), "rb") as handle:
                n_channels = handle.getnchannels()
                width = handle.getsampwidth()
                sample_rate = handle.getframerate()
                raw = handle.readframes(handle.getnframes())
        except wave.Error as exc:
            if backend == "wave":
                raise
            return _load_audio_soundfile(file_path, note=str(exc))
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        if width not in dtype_map:
            return _load_audio_soundfile(file_path, note=f"unsupported sample width {width}")
        flat = np.frombuffer(raw, dtype=dtype_map[width]).astype(np.float64)
        if width == 1:  # 8-bit PCM WAV is unsigned
            flat = flat - 128.0
            scale = 128.0
        else:
            scale = float(2 ** (8 * width - 1))
        data = flat.reshape(-1, n_channels).T / scale
        return AudioSession(np.ascontiguousarray(data), int(sample_rate), str(file_path))
    return _load_audio_soundfile(file_path)


def _load_audio_soundfile(file_path: Path, note: str = "") -> AudioSession:
    """Load audio through ``soundfile``, explaining clearly if it is absent."""
    import numpy as np

    try:
        import soundfile  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        extra = f" ({note})" if note else ""
        raise RuntimeError(
            f"cannot read {file_path.name}{extra}: the stdlib 'wave' module only "
            "reads PCM WAV and 'soundfile' is not installed in this environment. "
            "Either distribute eval-dev audio as PCM WAV, or install soundfile "
            "(libsndfile) before running the baselines. No download is attempted "
            "by this script."
        ) from exc
    data, sample_rate = soundfile.read(str(file_path), dtype="float64", always_2d=True)
    return AudioSession(
        np.ascontiguousarray(np.asarray(data, dtype=np.float64).T),
        int(sample_rate),
        str(file_path),
    )


# ---------------------------------------------------------------------------
# Eval-set discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseSpec:
    """One evaluation case: a session restricted to one channel subset."""

    case_id: str
    session_id: str
    mask_bits: str
    audio_path: str
    track: int = DEFAULT_TRACK
    mic_positions_m: tuple[tuple[float, float], ...] = ()
    geometry_source: str = "default_uca"
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> tuple[int, ...]:
        """Active channel indices, ascending.

        >>> CaseSpec("s__m1010", "s", "1010", "a.wav").channels
        (0, 2)
        """
        return spec.mask_to_channels(self.mask_bits)

    @property
    def n_active_channels(self) -> int:
        """``M``, the number of active microphones for this case."""
        return len(self.channels)

    def positions(self) -> tuple[tuple[float, float], ...]:
        """Mic coordinates, falling back to the default circular array."""
        return self.mic_positions_m or default_mic_positions_m()


class NotACaseListError(ValueError):
    """A candidate metadata file is a dataset *summary*, not a per-case list.

    The eval-dev generator writes two JSON files side by side: ``meta.jsonl``
    with one row per ``case_id``, and ``manifest.json`` with dataset-level
    counts, stratification and file hashes.  Auto-discovery must not mistake
    the summary for the case list, so a payload whose values are not case rows
    raises this and :func:`discover_cases` moves on to the next candidate.
    """


_MANIFEST_CANDIDATES: Final[tuple[str, ...]] = (
    # meta.jsonl first: that is the per-case input metadata the eval-dev
    # generator in this repository actually writes, and the manifest.json it
    # writes beside it is a dataset summary that must not be read as a case
    # list (see NotACaseListError).
    "meta.jsonl",
    "meta.json",
    "manifest.jsonl",
    "metadata.jsonl",
    "input_metadata.jsonl",
    "sessions.jsonl",
    "cases.jsonl",
    "index.jsonl",
    "manifest.json",
    "metadata.json",
    "input_metadata.json",
    "sessions.json",
    "cases.json",
    "index.json",
    "manifest.csv",
)
"""Case-list file names tried in order; no layout is frozen in ``spec.py``.

Every ``.jsonl`` candidate is tried before any ``.json`` one, because a
one-row-per-line file is a case list by construction while a single JSON
object is just as likely to be a summary.
"""

_AUDIO_SUFFIXES: Final[tuple[str, ...]] = (".wav", ".flac", ".ogg", ".mp3")

_KEY_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "case_id": ("case_id", "caseId", "case"),
    "session_id": ("session_id", "sessionId", "session", "id", "utt_id"),
    "mask_bits": ("mask_bits", "maskBits", "mask", "channel_mask"),
    "audio_path": ("audio_path", "audio", "wav", "wav_path", "path", "filename", "file"),
    "track": ("track", "track_id"),
    "active_channels": ("active_channels", "read_channels", "channels"),
    "mic_positions_m": (
        "mic_positions_m",
        "mic_positions_xy_m",
        "mic_positions",
        "mic_coords_m",
        "mic_coords",
        "array_geometry",
        "positions",
        "P",
    ),
    "sample_rate_hz": ("sample_rate_hz", "sample_rate", "sr", "fs"),
}


def _first_key(row: Mapping[str, Any], logical: str) -> Any:
    """Return the first present alias of ``logical`` in ``row``, else ``None``."""
    for key in _KEY_ALIASES[logical]:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _coerce_positions(raw: Any) -> tuple[tuple[float, float], ...]:
    """Coerce assorted geometry encodings into ``((x, y), ...)``.

    >>> _coerce_positions([[0.1, 0.0], [0.0, 0.1]])
    ((0.1, 0.0), (0.0, 0.1))
    >>> _coerce_positions({"0": {"x": 1.0, "y": 2.0}})
    ((1.0, 2.0),)
    >>> _coerce_positions(None)
    ()
    """
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        items = [raw[key] for key in sorted(raw, key=lambda k: int(k))]
    else:
        items = list(raw)
    out: list[tuple[float, float]] = []
    for item in items:
        if isinstance(item, Mapping):
            out.append((float(item["x"]), float(item["y"])))
        else:
            values = list(item)
            out.append((float(values[0]), float(values[1])))
    return tuple(out)


def _absolute_positions(
    positions: Sequence[Sequence[float]],
    mask_bits: str,
    active_channels: Sequence[int] | None = None,
    n_channels_max: int = spec.N_CHANNELS_MAX,
) -> tuple[tuple[float, float], ...]:
    """Return mic coordinates indexed by **absolute** channel index.

    Every consumer -- :func:`pair_spacings_m`, the steering table and the
    beamformer -- looks a microphone up by its absolute channel index, so that
    dropping channels never moves the coordinate frame.  A per-case manifest
    row may instead list only the *active* channels' coordinates in subset
    order; that is what the eval-dev generator's ``mic_positions_xy_m`` is.
    Those are scattered back onto their absolute slots.

    Slots belonging to inactive channels are filled with ``nan``: no consumer
    reads them for this mask, and if one ever does the arithmetic turns to
    ``nan`` instead of quietly using the wrong microphone.

    A row that already carries ``n_channels_max`` coordinates is taken as
    absolute and returned unchanged.

    >>> _absolute_positions([(1.0, 2.0)], "0010")
    ((nan, nan), (nan, nan), (1.0, 2.0), (nan, nan))
    >>> _absolute_positions([(0.0, 0.0), (1.0, 1.0)], "1010")[2]
    (1.0, 1.0)
    >>> _absolute_positions([(0.1, 0.0)] * 4, "1010")[1]
    (0.1, 0.0)
    >>> _absolute_positions([], "1111")
    ()
    >>> _absolute_positions([(0.0, 0.0)], "1100")
    Traceback (most recent call last):
        ...
    ValueError: mask '1100' activates 2 channel(s) but 1 coordinate(s) were given
    """
    coordinates = tuple((float(p[0]), float(p[1])) for p in positions)
    if not coordinates:
        return ()
    if len(coordinates) == n_channels_max:
        return coordinates
    channels = (
        tuple(int(c) for c in active_channels)
        if active_channels
        else spec.mask_to_channels(mask_bits)
    )
    if len(coordinates) != len(channels):
        raise ValueError(
            f"mask {mask_bits!r} activates {len(channels)} channel(s) but "
            f"{len(coordinates)} coordinate(s) were given"
        )
    nan = float("nan")
    out: list[tuple[float, float]] = [(nan, nan) for _ in range(n_channels_max)]
    for coordinate, channel in zip(coordinates, channels):
        if not 0 <= channel < n_channels_max:
            raise ValueError(
                f"channel index {channel} outside 0..{n_channels_max - 1} in a "
                f"case-list row for mask {mask_bits!r}"
            )
        out[channel] = coordinate
    return tuple(out)


def _iter_manifest_rows(manifest_path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield rows from a JSONL, JSON or CSV case list.

    Raises :class:`NotACaseListError` when a single JSON object is clearly a
    dataset summary rather than a collection of case rows.
    """
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".csv":
        yield from csv.DictReader(text.splitlines())
        return
    if manifest_path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    payload = json.loads(text)
    if isinstance(payload, Mapping):
        for key in ("sessions", "cases", "records", "items", "data"):
            if key in payload and isinstance(payload[key], list):
                yield from payload[key]
                return
        # The remaining reading is "mapping of session_id -> row". A summary
        # document also passes `isinstance(payload, Mapping)`, so require that
        # every value really is a row before treating it as one.
        if not payload or not all(isinstance(value, Mapping) for value in payload.values()):
            raise NotACaseListError(
                f"{manifest_path.name} is a mapping whose values are not case "
                "rows, so it is a dataset summary rather than a case list. "
                "Point --manifest at the per-case file (the eval-dev generator "
                "writes meta.jsonl)."
            )
        for session_id, row in payload.items():
            merged = dict(row)
            merged.setdefault("session_id", session_id)
            yield merged
        return
    yield from payload


def discover_cases(
    eval_dir: str | os.PathLike[str],
    manifest: str | os.PathLike[str] | None = None,
    track: int = DEFAULT_TRACK,
    masks: Sequence[str] = spec.EVAL_MASKS,
    limit: int | None = None,
) -> list[CaseSpec]:
    """Enumerate the evaluation cases found under ``eval_dir``.

    A case list is read from ``manifest`` or from the first of
    :data:`_MANIFEST_CANDIDATES` that exists *and parses as a case list* -- a
    candidate that turns out to be a dataset summary is skipped with a note
    (:class:`NotACaseListError`).  Rows that already name a ``case_id`` (or a
    ``mask_bits``) are taken as-is; rows that only name a session are expanded
    over ``masks``.  Either way the result is filtered to ``masks``, which
    defaults to the six frozen :data:`spec.EVAL_MASKS`, and then to ``track``.
    With no case list at all, audio files are globbed and expanded the same
    way, so the pipeline can be exercised on a directory of WAVs.

    Two shapes of row are tolerated beyond the flat one: ``audio`` may be an
    object carrying ``path`` and ``sample_rate_hz`` instead of a bare path
    string, and the microphone coordinates may cover only the mask's active
    channels in subset order, which :func:`_absolute_positions` scatters back
    onto absolute channel indices.  Both are what the eval-dev generator's
    ``meta.jsonl`` writes.
    """
    root = Path(eval_dir)
    if not root.exists():
        raise FileNotFoundError(f"eval dir not found: {root}")
    if track not in spec.MAX_SPEAKERS_PER_TRACK:
        raise ValueError(f"track must be one of {sorted(spec.MAX_SPEAKERS_PER_TRACK)}, got {track}")

    rows: list[Mapping[str, Any]] | None = None
    manifest_path: Path | None = Path(manifest) if manifest is not None else None
    if manifest_path is not None:
        rows = [dict(row) for row in _iter_manifest_rows(manifest_path)]
    else:
        for name in _MANIFEST_CANDIDATES:
            candidate = root / name
            if not candidate.exists():
                continue
            try:
                rows = [dict(row) for row in _iter_manifest_rows(candidate)]
            except NotACaseListError as exc:
                print(f"note: not a case list, skipping: {exc}", file=sys.stderr)
                continue
            manifest_path = candidate
            break

    if rows is not None:
        print(f"note: case list read from {manifest_path}", file=sys.stderr)
    else:
        audio_files = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in _AUDIO_SUFFIXES
        )
        if not audio_files:
            raise FileNotFoundError(
                f"no case list ({'/'.join(_MANIFEST_CANDIDATES[:4])}, ...) and no audio "
                f"files under {root}. Pass --manifest explicitly."
            )
        rows = [{"session_id": p.stem, "audio_path": str(p)} for p in audio_files]
        print(
            f"note: no case list found under {root}; globbed {len(rows)} audio file(s) "
            f"and expanded {len(masks)} mask(s) from spec.EVAL_MASKS",
            file=sys.stderr,
        )

    cases: list[CaseSpec] = []
    for row in rows:
        raw_case_id = _first_key(row, "case_id")
        mask_value = _first_key(row, "mask_bits")
        session_value = _first_key(row, "session_id")
        if raw_case_id:
            session_id, mask_bits = spec.parse_case_id(str(raw_case_id))
            row_masks: tuple[str, ...] = (mask_bits,)
        elif session_value is None:
            raise ValueError(f"manifest row names neither case_id nor session_id: {row!r}")
        else:
            session_id = str(session_value)
            row_masks = (str(mask_value),) if mask_value else tuple(masks)

        audio_value = _first_key(row, "audio_path")
        audio_info: Mapping[str, Any] | None = None
        if isinstance(audio_value, Mapping):
            # The eval-dev generator writes 'audio' as an object
            # ({"path": ..., "sample_rate_hz": ..., "read_channels": [...]}),
            # not a bare path string.
            audio_info = audio_value
            audio_value = _first_key(audio_info, "audio_path")
        if audio_value is None:
            found = [
                p
                for p in root.rglob(f"{session_id}.*")
                if p.suffix.lower() in _AUDIO_SUFFIXES
            ]
            if not found:
                raise FileNotFoundError(
                    f"no audio for session {session_id!r} under {root}; add an "
                    "'audio_path' column to the manifest"
                )
            audio_path = str(sorted(found)[0])
        else:
            candidate = Path(str(audio_value))
            audio_path = str(candidate if candidate.is_absolute() else root / candidate)

        raw_positions = _coerce_positions(_first_key(row, "mic_positions_m"))
        active = _first_key(row, "active_channels")
        if active is None and audio_info is not None:
            active = _first_key(audio_info, "active_channels")
        row_track = int(_first_key(row, "track") or track)
        extra = {
            key: value
            for key, value in row.items()
            if key not in {alias for aliases in _KEY_ALIASES.values() for alias in aliases}
        }
        sr = _first_key(row, "sample_rate_hz")
        if sr is None and audio_info is not None:
            sr = _first_key(audio_info, "sample_rate_hz")
        if sr is not None:
            extra["manifest_sample_rate_hz"] = int(sr)
        for mask_bits in row_masks:
            positions = _absolute_positions(raw_positions, mask_bits, active)
            cases.append(
                CaseSpec(
                    case_id=spec.make_case_id(session_id, mask_bits),
                    session_id=session_id,
                    mask_bits=mask_bits,
                    audio_path=audio_path,
                    track=row_track,
                    mic_positions_m=positions,
                    geometry_source="manifest" if positions else "default_uca",
                    extra=extra,
                )
            )
    requested_masks = tuple(masks)
    kept = [c for c in cases if c.mask_bits in requested_masks]
    if len(kept) != len(cases):
        print(
            f"note: --masks kept {len(kept)} of {len(cases)} case(s) from the case "
            f"list; {sorted({c.mask_bits for c in cases} - set(requested_masks))} "
            "were not requested",
            file=sys.stderr,
        )
    cases = [c for c in kept if c.track == track]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError(f"no cases for track {track} under {root}")
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise ValueError(f"duplicate case_id in case list: {case.case_id!r}")
        seen.add(case.case_id)
    return cases


# ---------------------------------------------------------------------------
# Distance models
# ---------------------------------------------------------------------------


@dataclass
class DistanceModel:
    """A distance predictor, either a fixed prior or a (ridge) regression.

    ``kind`` is one of ``"uniform"``, ``"prior"`` or ``"regression"``.  A
    regression maps ``features`` to ``log(distance)``; ``sigma_log`` is the
    residual spread used to widen the emitted distribution, so a badly fit
    model answers with an honestly broad distribution instead of a confident
    wrong bin.  ``feature_names`` documents the design matrix columns.
    """

    kind: str = "uniform"
    coefficients: tuple[float, ...] = ()
    feature_names: tuple[str, ...] = ()
    sigma_log: float = 0.5
    prior: tuple[float, ...] = ()
    alpha: float = 1.0
    n_train: int = 0
    note: str = ""

    def describe(self) -> dict[str, Any]:
        """Return JSON-safe metadata recording how distance was produced.

        >>> DistanceModel().describe()["distance_source"]
        'uniform'
        """
        return {
            "distance_source": self.kind,
            "distance_feature_names": list(self.feature_names),
            "distance_sigma_log": self.sigma_log,
            "distance_ridge_alpha": self.alpha,
            "distance_n_train": self.n_train,
            "distance_note": self.note,
        }

    def predict_distribution(self, features: Sequence[float]) -> list[float]:
        """Return a distance distribution over the frozen distance bins.

        ``uniform`` ignores the features, which is the honest fallback when no
        labelled data has been fitted yet: it scores ``log K_r`` and claims
        nothing.

        >>> m = DistanceModel()
        >>> len(m.predict_distribution([0.0])) == spec.K_DISTANCE
        True
        >>> p = DistanceModel(kind="prior", prior=tuple([1.0] * spec.K_DISTANCE))
        >>> abs(sum(p.predict_distribution([])) - 1.0) < spec.PROB_SUM_TOLERANCE
        True
        """
        if self.kind == "uniform":
            return uniform_distribution(spec.K_DISTANCE)
        if self.kind == "prior":
            if len(self.prior) != spec.K_DISTANCE:
                raise ValueError(
                    f"prior must have {spec.K_DISTANCE} entries, got {len(self.prior)}"
                )
            return normalise_distribution(self.prior)
        if self.kind == "regression":
            return kernel_distribution(
                distance_centers(),
                self.predict_point(features),
                self.sigma_log,
                log_domain=True,
            )
        raise ValueError(f"unknown distance model kind {self.kind!r}")

    def predict_point(self, features: Sequence[float]) -> float:
        """Return the point distance estimate in meters, clipped to the range.

        >>> m = DistanceModel(kind="regression", coefficients=(0.0, 0.0),
        ...                   feature_names=("f",))
        >>> round(m.predict_point([3.0]), 6)
        1.0
        """
        low, high = spec.DISTANCE_RANGE_M
        if self.kind == "regression":
            if len(self.coefficients) != len(features) + 1:
                raise ValueError(
                    f"expected {len(self.coefficients) - 1} features, got {len(features)}"
                )
            acc = self.coefficients[-1]
            for coef, value in zip(self.coefficients[:-1], features):
                acc += coef * value
            return float(min(high, max(low, math.exp(acc))))
        return map_point(self.predict_distribution(features), distance_centers())

    def to_json(self) -> dict[str, Any]:
        """Serialise the model for ``--save-distance-model``."""
        return {
            "kind": self.kind,
            "coefficients": list(self.coefficients),
            "feature_names": list(self.feature_names),
            "sigma_log": self.sigma_log,
            "prior": list(self.prior),
            "alpha": self.alpha,
            "n_train": self.n_train,
            "note": self.note,
            "spec_version": spec.SPEC_VERSION,
        }


def load_distance_model(path: str | os.PathLike[str]) -> DistanceModel:
    """Load a :class:`DistanceModel` written by ``--save-distance-model``.

    A bare list or dict of ``K_r`` numbers is accepted as a training-set prior,
    which is the common hand-written case.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return DistanceModel(
            kind="prior",
            prior=tuple(float(v) for v in payload),
            note=f"train-set prior from {Path(path).name}",
        )
    if "kind" not in payload:
        if "prior" in payload:
            return DistanceModel(
                kind="prior",
                prior=tuple(float(v) for v in payload["prior"]),
                note=f"train-set prior from {Path(path).name}",
            )
        counts = [float(payload[str(j)]) for j in range(spec.K_DISTANCE)]
        return DistanceModel(
            kind="prior", prior=tuple(counts), note=f"train-set prior from {Path(path).name}"
        )
    return DistanceModel(
        kind=str(payload["kind"]),
        coefficients=tuple(float(v) for v in payload.get("coefficients", ())),
        feature_names=tuple(str(v) for v in payload.get("feature_names", ())),
        sigma_log=float(payload.get("sigma_log", 0.5)),
        prior=tuple(float(v) for v in payload.get("prior", ())),
        alpha=float(payload.get("alpha", 1.0)),
        n_train=int(payload.get("n_train", 0)),
        note=str(payload.get("note", "")) or f"loaded from {Path(path).name}",
    )


# ---------------------------------------------------------------------------
# Frozen ASR backend (shared controlled variable)
# ---------------------------------------------------------------------------

DEFAULT_ASR_MODEL: Final[str] = "facebook/wav2vec2-base-960h"
"""The frozen CTC checkpoint both controlled baselines use.

Held identical between Baseline 1 and Baseline 2 on purpose: the comparison is
about the spatial front end, so the recogniser must not vary.  Nothing here
downloads it -- see :class:`FrozenASR`.
"""

SYNTHETIC_TRANSCRIPT: Final[str] = "<synthetic-asr-placeholder>"


@dataclass
class FrozenASR:
    """Thin wrapper around a frozen pretrained CTC ASR checkpoint.

    The checkpoint is resolved with ``local_files_only=True`` and
    ``HF_HUB_OFFLINE=1``, so a missing checkpoint is an immediate, explained
    failure instead of a silent multi-hundred-megabyte download.  ``torch``,
    ``transformers`` and ``numpy`` are imported inside :meth:`_ensure_loaded`.

    With ``synthetic=True`` the pipeline runs end to end without any
    checkpoint, emitting :data:`SYNTHETIC_TRANSCRIPT` for every segment.  Those
    records are stamped ``wer_reportable=false``.
    """

    model_name: str = DEFAULT_ASR_MODEL
    synthetic: bool = False
    target_sample_rate_hz: int = 16000
    _bundle: Any = field(default=None, repr=False, compare=False)

    @property
    def backend_id(self) -> str:
        """Identifier recorded in the output metadata.

        >>> FrozenASR(synthetic=True).backend_id
        'synthetic-placeholder'
        """
        return "synthetic-placeholder" if self.synthetic else self.model_name

    def _ensure_loaded(self) -> Any:
        """Load processor and model once, or explain why it is impossible."""
        if self._bundle is not None:
            return self._bundle
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import torch  # noqa: F401  (loaded for its side effect on transformers)
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as exc:
            raise SystemExit(
                f"frozen ASR unavailable: {exc}. The controlled baselines need "
                "'torch' and 'transformers' with the checkpoint "
                f"{self.model_name!r} already present in the local cache. This "
                "script never downloads and never installs. Either provision the "
                "checkpoint offline and re-run, or re-run with --synthetic-asr to "
                "exercise the pipeline without transcription (WER is then not "
                "reportable)."
            ) from exc
        try:
            processor = Wav2Vec2Processor.from_pretrained(self.model_name, local_files_only=True)
            model = Wav2Vec2ForCTC.from_pretrained(self.model_name, local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - any resolution failure is terminal
            raise SystemExit(
                f"frozen ASR checkpoint {self.model_name!r} is not in the local "
                f"cache ({type(exc).__name__}: {exc}). Provision it offline, pass "
                "--asr-model with a local path, or re-run with --synthetic-asr."
            ) from exc
        model.eval()
        self._bundle = (processor, model)
        return self._bundle

    def transcribe(self, mono: FloatArray, sample_rate_hz: int) -> str:
        """Transcribe one mono segment and return normalised-ish raw text.

        Text normalisation for scoring is the evaluator's job
        (:data:`spec.TEXT_NORMALISATION`); this returns the recogniser's own
        output, lowercased only because the CTC vocabulary is upper case.

        >>> import numpy as np
        >>> FrozenASR(synthetic=True).transcribe(np.zeros(16000), 16000)
        '<synthetic-asr-placeholder>'
        """
        import numpy as np

        if self.synthetic:
            return SYNTHETIC_TRANSCRIPT
        processor, model = self._ensure_loaded()
        import torch

        audio = np.asarray(mono, dtype=np.float64).ravel()
        if sample_rate_hz != self.target_sample_rate_hz:
            audio = _resample_linear(audio, sample_rate_hz, self.target_sample_rate_hz)
        inputs = processor(
            audio.astype("float32"),
            sampling_rate=self.target_sample_rate_hz,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        ids = logits.argmax(dim=-1)
        return str(processor.batch_decode(ids)[0]).strip().lower()


def _resample_linear(signal: FloatArray, source_hz: int, target_hz: int) -> FloatArray:
    """Linearly resample a mono signal.

    Good enough to feed a 16 kHz CTC model from a nearby rate; it is not a
    high-quality resampler and the eval set is expected to be 16 kHz already,
    in which case this is never called.

    >>> import numpy as np
    >>> _resample_linear(np.array([0.0, 1.0, 2.0, 3.0]), 4, 2).tolist()
    [0.0, 2.0]
    """
    import numpy as np

    arr = np.asarray(signal, dtype=np.float64).ravel()
    if source_hz == target_hz or arr.size == 0:
        return arr
    n_out = int(round(arr.size * target_hz / source_hz))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float64)
    src_idx = np.arange(arr.size, dtype=np.float64)
    dst_idx = np.arange(n_out, dtype=np.float64) * (source_hz / target_hz)
    return np.interp(dst_idx, src_idx, arr)


# ---------------------------------------------------------------------------
# Output records
# ---------------------------------------------------------------------------


def build_timestep_record(
    case_id: str,
    eval_time_sec: float,
    speaker_id: str,
    azimuth_probs: Sequence[float],
    distance_probs: Sequence[float],
    pred_azimuth_deg: float | None = None,
    pred_distance_m: float | None = None,
    include_record_type: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``(case_id, eval_time_sec)`` prediction record.

    The explicit point fields are filled from the distributions' MAP bin
    centers when not supplied, so an omitted point can never be mistaken for a
    deliberate abstention.  Lengths are validated against the frozen bin counts
    before the record leaves this function.

    >>> import spec
    >>> rec = build_timestep_record(
    ...     "s__m1000", 0.25, "spk_0",
    ...     uniform_distribution(spec.K_AZIMUTH),
    ...     uniform_distribution(spec.K_DISTANCE))
    >>> rec["record_type"], rec["eval_time_sec"], rec["speaker_id"]
    ('timepoint', 0.25, 'spk_0')
    >>> rec["pred_azimuth_deg"]
    -175.0
    """
    if len(azimuth_probs) != spec.K_AZIMUTH:
        raise ValueError(f"azimuth_probs must have {spec.K_AZIMUTH} entries")
    if len(distance_probs) != spec.K_DISTANCE:
        raise ValueError(f"distance_probs must have {spec.K_DISTANCE} entries")
    for name, vector in (("azimuth_probs", azimuth_probs), ("distance_probs", distance_probs)):
        err = abs(math.fsum(vector) - 1.0)
        if err >= spec.PROB_SUM_TOLERANCE:
            raise ValueError(f"{name} sums to 1 +- {err:g}, outside PROB_SUM_TOLERANCE")
    record: dict[str, Any] = {}
    if include_record_type:
        record["record_type"] = RECORD_TYPE_TIMEPOINT
    record.update(
        {
            "case_id": case_id,
            "eval_time_sec": eval_time_sec,
            "speaker_id": speaker_id,
            "azimuth_probs": list(azimuth_probs),
            "distance_probs": list(distance_probs),
            "pred_azimuth_deg": (
                map_point(azimuth_probs, azimuth_centers())
                if pred_azimuth_deg is None
                else float(pred_azimuth_deg)
            ),
            "pred_distance_m": (
                map_point(distance_probs, distance_centers())
                if pred_distance_m is None
                else float(pred_distance_m)
            ),
        }
    )
    if extra:
        record.update(extra)
    return record


def build_final_record(
    case_id: str,
    final_transcripts: Mapping[str, str],
    observation_mode: str,
    reported_rtf: float,
    system: str,
    track: int = DEFAULT_TRACK,
    wer_reportable: bool = True,
    include_record_type: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the one ``final_transcripts`` record for a case.

    The speaker-slot count is validated against
    :data:`spec.MAX_SPEAKERS_PER_TRACK` here rather than left to the evaluator,
    so an over-segmenting tracker fails loudly at write time.

    >>> rec = build_final_record("s__m1000", {"spk_0": "hello"}, "causal",
    ...                          0.1, "baseline1")
    >>> rec["record_type"], sorted(rec["final_transcripts"])
    ('final_transcripts', ['spk_0'])
    >>> build_final_record("s__m1000", {"spk_0": "a", "spk_1": "b"}, "causal",
    ...                    0.1, "b1", track=1)
    Traceback (most recent call last):
        ...
    ValueError: track 1 allows at most 1 speaker slot(s), got 2
    """
    limit = spec.MAX_SPEAKERS_PER_TRACK[track]
    if len(final_transcripts) > limit:
        raise ValueError(
            f"track {track} allows at most {limit} speaker slot(s), "
            f"got {len(final_transcripts)}"
        )
    if observation_mode not in spec.OBSERVATION_MODES:
        raise ValueError(f"observation_mode must be in {spec.OBSERVATION_MODES}")
    record: dict[str, Any] = {}
    if include_record_type:
        record["record_type"] = RECORD_TYPE_FINAL_TRANSCRIPTS
    record.update(
        {
            "case_id": case_id,
            "final_transcripts": dict(final_transcripts),
            "observation_mode": observation_mode,
            "reported_rtf": float(reported_rtf),
            "hardware": hardware_tag(),
            "system": system,
            "track": track,
            "spec_version": spec.SPEC_VERSION,
            "baseline_impl_version": BASELINE_IMPL_VERSION,
            "wer_reportable": bool(wer_reportable),
            "verification_tag": "self_reported",
        }
    )
    if meta:
        record["meta"] = dict(meta)
    return record


def write_predictions(
    out_path: str | os.PathLike[str],
    records: Iterable[Mapping[str, Any]],
) -> int:
    """Write ``records`` to ``out_path`` as JSONL and return the line count.

    Floats are written with ``repr`` (json's default), so every probability
    round-trips exactly and the sum the evaluator computes is the sum this
    module validated.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False))
            handle.write("\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# CLI plumbing shared by both baselines
# ---------------------------------------------------------------------------


def add_common_cli_args(parser: Any) -> Any:
    """Add the CLI surface both baselines share to an ``ArgumentParser``.

    Kept here so the two baselines cannot drift on the arguments that define
    the controlled comparison (scope, track, ASR checkpoint, masks).
    """
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="eval-dev directory (audio plus input metadata / case list)",
    )
    parser.add_argument("--out", required=True, help="output predictions.jsonl path")
    parser.add_argument(
        "--scope",
        default=spec.OFFICIAL_SCOPES[0],
        choices=list(spec.OBSERVATION_MODES),
        help=f"observation scope; official: {spec.OFFICIAL_SCOPES}",
    )
    parser.add_argument(
        "--track",
        type=int,
        default=DEFAULT_TRACK,
        choices=sorted(spec.MAX_SPEAKERS_PER_TRACK),
        help="benchmark track (this round freezes Track 1)",
    )
    parser.add_argument(
        "--lookahead-sec",
        type=float,
        default=spec.CHUNK_HOP_SEC,
        help="lookahead W for --scope windowed (unofficial; W is not frozen)",
    )
    parser.add_argument("--manifest", default=None, help="explicit case-list path")
    parser.add_argument(
        "--masks",
        default=",".join(spec.EVAL_MASKS),
        help="comma-separated channel masks to expand sessions over",
    )
    parser.add_argument("--limit", type=int, default=None, help="process at most N cases")
    parser.add_argument(
        "--asr-model",
        default=DEFAULT_ASR_MODEL,
        help="frozen ASR checkpoint id or local path (never downloaded)",
    )
    parser.add_argument(
        "--synthetic-asr",
        action="store_true",
        help="run the pipeline with placeholder transcripts; WER is NOT reportable",
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        help="skip transcription entirely and emit empty final transcripts",
    )
    parser.add_argument(
        "--audio-backend",
        default="auto",
        choices=["auto", "wave", "soundfile"],
        help="audio reader; 'wave' is stdlib PCM WAV only",
    )
    parser.add_argument(
        "--distance-model",
        default=None,
        help="JSON distance model or train-set prior (default: uniform fallback)",
    )
    parser.add_argument(
        "--prob-decimals",
        type=int,
        default=DEFAULT_PROB_DECIMALS,
        help="round emitted probabilities to N decimals before the exact-sum repair",
    )
    parser.add_argument(
        "--omit-record-type",
        action="store_true",
        help="do not emit the 'record_type' discriminator field",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run doctests and the synthetic smoke tests, then exit",
    )
    return parser


def resolve_masks(raw: str) -> tuple[str, ...]:
    """Parse a ``--masks`` value, validating each mask against the contract.

    >>> resolve_masks("1000,1111")
    ('1000', '1111')
    >>> resolve_masks("111")
    Traceback (most recent call last):
        ...
    ValueError: mask_bits must be 4 binary digits, got '111'
    """
    masks = tuple(part.strip() for part in raw.split(",") if part.strip())
    for mask in masks:
        spec.mask_to_channels(mask)
    return masks


class CaseTimer:
    """Wall-clock timer that converts to the ``reported_rtf`` metadata field.

    >>> t = CaseTimer()
    >>> with t:
    ...     pass
    >>> t.rtf(0.0)
    0.0
    """

    def __init__(self) -> None:
        self.elapsed_sec: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "CaseTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> Literal[False]:
        self.elapsed_sec += time.perf_counter() - self._start
        return False

    def rtf(self, audio_sec: float) -> float:
        """Return elapsed wall clock divided by processed audio seconds."""
        if audio_sec <= 0.0:
            return 0.0
        return self.elapsed_sec / audio_sec


# This file is a package __init__ and deliberately has no __main__ block:
# running it as a script would put baselines/ on sys.path instead of the
# repository root, so `import spec` above would fail before any block here
# could run. Its doctests are collected from the repository root instead, by
# `python3 -m doctest baselines/__init__.py` (what `make test` runs) and by
# either baseline's `--self-test`.
