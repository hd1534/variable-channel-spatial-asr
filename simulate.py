"""Multi-channel spatial acoustic session simulator for the VC-SASR benchmark.

The module is split into two layers on purpose:

**Plan layer (pure standard library).**  Room geometry, array geometry, source
placement, stratification, 2x2 confound cells, gains, start offsets and
train/val/eval split assignment are all derived deterministically from an
integer seed with :class:`random.Random`.  Nothing in this layer imports
``numpy``, ``pyroomacoustics`` or ``soundfile``, so every invariant the
benchmark depends on can be verified with ``python3 simulate.py --self-test``
on a bare interpreter.

**Render layer (lazy imports).**  RIR computation needs
``pyroomacoustics``; reading LibriSpeech flac needs ``soundfile``; convolution
needs ``numpy``.  All three are imported *inside* functions, with an actionable
message when they are missing.  Nothing here installs packages or touches the
network.

Every quantity that ``evaluate.py`` and the baselines must agree on is taken
from :mod:`spec` -- no contract constant is restated here.  Constants that
:mod:`spec` does not fix (sample rate, room size range, start-offset range,
...) are module-level constants named ``*_LOCAL`` free or documented below with
a ``[spec gap]`` marker, and are stamped into every manifest so a later
contract change is detectable.

Design decisions that protect the headline claim ("M>=2 beats M=1"):

1.  One horizontal plane.  Every microphone and every source sits at
    ``spec.PLANE_HEIGHT_M``, so the 2D distance *is* the true slant range.
    :func:`verify_plane_consistency` asserts it.
2.  Fixed coordinate frame.  The array reference point is the constant
    ``(0, 0)``; it is never recomputed as the centroid of the *active*
    microphones, so dropping a channel cannot move the origin or rotate the
    axes.  :func:`verify_frame_invariance` asserts it over all
    ``spec.EVAL_MASKS``.
3.  Stratified labels.  Azimuth bins and distance bins are filled by an even
    schedule, so an audio-blind model that emits the training prior cannot beat
    ``log K``.  :func:`verify_stratification` asserts it and
    :func:`prior_only_nll` reports exactly how much a memoriser would gain.
4.  Loudness is not a distance cue.  A session-level common gain is
    randomised over a wide range, while per-channel mismatch is capped at
    ``CHANNEL_GAIN_MISMATCH_DB`` so the inter-channel level difference (a real
    spatial cue) survives.
5.  Distance and RT60 are crossed, not confounded: each room block holds one
    near and one far source position, each rendered under a low-RT60 and a
    high-RT60 absorption setting.
6.  RT60 labels are *measured* from the generated RIR, never the requested
    value; both are recorded.
7.  Start offsets are drawn continuously from ``U(0, MAX_START_OFFSET_SEC)``
    and are never quantised to a multiple of ``spec.DELTA_T_SEC``; the same
    offset is applied to every channel of a case so the TDOA survives.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Literal, Sequence

if __package__ in (None, "") and str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import spec

__all__ = [
    "GENERATOR_VERSION",
    "MicArrayGeometry",
    "SourcePlacement",
    "RoomBlock",
    "SessionPlan",
    "EvalPlan",
    "even_bin_schedule",
    "make_mic_array",
    "sample_azimuth_in_bin",
    "sample_distance_in_bin",
    "build_eval_plan",
    "verify_plan",
    "assign_splits",
    "position_usage",
    "verify_position_balance",
    "session_counts_allowing_stratification",
    "exact_stratification_n_sessions",
    "verify_plane_consistency",
    "verify_array_geometry",
    "verify_frame_invariance",
    "verify_stratification",
    "verify_confound_cells",
    "verify_split_isolation",
    "verify_channel_gains",
    "verify_start_offsets",
    "prior_only_nll",
    "grid_times_sec",
    "local_constants",
    "synthesise_utterance",
    "load_librispeech_utterance",
    "compute_session_rirs",
    "render_session",
    "activity_grid",
    "save_rir_bundle",
    "load_rir_bundle",
    "write_wav_pcm16",
    "sha256_file",
]

GENERATOR_VERSION: Final[str] = "0.1.0"
"""Bumped whenever generated audio changes for a fixed seed."""


# ---------------------------------------------------------------------------
# Local constants -- generator choices, not contract values.
#
# Each one is reported by local_constants() and stamped into every manifest.
# Items marked [spec gap] belong in spec.py before the official set is frozen.
# ---------------------------------------------------------------------------

SAMPLE_RATE_HZ: Final[int] = spec.SAMPLE_RATE_HZ
"""Re-export of the contract's rate, kept so call sites in this file stay short.

It was a local constant until ``spec.py`` owned one, which meant two conforming
generators could ship 16 kHz and 48 kHz data under the same SPEC_VERSION.
"""

MAX_START_OFFSET_SEC: Final[float] = 2.0
"""[spec gap] Upper bound of the continuous utterance start offset."""

SESSION_GAIN_DBFS_RANGE: Final[tuple[float, float]] = (-26.0, -14.0)
"""Session-level common-gain target range, applied identically to all channels."""

CHANNEL_GAIN_MISMATCH_DB: Final[float] = 0.5
"""Hard cap on per-channel gain mismatch. Larger would erase the ILD cue;
zero would make every channel equally loud and remove it too."""

PEAK_CEILING: Final[float] = 0.999
"""Anti-clip ceiling. A single scalar is applied to *all* channels so the
inter-channel level difference is preserved."""

ROOM_DIM_MIN_M: Final[tuple[float, float, float]] = (4.0, 3.0, 2.5)
ROOM_DIM_MAX_M: Final[tuple[float, float, float]] = (10.0, 8.0, 3.5)
"""[spec gap] Shoe-box room size range."""

SOURCE_WALL_MARGIN_M: Final[float] = 0.5
"""Minimum clearance between any source or microphone and any wall."""

RT60_RANGE_LOW_S: Final[tuple[float, float]] = (0.15, 0.30)
RT60_RANGE_HIGH_S: Final[tuple[float, float]] = (0.55, 0.85)
"""[spec gap] The two absorption levels of the 2x2 confound design."""

MIN_PAIR_SPACING_M: Final[float] = 0.04
"""Smallest allowed microphone pair spacing. 0.04 m is ~1.9 samples of TDOA at
16 kHz: resolvable with interpolation, yet clearly worse than the long pair,
which is what makes the ``1100`` vs ``1010`` contrast in spec.EVAL_MASKS
informative."""

MIN_PAIR_SPACING_GAP_M: Final[float] = 0.002
"""Minimum separation between any two pairwise spacings, so the array is
genuinely asymmetric and no two channel pairs are interchangeable."""

MIN_TRIANGLE_AREA_M2: Final[float] = 1.0e-4
"""Rejects near-collinear arrays, for which M>=3 adds no 2D information."""

VAD_REL_THRESHOLD_DB: Final[float] = spec.VAD_REL_THRESHOLD_DB
VAD_REF_PERCENTILE: Final[float] = spec.VAD_REF_PERCENTILE
"""Re-exports of :data:`spec.T_EVAL_RULE`'s two parameters.

They were local while ``spec.py`` said nothing about which grid points carry a
label -- which made them the undeclared denominator of every spatial metric.
"""

RIR_STORE_SEC: Final[float] = 0.5
"""Default truncation length for *cached* RIR bundles (4 ch x 0.5 s float32
~= 128 KiB). RT60 is always measured on the untruncated RIR, and eval-dev
audio is always rendered from the untruncated RIR, so truncation only ever
affects on-the-fly training loads."""

SYNTHETIC_DURATION_RANGE_SEC: Final[tuple[float, float]] = (8.0, 15.0)
"""Duration range of the ``--synthetic`` placeholder utterances."""

FALLBACK_ISM_MAX_ORDER_CAP: Final[int] = 8
"""Image-source order cap of the non-official numpy fallback engine.

Image count grows as ``(2n+1)**3 * 8``, so a pure image-source model cannot
reach RT60 = 0.8 s at any tolerable cost: at order 8 the tail stops because the
images ran out, not because the sound decayed, and the *measured* RT60 then
reflects the cap instead of the absorption -- it even inverts, with the
high-RT60 setting measuring shorter than the low-RT60 one.  Hence the hybrid
diffuse tail below.
"""

FALLBACK_TAIL_COHERENCE: Final[float] = 0.5
"""Fraction of the fallback engine's diffuse-tail *energy* that is coherent
across microphones.

In a real room the spatial coherence of a diffuse field between two mics
``d`` apart is ``sinc(2 pi f d / c)``, which is close to 1 below
``c / (2 d)`` -- about 1 kHz for this array, i.e. exactly the band below
``spec.ALIASING_LIMIT_HZ`` where a small array gets its direction cue.  A tail
built from independent noise per channel would therefore be far *less*
informative than reality and would make the reverberant cases artificially
unsolvable.  Splitting the tail into a shared part (delayed by each
microphone's own direct-path delay) and an independent part is a deliberately
crude, frequency-flat stand-in for that coherence; it is one more reason this
engine is not official.
"""

FALLBACK_MIXING_TIME_SEC: Final[float] = 0.020
"""Where the fallback engine hands over from image sources to a diffuse tail.

The direct path and the first 20 ms of reflections -- which carry the TDOA, the
inter-channel level difference and the direct-to-reverberant ratio, i.e. every
cue this benchmark measures -- stay exactly as the image-source model computed
them.  Only the late field, which is diffuse and mutually incoherent between
microphones in reality, is replaced by exponentially decaying noise matched to
the local energy at the splice point.
"""

RIR_ENGINES: Final[tuple[str, ...]] = ("pyroomacoustics", "fallback_ism")
"""``pyroomacoustics`` (ShoeBox / Image Source Method) is the only engine whose
output is official. ``fallback_ism`` is a self-contained numpy image-source
implementation used to exercise the pipeline where pyroomacoustics is not
installed; it stamps ``official: false`` into the manifest."""

DISTANCE_LEVELS: Final[tuple[str, str]] = ("near", "far")
ABSORPTION_LEVELS: Final[tuple[str, str]] = ("low_rt60", "high_rt60")

CELL_ORDER: Final[tuple[tuple[str, str], ...]] = (
    ("near", "low_rt60"),
    ("far", "high_rt60"),
    ("near", "high_rt60"),
    ("far", "low_rt60"),
)
"""Fill order of the 2x2 cells inside a room block.

Chosen so a *partial* block is still useful: two sessions already give "same
room, two distances", and three give "same position, two absorptions" as well.
"""

SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "val", "eval")
DEFAULT_SPLIT_FRACTIONS: Final[tuple[float, float, float]] = (0.70, 0.15, 0.15)


def local_constants() -> dict[str, Any]:
    """Return every non-contract constant, for stamping into a manifest.

    >>> c = local_constants()
    >>> c["sample_rate_hz"] == SAMPLE_RATE_HZ
    True
    >>> c["spec_version"] == spec.SPEC_VERSION
    True
    """
    return {
        "spec_version": spec.SPEC_VERSION,
        "generator_version": GENERATOR_VERSION,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "max_start_offset_sec": MAX_START_OFFSET_SEC,
        "session_gain_dbfs_range": list(SESSION_GAIN_DBFS_RANGE),
        "channel_gain_mismatch_db": CHANNEL_GAIN_MISMATCH_DB,
        "peak_ceiling": PEAK_CEILING,
        "room_dim_min_m": list(ROOM_DIM_MIN_M),
        "room_dim_max_m": list(ROOM_DIM_MAX_M),
        "source_wall_margin_m": SOURCE_WALL_MARGIN_M,
        "rt60_range_low_s": list(RT60_RANGE_LOW_S),
        "rt60_range_high_s": list(RT60_RANGE_HIGH_S),
        "min_pair_spacing_m": MIN_PAIR_SPACING_M,
        "vad_rel_threshold_db": VAD_REL_THRESHOLD_DB,
        "vad_ref_percentile": VAD_REF_PERCENTILE,
        "rir_store_sec": RIR_STORE_SEC,
        "synthetic_duration_range_sec": list(SYNTHETIC_DURATION_RANGE_SEC),
        "aliasing_limit_hz": spec.ALIASING_LIMIT_HZ,
        "max_tdoa_sec": spec.MAX_PAIR_SPACING_M / spec.SPEED_OF_SOUND_M_S,
        "max_tdoa_samples": SAMPLE_RATE_HZ * spec.MAX_PAIR_SPACING_M / spec.SPEED_OF_SOUND_M_S,
    }


# ---------------------------------------------------------------------------
# Array geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicArrayGeometry:
    """An asymmetric planar microphone array in the fixed array frame.

    ``positions_xy[i]`` is channel ``i``'s offset from the array reference
    point, which is the constant ``(0, 0)`` by definition -- never the centroid
    of a channel subset.

    >>> g = MicArrayGeometry(((0.06, 0.0), (0.0, 0.05), (-0.07, 0.0), (0.0, -0.09)))
    >>> g.n_channels
    4
    >>> g.reference_point_xy
    (0.0, 0.0)
    >>> round(g.max_pair_spacing_m(), 3)
    0.14
    >>> g.for_mask("1010").positions_xy
    ((0.06, 0.0), (-0.07, 0.0))
    """

    positions_xy: tuple[tuple[float, float], ...]

    @property
    def n_channels(self) -> int:
        """Number of microphones in this (possibly masked) geometry."""
        return len(self.positions_xy)

    @property
    def reference_point_xy(self) -> tuple[float, float]:
        """The array frame origin. Constant by contract; see rule 2."""
        return (0.0, 0.0)

    def positions_xyz(self, plane_height_m: float = spec.PLANE_HEIGHT_M) -> tuple[tuple[float, float, float], ...]:
        """Lift the planar coordinates onto the common horizontal plane.

        >>> g = MicArrayGeometry(((0.06, 0.0),))
        >>> g.positions_xyz()[0][2] == spec.PLANE_HEIGHT_M
        True
        """
        return tuple((x, y, plane_height_m) for x, y in self.positions_xy)

    def pair_spacings_m(self) -> dict[tuple[int, int], float]:
        """Return ``{(i, j): distance}`` for every ``i < j``.

        >>> sorted(MicArrayGeometry(((0.05, 0.0), (-0.05, 0.0))).pair_spacings_m().items())
        [((0, 1), 0.1)]
        """
        out: dict[tuple[int, int], float] = {}
        for i, j in itertools.combinations(range(self.n_channels), 2):
            (xi, yi), (xj, yj) = self.positions_xy[i], self.positions_xy[j]
            out[(i, j)] = math.hypot(xi - xj, yi - yj)
        return out

    def max_pair_spacing_m(self) -> float:
        """Largest pairwise microphone spacing."""
        spacings = self.pair_spacings_m()
        return max(spacings.values()) if spacings else 0.0

    def min_pair_spacing_m(self) -> float:
        """Smallest pairwise microphone spacing."""
        spacings = self.pair_spacings_m()
        return min(spacings.values()) if spacings else 0.0

    def for_mask(self, mask_bits: str) -> "MicArrayGeometry":
        """Return the sub-array selected by ``mask_bits``, same frame and origin.

        >>> g = MicArrayGeometry(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))
        >>> g.for_mask("1100").positions_xy
        ((1.0, 0.0), (0.0, 1.0))
        """
        channels = spec.mask_to_channels(mask_bits)
        return MicArrayGeometry(tuple(self.positions_xy[c] for c in channels))

    def as_json(self) -> list[list[float]]:
        """Return the positions as plain JSON-serialisable lists."""
        return [[float(x), float(y)] for x, y in self.positions_xy]


def _label_channels_for_masks(points: Sequence[tuple[float, float]]) -> tuple[int, ...] | None:
    """Find a channel labelling consistent with the semantics of spec.EVAL_MASKS.

    ``spec.EVAL_MASKS`` documents ``"1100"`` as the *shortest* M=2 pair and
    ``"1010"`` as the *longest*, so the geometry must satisfy
    ``d(0, 1) == min`` and ``d(0, 2) == max`` over all pairs.  Returns the
    permutation mapping new channel index -> input index, or ``None`` when no
    labelling works (which happens whenever the min and max pair share no
    microphone).

    >>> pts = [(0.0, 0.0), (0.05, 0.0), (0.0, 0.05), (-0.1, 0.0)]
    >>> _label_channels_for_masks(pts) is None
    False
    """
    n = len(points)

    def dist(a: int, b: int) -> float:
        return math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1])

    all_pairs = list(itertools.combinations(range(n), 2))
    d_min = min(dist(*p) for p in all_pairs)
    d_max = max(dist(*p) for p in all_pairs)
    for perm in itertools.permutations(range(n)):
        if not math.isclose(dist(perm[0], perm[1]), d_min, rel_tol=0.0, abs_tol=1e-12):
            continue
        if not math.isclose(dist(perm[0], perm[2]), d_max, rel_tol=0.0, abs_tol=1e-12):
            continue
        return perm
    return None


def make_mic_array(
    rng: random.Random,
    n_channels: int = spec.N_CHANNELS_MAX,
    max_attempts: int = 4000,
) -> MicArrayGeometry:
    """Draw an asymmetric planar array satisfying every geometric constraint.

    Radii come from ``spec.ARRAY_RADIUS_RANGE_M`` and the angular gaps are
    deliberately unequal, so no two channel pairs are interchangeable and the
    front/back degeneracy of a symmetric array is avoided.

    >>> g = make_mic_array(random.Random(7))
    >>> g.n_channels == spec.N_CHANNELS_MAX
    True
    >>> g.max_pair_spacing_m() <= spec.MAX_PAIR_SPACING_M
    True
    >>> spacings = g.pair_spacings_m()
    >>> spacings[(0, 1)] == min(spacings.values())  # "1100" is the short pair
    True
    >>> spacings[(0, 2)] == max(spacings.values())  # "1010" is the long pair
    True
    """
    r_lo, r_hi = spec.ARRAY_RADIUS_RANGE_M
    for _ in range(max_attempts):
        radii = [rng.uniform(r_lo, r_hi) for _ in range(n_channels)]
        gaps = [rng.uniform(0.45, 2.0) for _ in range(n_channels)]
        total = sum(gaps)
        angle = rng.uniform(0.0, 360.0)
        points: list[tuple[float, float]] = []
        for radius, gap in zip(radii, gaps):
            rad = math.radians(angle)
            points.append((radius * math.cos(rad), radius * math.sin(rad)))
            angle += 360.0 * gap / total
        perm = _label_channels_for_masks(points)
        if perm is None:
            continue
        geom = MicArrayGeometry(tuple(points[i] for i in perm))
        try:
            verify_array_geometry(geom)
        except AssertionError:
            continue
        return geom
    raise RuntimeError(
        f"could not draw a valid {n_channels}-channel array in {max_attempts} attempts; "
        f"check ARRAY_RADIUS_RANGE_M={spec.ARRAY_RADIUS_RANGE_M} against "
        f"MIN_PAIR_SPACING_M={MIN_PAIR_SPACING_M}"
    )


def verify_array_geometry(geom: MicArrayGeometry) -> dict[str, Any]:
    """Assert every array constraint and return the measured values.

    Checks the radius range, the ``spec.MAX_PAIR_SPACING_M`` ceiling, the
    short/long pair labelling required by ``spec.EVAL_MASKS``, asymmetry (all
    six spacings distinct) and non-collinearity.

    >>> report = verify_array_geometry(make_mic_array(random.Random(3)))
    >>> report["asymmetric"] and report["non_collinear"]
    True
    """
    r_lo, r_hi = spec.ARRAY_RADIUS_RANGE_M
    radii = [math.hypot(x, y) for x, y in geom.positions_xy]
    for i, radius in enumerate(radii):
        assert r_lo - 1e-12 <= radius <= r_hi + 1e-12, (
            f"channel {i} radius {radius:.4f} m outside spec.ARRAY_RADIUS_RANGE_M {spec.ARRAY_RADIUS_RANGE_M}"
        )

    spacings = geom.pair_spacings_m()
    max_spacing = max(spacings.values())
    min_spacing = min(spacings.values())
    assert max_spacing <= spec.MAX_PAIR_SPACING_M + 1e-12, (
        f"max pair spacing {max_spacing:.4f} m exceeds spec.MAX_PAIR_SPACING_M {spec.MAX_PAIR_SPACING_M}"
    )
    assert min_spacing >= MIN_PAIR_SPACING_M, (
        f"min pair spacing {min_spacing:.4f} m below MIN_PAIR_SPACING_M {MIN_PAIR_SPACING_M}"
    )

    assert math.isclose(spacings[(0, 1)], min_spacing, abs_tol=1e-12), (
        "spec.EVAL_MASKS documents '1100' as the shortest M=2 pair, so d(0,1) must be the minimum"
    )
    assert math.isclose(spacings[(0, 2)], max_spacing, abs_tol=1e-12), (
        "spec.EVAL_MASKS documents '1010' as the longest M=2 pair, so d(0,2) must be the maximum"
    )

    ordered = sorted(spacings.values())
    asymmetric = all(
        ordered[i + 1] - ordered[i] >= MIN_PAIR_SPACING_GAP_M for i in range(len(ordered) - 1)
    )
    assert asymmetric, f"array is too symmetric: spacings {[round(s, 4) for s in ordered]}"

    areas = []
    for a, b, c in itertools.combinations(range(geom.n_channels), 3):
        (xa, ya), (xb, yb), (xc, yc) = (geom.positions_xy[i] for i in (a, b, c))
        areas.append(abs((xb - xa) * (yc - ya) - (xc - xa) * (yb - ya)) / 2.0)
    non_collinear = max(areas) >= MIN_TRIANGLE_AREA_M2
    assert non_collinear, f"array is near-collinear: max triangle area {max(areas):.2e} m^2"

    return {
        "radii_m": [round(r, 6) for r in radii],
        "min_pair_spacing_m": min_spacing,
        "max_pair_spacing_m": max_spacing,
        "pair_spacings_m": {f"{i}{j}": v for (i, j), v in sorted(spacings.items())},
        "asymmetric": asymmetric,
        "non_collinear": non_collinear,
        "aliasing_limit_hz": spec.aliasing_limit_hz(max_spacing),
    }


def verify_frame_invariance(geom: MicArrayGeometry, masks: Iterable[str] = spec.EVAL_MASKS) -> dict[str, Any]:
    """Assert that masking channels changes neither the origin nor the axes.

    This is the coordinate-system invariance rule: if the origin were
    recomputed as the centroid of the active microphones, every azimuth and
    distance label would silently shift between M=1 and M=4, and the headline
    comparison would be measuring a moved frame rather than added channels.

    >>> g = make_mic_array(random.Random(11))
    >>> verify_frame_invariance(g)["origin_shift_max_m"]
    0.0
    """
    base = geom.reference_point_xy
    shifts = []
    for mask_bits in masks:
        sub = geom.for_mask(mask_bits)
        assert sub.reference_point_xy == base, f"mask {mask_bits} moved the array origin"
        channels = spec.mask_to_channels(mask_bits)
        for local_index, channel in enumerate(channels):
            assert sub.positions_xy[local_index] == geom.positions_xy[channel], (
                f"mask {mask_bits} changed channel {channel}'s coordinates"
            )
        shifts.append(
            math.hypot(sub.reference_point_xy[0] - base[0], sub.reference_point_xy[1] - base[1])
        )
    return {"masks_checked": list(masks), "origin_shift_max_m": max(shifts) if shifts else 0.0}


def verify_plane_consistency(
    geom: MicArrayGeometry,
    source_xy: tuple[float, float],
    plane_height_m: float = spec.PLANE_HEIGHT_M,
) -> dict[str, Any]:
    """Assert the 2D label distance equals the true 3D slant range.

    Every microphone and the source share ``z = spec.PLANE_HEIGHT_M``.  Placing
    an array at 1.0 m and a mouth at 1.6 m would make the slant range to a
    point 1 m away horizontally 1.166 m -- a systematic bias in *every*
    distance label that no downstream test would catch.

    >>> g = make_mic_array(random.Random(5))
    >>> verify_plane_consistency(g, (2.0, 1.0))["max_abs_error_m"] < 1e-12
    True
    """
    sx, sy = source_xy
    src_z = plane_height_m
    errors = []
    for (mx, my), (_, _, mz) in zip(geom.positions_xy, geom.positions_xyz(plane_height_m)):
        assert mz == plane_height_m, f"microphone z {mz} != plane {plane_height_m}"
        d2 = math.hypot(sx - mx, sy - my)
        d3 = math.sqrt((sx - mx) ** 2 + (sy - my) ** 2 + (src_z - mz) ** 2)
        errors.append(abs(d3 - d2))
    max_error = max(errors) if errors else 0.0
    assert max_error < 1e-9, f"2D/3D distance mismatch {max_error:.3e} m: sources and mics are not coplanar"

    origin_2d = math.hypot(sx, sy)
    origin_3d = math.sqrt(sx * sx + sy * sy + (src_z - plane_height_m) ** 2)
    assert abs(origin_2d - origin_3d) < 1e-9, "label distance to the array origin is not the slant range"
    return {
        "plane_height_m": plane_height_m,
        "max_abs_error_m": max_error,
        "label_distance_m": origin_2d,
    }


# ---------------------------------------------------------------------------
# Stratified source placement
# ---------------------------------------------------------------------------


def even_bin_schedule(n_items: int, k_bins: int, bin_offset: int = 0) -> tuple[int, ...]:
    """Spread ``n_items`` over ``k_bins`` as evenly as integers allow.

    When ``n_items <= k_bins`` every item lands in a *distinct* bin at an even
    stride.  Otherwise each bin gets ``n // k`` items and the ``n % k``
    remainder is spread at an even stride, so ``max(count) - min(count) <= 1``
    always and all counts are equal exactly when ``k`` divides ``n``.

    >>> even_bin_schedule(6, 36)
    (0, 6, 12, 18, 24, 30)
    >>> even_bin_schedule(3, 4)
    (0, 1, 2)
    >>> even_bin_schedule(8, 4)
    (0, 0, 1, 1, 2, 2, 3, 3)
    >>> even_bin_schedule(6, 4)
    (0, 0, 1, 2, 2, 3)
    >>> even_bin_schedule(2, 4, bin_offset=4)
    (4, 6)
    """
    if n_items < 0:
        raise ValueError(f"n_items must be >= 0, got {n_items}")
    if k_bins <= 0:
        raise ValueError(f"k_bins must be > 0, got {k_bins}")
    if n_items == 0:
        return ()
    if n_items <= k_bins:
        return tuple(bin_offset + (i * k_bins) // n_items for i in range(n_items))
    base, remainder = divmod(n_items, k_bins)
    counts = [base] * k_bins
    for j in range(remainder):
        counts[(j * k_bins) // remainder] += 1
    out: list[int] = []
    for b, count in enumerate(counts):
        out.extend([bin_offset + b] * count)
    return tuple(out)


def sample_azimuth_in_bin(rng: random.Random, bin_index: int) -> float:
    """Draw a continuous azimuth inside azimuth bin ``bin_index``.

    The value is guaranteed to round-trip through ``spec.azimuth_bin_index``,
    including in the last bin whose upper edge ``+180`` wraps to ``-180``.

    >>> az = sample_azimuth_in_bin(random.Random(0), 35)
    >>> spec.azimuth_bin_index(az)
    35
    >>> lo, hi = spec.AZIMUTH_RANGE_DEG
    >>> lo <= az < hi
    True
    """
    if not 0 <= bin_index < spec.K_AZIMUTH:
        raise ValueError(f"azimuth bin {bin_index} outside [0, {spec.K_AZIMUTH})")
    lo = spec.AZIMUTH_BIN_EDGES_DEG[bin_index]
    hi = spec.AZIMUTH_BIN_EDGES_DEG[bin_index + 1]
    value = min(rng.uniform(lo, hi), math.nextafter(hi, lo))
    assert spec.azimuth_bin_index(value) == bin_index, (
        f"azimuth {value} fell outside its own bin {bin_index}"
    )
    return value


def sample_distance_in_bin(rng: random.Random, bin_index: int) -> float:
    """Draw a continuous distance inside distance bin ``bin_index``.

    Log-uniform within the bin, matching the log spacing of
    ``spec.DISTANCE_BIN_EDGES_M``: the usable distance cue in a small array is
    the direct-to-reverberant ratio, which falls roughly as ``1 / r``.

    >>> r = sample_distance_in_bin(random.Random(0), 5)
    >>> spec.distance_bin_index(r)
    5
    >>> lo, hi = spec.DISTANCE_RANGE_M
    >>> lo <= r <= hi
    True
    """
    if not 0 <= bin_index < spec.K_DISTANCE:
        raise ValueError(f"distance bin {bin_index} outside [0, {spec.K_DISTANCE})")
    lo = spec.DISTANCE_BIN_EDGES_M[bin_index]
    hi = spec.DISTANCE_BIN_EDGES_M[bin_index + 1]
    value = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    is_last = bin_index == spec.K_DISTANCE - 1
    value = min(value, hi) if is_last else min(value, math.nextafter(hi, lo))
    value = max(value, lo)
    assert spec.distance_bin_index(value) == bin_index, (
        f"distance {value} fell outside its own bin {bin_index}"
    )
    r_lo, r_hi = spec.DISTANCE_RANGE_M
    assert r_lo <= value <= r_hi, f"distance {value} outside spec.DISTANCE_RANGE_M"
    return value


@dataclass(frozen=True)
class SourcePlacement:
    """One stationary source position in the array frame.

    >>> p = SourcePlacement("p0", "near", 12.5, 1.2, 19, 3)
    >>> round(p.xy[0], 4), round(p.xy[1], 4)
    (1.1716, 0.2597)
    """

    position_id: str
    distance_level: str
    azimuth_deg: float
    distance_m: float
    azimuth_bin: int
    distance_bin: int

    @property
    def xy(self) -> tuple[float, float]:
        """Cartesian offset from the array reference point."""
        rad = math.radians(self.azimuth_deg)
        return (self.distance_m * math.cos(rad), self.distance_m * math.sin(rad))

    def as_json(self) -> dict[str, Any]:
        """JSON-serialisable view of the placement."""
        return {
            "position_id": self.position_id,
            "distance_level": self.distance_level,
            "azimuth_deg": self.azimuth_deg,
            "distance_m": self.distance_m,
            "azimuth_bin": self.azimuth_bin,
            "distance_bin": self.distance_bin,
        }


@dataclass(frozen=True)
class RoomBlock:
    """One shoe-box room shared by up to four sessions.

    The block is the unit that crosses distance with absorption: it holds one
    ``near`` and one ``far`` source position (same room -> distance varies with
    acoustics fixed) and one low and one high RT60 target (same position ->
    absorption varies with distance fixed).
    """

    room_id: str
    dims_xyz_m: tuple[float, float, float]
    array_center_xy_m: tuple[float, float]
    geometry: MicArrayGeometry
    sources: dict[str, SourcePlacement]
    rt60_requested_s: dict[str, float]

    def source_room_xyz(self, distance_level: str) -> tuple[float, float, float]:
        """Absolute room coordinates of one source, on the common plane."""
        sx, sy = self.sources[distance_level].xy
        cx, cy = self.array_center_xy_m
        return (cx + sx, cy + sy, spec.PLANE_HEIGHT_M)

    def mic_room_xyz(self) -> tuple[tuple[float, float, float], ...]:
        """Absolute room coordinates of all microphones, on the common plane."""
        cx, cy = self.array_center_xy_m
        return tuple((cx + x, cy + y, spec.PLANE_HEIGHT_M) for x, y in self.geometry.positions_xy)


@dataclass(frozen=True)
class SessionPlan:
    """Everything needed to render one 4-channel session, before any audio."""

    session_id: str
    split: str
    block: RoomBlock
    distance_level: str
    absorption_level: str
    cell: tuple[int, int]
    source_utt_id: str
    speaker_id: str
    variant_group: str
    start_offset_sec: float
    common_gain_dbfs: float
    channel_gain_db: tuple[float, ...]

    @property
    def source(self) -> SourcePlacement:
        """The source placement used by this session."""
        return self.block.sources[self.distance_level]

    @property
    def rt60_requested_s(self) -> float:
        """Requested (Sabine) RT60 for this session's absorption level."""
        return self.block.rt60_requested_s[self.absorption_level]

    @property
    def acoustic_id(self) -> str:
        """Identifier of the (room geometry, absorption) pair."""
        return f"{self.block.room_id}_{self.absorption_level}"

    def case_ids(self, masks: Iterable[str] = spec.EVAL_MASKS) -> tuple[str, ...]:
        """Return this session's ``case_id`` for every evaluated channel subset.

        >>> plan = build_eval_plan(4, seed=1).sessions[0]
        >>> len(plan.case_ids()) == len(spec.EVAL_MASKS)
        True
        >>> plan.case_ids()[0].endswith("__m" + spec.EVAL_MASKS[0])
        True
        """
        return tuple(spec.make_case_id(self.session_id, m) for m in masks)


@dataclass(frozen=True)
class EvalPlan:
    """A complete, verified evaluation-set plan."""

    seed: int
    sessions: tuple[SessionPlan, ...]
    blocks: tuple[RoomBlock, ...]
    reports: dict[str, Any] = field(default_factory=dict)

    @property
    def n_sessions(self) -> int:
        """Number of sessions in the plan."""
        return len(self.sessions)

    @property
    def n_cases(self) -> int:
        """Number of evaluation cases (sessions x evaluated channel subsets)."""
        return len(self.sessions) * len(spec.EVAL_MASKS)


def _fit_room_to_block(
    rng: random.Random,
    geom: MicArrayGeometry,
    sources: dict[str, SourcePlacement],
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """Choose room dimensions and the array centre so nothing is near a wall.

    The room is sized from the bounding box of the microphones and *all* of the
    block's sources plus ``SOURCE_WALL_MARGIN_M`` on every side, then padded
    with random slack and clamped into ``[ROOM_DIM_MIN_M, ROOM_DIM_MAX_M]``.
    The array centre is placed at a random feasible offset, so room size does
    not encode where in the room the array sits.
    """
    points = list(geom.positions_xy) + [s.xy for s in sources.values()]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    margin = SOURCE_WALL_MARGIN_M
    needed_x = (max(xs) - min(xs)) + 2 * margin
    needed_y = (max(ys) - min(ys)) + 2 * margin

    if needed_x > ROOM_DIM_MAX_M[0] or needed_y > ROOM_DIM_MAX_M[1]:
        raise ValueError(
            "block does not fit any allowed room: needs "
            f"{needed_x:.2f} x {needed_y:.2f} m but ROOM_DIM_MAX_M is {ROOM_DIM_MAX_M[:2]}; "
            f"sources {[s.as_json() for s in sources.values()]}"
        )

    lx = min(ROOM_DIM_MAX_M[0], max(ROOM_DIM_MIN_M[0], needed_x + rng.uniform(0.0, 1.5)))
    ly = min(ROOM_DIM_MAX_M[1], max(ROOM_DIM_MIN_M[1], needed_y + rng.uniform(0.0, 1.5)))
    lz = rng.uniform(ROOM_DIM_MIN_M[2], ROOM_DIM_MAX_M[2])

    free_x = lx - needed_x
    free_y = ly - needed_y
    cx = margin + rng.uniform(0.0, max(0.0, free_x)) - min(xs)
    cy = margin + rng.uniform(0.0, max(0.0, free_y)) - min(ys)

    for px, py in points:
        assert margin - 1e-9 <= cx + px <= lx - margin + 1e-9, "x placement violates the wall margin"
        assert margin - 1e-9 <= cy + py <= ly - margin + 1e-9, "y placement violates the wall margin"
    assert margin <= spec.PLANE_HEIGHT_M <= lz - margin, (
        f"spec.PLANE_HEIGHT_M={spec.PLANE_HEIGHT_M} leaves less than {margin} m to the ceiling of a {lz:.2f} m room"
    )
    return (lx, ly, lz), (cx, cy)


def _dither_off_grid(offset_samples: int, grid_samples: int, limit: int) -> int:
    """Nudge a sample offset off the ``spec.DELTA_T_SEC`` grid.

    A start offset that lands exactly on a grid multiple would leak the
    utterance onset to a model that only has to inspect grid boundaries, which
    is the shortcut rule 7 exists to close.  Continuous sampling makes this
    astronomically rare; the nudge makes it impossible.

    >>> _dither_off_grid(4000, 4000, 320000)
    4001
    >>> _dither_off_grid(4123, 4000, 320000)
    4123
    """
    if grid_samples <= 1:
        return offset_samples
    if offset_samples % grid_samples != 0:
        return offset_samples
    return min(offset_samples + 1, max(0, limit - 1))


def build_eval_plan(
    n_sessions: int,
    seed: int,
    split: str = "eval",
    speaker_pool: Sequence[str] | None = None,
    masks: Iterable[str] = spec.EVAL_MASKS,
    allow_unbalanced: bool = False,
) -> EvalPlan:
    """Build a fully verified evaluation-set plan from a seed. Pure stdlib.

    Sessions are emitted in blocks of four.  Each block owns one room, one
    microphone array, one ``near`` and one ``far`` source position and one
    low/high RT60 pair, and its four sessions walk :data:`CELL_ORDER`, so the
    distance x absorption design is crossed by construction rather than by
    chance.

    >>> plan = build_eval_plan(12, seed=20261029)
    >>> plan.n_sessions, plan.n_cases
    (12, 72)
    >>> plan.reports["azimuth_stratification"]["uniform_over_used"]
    True
    >>> plan.reports["confound_cells"]["cells_populated"]
    4
    >>> sorted(plan.reports["confound_cells"]["counts"].values())
    [3, 3, 3, 3]
    >>> plan.reports["frame_invariance"]["origin_shift_max_m"]
    0.0

    A session count whose trailing block cannot be balanced is rejected up
    front rather than silently producing a skewed label prior:

    >>> build_eval_plan(3, seed=1)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: n_sessions=3 cannot be stratified: ...
    >>> build_eval_plan(3, seed=1, allow_unbalanced=True).n_sessions
    3
    """
    if n_sessions <= 0:
        raise ValueError(f"n_sessions must be > 0, got {n_sessions}")
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    position_report = verify_position_balance(n_sessions, require=not allow_unbalanced)

    rng = random.Random(seed)
    n_blocks = -(-n_sessions // len(CELL_ORDER))

    k_near = spec.K_DISTANCE // 2
    k_far = spec.K_DISTANCE - k_near
    near_bins = list(even_bin_schedule(n_blocks, k_near))
    far_bins = list(even_bin_schedule(n_blocks, k_far, bin_offset=k_near))
    rng.shuffle(near_bins)
    rng.shuffle(far_bins)

    azimuth_bins = list(even_bin_schedule(2 * n_blocks, spec.K_AZIMUTH))
    rng.shuffle(azimuth_bins)

    if speaker_pool is None:
        speaker_pool = tuple(f"syn_spk_{i:03d}" for i in range(max(4, 2 * n_blocks)))
    if not speaker_pool:
        raise ValueError("speaker_pool must not be empty")

    blocks: list[RoomBlock] = []
    sessions: list[SessionPlan] = []
    grid_samples = int(round(spec.DELTA_T_SEC * SAMPLE_RATE_HZ))
    session_samples = int(round(spec.SESSION_LENGTH_SEC * SAMPLE_RATE_HZ))

    for b in range(n_blocks):
        geom = make_mic_array(rng)
        sources = {
            "near": SourcePlacement(
                position_id=f"pos_{b:03d}_near",
                distance_level="near",
                azimuth_deg=sample_azimuth_in_bin(rng, azimuth_bins[2 * b]),
                distance_m=sample_distance_in_bin(rng, near_bins[b]),
                azimuth_bin=azimuth_bins[2 * b],
                distance_bin=near_bins[b],
            ),
            "far": SourcePlacement(
                position_id=f"pos_{b:03d}_far",
                distance_level="far",
                azimuth_deg=sample_azimuth_in_bin(rng, azimuth_bins[2 * b + 1]),
                distance_m=sample_distance_in_bin(rng, far_bins[b]),
                azimuth_bin=azimuth_bins[2 * b + 1],
                distance_bin=far_bins[b],
            ),
        }
        dims, center = _fit_room_to_block(rng, geom, sources)
        block = RoomBlock(
            room_id=f"room_{b:03d}",
            dims_xyz_m=dims,
            array_center_xy_m=center,
            geometry=geom,
            sources=sources,
            rt60_requested_s={
                "low_rt60": rng.uniform(*RT60_RANGE_LOW_S),
                "high_rt60": rng.uniform(*RT60_RANGE_HIGH_S),
            },
        )
        blocks.append(block)

        # One utterance per source position, shared by the two absorption
        # levels, so "same distance, different absorption" differs *only* in
        # absorption. The start offset travels with the utterance (rule 7).
        utt_of_level: dict[str, tuple[str, str, float]] = {}
        for level_index, level in enumerate(DISTANCE_LEVELS):
            speaker = speaker_pool[(2 * b + level_index) % len(speaker_pool)]
            utt_id = f"{speaker}_{b:03d}{level_index}"
            offset = rng.uniform(0.0, MAX_START_OFFSET_SEC)
            offset_samples = _dither_off_grid(
                int(round(offset * SAMPLE_RATE_HZ)), grid_samples, session_samples
            )
            utt_of_level[level] = (utt_id, speaker, offset_samples / SAMPLE_RATE_HZ)

        for slot, (distance_level, absorption_level) in enumerate(CELL_ORDER):
            index = b * len(CELL_ORDER) + slot
            if index >= n_sessions:
                break
            utt_id, speaker, offset = utt_of_level[distance_level]
            sessions.append(
                SessionPlan(
                    session_id=f"sess_{index:04d}",
                    split=split,
                    block=block,
                    distance_level=distance_level,
                    absorption_level=absorption_level,
                    cell=(DISTANCE_LEVELS.index(distance_level), ABSORPTION_LEVELS.index(absorption_level)),
                    source_utt_id=utt_id,
                    speaker_id=speaker,
                    variant_group=f"{utt_id}@{sources[distance_level].position_id}",
                    start_offset_sec=offset,
                    common_gain_dbfs=rng.uniform(*SESSION_GAIN_DBFS_RANGE),
                    channel_gain_db=tuple(
                        rng.uniform(-CHANNEL_GAIN_MISMATCH_DB, CHANNEL_GAIN_MISMATCH_DB)
                        for _ in range(spec.N_CHANNELS_MAX)
                    ),
                )
            )

    plan = EvalPlan(seed=seed, sessions=tuple(sessions), blocks=tuple(blocks))
    plan.reports.update(verify_plan(plan, masks=masks, require_stratification=not allow_unbalanced))
    plan.reports["position_balance"] = position_report
    plan.reports["allow_unbalanced"] = allow_unbalanced
    return plan


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_stratification(
    counts: dict[int, int], k_bins: int, axis: str = "", require: bool = True
) -> dict[str, Any]:
    """Assert that a bin histogram is as uniform as the item count allows.

    Returns three distinct flags, because with fewer sessions than bins "equal
    counts in every bin" is arithmetically impossible and quietly pretending
    otherwise would be the real failure:

    ``strict_uniform``  every one of the ``k_bins`` bins has the same count.
    ``uniform_over_used``  every *occupied* bin has the same count.
    ``balanced``  ``max - min <= 1`` over all bins.

    With ``require`` the function asserts ``uniform_over_used or balanced``.
    Only ``strict_uniform`` fully removes the memorisation gain described in
    :func:`prior_only_nll`.

    >>> verify_stratification({0: 2, 1: 2, 2: 2}, 3)["strict_uniform"]
    True
    >>> r = verify_stratification({0: 2, 1: 2, 2: 0}, 3)
    >>> r["strict_uniform"], r["uniform_over_used"], r["n_used_bins"]
    (False, True, 2)
    >>> verify_stratification({0: 2, 1: 1, 2: 0}, 3, "azimuth", require=False)["uniform_over_used"]
    False
    """
    full = {b: int(counts.get(b, 0)) for b in range(k_bins)}
    used = {b: c for b, c in full.items() if c > 0}
    values = list(full.values())
    used_values = list(used.values())
    strict_uniform = len(set(values)) <= 1
    uniform_over_used = len(set(used_values)) <= 1 if used_values else True
    balanced = (max(values) - min(values)) <= 1 if values else True
    if require:
        assert uniform_over_used or balanced, (
            f"{axis or 'bin'} stratification violated: counts {full}"
        )
    return {
        "axis": axis,
        "k_bins": k_bins,
        "counts": full,
        "n_items": sum(values),
        "n_used_bins": len(used),
        "min_count": min(values) if values else 0,
        "max_count": max(values) if values else 0,
        "strict_uniform": strict_uniform,
        "uniform_over_used": uniform_over_used,
        "balanced": balanced,
    }


def prior_only_nll(counts: dict[int, int] | int, k_bins: int) -> dict[str, float]:
    """Report the NLL an audio-blind memoriser achieves, versus uniform.

    A model that never listens and emits the empirical label prior scores the
    *entropy* of that prior.  When it falls below ``log(k_bins)`` the
    "M>=2 beats M=1" comparison contains pure memorisation gain, so the set is
    too small or too skewed to settle the claim and the gap must be reported.

    ``counts`` may be a bin histogram, in which case the true entropy is used,
    or an integer count of occupied bins, which assumes they are equally
    filled and gives ``log(n_used_bins)``.

    >>> round(prior_only_nll(36, 36)["memorisation_gain_nats"], 9)
    0.0
    >>> round(prior_only_nll(6, 36)["prior_only_nll_nats"], 4)
    1.7918

    A skewed histogram is worth slightly more to a memoriser than an even one
    over the same bins:

    >>> even = prior_only_nll({0: 5, 1: 5}, 2)["memorisation_gain_nats"]
    >>> skew = prior_only_nll({0: 6, 1: 4}, 2)["memorisation_gain_nats"]
    >>> round(even, 9), skew > even
    (0.0, True)
    """
    if isinstance(counts, int):
        entropy = math.log(max(1, counts))
        n_used = max(1, counts)
    else:
        total = sum(counts.values())
        occupied = [c for c in counts.values() if c > 0]
        n_used = len(occupied)
        entropy = (
            -sum((c / total) * math.log(c / total) for c in occupied) if total > 0 else 0.0
        )
    return {
        "prior_only_nll_nats": entropy,
        "uniform_nll_nats": math.log(k_bins),
        "memorisation_gain_nats": math.log(k_bins) - entropy,
        "n_used_bins": n_used,
    }


def position_usage(n_sessions: int) -> dict[str, int]:
    """Return ``{position_id: number of sessions}`` for a plan of this size.

    Each room block owns ``len(DISTANCE_LEVELS)`` source positions and fills
    :data:`CELL_ORDER`, so in a *full* block every position carries exactly two
    sessions.  A trailing partial block alongside full blocks mixes counts, and
    because one position *is* one azimuth bin and one distance bin, that mixture
    is exactly the non-uniform label prior rule 3 forbids.  The keys match the
    ``position_id`` values :func:`build_eval_plan` assigns.

    >>> sorted(set(position_usage(12).values()))
    [2]
    >>> position_usage(2)
    {'pos_000_near': 1, 'pos_000_far': 1}
    >>> position_usage(3)
    {'pos_000_near': 2, 'pos_000_far': 1}
    >>> sorted(set(position_usage(6).values()))  # full block + half block
    [1, 2]
    """
    n_blocks = -(-n_sessions // len(CELL_ORDER))
    usage: dict[str, int] = {}
    for b in range(n_blocks):
        for level in DISTANCE_LEVELS:
            usage[f"pos_{b:03d}_{level}"] = 0
    for index in range(n_sessions):
        block_index = index // len(CELL_ORDER)
        distance_level, _ = CELL_ORDER[index % len(CELL_ORDER)]
        usage[f"pos_{block_index:03d}_{distance_level}"] += 1
    return usage


def session_counts_allowing_stratification(around: int, span: int = 8) -> tuple[int, ...]:
    """Session counts near ``around`` for which every source position is used equally.

    With :data:`CELL_ORDER` this works out to the multiples of
    ``len(CELL_ORDER)``, plus the one- and two-session degenerate plans that
    consist of a single partial block.

    >>> session_counts_allowing_stratification(3, span=2)
    (1, 2, 4)
    >>> session_counts_allowing_stratification(12, span=4)
    (8, 12, 16)
    """
    out = []
    for n in range(max(1, around - span), around + span + 1):
        usage = [v for v in position_usage(n).values() if v > 0]
        if usage and len(set(usage)) == 1:
            out.append(n)
    return tuple(out)


def verify_position_balance(n_sessions: int, require: bool = True) -> dict[str, Any]:
    """Assert every used source position carries the same number of sessions.

    >>> verify_position_balance(12)["balanced"]
    True
    >>> verify_position_balance(3, require=False)["balanced"]
    False
    >>> verify_position_balance(3)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: n_sessions=3 cannot be stratified: ...Use one of (1, 2, 4, 8) sessions...
    """
    usage = position_usage(n_sessions)
    used = [v for v in usage.values() if v > 0]
    balanced = bool(used) and len(set(used)) == 1
    if require and not balanced:
        raise ValueError(
            f"n_sessions={n_sessions} cannot be stratified: the trailing partial block would use "
            f"its source positions {usage} times, so some azimuth and distance bins would carry "
            f"more sessions than others. Use one of "
            f"{session_counts_allowing_stratification(n_sessions)} sessions, or pass "
            f"allow_unbalanced=True and accept a non-uniform label prior."
        )
    return {"usage": usage, "balanced": balanced, "sessions_per_position": used[0] if balanced else None}


def exact_stratification_n_sessions() -> int:
    """Smallest session count that makes both axes ``strict_uniform``.

    Each room block contributes two source positions and four sessions, so with
    ``P`` distinct positions there are ``2 * P`` sessions; ``P`` must be a
    multiple of both ``spec.K_AZIMUTH`` and ``spec.K_DISTANCE``, and the
    session count a multiple of ``len(CELL_ORDER)``.

    >>> exact_stratification_n_sessions()
    144
    """
    positions = math.lcm(spec.K_AZIMUTH, spec.K_DISTANCE)
    return math.lcm(2 * positions, len(CELL_ORDER))


def verify_confound_cells(sessions: Sequence[SessionPlan]) -> dict[str, Any]:
    """Assert the distance x absorption design is actually crossed.

    Two things must hold for the distance label and the reverberation label to
    be separable at all:

    (a) at least one room contains two different distances at a fixed
        absorption, and
    (b) at least one source position is rendered at two absorptions with the
        distance fixed.

    With fewer than three sessions neither can hold, so the assertion is
    skipped and ``complete`` reports ``False``.

    >>> r = verify_confound_cells(build_eval_plan(12, seed=5).sessions)
    >>> r["has_same_room_distance_contrast"], r["has_same_position_absorption_contrast"]
    (True, True)
    >>> r["complete"]
    True
    """
    counts: dict[str, int] = {f"{d}|{a}": 0 for d in DISTANCE_LEVELS for a in ABSORPTION_LEVELS}
    by_room_abs: dict[tuple[str, str], set[int]] = {}
    by_position: dict[str, set[str]] = {}
    for s in sessions:
        counts[f"{s.distance_level}|{s.absorption_level}"] += 1
        by_room_abs.setdefault((s.block.room_id, s.absorption_level), set()).add(s.source.distance_bin)
        by_position.setdefault(s.source.position_id, set()).add(s.absorption_level)

    same_room_distance = any(len(v) >= 2 for v in by_room_abs.values())
    same_position_absorption = any(len(v) >= 2 for v in by_position.values())
    populated = sum(1 for v in counts.values() if v > 0)
    complete = populated == len(counts) and same_room_distance and same_position_absorption

    if len(sessions) >= len(CELL_ORDER):
        assert same_room_distance, (
            "no room holds two distances at a fixed absorption: distance and RT60 are confounded"
        )
        assert same_position_absorption, (
            "no source position is rendered at two absorptions: RT60 and distance are confounded"
        )
        assert populated == len(counts), f"2x2 design incomplete: cell counts {counts}"
    return {
        "counts": counts,
        "cells_populated": populated,
        "has_same_room_distance_contrast": same_room_distance,
        "has_same_position_absorption_contrast": same_position_absorption,
        "complete": complete,
        "n_rooms": len({s.block.room_id for s in sessions}),
    }


def verify_split_isolation(splits: dict[str, Sequence[SessionPlan]]) -> dict[str, Any]:
    """Assert speakers and source utterances never cross a split boundary.

    Checks three things: pairwise-disjoint speaker sets, pairwise-disjoint
    source-utterance sets, and that every variant group (all renders of one
    utterance at one position) lives in exactly one split.

    >>> plan = build_eval_plan(8, seed=2)
    >>> r = verify_split_isolation({"eval": plan.sessions, "train": []})
    >>> r["max_speaker_overlap"], r["max_utterance_overlap"]
    (0, 0)
    """
    speakers = {name: {s.speaker_id for s in items} for name, items in splits.items()}
    utterances = {name: {s.source_utt_id for s in items} for name, items in splits.items()}

    overlaps: dict[str, dict[str, int]] = {}
    max_speaker = 0
    max_utterance = 0
    for a, b in itertools.combinations(sorted(splits), 2):
        spk = speakers[a] & speakers[b]
        utt = utterances[a] & utterances[b]
        overlaps[f"{a}|{b}"] = {"speakers": len(spk), "utterances": len(utt)}
        max_speaker = max(max_speaker, len(spk))
        max_utterance = max(max_utterance, len(utt))
        assert not spk, f"speakers leak between {a} and {b}: {sorted(spk)[:5]}"
        assert not utt, f"source utterances leak between {a} and {b}: {sorted(utt)[:5]}"

    group_splits: dict[str, set[str]] = {}
    for name, items in splits.items():
        for s in items:
            group_splits.setdefault(s.variant_group, set()).add(name)
    straddling = {g: sorted(v) for g, v in group_splits.items() if len(v) > 1}
    assert not straddling, f"variant groups straddle splits: {list(straddling)[:5]}"

    return {
        "sizes": {name: len(items) for name, items in splits.items()},
        "n_speakers": {name: len(v) for name, v in speakers.items()},
        "n_utterances": {name: len(v) for name, v in utterances.items()},
        "pairwise_overlap": overlaps,
        "max_speaker_overlap": max_speaker,
        "max_utterance_overlap": max_utterance,
        "n_variant_groups": len(group_splits),
    }


def assign_splits(
    speaker_ids: Sequence[str],
    fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
    seed: int = 0,
) -> dict[str, str]:
    """Partition speakers into train/val/eval, one split per speaker.

    Splitting by *speaker* is the strong form of rule 9: it implies utterance
    disjointness and it also stops a model from recognising a voice it was
    trained on and recalling where that voice used to stand.

    >>> mapping = assign_splits([f"s{i}" for i in range(20)], seed=1)
    >>> sorted(set(mapping.values()))
    ['eval', 'train', 'val']
    >>> len(mapping)
    20
    """
    if len(fractions) != len(SPLIT_NAMES):
        raise ValueError(f"fractions must have {len(SPLIT_NAMES)} entries, got {fractions}")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {sum(fractions)}")
    unique = sorted(set(speaker_ids))
    if len(unique) < len(SPLIT_NAMES):
        raise ValueError(
            f"need at least {len(SPLIT_NAMES)} distinct speakers to isolate splits, got {len(unique)}"
        )
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)

    counts = [max(1, int(round(f * len(shuffled)))) for f in fractions]
    while sum(counts) > len(shuffled):
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < len(shuffled):
        counts[0] += 1

    mapping: dict[str, str] = {}
    cursor = 0
    for name, count in zip(SPLIT_NAMES, counts):
        for speaker in shuffled[cursor : cursor + count]:
            mapping[speaker] = name
        cursor += count
    return mapping


def verify_channel_gains(channel_gain_db: Sequence[float]) -> dict[str, Any]:
    """Assert per-channel gain mismatch stays inside the allowed band.

    The cap exists in both directions: a large mismatch would swamp the
    inter-channel level difference, and forcing every channel to equal loudness
    would delete that cue, which is one of the few things an M>=2 model has
    that an M=1 model does not.

    >>> verify_channel_gains([0.3, -0.2, 0.0, 0.49])["max_abs_db"]
    0.49
    >>> verify_channel_gains([0.9])
    Traceback (most recent call last):
        ...
    AssertionError: channel gain mismatch 0.900 dB exceeds CHANNEL_GAIN_MISMATCH_DB 0.5
    """
    max_abs = max(abs(float(g)) for g in channel_gain_db)
    assert max_abs <= CHANNEL_GAIN_MISMATCH_DB + 1e-12, (
        f"channel gain mismatch {max_abs:.3f} dB exceeds CHANNEL_GAIN_MISMATCH_DB {CHANNEL_GAIN_MISMATCH_DB}"
    )
    values = [float(g) for g in channel_gain_db]
    return {
        "max_abs_db": max_abs,
        "spread_db": max(values) - min(values),
        "cap_db": CHANNEL_GAIN_MISMATCH_DB,
    }


def verify_start_offsets(offsets_sec: Sequence[float]) -> dict[str, Any]:
    """Assert start offsets are continuous and never on the ``T_all`` grid.

    >>> verify_start_offsets([0.113, 1.887])["min_grid_residual_sec"] > 0
    True
    >>> verify_start_offsets([0.5])
    Traceback (most recent call last):
        ...
    AssertionError: start offset 0.500000 s is a multiple of spec.DELTA_T_SEC=0.25
    """
    residuals = []
    for offset in offsets_sec:
        assert 0.0 <= offset <= MAX_START_OFFSET_SEC + 1e-12, (
            f"start offset {offset} s outside [0, {MAX_START_OFFSET_SEC}]"
        )
        steps = offset / spec.DELTA_T_SEC
        residual = abs(steps - round(steps)) * spec.DELTA_T_SEC
        assert residual > 1e-9, (
            f"start offset {offset:.6f} s is a multiple of spec.DELTA_T_SEC={spec.DELTA_T_SEC}"
        )
        residuals.append(residual)
    return {
        "n_offsets": len(residuals),
        "min_grid_residual_sec": min(residuals) if residuals else 0.0,
        "max_offset_sec": max(offsets_sec) if offsets_sec else 0.0,
        "quantised_to_grid": False,
    }


def verify_plan(
    plan: EvalPlan,
    masks: Iterable[str] = spec.EVAL_MASKS,
    require_stratification: bool = True,
) -> dict[str, Any]:
    """Run every plan-level invariant and return the collected reports.

    >>> reports = verify_plan(build_eval_plan(4, seed=9))
    >>> sorted(reports)[:3]
    ['array_geometry', 'azimuth_stratification', 'case_ids']
    """
    masks = tuple(masks)
    sessions = plan.sessions

    # The stratification unit is the *source position*: one position carries
    # exactly one azimuth bin and one distance bin, and contributes a fixed
    # number of sessions (two in a full block, one in a single partial block).
    # Gating on session counts instead would reject a perfectly stratified
    # design whose position histogram is balanced but whose session histogram
    # is that same histogram doubled -- e.g. 40 sessions give positions
    # 3,2,3,2 per distance bin and sessions 6,4,6,4.  Because
    # sessions-per-position is constant (verify_position_balance), the two
    # histograms are proportional, so the uniformity flags describe both.
    azimuth_counts: dict[int, int] = {}
    distance_counts: dict[int, int] = {}
    position_azimuth: dict[int, int] = {}
    position_distance: dict[int, int] = {}
    seen_positions: set[str] = set()
    for s in sessions:
        azimuth_counts[s.source.azimuth_bin] = azimuth_counts.get(s.source.azimuth_bin, 0) + 1
        distance_counts[s.source.distance_bin] = distance_counts.get(s.source.distance_bin, 0) + 1
        if s.source.position_id not in seen_positions:
            seen_positions.add(s.source.position_id)
            position_azimuth[s.source.azimuth_bin] = (
                position_azimuth.get(s.source.azimuth_bin, 0) + 1
            )
            position_distance[s.source.distance_bin] = (
                position_distance.get(s.source.distance_bin, 0) + 1
            )

    az_report = verify_stratification(
        position_azimuth, spec.K_AZIMUTH, "azimuth (per source position)", require=require_stratification
    )
    dist_report = verify_stratification(
        position_distance,
        spec.K_DISTANCE,
        "distance (per source position)",
        require=require_stratification,
    )
    az_report["position_counts"] = az_report.pop("counts")
    dist_report["position_counts"] = dist_report.pop("counts")
    az_report["counts"] = {b: azimuth_counts.get(b, 0) for b in range(spec.K_AZIMUTH)}
    dist_report["counts"] = {b: distance_counts.get(b, 0) for b in range(spec.K_DISTANCE)}
    az_report["n_positions"] = len(seen_positions)
    dist_report["n_positions"] = len(seen_positions)
    az_report["n_sessions"] = len(sessions)
    dist_report["n_sessions"] = len(sessions)
    az_report["prior_only"] = prior_only_nll(az_report["counts"], spec.K_AZIMUTH)
    dist_report["prior_only"] = prior_only_nll(dist_report["counts"], spec.K_DISTANCE)

    geometry_reports = []
    plane_reports = []
    frame_report: dict[str, Any] = {}
    for block in plan.blocks:
        geometry_reports.append(verify_array_geometry(block.geometry))
        frame_report = verify_frame_invariance(block.geometry, masks)
        for source in block.sources.values():
            plane_reports.append(verify_plane_consistency(block.geometry, source.xy))

    for s in sessions:
        verify_channel_gains(s.channel_gain_db)
        assert len(s.channel_gain_db) == spec.N_CHANNELS_MAX, (
            f"session {s.session_id} has {len(s.channel_gain_db)} channel gains, expected {spec.N_CHANNELS_MAX}"
        )
        assert spec.azimuth_bin_index(s.source.azimuth_deg) == s.source.azimuth_bin, (
            f"session {s.session_id} azimuth label and bin disagree"
        )
        assert spec.distance_bin_index(s.source.distance_m) == s.source.distance_bin, (
            f"session {s.session_id} distance label and bin disagree"
        )

    case_ids = [cid for s in sessions for cid in s.case_ids(masks)]
    assert len(case_ids) == len(set(case_ids)), "duplicate case_id in plan"
    for cid in case_ids:
        session_id, mask_bits = spec.parse_case_id(cid)
        assert mask_bits in masks, f"case_id {cid} carries a mask outside the evaluated set"
        assert any(s.session_id == session_id for s in sessions), f"case_id {cid} has no session"

    offsets = sorted({s.start_offset_sec for s in sessions})

    return {
        "azimuth_stratification": az_report,
        "distance_stratification": dist_report,
        "confound_cells": verify_confound_cells(sessions),
        "split_isolation": verify_split_isolation(
            {name: [s for s in sessions if s.split == name] for name in SPLIT_NAMES}
        ),
        "start_offsets": verify_start_offsets(offsets),
        "array_geometry": geometry_reports[0] if geometry_reports else {},
        "n_arrays_checked": len(geometry_reports),
        "plane_consistency": {
            "n_checked": len(plane_reports),
            "max_abs_error_m": max((r["max_abs_error_m"] for r in plane_reports), default=0.0),
            "plane_height_m": spec.PLANE_HEIGHT_M,
        },
        "frame_invariance": frame_report,
        "case_ids": {"n_cases": len(case_ids), "n_unique": len(set(case_ids)), "masks": list(masks)},
        "exact_stratification_n_sessions": exact_stratification_n_sessions(),
    }


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------


def grid_times_sec(
    session_length_sec: float = spec.SESSION_LENGTH_SEC,
    delta_t_sec: float = spec.DELTA_T_SEC,
) -> tuple[float, ...]:
    """Return ``T_all``: the uniform inference grid from ``session_start``.

    >>> t = grid_times_sec()
    >>> len(t), t[0], t[1], t[-1]
    (80, 0.0, 0.25, 19.75)
    """
    n = int(round(session_length_sec / delta_t_sec))
    return tuple(k * delta_t_sec for k in range(n))


# ---------------------------------------------------------------------------
# Render layer -- lazy imports only below this line
# ---------------------------------------------------------------------------


def _require_numpy():  # noqa: ANN202 - module object
    """Import numpy lazily, with an actionable message when it is missing."""
    try:
        import numpy  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "numpy is required to render audio but is not installed.\n"
            "Plan-only verification works without it: python3 simulate.py --self-test"
        ) from exc
    return numpy


def _require_pyroomacoustics():  # noqa: ANN202 - module object
    """Import pyroomacoustics lazily, with an actionable message."""
    try:
        import pyroomacoustics  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "pyroomacoustics is not installed, so no official RIR can be computed.\n"
            "This generator does not install packages. Either:\n"
            "  1. install it yourself:  python3 -m pip install pyroomacoustics\n"
            "  2. or run the pipeline with the non-official numpy engine:\n"
            "       --rir-engine fallback_ism\n"
            "     which stamps official=false into the manifest and must not be\n"
            "     used for reported numbers."
        ) from exc
    return pyroomacoustics


def _require_soundfile():  # noqa: ANN202 - module object
    """Import soundfile lazily, with an actionable message."""
    try:
        import soundfile  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "soundfile is not installed, so LibriSpeech flac cannot be read.\n"
            "This generator does not install packages. Either install it\n"
            "(python3 -m pip install soundfile) or run with --synthetic, which\n"
            "needs numpy only."
        ) from exc
    return soundfile


def _fft_convolve(x, h):  # noqa: ANN001, ANN202 - numpy arrays
    """Linear convolution of two 1-D signals through the real FFT.

    Direct convolution of a 20 s session with an 8000-tap RIR is ~2.6e9
    multiply-adds per channel; the FFT path keeps a full session under a
    second without pulling in scipy.
    """
    np = _require_numpy()
    n = int(len(x) + len(h) - 1)
    size = 1 << (n - 1).bit_length()
    spec_x = np.fft.rfft(x, size)
    spec_h = np.fft.rfft(h, size)
    return np.fft.irfft(spec_x * spec_h, size)[:n]


def synthesise_utterance(
    rng: random.Random,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    duration_sec: float | None = None,
) -> tuple[Any, str, dict[str, Any]]:
    """Synthesise a voiced-like placeholder utterance and its transcript.

    The signal is additive harmonic synthesis of a drifting F0 with three
    formant-like resonances, gated by a syllable envelope with silences, so the
    pipeline produces a broadband signal with real pauses (which ``T_eval``
    needs) without any model checkpoint, corpus download or network access.

    The transcript is deterministic placeholder text with one word per
    syllable, so WER is computable end to end.

    >>> sig, text, meta = synthesise_utterance(random.Random(0), 16000, 2.0)
    >>> int(sig.shape[0]), meta["n_syllables"] == len(text.split())
    (32000, True)
    >>> float(abs(sig).max()) <= 1.0
    True
    """
    np = _require_numpy()
    if duration_sec is None:
        duration_sec = rng.uniform(*SYNTHETIC_DURATION_RANGE_SEC)
    n = int(round(duration_sec * sample_rate_hz))
    t = np.arange(n, dtype=np.float64) / float(sample_rate_hz)

    f0_base = rng.uniform(95.0, 210.0)
    drift = 1.0 + 0.10 * np.sin(2.0 * math.pi * rng.uniform(0.2, 0.6) * t + rng.uniform(0.0, 6.28))
    phase = 2.0 * math.pi * np.cumsum(f0_base * drift) / float(sample_rate_hz)

    formants = [rng.uniform(450.0, 800.0), rng.uniform(1100.0, 1900.0), rng.uniform(2400.0, 3200.0)]
    bandwidth = 420.0
    signal = np.zeros(n, dtype=np.float64)
    n_harmonics = max(1, int(0.45 * sample_rate_hz / f0_base))
    for k in range(1, n_harmonics + 1):
        freq = k * f0_base
        if freq >= 0.45 * sample_rate_hz:
            break
        weight = 1.0 / (k**1.2)
        for center in formants:
            weight += 0.9 / (1.0 + ((freq - center) / bandwidth) ** 2) / (k**0.5)
        signal += weight * np.sin(k * phase + rng.uniform(0.0, 6.28))

    syllable_hz = rng.uniform(3.0, 5.0)
    syllable = 0.5 * (1.0 - np.cos(2.0 * math.pi * syllable_hz * t))
    envelope = np.clip(syllable, 0.0, None) ** 1.5

    n_syllables = max(1, int(round(duration_sec * syllable_hz)))
    pause_mask = np.ones(n, dtype=np.float64)
    for _ in range(max(1, n_syllables // 6)):
        start = rng.uniform(0.0, max(0.0, duration_sec - 0.6))
        length = rng.uniform(0.18, 0.45)
        a = int(start * sample_rate_hz)
        b = min(n, int((start + length) * sample_rate_hz))
        pause_mask[a:b] = 0.0

    signal *= envelope * pause_mask
    peak = float(np.abs(signal).max())
    if peak > 0.0:
        signal = 0.9 * signal / peak

    # Drawn, not cycled: a fixed repeating word cycle would let a trivial
    # constant hypothesis score a near-zero WER, which would make the
    # placeholder useless even for smoke-testing the scorer.
    words = ("ALPHA", "BRAVO", "DELTA", "ECHO", "GAMMA", "KILO", "LIMA", "NOVA", "OSCAR", "SIERRA")
    transcript = " ".join(rng.choice(words) for _ in range(n_syllables))
    meta = {
        "source": "synthetic",
        "duration_sec": n / float(sample_rate_hz),
        "sample_rate_hz": sample_rate_hz,
        "f0_base_hz": f0_base,
        "syllable_rate_hz": syllable_hz,
        "n_syllables": n_syllables,
        "n_words": len(transcript.split()),
    }
    return signal.astype(np.float64), transcript, meta


def load_librispeech_utterance(flac_path: str | os.PathLike[str], sample_rate_hz: int = SAMPLE_RATE_HZ):  # noqa: ANN201
    """Read one LibriSpeech utterance as mono float64 at the given rate.

    Raises ``SystemExit`` with instructions when ``soundfile`` is missing and
    ``ValueError`` when the file's rate differs, rather than resampling
    silently -- an unnoticed resample would move every TDOA.
    """
    np = _require_numpy()
    sf = _require_soundfile()
    audio, rate = sf.read(str(flac_path), dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(rate) != int(sample_rate_hz):
        raise ValueError(
            f"{flac_path} is {rate} Hz but the generator is configured for {sample_rate_hz} Hz; "
            "resample the corpus offline instead of letting the generator do it silently"
        )
    return np.asarray(audio, dtype=np.float64)


def _sabine_absorption(rt60_s: float, dims: Sequence[float], speed_m_s: float) -> tuple[float, int]:
    """Return ``(absorption, max_order)`` for a Sabine RT60 target.

    >>> a, order = _sabine_absorption(0.4, (6.0, 5.0, 2.8), 343.0)
    >>> 0.0 < a < 1.0 and order >= 1
    True
    """
    lx, ly, lz = (float(d) for d in dims)
    volume = lx * ly * lz
    surface = 2.0 * (lx * ly + lx * lz + ly * lz)
    absorption = 24.0 * math.log(10.0) * volume / (speed_m_s * surface * rt60_s)
    absorption = min(max(absorption, 1e-3), 0.95)
    max_order = int(math.ceil(speed_m_s * rt60_s / min(lx, ly, lz)))
    return absorption, max(1, max_order)


def _schroeder_rt60(rir, sample_rate_hz: int, decay_db: float = 30.0) -> float:
    """Estimate RT60 by Schroeder backward integration. Pure numpy fallback.

    Used only when ``pyroomacoustics.measure_rt60`` is unavailable; which path
    produced a label is always recorded as ``rt60_method``.
    """
    np = _require_numpy()
    h = np.asarray(rir, dtype=np.float64)
    energy = np.cumsum(h[::-1] ** 2)[::-1]
    total = float(energy[0])
    if total <= 0.0:
        return float("nan")
    curve = 10.0 * np.log10(np.maximum(energy / total, 1e-20))
    try:
        i5 = int(np.flatnonzero(curve <= -5.0)[0])
        i_end = int(np.flatnonzero(curve <= -(5.0 + decay_db))[0])
    except IndexError:
        return float("nan")
    if i_end <= i5:
        return float("nan")
    slope_db_per_sample = (curve[i_end] - curve[i5]) / float(i_end - i5)
    if slope_db_per_sample >= 0.0:
        return float("nan")
    return float(-60.0 / slope_db_per_sample / sample_rate_hz)


def _measure_rt60_pra(room, rir, sample_rate_hz: int, channel: int) -> tuple[float, str]:
    """Measure RT60 with pyroomacoustics, falling back to Schroeder.

    Requirement: the RT60 *label* is a measurement on the generated RIR, never
    the requested value.  Several pyroomacoustics releases expose this through
    different entry points, so each is tried in turn and the one that worked is
    reported.
    """
    for call, name in (
        (lambda: room.measure_rt60(decay_db=30), "Room.measure_rt60(decay_db=30)"),
        (lambda: room.measure_rt60(), "Room.measure_rt60()"),
    ):
        try:
            value = call()
        except Exception:  # noqa: BLE001 - probing an unstable third-party API
            continue
        try:
            arr = value
            if hasattr(arr, "shape") and getattr(arr, "ndim", 0) >= 1:
                flat = arr[channel, 0] if getattr(arr, "ndim", 0) == 2 else arr[channel]
            else:
                flat = arr
            measured = float(flat)
        except Exception:  # noqa: BLE001
            continue
        if measured == measured and measured > 0.0:
            return measured, name
    try:
        from pyroomacoustics.experimental.rt60 import measure_rt60 as pra_measure  # noqa: PLC0415

        measured = float(pra_measure(rir, fs=sample_rate_hz, decay_db=30))
        if measured == measured and measured > 0.0:
            return measured, "pyroomacoustics.experimental.rt60.measure_rt60"
    except Exception:  # noqa: BLE001
        pass
    return _schroeder_rt60(rir, sample_rate_hz), "schroeder_fallback"


def _rirs_pyroomacoustics(
    block: RoomBlock, distance_level: str, rt60_s: float, sample_rate_hz: int
) -> tuple[list[Any], list[float], str, dict[str, Any]]:
    """Compute 4 RIRs with pyroomacoustics ShoeBox / Image Source Method."""
    np = _require_numpy()
    pra = _require_pyroomacoustics()

    dims = list(block.dims_xyz_m)
    requested = float(rt60_s)
    absorption: float
    max_order: int
    inverse_note = "pra.inverse_sabine"
    try:
        absorption, max_order = pra.inverse_sabine(requested, dims)
    except Exception:  # noqa: BLE001 - target may be unreachable for this volume
        absorption, max_order = _sabine_absorption(requested, dims, spec.SPEED_OF_SOUND_M_S)
        inverse_note = "local_sabine (pra.inverse_sabine rejected the target)"

    room = pra.ShoeBox(
        dims,
        fs=sample_rate_hz,
        materials=pra.Material(float(absorption)),
        max_order=int(max_order),
        air_absorption=False,
        ray_tracing=False,
    )
    mic_locs = np.asarray(block.mic_room_xyz(), dtype=np.float64).T
    room.add_microphone_array(pra.MicrophoneArray(mic_locs, sample_rate_hz))
    room.add_source(list(block.source_room_xyz(distance_level)))
    room.compute_rir()

    rirs = [np.asarray(room.rir[m][0], dtype=np.float64) for m in range(spec.N_CHANNELS_MAX)]
    measured: list[float] = []
    methods: set[str] = set()
    for m, rir in enumerate(rirs):
        value, method = _measure_rt60_pra(room, rir, sample_rate_hz, m)
        measured.append(value)
        methods.add(method)
    info = {
        "absorption": float(absorption),
        "max_order": int(max_order),
        "inverse_sabine": inverse_note,
        "rt60_method": sorted(methods),
    }
    return rirs, measured, "pyroomacoustics", info


def _rirs_fallback_ism(
    block: RoomBlock, distance_level: str, rt60_s: float, sample_rate_hz: int
) -> tuple[list[Any], list[float], str, dict[str, Any]]:
    """Compute 4 RIRs with a self-contained numpy image-source model.

    NOT OFFICIAL.  This exists so the whole pipeline -- plan, render, wav,
    metadata, manifest, hashes -- can be exercised on a machine without
    pyroomacoustics, and every artefact it produces is stamped
    ``official: false``.  Rigid shoe-box walls, frequency-independent
    reflection, a windowed-sinc fractional delay, an order cap of
    ``FALLBACK_ISM_MAX_ORDER_CAP`` and, past
    ``FALLBACK_MIXING_TIME_SEC``, an exponentially decaying diffuse tail that
    restores the decay rate the order cap would otherwise destroy.

    RT60 is still *measured* on the result rather than assumed, so the label
    describes the impulse response that actually produced the audio.
    """
    np = _require_numpy()
    dims = np.asarray(block.dims_xyz_m, dtype=np.float64)
    absorption, max_order = _sabine_absorption(rt60_s, block.dims_xyz_m, spec.SPEED_OF_SOUND_M_S)
    max_order = min(max_order, FALLBACK_ISM_MAX_ORDER_CAP)
    reflection = math.sqrt(max(0.0, 1.0 - absorption))
    src = np.asarray(block.source_room_xyz(distance_level), dtype=np.float64)

    grid = np.arange(-max_order, max_order + 1, dtype=np.float64)
    mg, qg = np.meshgrid(grid, np.asarray([0.0, 1.0]), indexing="ij")
    per_axis = []
    for axis in range(3):
        pos = (1.0 - 2.0 * qg) * src[axis] + 2.0 * mg * dims[axis]
        order = np.abs(mg - qg) + np.abs(mg)
        per_axis.append((pos.ravel(), order.ravel()))

    px, ox = per_axis[0]
    py, oy = per_axis[1]
    pz, oz = per_axis[2]
    ix, iy, iz = np.meshgrid(
        np.arange(px.size), np.arange(py.size), np.arange(pz.size), indexing="ij"
    )
    ix, iy, iz = ix.ravel(), iy.ravel(), iz.ravel()
    orders = ox[ix] + oy[iy] + oz[iz]
    keep = orders <= max_order
    images = np.stack([px[ix][keep], py[iy][keep], pz[iz][keep]], axis=1)
    attenuation = reflection ** orders[keep]

    rir_len = max(int(round(1.5 * max(rt60_s, 0.05) * sample_rate_hz)), 512)
    taps = np.arange(-4, 5, dtype=np.float64)
    tau_amplitude = max(rt60_s, 1e-3) / (60.0 / (20.0 * math.log10(math.e)))
    volume = float(dims[0] * dims[1] * dims[2])
    critical_distance = 0.0566 * math.sqrt(volume / max(rt60_s, 1e-3))
    rirs: list[Any] = []
    splice_points: list[int] = []
    drr_db: list[float] = []
    shared_key = f"{block.room_id}:{distance_level}:{rt60_s:.6f}:shared"
    shared_seed = int.from_bytes(hashlib.sha256(shared_key.encode("utf-8")).digest()[:8], "big")
    shared_noise = np.random.default_rng(shared_seed).standard_normal(rir_len + 32)
    for mic_index, mic in enumerate(np.asarray(block.mic_room_xyz(), dtype=np.float64)):
        distances = np.sqrt(((images - mic[None, :]) ** 2).sum(axis=1))
        distances = np.maximum(distances, 1e-3)
        amplitudes = attenuation / (4.0 * math.pi * distances)
        delays = distances / spec.SPEED_OF_SOUND_M_S * sample_rate_hz
        h = np.zeros(rir_len + 16, dtype=np.float64)
        base = np.floor(delays).astype(np.int64)
        frac = delays - base
        for tap in taps:
            idx = base + int(tap)
            arg = tap - frac
            sinc = np.sinc(arg) * (0.5 + 0.5 * np.cos(math.pi * np.clip(arg / 5.0, -1.0, 1.0)))
            valid = (idx >= 0) & (idx < h.size)
            np.add.at(h, idx[valid], (amplitudes * sinc)[valid])

        # The diffuse tail's *total energy* is set by the Sabine critical
        # distance, not by matching a local RMS: the direct-to-reverberant
        # ratio falls as (r_c / r)**2 and is the only usable distance cue in a
        # small array, so getting its absolute level right is what makes the
        # distance labels learnable rather than misleading.  Scaling the tail
        # to a local RMS instead put more energy in the tail than in the
        # direct path and moved the RIR peak off the direct arrival entirely.
        direct_distance = float(np.linalg.norm(src - mic))
        direct_samples = int(
            math.floor(direct_distance / spec.SPEED_OF_SOUND_M_S * sample_rate_hz)
        )
        splice = direct_samples + int(round(FALLBACK_MIXING_TIME_SEC * sample_rate_hz))
        splice_points.append(splice)
        window_lo = max(0, direct_samples - 8)
        window_hi = min(h.size, direct_samples + 9)
        direct_energy = float((h[window_lo:window_hi] ** 2).sum())
        target_reverb_energy = direct_energy * (direct_distance / max(critical_distance, 1e-6)) ** 2
        if splice < h.size:
            deficit = target_reverb_energy - float((h[splice:] ** 2).sum())
            if deficit > 0.0 and direct_energy > 0.0:
                n_tail = h.size - splice
                envelope = np.exp(
                    -np.arange(n_tail, dtype=np.float64) / (tau_amplitude * sample_rate_hz)
                )
                key = f"{block.room_id}:{distance_level}:{rt60_s:.6f}:{mic_index}"
                seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
                independent = np.random.default_rng(seed).standard_normal(n_tail)
                # The shared part is indexed from this microphone's own splice,
                # so it arrives with that microphone's direct-path delay and
                # carries the same TDOA as the direct sound.
                rho = min(max(FALLBACK_TAIL_COHERENCE, 0.0), 1.0)
                noise = (
                    math.sqrt(rho) * shared_noise[:n_tail]
                    + math.sqrt(1.0 - rho) * independent
                ) * envelope
                energy = float((noise**2).sum())
                if energy > 0.0:
                    h[splice:] += noise * math.sqrt(deficit / energy)
        reverb_energy = float((h[splice:] ** 2).sum()) if splice < h.size else 0.0
        drr_db.append(
            10.0 * math.log10(direct_energy / reverb_energy)
            if reverb_energy > 0.0 and direct_energy > 0.0
            else float("inf")
        )
        rirs.append(h[:rir_len])

    measured = [_schroeder_rt60(h, sample_rate_hz) for h in rirs]
    info = {
        "absorption": float(absorption),
        "max_order": int(max_order),
        "order_capped": int(max_order) >= FALLBACK_ISM_MAX_ORDER_CAP,
        "inverse_sabine": "local_sabine",
        "rt60_method": ["schroeder_fallback"],
        "tail_model": "exponential_noise_hybrid",
        "tail_level_rule": "sabine_critical_distance",
        "tail_coherence": FALLBACK_TAIL_COHERENCE,
        "critical_distance_m": critical_distance,
        "mixing_time_sec": FALLBACK_MIXING_TIME_SEC,
        "splice_samples": splice_points,
        "drr_db_per_channel": drr_db,
        "official": False,
    }
    return rirs, measured, "fallback_ism", info


def compute_session_rirs(
    session: SessionPlan,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    engine: str = "pyroomacoustics",
) -> dict[str, Any]:
    """Compute this session's four RIRs plus measured RT60 and a geometry check.

    Returns ``{"rirs", "rt60_measured_s", "rt60_requested_s", "engine",
    "engine_info", "geometry_check", "direct_delay_samples"}``.  The RT60 label
    is measured on the untruncated RIR before any storage truncation, so the
    label is a property of the acoustics rather than of the cache format.
    """
    if engine not in RIR_ENGINES:
        raise ValueError(f"engine must be one of {RIR_ENGINES}, got {engine!r}")
    block = session.block
    geometry_check = verify_plane_consistency(block.geometry, session.source.xy)
    verify_array_geometry(block.geometry)

    builder = _rirs_pyroomacoustics if engine == "pyroomacoustics" else _rirs_fallback_ism
    rirs, measured, engine_name, info = builder(
        block, session.distance_level, session.rt60_requested_s, sample_rate_hz
    )

    src = block.source_room_xyz(session.distance_level)
    direct = [
        math.dist(src, mic) / spec.SPEED_OF_SOUND_M_S * sample_rate_hz for mic in block.mic_room_xyz()
    ]
    return {
        "rirs": rirs,
        "rt60_measured_s": [float(v) for v in measured],
        "rt60_requested_s": float(session.rt60_requested_s),
        "engine": engine_name,
        "engine_info": info,
        "geometry_check": geometry_check,
        "direct_delay_samples": direct,
        "max_tdoa_samples": max(direct) - min(direct),
    }


def activity_grid(
    dry, sample_rate_hz: int, offset_samples: int, session_samples: int
):  # noqa: ANN001, ANN201
    """Return per-grid-point speech activity of the dry source, as ``T_all`` flags.

    ``spec.py`` fixes ``T_all`` and the ``session_start`` time origin but not
    which grid points carry a label, so the rule is local and documented:
    ``t_k`` is evaluated when the dry RMS over ``[t_k, t_k + DELTA_T_SEC)`` is
    within :data:`VAD_REL_THRESHOLD_DB` of the utterance's loud reference
    level.  The *dry* signal is used so a reverberant tail cannot extend
    ``T_eval`` past the talker.
    """
    np = _require_numpy()
    placed = np.zeros(session_samples, dtype=np.float64)
    end = min(session_samples, offset_samples + len(dry))
    if end > offset_samples:
        placed[offset_samples:end] = dry[: end - offset_samples]

    hop = int(round(spec.DELTA_T_SEC * sample_rate_hz))
    times = grid_times_sec()
    rms = np.zeros(len(times), dtype=np.float64)
    for k in range(len(times)):
        a = k * hop
        b = min(session_samples, a + hop)
        if b > a:
            rms[k] = float(np.sqrt(np.mean(placed[a:b] ** 2)))
    nonzero = rms[rms > 0.0]
    if nonzero.size == 0:
        return tuple(False for _ in times), rms
    reference = float(np.percentile(nonzero, VAD_REF_PERCENTILE))
    threshold = reference * (10.0 ** (VAD_REL_THRESHOLD_DB / 20.0))
    return tuple(bool(v > threshold) for v in rms), rms


def render_session(
    session: SessionPlan,
    rirs: Sequence[Any],
    dry,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
):  # noqa: ANN001, ANN201
    """Convolve, place, gain and clip-protect one session into ``(n, 4)`` audio.

    The single start offset is applied to the dry signal *once*, before the
    per-channel convolution, so every channel of a case shares it and the TDOA
    carried by the RIRs survives untouched (rule 7).  The common gain is one
    scalar for all channels, the per-channel mismatch is capped, and any
    anti-clip rescale is also a single scalar -- so the inter-channel level
    difference is never flattened (rule 4).

    Returns ``(audio, info)``.
    """
    np = _require_numpy()
    if len(rirs) != spec.N_CHANNELS_MAX:
        raise ValueError(f"expected {spec.N_CHANNELS_MAX} RIRs, got {len(rirs)}")
    verify_channel_gains(session.channel_gain_db)
    verify_start_offsets([session.start_offset_sec])

    session_samples = int(round(spec.SESSION_LENGTH_SEC * sample_rate_hz))
    grid_samples = int(round(spec.DELTA_T_SEC * sample_rate_hz))
    offset_samples = int(round(session.start_offset_sec * sample_rate_hz))
    assert offset_samples % grid_samples != 0, (
        f"session {session.session_id} start offset landed on the T_all grid"
    )

    audio = np.zeros((session_samples, spec.N_CHANNELS_MAX), dtype=np.float64)
    truncated = False
    for channel, rir in enumerate(rirs):
        wet = _fft_convolve(np.asarray(dry, dtype=np.float64), np.asarray(rir, dtype=np.float64))
        end = min(session_samples, offset_samples + len(wet))
        if offset_samples + len(wet) > session_samples:
            truncated = True
        if end > offset_samples:
            audio[offset_samples:end, channel] = wet[: end - offset_samples]

    reference_rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    target_rms = 10.0 ** (session.common_gain_dbfs / 20.0)
    common_gain = target_rms / reference_rms if reference_rms > 0.0 else 1.0
    audio *= common_gain

    mismatch = np.asarray(
        [10.0 ** (g / 20.0) for g in session.channel_gain_db], dtype=np.float64
    )
    audio *= mismatch[None, :]

    peak = float(np.abs(audio).max()) if audio.size else 0.0
    anti_clip = 1.0
    if peak > PEAK_CEILING:
        anti_clip = PEAK_CEILING / peak
        audio *= anti_clip

    channel_rms = [float(np.sqrt(np.mean(audio[:, c] ** 2))) for c in range(spec.N_CHANNELS_MAX)]
    info = {
        "offset_samples": offset_samples,
        "offset_sec_exact": offset_samples / float(sample_rate_hz),
        "common_gain_linear": common_gain,
        "common_gain_dbfs_target": session.common_gain_dbfs,
        "channel_gain_db": list(session.channel_gain_db),
        "anti_clip_scale": anti_clip,
        "peak": float(np.abs(audio).max()) if audio.size else 0.0,
        "channel_rms": channel_rms,
        "channel_rms_dbfs": [
            (20.0 * math.log10(v) if v > 0.0 else float("-inf")) for v in channel_rms
        ],
        "ild_spread_db": (
            20.0 * math.log10(max(channel_rms) / min(channel_rms))
            if min(channel_rms) > 0.0
            else float("inf")
        ),
        "tail_truncated": truncated,
    }
    return audio.astype(np.float32), info


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def save_rir_bundle(
    path: str | os.PathLike[str],
    rirs: Sequence[Any],
    meta: dict[str, Any],
    store_sec: float = RIR_STORE_SEC,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> dict[str, Any]:
    """Store one session's RIRs as a truncated float32 ``.npz`` plus metadata.

    Rule 10's storage strategy: only RIRs are pre-computed, and the utterance
    convolution, gain and offset happen at load time.  Pre-rendering every
    training variant to wav would be tens of gigabytes; 4 channels x 0.5 s of
    float32 is ~128 KiB.  The kept-energy fraction is recorded so a truncation
    that would bias the reverberation cue is visible rather than silent.
    """
    np = _require_numpy()
    length = int(round(store_sec * sample_rate_hz))
    stacked = np.zeros((len(rirs), length), dtype=np.float32)
    kept = []
    for i, rir in enumerate(rirs):
        h = np.asarray(rir, dtype=np.float64)
        total = float((h**2).sum())
        n = min(length, h.size)
        stacked[i, :n] = h[:n].astype(np.float32)
        kept.append(float((h[:n] ** 2).sum() / total) if total > 0.0 else 1.0)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(meta)
    payload.update(
        {
            "sample_rate_hz": sample_rate_hz,
            "store_sec": store_sec,
            "n_channels": len(rirs),
            "original_lengths": [int(np.asarray(r).size) for r in rirs],
            "energy_kept_fraction": kept,
            "truncated": any(int(np.asarray(r).size) > length for r in rirs),
        }
    )
    np.savez_compressed(path, rirs=stacked, meta_json=json.dumps(payload, sort_keys=True))
    return payload


def load_rir_bundle(path: str | os.PathLike[str]) -> tuple[Any, dict[str, Any]]:
    """Load a ``.npz`` RIR bundle written by :func:`save_rir_bundle`."""
    np = _require_numpy()
    with np.load(Path(path), allow_pickle=False) as data:
        return np.asarray(data["rirs"], dtype=np.float64), json.loads(str(data["meta_json"]))


def write_wav_pcm16(
    path: str | os.PathLike[str], audio, sample_rate_hz: int = SAMPLE_RATE_HZ
) -> dict[str, Any]:  # noqa: ANN001
    """Write an ``(n, channels)`` float array as 16-bit PCM using stdlib ``wave``.

    Deliberately avoids ``soundfile`` for *writing*: the stdlib handles
    multi-channel 16-bit PCM, which keeps eval-dev generation possible on a
    machine that only has numpy.
    """
    np = _require_numpy()
    arr = np.asarray(audio)
    if arr.ndim != 2:
        raise ValueError(f"audio must be (n_samples, n_channels), got shape {arr.shape}")
    clipped = np.clip(arr, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(arr.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate_hz))
        handle.writeframes(np.ascontiguousarray(pcm).tobytes())
    return {
        "path": str(path),
        "n_samples": int(arr.shape[0]),
        "n_channels": int(arr.shape[1]),
        "sample_rate_hz": int(sample_rate_hz),
        "encoding": "pcm_s16le",
        "n_clipped_samples": int((np.abs(arr) > 1.0).sum()),
    }


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA256 of a file, read in chunks.

    >>> import tempfile, pathlib
    >>> d = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (d / "x.txt").write_bytes(b"abc")
    >>> sha256_file(d / "x.txt")[:16]
    'ba7816bf8f01cfea'
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def self_test(
    n_sessions: int = 12,
    seed: int = 20261029,
    verbose: bool = True,
    allow_unbalanced: bool = False,
) -> dict[str, Any]:
    """Run every plan-layer invariant on a generated plan. Pure standard library.

    >>> report = self_test(8, seed=3, verbose=False)
    >>> report["ok"]
    True

    The two stratification gates catch different faults.  With 6 sessions each
    source position still holds its own azimuth bin, so the *position*
    histogram is uniform; what breaks is the number of sessions per position,
    which is what :func:`verify_position_balance` rejects:

    >>> loose = self_test(6, seed=3, verbose=False, allow_unbalanced=True)
    >>> loose["azimuth"]["uniform_over_used"], loose["position_balance"]["balanced"]
    (True, False)
    >>> sorted(loose["azimuth"]["counts"].values())[-2:]  # sessions per bin differ
    [2, 2]
    """
    plan = build_eval_plan(n_sessions, seed=seed, allow_unbalanced=allow_unbalanced)
    reports = plan.reports

    pool = [f"syn_spk_{i:03d}" for i in range(24)]
    mapping = assign_splits(pool, seed=seed)
    by_split: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for speaker, name in mapping.items():
        by_split[name].append(speaker)
    assert all(by_split[name] for name in SPLIT_NAMES), "a split came out empty"
    for a, b in itertools.combinations(SPLIT_NAMES, 2):
        assert not (set(by_split[a]) & set(by_split[b])), f"speaker leak {a}/{b}"

    for mask in spec.EVAL_MASKS:
        channels = spec.mask_to_channels(mask)
        sub = plan.blocks[0].geometry.for_mask(mask)
        assert sub.n_channels == len(channels)
        assert sub.reference_point_xy == (0.0, 0.0)

    spacings = plan.blocks[0].geometry.pair_spacings_m()
    short_pair = spacings[(0, 1)]
    long_pair = spacings[(0, 2)]
    assert short_pair < long_pair, "mask '1100' must be the shorter pair than '1010'"

    times = grid_times_sec()
    assert len(times) == int(round(spec.SESSION_LENGTH_SEC / spec.DELTA_T_SEC))

    summary = {
        "ok": True,
        "n_sessions": plan.n_sessions,
        "n_cases": plan.n_cases,
        "n_rooms": len(plan.blocks),
        "spec_version": spec.SPEC_VERSION,
        "generator_version": GENERATOR_VERSION,
        "grid_points": len(times),
        "azimuth": reports["azimuth_stratification"],
        "distance": reports["distance_stratification"],
        "confound_cells": reports["confound_cells"],
        "split_isolation": reports["split_isolation"],
        "start_offsets": reports["start_offsets"],
        "plane_consistency": reports["plane_consistency"],
        "frame_invariance": reports["frame_invariance"],
        "array_geometry": reports["array_geometry"],
        "exact_stratification_n_sessions": reports["exact_stratification_n_sessions"],
        "position_balance": reports["position_balance"],
        "split_assignment_sizes": {k: len(v) for k, v in by_split.items()},
    }
    if verbose:
        az = reports["azimuth_stratification"]
        dist = reports["distance_stratification"]
        cells = reports["confound_cells"]
        print(f"spec {spec.SPEC_VERSION} / generator {GENERATOR_VERSION}")
        print(f"  sessions            : {plan.n_sessions} in {len(plan.blocks)} rooms -> {plan.n_cases} cases")
        print(f"  grid (T_all)        : {len(times)} points, {times[0]} .. {times[-1]} s")
        print(
            f"  azimuth strat       : {az['n_used_bins']}/{spec.K_AZIMUTH} bins used, "
            f"{az['min_count']}..{az['max_count']} positions/bin, strict={az['strict_uniform']}, "
            f"uniform_over_used={az['uniform_over_used']}"
        )
        print(
            f"  distance strat      : {dist['n_used_bins']}/{spec.K_DISTANCE} bins used, "
            f"{dist['min_count']}..{dist['max_count']} positions/bin, strict={dist['strict_uniform']}, "
            f"uniform_over_used={dist['uniform_over_used']}"
        )
        print(
            f"  memorisation gain   : azimuth {az['prior_only']['memorisation_gain_nats']:.3f} nats, "
            f"distance {dist['prior_only']['memorisation_gain_nats']:.3f} nats "
            f"(0 requires n_sessions={reports['exact_stratification_n_sessions']})"
        )
        print(f"  2x2 cells           : {cells['counts']} complete={cells['complete']}")
        print(f"  plane consistency   : max |3D-2D| = {reports['plane_consistency']['max_abs_error_m']:.2e} m")
        print(f"  frame invariance    : origin shift {reports['frame_invariance']['origin_shift_max_m']} m")
        print(
            f"  array (room_000)    : spacings "
            f"{[round(v, 4) for v in reports['array_geometry']['pair_spacings_m'].values()]}, "
            f"aliasing {reports['array_geometry']['aliasing_limit_hz']:.1f} Hz"
        )
        print(
            f"  start offsets       : min grid residual "
            f"{reports['start_offsets']['min_grid_residual_sec']*1e3:.3f} ms (0 would be a leak)"
        )
        print(f"  split isolation     : overlaps {reports['split_isolation']['pairwise_overlap']}")
        print("  all plan invariants OK")
    return summary


def _main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``--self-test`` and ``--doctest``."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="VC-SASR multi-channel spatial session simulator (library + self test)."
    )
    parser.add_argument("--self-test", action="store_true", help="run every plan-layer invariant")
    parser.add_argument("--doctest", action="store_true", help="run the module doctests")
    parser.add_argument("--n-sessions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--json", action="store_true", help="print the self-test report as JSON")
    parser.add_argument(
        "--allow-unbalanced",
        action="store_true",
        help="permit a session count whose trailing block cannot be stratified",
    )
    args = parser.parse_args(argv)

    status = 0
    if args.doctest or not (args.self_test or args.doctest):
        import doctest  # noqa: PLC0415

        failures, tests = doctest.testmod(verbose=False)
        print(f"doctest: {tests - failures}/{tests} passed")
        status |= 1 if failures else 0
    if args.self_test:
        report = self_test(
            args.n_sessions,
            args.seed,
            verbose=not args.json,
            allow_unbalanced=args.allow_unbalanced,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return status


if __name__ == "__main__":
    raise SystemExit(_main())
