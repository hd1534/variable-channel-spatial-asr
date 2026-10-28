"""Frozen evaluation contract for the Variable-Channel Spatial ASR benchmark.

This module is the single source of truth for every constant and rule that
``evaluate.py``, the data generator and the baselines must agree on.  The
project proposal (v1.7) left several of these values open; they are closed here
with concrete numbers so that the evaluator can be implemented at all.

Every value in this file is a *decision*, not a measurement.  Changing any of
them invalidates previously generated data and previously reported scores, so
each change must bump :data:`SPEC_VERSION`.

Rationale notes marked ``[v1.7 gap]`` record a value the proposal did not fix.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Final, Literal

SPEC_VERSION: Final[str] = "0.2.0"
"""Semantic version of this contract. Stamped into every score report."""


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------

DELTA_T_SEC: Final[float] = 0.25
"""Spacing of the uniform inference grid ``T_all``.

0.25 is exactly representable in binary floating point, so ``k * DELTA_T_SEC``
is exact and ``eval_time_sec`` can safely be used as a dictionary key.
"""

SESSION_LENGTH_SEC: Final[float] = 20.0
"""Fixed session length. [v1.7 gap] The proposal never fixed a session length,
which left the number of evaluation records per case undefined."""

CHUNK_LENGTH_SEC: Final[float] = 1.0
"""Spatial/speaker model chunk length. [v1.7 gap] ``Exp 2`` manipulates
"reset state every chunk" vs "carry state across chunks" but never said how
long a chunk is, which made the result uninterpretable."""

CHUNK_HOP_SEC: Final[float] = 0.25
"""Chunk hop, equal to :data:`DELTA_T_SEC` so every grid point gets one
prediction."""

ASR_MIN_SEGMENT_SEC: Final[float] = 1.0
"""Minimum VAD segment length handed to the ASR backend.

``final_transcripts`` is built by running the frozen ASR once per VAD segment
and attributing each segment transcript to the majority speaker slot over that
segment. [v1.7 gap] The proposal scored ``final_transcripts`` but never said
who produces it.
"""

GRID_ENDPOINT: Final[Literal["exclusive"]] = "exclusive"
"""``T_all = {k * DELTA_T_SEC : 0 <= k * DELTA_T_SEC < SESSION_LENGTH_SEC}``.

The session end is excluded, so a 20 s session has exactly 80 grid instants,
the first at 0.0 and the last at 19.75.

[v1.7 gap] The proposal gave the spacing and the session length but not
whether the endpoint is included.  80 against 81 is a silent off-by-one in
every per-case denominator, and an instant at exactly 20.0 s has no audio
under it: a causal model would be charged :data:`PENALTY_NLL` for a frame
that does not exist.
"""

N_GRID_POINTS: Final[int] = int(SESSION_LENGTH_SEC / DELTA_T_SEC)
"""80 instants per session, from :data:`GRID_ENDPOINT`."""

T_EVAL_RULE: Final[Literal["dry_source_rms_gate"]] = "dry_source_rms_gate"
"""How ``T_eval`` is selected out of ``T_all``.

A grid instant ``t_k`` carries a label when the **dry source** RMS over
``[t_k, t_k + DELTA_T_SEC)`` is within :data:`VAD_REL_THRESHOLD_DB` of that
utterance's loud reference level, defined as the
:data:`VAD_REF_PERCENTILE`-th percentile of its non-silent frame RMS.

[v1.7 gap] ``SPATIAL_AGGREGATION`` weights every instant by ``1 / |T_eval|``
and the proposal never said how ``T_eval`` is chosen.  That makes it the
denominator of every spatial metric: a looser gate admits near-silent instants
where no system can localise anything and drags every score down, a tighter one
scores only the easy loud frames.  Two conforming generators would therefore
publish different leaderboards from the same model.

The gate reads the **dry** source, before room propagation, so the label set
does not depend on reverberation -- otherwise the high-RT60 half of the 2x2
confound design would be scored on a different instant set than the low-RT60
half, which would confound exactly the contrast it exists to measure.

``T_eval`` is evaluator-internal: a submission is expected to answer every
instant in ``T_all`` and is never told which ones count, so Oracle VAD cannot
leak through it.
"""

VAD_REL_THRESHOLD_DB: Final[float] = -30.0
"""Gate width of :data:`T_EVAL_RULE`, in dB below the loud reference level.

-30 dB keeps ordinary speech decay and inter-word gaps inside a word, while
excluding the inter-utterance silence where a direction label is meaningless.
"""

VAD_REF_PERCENTILE: Final[float] = 90.0
"""Percentile of non-silent per-frame RMS used as the loud reference level.

The 90th rather than the maximum, so one glottal peak or one clipped sample
cannot raise the reference and silence half an utterance.
"""

TIME_ORIGIN: Final[Literal["session_start"]] = "session_start"
"""``eval_time_sec`` is measured from the first sample of the session audio,
not from the first speech frame. [v1.7 gap]"""


# ---------------------------------------------------------------------------
# Array geometry and acoustics
# ---------------------------------------------------------------------------

N_CHANNELS_MAX: Final[int] = 4
"""Maximum number of microphones ``N``."""

ARRAY_RADIUS_RANGE_M: Final[tuple[float, float]] = (0.05, 0.12)
"""Planar array radius range."""

MAX_PAIR_SPACING_M: Final[float] = 0.24
"""Largest microphone pair spacing (array diameter at the maximum radius)."""

SPEED_OF_SOUND_M_S: Final[float] = 343.0

PLANE_HEIGHT_M: Final[float] = 1.5
"""Common ``z`` for every microphone and every source.

[v1.7 gap] The proposal defines coordinates on a 2D horizontal plane but
``pyroomacoustics`` rooms are 3D.  Placing the array at 1.0 m and a mouth at
1.6 m would make the true slant range to a point 1 m away horizontally equal
1.166 m -- a 17 % systematic bias in every distance label that no test would
catch.  Keeping one plane makes the 2D distance the true distance.
"""


def aliasing_limit_hz(
    spacing_m: float = MAX_PAIR_SPACING_M,
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S,
) -> float:
    """Return the narrowband half-wavelength spatial aliasing limit.

    Above this frequency the pair spacing exceeds lambda/2 and phase differences
    wrap, so the array cannot by itself resolve direction unambiguously.

    >>> round(aliasing_limit_hz(), 1)
    714.6
    """
    return speed_of_sound_m_s / (2.0 * spacing_m)


ALIASING_LIMIT_HZ: Final[float] = aliasing_limit_hz()
"""~714.6 Hz for a 24 cm maximum pair spacing."""


# ---------------------------------------------------------------------------
# Audio format
# ---------------------------------------------------------------------------

SAMPLE_RATE_HZ: Final[int] = 16_000
"""Sample rate of every distributed ``.wav`` and of every RIR.

[v1.7 gap] The proposal never fixed a sample rate, and the value lived only in
``simulate.py``.  That made two of this contract's own numbers unverifiable from
the contract: ``MAX_TDOA_SAMPLES`` below, and the band in which
:data:`ALIASING_LIMIT_HZ` sits relative to Nyquist.  16 kHz is the LibriSpeech
rate, so no resampling is needed on the source side.
"""

AUDIO_ENCODING: Final[Literal["pcm_s16le"]] = "pcm_s16le"
"""Signed 16-bit little-endian PCM, interleaved, ``N`` channels per file."""

MAX_TDOA_SEC: Final[float] = MAX_PAIR_SPACING_M / SPEED_OF_SOUND_M_S
"""Largest possible inter-microphone delay, ~0.6997 ms."""

MAX_TDOA_SAMPLES: Final[float] = MAX_TDOA_SEC * SAMPLE_RATE_HZ
"""~11.2 samples at 16 kHz.

The whole usable direction cue therefore spans about 22 lags.  A cross
correlation evaluated on the integer lag grid alone quantises azimuth far more
coarsely than :data:`AZIMUTH_BIN_WIDTH_DEG`, so sub-sample interpolation is a
requirement for any TDOA front-end, not an optimisation.
"""


# ---------------------------------------------------------------------------
# Fixed evaluation channel subsets
# ---------------------------------------------------------------------------

EVAL_MASKS: Final[tuple[str, ...]] = (
    "1000",  # M=1, channel 0
    "0010",  # M=1, channel 2
    "1100",  # M=2, shortest pair spacing
    "1010",  # M=2, longest pair spacing
    "1110",  # M=3
    "1111",  # M=4
)
"""The six channel subsets that are actually evaluated.

[v1.7 gap] The proposal said "evaluate all of a fixed set of subsets"; with
``N = 4`` there are 15 non-empty subsets, which multiplies every inference run
and every eval-dev record by 15.  Six subsets keep both M=1 channels, both the
shortest and longest M=2 pair (so the aliasing argument is testable), one M=3
and the full array.

The channel referenced by bit ``i`` is channel ``i``, counting from the left.
The shortest/longest pair claims above hold under
:data:`DEFAULT_ARRAY_GEOMETRY`.
"""

DEFAULT_ARRAY_GEOMETRY: Final[Literal["uniform_circular"]] = "uniform_circular"
"""Microphone ``i`` of ``N_CHANNELS_MAX`` sits at ``360 * i / N`` degrees on a
circle of diameter :data:`MAX_PAIR_SPACING_M`, in the plane
:data:`PLANE_HEIGHT_M`.

[v1.7 gap] :data:`EVAL_MASKS` claims ``1100`` is the shortest pair and ``1010``
the longest -- claims that are only true for a specific channel ordering, which
the proposal never gave.  Under a uniform circular array, adjacent channels are
a chord apart and opposite channels a diameter apart, so the claim holds and the
aliasing argument has a concrete spacing behind it.  Each distributed case also
carries its own measured coordinates; this is the fallback when they are absent
and the reference geometry the mask comments assume.
"""


def mask_to_channels(mask_bits: str) -> tuple[int, ...]:
    """Return the active channel indices encoded by ``mask_bits``.

    >>> mask_to_channels("1010")
    (0, 2)
    """
    if len(mask_bits) != N_CHANNELS_MAX or set(mask_bits) - {"0", "1"}:
        raise ValueError(f"mask_bits must be {N_CHANNELS_MAX} binary digits, got {mask_bits!r}")
    return tuple(i for i, bit in enumerate(mask_bits) if bit == "1")


def make_case_id(session_id: str, mask_bits: str) -> str:
    """Build the primary evaluation key ``<session_id>__m<mask_bits>``.

    >>> make_case_id("sess_001", "1010")
    'sess_001__m1010'
    """
    mask_to_channels(mask_bits)  # validates
    if "__" in session_id:
        raise ValueError(f"session_id must not contain '__': {session_id!r}")
    return f"{session_id}__m{mask_bits}"


def parse_case_id(case_id: str) -> tuple[str, str]:
    """Split a ``case_id`` back into ``(session_id, mask_bits)``.

    >>> parse_case_id("sess_001__m1010")
    ('sess_001', '1010')
    """
    session_id, _, tail = case_id.partition("__m")
    if not session_id or not tail:
        raise ValueError(f"malformed case_id: {case_id!r}")
    mask_to_channels(tail)
    return session_id, tail


# ---------------------------------------------------------------------------
# Azimuth discretisation
# ---------------------------------------------------------------------------

K_AZIMUTH: Final[int] = 36
"""Number of azimuth bins."""

AZIMUTH_BIN_WIDTH_DEG: Final[float] = 10.0

AZIMUTH_RANGE_DEG: Final[tuple[float, float]] = (-180.0, 180.0)
"""Half-open ``[-180, 180)``. ``+180`` is identical to ``-180``."""

AZIMUTH_CONVENTION: Final[Literal["atan2_y_x_ccw"]] = "atan2_y_x_ccw"
"""``azimuth_deg = degrees(atan2(y, x))`` about the array origin.

Zero points along ``+x``, angles increase counter-clockwise, and the inverse
is ``(x, y) = (r cos az, r sin az)`` -- which is what ``pos_error_2d_m``
needs to be computable from a polar submission at all.

[v1.7 gap] The proposal binned azimuth without naming a zero reference or a
sign.  A clockwise submission is not slightly wrong: it is mirrored, so a
model with perfect direction-of-arrival estimation would score an azimuth MAE
near 90 deg and look like chance.  The "DSP front-end beats single channel"
sanity check cannot distinguish a broken model from a mirrored convention.
"""

AZIMUTH_BIN_EDGES_DEG: Final[tuple[float, ...]] = tuple(
    -180.0 + AZIMUTH_BIN_WIDTH_DEG * i for i in range(K_AZIMUTH + 1)
)
"""37 edges. Bin ``i`` covers ``[edges[i], edges[i + 1])``. [v1.7 gap] The
proposal never stated the index-to-interval convention, so two conforming
evaluators could disagree on every NLL."""

AZIMUTH_BIN_CENTERS_DEG: Final[tuple[float, ...]] = tuple(
    -180.0 + AZIMUTH_BIN_WIDTH_DEG * (i + 0.5) for i in range(K_AZIMUTH)
)
"""Bin centers, i.e. ``-175, -165, ..., +175``. Used as the MAP representative
point when no explicit point prediction is submitted."""


def azimuth_bin_index(azimuth_deg: float) -> int:
    """Return the bin index containing ``azimuth_deg``.

    The angle is first wrapped into ``[-180, 180)``.

    >>> azimuth_bin_index(-180.0), azimuth_bin_index(0.0), azimuth_bin_index(179.9)
    (0, 18, 35)
    >>> azimuth_bin_index(180.0)  # wraps to -180
    0
    """
    wrapped = (azimuth_deg + 180.0) % 360.0
    index = int(wrapped // AZIMUTH_BIN_WIDTH_DEG)
    return min(index, K_AZIMUTH - 1)


def circular_error_deg(truth_deg: float, pred_deg: float) -> float:
    """Return ``min(|d|, 360 - |d|)``, the azimuth error used by every metric.

    >>> circular_error_deg(-179.0, 179.0)
    2.0
    >>> circular_error_deg(-55.0, 55.0)
    110.0
    """
    diff = abs((truth_deg - pred_deg) % 360.0)
    return min(diff, 360.0 - diff)


AZIMUTH_ACC_THRESHOLDS_DEG: Final[tuple[float, ...]] = (10.0, 20.0)
"""Thresholds of the ``azimuth_acc_<t>deg`` metrics.

An instant counts as correct when :func:`circular_error_deg` is ``<=`` the
threshold.

[v1.7 gap] The thresholds appeared only inside the metric *names* in
:data:`OFFICIAL_METRICS`, so the evaluator had to recover them with a regular
expression over its own metric list.  They are declared here instead; the two
lists are cross-checked at import time.  10 deg is exactly one bin, so
``azimuth_acc_10deg`` is the accuracy of the bin decision itself.
"""


# ---------------------------------------------------------------------------
# Distance discretisation
# ---------------------------------------------------------------------------

DISTANCE_RANGE_M: Final[tuple[float, float]] = (0.5, 5.0)
"""Evaluated source-to-origin distance range. [v1.7 gap] The proposal only
mentioned 5.0 m inside the missing-value penalty and never bounded the range."""

DISTANCE_BIN_EDGES_M: Final[tuple[float, ...]] = (
    0.5, 0.7, 1.0, 1.4, 1.9, 2.5, 3.2, 4.0, 5.0,
)
"""Nine edges, so ``K_r = 8`` roughly log-spaced bins.

[v1.7 gap] ``K_r`` stayed the symbol ``K_r`` through the whole proposal, which
made ``Distance NLL``, the ``log K_r`` missing penalty and the ``delta_r / 2``
quantisation check in test 5(b) all uncomputable.  Log spacing is used because
the only usable distance cue in a small array is the direct-to-reverberant
ratio, which falls roughly with ``1 / r``: equal-ratio bins carry roughly equal
DRR difference, while linear bins would waste resolution far away.

Bin ``j`` covers ``[edges[j], edges[j + 1])``; the last bin is closed.
"""

K_DISTANCE: Final[int] = len(DISTANCE_BIN_EDGES_M) - 1
"""8."""

DISTANCE_BIN_CENTERS_M: Final[tuple[float, ...]] = tuple(
    math.sqrt(DISTANCE_BIN_EDGES_M[j] * DISTANCE_BIN_EDGES_M[j + 1])
    for j in range(K_DISTANCE)
)
"""Geometric bin centers, consistent with log spacing.

Used as the MAP representative point when no explicit point prediction is
submitted, exactly as :data:`AZIMUTH_BIN_CENTERS_DEG` is.  [v1.7 gap] The
azimuth side said so and this one did not, so the symmetry was a convention
rather than a rule.
"""

DISTANCE_BIN_WIDTHS_M: Final[tuple[float, ...]] = tuple(
    DISTANCE_BIN_EDGES_M[j + 1] - DISTANCE_BIN_EDGES_M[j] for j in range(K_DISTANCE)
)


def distance_bin_index(distance_m: float) -> int:
    """Return the bin index containing ``distance_m``, clipped to the range.

    >>> distance_bin_index(0.5), distance_bin_index(1.0), distance_bin_index(5.0)
    (0, 2, 7)
    >>> distance_bin_index(0.1), distance_bin_index(99.0)  # clipped
    (0, 7)
    """
    if distance_m < DISTANCE_BIN_EDGES_M[0]:
        return 0
    for j in range(K_DISTANCE):
        if distance_m < DISTANCE_BIN_EDGES_M[j + 1]:
            return j
    return K_DISTANCE - 1


# ---------------------------------------------------------------------------
# Probability vector validation and NLL
# ---------------------------------------------------------------------------

PROB_SUM_TOLERANCE: Final[float] = 1e-4
"""A submitted distribution must satisfy ``|sum(p) - 1| < PROB_SUM_TOLERANCE``."""

NLL_EPSILON: Final[float] = 1e-12
"""Underflow floor applied to the true bin only, via ``max(p, NLL_EPSILON)``."""

DUPLICATE_RECORD_POLICY: Final[Literal["reject_file"]] = "reject_file"
"""A repeated ``case_id``, or a repeated ``(case_id, eval_time_sec)``, rejects
the **whole file**.

[v1.7 gap] The proposal named an ``R-DUP`` rule without saying whether a
duplicate is refused or resolved.  "Last one wins" and "first one wins" give
different scores for the same file, and a submission written by a crashed-and-
resumed inference loop is exactly the case that hits this.  Refusing the file
is the only option that cannot silently pick a record for the participant.
"""

NORMALISE_SUBMITTED_PROBS: Final[bool] = False
"""The evaluator does **not** rescale submitted distributions.

[v1.7 gap] The proposal said both "validate the sum, then normalise" and "the
evaluator does not alter submitted distributions", which contradict each other.
Resolved toward rejection: a vector that fails validation raises ``ValueError``
instead of being silently repaired, so a participant can never discover that
their malformed submission was quietly fixed differently than they assumed.
"""


# ---------------------------------------------------------------------------
# Missing-prediction penalty
# ---------------------------------------------------------------------------

PENALTY_AZIMUTH_DEG: Final[float] = 180.0
"""Maximum possible circular error, so omission is never better than a guess."""

PENALTY_DISTANCE_M: Final[float] = 5.0
"""At or above the worst achievable error in the range (4.5 m)."""

PENALTY_SPEAKER_CORRECT: Final[bool] = False

PENALTY_NLL: Final[float] = -math.log(NLL_EPSILON)
"""~27.631 nats -- the worst value the clipping floor allows.

[v1.7 gap] The proposal charged a *uniform* distribution's NLL
(``log 36 = 3.5835``) for a missing prediction.  A confidently wrong submitted
distribution scores worse than that, so omitting the time steps you are unsure
about was strictly profitable.  Charging the clipping floor instead closes the
hole: no submission can ever score worse than a missing one.
"""


AZIMUTH_SUPPORT_WIDTH_DEG: Final[float] = float(K_AZIMUTH * AZIMUTH_BIN_WIDTH_DEG)
"""Total azimuth support, 360 deg -- the widest possible HPD set."""

DISTANCE_SUPPORT_WIDTH_M: Final[float] = math.fsum(DISTANCE_BIN_WIDTHS_M)
"""Total distance support, 4.5 m -- the widest possible HPD set."""

PENALTY_POS_ERROR_2D_M: Final[float] = 2.0 * DISTANCE_RANGE_M[1]
"""10.0 m, charged to ``pos_error_2d_m`` for a missing prediction.

[v1.7 gap] The proposal fixed an azimuth and a distance penalty but none for the
joint 2-D error, which is not recoverable from the other two: the error for
(180 deg, 5 m) against a truth at 5 m is 10 m, not 5 m.  Two points inside the
evaluated annulus are at most ``2 * max distance`` apart, so 10 m matches the
"at or above the worst achievable error" rule the other penalties follow.
"""

PENALTY_COVERAGE: Final[float] = 0.0
"""A missing prediction covers nothing, mirroring
:data:`PENALTY_SPEAKER_CORRECT`. [v1.7 gap]"""

PENALTY_AZIMUTH_INTERVAL_WIDTH_DEG: Final[float] = AZIMUTH_SUPPORT_WIDTH_DEG
"""A missing prediction is charged the whole support, 360 deg.

[v1.7 gap] Interval Width had no missing-value rule.  Charging 0 would make
omission the cheapest possible interval, which inverts the metric: a submission
could win the width column by answering nothing.
"""

PENALTY_DISTANCE_INTERVAL_WIDTH_M: Final[float] = DISTANCE_SUPPORT_WIDTH_M
"""A missing prediction is charged the whole support, 4.5 m. [v1.7 gap]"""


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

SPATIAL_AGGREGATION: Final[Literal["macro_per_case"]] = "macro_per_case"
"""Spatial and speaker metrics: every ``t_k`` in ``T_eval`` is weighted
``1 / |T_eval|`` within a case, then every case is weighted equally.

[v1.7 gap] The proposal fixed the weighting *within* a case and said nothing
about combining cases, so two conforming evaluators would print different
leaderboard numbers for the same submission file.
"""

WER_AGGREGATION: Final[Literal["micro_global"]] = "micro_global"
"""``WER`` and ``cpWER``: total edit distance over total reference words across
all cases, not the mean of per-case rates."""

MISSING_TRANSCRIPT_POLICY: Final[Literal["count_all_deletions"]] = "count_all_deletions"
"""A case with no ``final_transcripts`` record scores every reference word as a
deletion. The case is never dropped, which would otherwise reward omission."""


MULTI_SPEAKER_WER_CONCAT_ORDER: Final[Literal["ascending_label"]] = "ascending_label"
"""How the speaker-agnostic ``WER`` stream is built from a multi-slot mapping.

Slots are concatenated in ascending speaker-label order (Python string sort of
the mapping keys), both for the hypothesis and for the reference.

[v1.7 gap] ``WER`` is defined as speaker agnostic, but with two slots the token
stream depends on concatenation order and so does the edit distance.  Track 1
has one slot, so the choice is first visible in Track 2.  Ascending label is
chosen over chronological order because a submission need not report any
timing for ``final_transcripts``.
"""


# ---------------------------------------------------------------------------
# Speaker permutation
# ---------------------------------------------------------------------------

SPEAKER_PAD_TOKEN: Final[str] = ""
"""Both sides are padded to ``max(len(pred), len(ref))`` with empty transcripts
before the optimal one-to-one assignment is searched.

[v1.7 gap] The proposal required a bijection between predicted and reference
speaker ids, which is undefined as soon as an online tracker over- or
under-segments -- the most common failure mode there is.  Padding makes the
assignment always well defined: an unmatched predicted speaker's words all
count as insertions, an unmatched reference speaker's words all as deletions.
"""

MAX_SPEAKERS_PER_TRACK: Final[dict[int, int]] = {1: 1, 2: 2}
"""Schema validation rejects a ``final_transcripts`` mapping with more keys than
the track allows, so a model cannot scatter guesses across many slots."""

MAX_PERMUTATION_SLOTS: Final[int] = 8
"""Hard cap on ``max(len(predicted slots), len(reference slots))`` per case.

Beyond it the exhaustive search over assignments is refused rather than
approximated, so a reported ``cpWER`` is always the true optimum and never a
greedy stand-in.

[v1.7 gap] :data:`MAX_SPEAKERS_PER_TRACK` caps what a *track* allows, but the
padded assignment is sized by ``max(pred, ref)``, so an over-segmenting tracker
can present more slots than its track permits.  Without a cap the search is
``O(n!)``: 8 slots is 40320 assignments, 12 would be 479 million.  The cap is a
rejection a participant can hit, so it belongs in the contract rather than in
the evaluator.
"""

PERMUTATION_TIE_BREAK: Final[tuple[str, str, str]] = (
    "min_cpwer",
    "max_speaker_match_count",
    "lexicographic",
)
"""The three-stage deterministic rule from the proposal, kept as is."""


PERMUTATION_LEXICOGRAPHIC_KEY: Final[
    Literal["assigned_reference_slot_indices"]
] = "assigned_reference_slot_indices"
"""What stage 3 of :data:`PERMUTATION_TIE_BREAK` actually orders.

The winning candidate is the one with the smallest tuple of assigned
reference-slot indices, read over the submitted labels in ascending order, with
padded slots last.

[v1.7 gap] The proposal said "lexicographic" without naming the sequence being
ordered.  Over labels, over reference indices and over per-slot WERs are three
different rules that pick three different permutations on a genuine tie.
"""


# ---------------------------------------------------------------------------
# Text normalisation for WER / cpWER
# ---------------------------------------------------------------------------

TEXT_NORMALISATION: Final[tuple[str, ...]] = (
    "unicode_nfkc",
    "casefold",
    "strip_punctuation",
    "digits_to_words",
    "collapse_whitespace",
)
"""The ordered normalisation pipeline applied to both hypothesis and reference.

[v1.7 gap] The proposal scored ``WER`` without defining normalisation, so
casing and punctuation alone could move the number by several points.
"""


PUNCTUATION_TO_SPACE_CATEGORIES: Final[tuple[str, ...]] = ("Pd", "Pc")
"""Unicode categories replaced by a space during ``strip_punctuation``:
dashes and connectors.  ``well-known`` therefore becomes two tokens."""

PUNCTUATION_TO_SPACE_CHARS: Final[str] = "/"
"""Individual characters replaced by a space, regardless of category.
``and/or`` becomes two tokens."""

PUNCTUATION_DROP_RULE: Final[Literal["drop_remaining_p_categories"]] = (
    "drop_remaining_p_categories"
)
"""Every other character whose Unicode category starts with ``P`` is deleted
without leaving a space, so ``don't`` becomes the single token ``dont`` and
``U.S.`` becomes ``us``.

[v1.7 gap] ``strip_punctuation`` was a pipeline stage name with no character
rules.  Delete-or-split is not a cosmetic choice: ``well-known`` is one token
under deletion and two under splitting, and the two readings give different
word counts for the same reference, hence different WER denominators.
"""

DIGITS_TO_WORDS_MAX_VALUE: Final[int] = 999
"""Largest integer spelled as a number.  Above it, and for any run with a
significant leading zero, the run is read digit by digit.

[v1.7 gap] ``digits_to_words`` was named but never specified.  ``1999`` as
"one thousand nine hundred ninety nine" and as "one nine nine nine" differ by
four tokens on one word, and no scale convention above 999 is uncontroversial
(short scale, Indian grouping, year reading).  Reading long runs digit by digit
needs no convention at all.  ``007`` is an identifier rather than seven, so a
leading zero forces the same digit-by-digit reading.
"""

NUMBER_WORDS_LANGUAGE: Final[Literal["en"]] = "en"
"""The number words are English, matching the LibriSpeech references."""


# ---------------------------------------------------------------------------
# Uncertainty interval construction
# ---------------------------------------------------------------------------

COVERAGE_LEVEL: Final[float] = 0.90

INTERVAL_RULE: Final[Literal["hpd_greedy"]] = "hpd_greedy"
"""Highest-posterior-density set, built greedily.

Sort bins by probability descending, breaking ties toward the smaller index;
accumulate until the cumulative mass is ``>= COVERAGE_LEVEL``; the accumulated
bins form the set ``S``.  ``Coverage@90%`` is ``1`` when the true bin is in
``S``, ``Interval Width`` is the summed width of the bins in ``S`` (degrees for
azimuth, meters for distance).  ``S`` need not be contiguous, which is the whole
point for the bimodal ``M = 2`` case.

[v1.7 gap] ``Coverage@90%`` and ``Interval Width`` were listed as metrics with
no construction rule at all.  A quantile interval would be meaningless on a
circular variable and would hide exactly the bimodality this benchmark is
about.
"""


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------
#
# [v1.7 gap] This section is the one the proposal left emptiest.  It fixed the
# *semantics* of every output -- distributions, point estimates, transcripts --
# and none of the names they travel under.  The consequence was not theoretical:
# the generator, the two baselines and the evaluator each invented their own
# vocabulary, the evaluator grew an alias table to bridge them, and for a while
# it read none of the records the baselines wrote.  Everything below is
# therefore part of the contract, not an implementation detail.

RECORD_TYPE_TIMEPOINT: Final[str] = "timepoint"
"""One instant on the inference grid. At most one per ``(case_id, eval_time_sec)``."""

RECORD_TYPE_FINAL_TRANSCRIPTS: Final[str] = "final_transcripts"
"""The whole-case transcript mapping. At most one per ``case_id``."""

RECORD_TYPES: Final[tuple[str, ...]] = (
    RECORD_TYPE_TIMEPOINT,
    RECORD_TYPE_FINAL_TRANSCRIPTS,
)
"""The two canonical ``record_type`` values."""

RECORD_TYPE_ALIASES: Final[dict[str, str]] = {
    "timestep": RECORD_TYPE_TIMEPOINT,
    "final_transcript": RECORD_TYPE_FINAL_TRANSCRIPTS,
}
"""Deprecated spellings that are still read.

``timestep`` and ``final_transcript`` are what the baselines wrote before this
section existed.  They stay readable so that already-generated submission files
keep scoring, but they are not produced any more.  An unrecognised value is a
hard rejection, not a third alias.
"""


def canonical_record_type(value: str) -> str:
    """Map a declared ``record_type`` to its canonical spelling.

    >>> canonical_record_type("timestep")
    'timepoint'
    >>> canonical_record_type("final_transcripts")
    'final_transcripts'
    >>> canonical_record_type("frame")
    Traceback (most recent call last):
        ...
    ValueError: unknown record_type: 'frame'
    """
    if value in RECORD_TYPES:
        return value
    if value in RECORD_TYPE_ALIASES:
        return RECORD_TYPE_ALIASES[value]
    raise ValueError(f"unknown record_type: {value!r}")


# --- canonical field names -------------------------------------------------

F_RECORD_TYPE: Final[str] = "record_type"
F_CASE_ID: Final[str] = "case_id"
F_EVAL_TIME: Final[str] = "eval_time_sec"
F_OBSERVATION_MODE: Final[str] = "observation_mode"
F_RTF: Final[str] = "rtf"
F_FINAL_TRANSCRIPTS: Final[str] = "final_transcripts"

F_AZIMUTH_DEG: Final[str] = "azimuth_deg"
F_DISTANCE_M: Final[str] = "distance_m"
F_X_M: Final[str] = "x_m"
F_Y_M: Final[str] = "y_m"
F_SPEAKER: Final[str] = "speaker"

F_PRED_AZIMUTH_DEG: Final[str] = "pred_azimuth_deg"
F_PRED_DISTANCE_M: Final[str] = "pred_distance_m"
F_PRED_X_M: Final[str] = "pred_x_m"
F_PRED_Y_M: Final[str] = "pred_y_m"
F_PRED_SPEAKER: Final[str] = "pred_speaker"
F_AZIMUTH_PROBS: Final[str] = "azimuth_probs"
F_DISTANCE_PROBS: Final[str] = "distance_probs"

F_AZIMUTH_BIN: Final[str] = "azimuth_bin"
F_DISTANCE_BIN: Final[str] = "distance_bin"
F_FRAMES: Final[str] = "frames"
F_TRANSCRIPTS_REFERENCE: Final[str] = "final_transcripts_reference"
F_SESSION_ID: Final[str] = "session_id"
F_MASK_BITS: Final[str] = "mask_bits"

FIELD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    F_SPEAKER: (F_SPEAKER, "speaker_id"),
    F_PRED_SPEAKER: (F_PRED_SPEAKER, "speaker_id", F_SPEAKER),
    F_RTF: (F_RTF, "reported_rtf"),
    F_FINAL_TRANSCRIPTS: (F_FINAL_TRANSCRIPTS, F_TRANSCRIPTS_REFERENCE),
}
"""Accepted spellings per canonical field, in resolution order.

The order is part of the contract: the first spelling present in a record wins,
and a record carrying two spellings of one field is resolved, not rejected.
``speaker_id`` is the generator's and both baselines' name for the label.
"""


def accepted_field_names(canonical: str) -> tuple[str, ...]:
    """Return every spelling accepted for ``canonical``, in resolution order.

    >>> accepted_field_names("rtf")
    ('rtf', 'reported_rtf')
    >>> accepted_field_names("case_id")
    ('case_id',)
    """
    return FIELD_ALIASES.get(canonical, (canonical,))


def present_field_name(record: Mapping[str, Any], canonical: str) -> str | None:
    """Return the spelling ``record`` actually uses for ``canonical``, or ``None``.

    Lets a caller tell "absent" from "present and null", which matters because
    an explicit ``null`` point estimate is a valid abstention.

    >>> present_field_name({"reported_rtf": 0.4}, "rtf")
    'reported_rtf'
    >>> present_field_name({"rtf": None}, "rtf")
    'rtf'
    >>> present_field_name({}, "rtf") is None
    True
    """
    for name in accepted_field_names(canonical):
        if name in record:
            return name
    return None


def resolve_field(
    record: Mapping[str, Any], canonical: str, default: Any = None
) -> Any:
    """Return ``record``'s value for ``canonical`` under any accepted spelling.

    >>> resolve_field({"speaker_id": "spk_0"}, "speaker")
    'spk_0'
    >>> resolve_field({}, "speaker", default="?")
    '?'
    """
    name = present_field_name(record, canonical)
    return default if name is None else record[name]


# --- required and optional fields -----------------------------------------

REQUIRED_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    RECORD_TYPE_TIMEPOINT: (F_CASE_ID, F_EVAL_TIME),
    RECORD_TYPE_FINAL_TRANSCRIPTS: (F_CASE_ID, F_FINAL_TRANSCRIPTS),
}
"""Fields every record of that type must carry, on either side."""

POSITION_FIELD_GROUPS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    "ground_truth": ((F_AZIMUTH_DEG, F_DISTANCE_M), (F_X_M, F_Y_M)),
    "prediction": (
        (F_PRED_AZIMUTH_DEG, F_PRED_DISTANCE_M),
        (F_PRED_X_M, F_PRED_Y_M),
    ),
}
"""A position may be given as polar or as cartesian; the evaluator converts.

[v1.7 gap] The proposal's I/O table listed azimuth and distance only, while its
2-D error metric is cartesian.  Accepting either and converting removes a
rounding asymmetry: a model that regresses ``(x, y)`` directly would otherwise
have to quantise to polar before submitting.
"""

OPTIONAL_PREDICTION_FIELDS: Final[tuple[str, ...]] = (
    F_PRED_SPEAKER,
    F_AZIMUTH_PROBS,
    F_DISTANCE_PROBS,
    F_OBSERVATION_MODE,
    F_RTF,
)
"""Prediction fields that may be omitted.

Omitting a distribution costs the distribution metrics for that instant
(:data:`PENALTY_NLL`), except when the whole submission is point-only -- see
:data:`MISSING_PROBS_REASON`.
"""

CASE_ROW_FRAME_FIELDS: Final[tuple[str, ...]] = (
    F_EVAL_TIME,
    F_AZIMUTH_DEG,
    F_DISTANCE_M,
    F_AZIMUTH_BIN,
    F_DISTANCE_BIN,
)
"""Fields of one entry in a case-row ground truth's :data:`F_FRAMES` list.

Ground truth may be written either as one ``timepoint`` record per instant or
as one row per ``case_id`` with the instants nested under ``frames`` -- the
shape ``make_eval_dev.py`` produces.  A declared ``azimuth_bin`` /
``distance_bin`` that disagrees with this module's own binning of the
continuous label is a rejection, so the two can never drift apart silently.
"""

PROB_FIELD_LENGTHS: Final[dict[str, int]] = {
    F_AZIMUTH_PROBS: K_AZIMUTH,
    F_DISTANCE_PROBS: K_DISTANCE,
}
"""Required length of each submitted distribution vector."""


# --- speaker labels --------------------------------------------------------

SPEAKER_LABELS_ARE_OPAQUE: Final[bool] = True
"""Submitted speaker labels carry no meaning beyond the partition they induce.

Every speaker metric is computed after the optimal assignment of
:data:`PERMUTATION_TIE_BREAK`, so a submission may name its slots anything.  The
ground truth's labels are source identities and a submission is not expected to
guess them -- that is what "anonymous speaker id" in the proposal means, and it
is why ``speaker_attribution_accuracy`` compares mapped labels rather than raw
ones.
"""

SPEAKER_SLOT_FORMAT: Final[str] = "spk_{index}"
"""Recommended slot label format, zero-based. Only a convention: see
:data:`SPEAKER_LABELS_ARE_OPAQUE`."""

PAD_LABEL_PREFIX: Final[str] = "__pad__"
"""Prefix of the synthetic slots used for :data:`SPEAKER_PAD_TOKEN` padding.

A submitted or reference label starting with it is rejected, so a padded slot
can never be mistaken for a real one.
"""


def speaker_slot(index: int) -> str:
    """Return the conventional label for slot ``index``.

    >>> speaker_slot(0), speaker_slot(1)
    ('spk_0', 'spk_1')
    """
    if index < 0:
        raise ValueError(f"speaker slot index must be >= 0, got {index}")
    return SPEAKER_SLOT_FORMAT.format(index=index)


# --- keying ----------------------------------------------------------------

TIME_KEY_DECIMALS: Final[int] = 6
"""Decimals used when ``eval_time_sec`` is rounded into a dictionary key.

:data:`DELTA_T_SEC` is exactly representable, so grid instants round-trip
unchanged; the rounding only makes off-grid instants and duplicate detection
robust against a submission written by a float-formatting JSON writer.
"""


# ---------------------------------------------------------------------------
# Official reporting scope
# ---------------------------------------------------------------------------

OFFICIAL_SCOPES: Final[tuple[str, ...]] = ("causal",)
"""Only ``causal`` is official for the minimum viable submission.

[v1.7 gap] ``Windowed`` has no fixed ``W``, so submissions with different
lookahead would be ranked in one group -- which defeats the separation the
proposal asked for.  ``windowed`` and ``full_audio`` are accepted as submission
values and reported in separate tables, but they are not part of the official
comparison until ``W`` is fixed.
"""

OBSERVATION_MODES: Final[tuple[str, ...]] = ("causal", "windowed", "full_audio")

OBSERVATION_MODE_IS_RECORD_FIELD: Final[bool] = True
"""The scope is a field of each record, not a switch on the scoring run.

A submitter *writes* ``observation_mode`` into every record; the evaluator
*reads* it and splits the official table from the rest.  The evaluator
therefore takes no scope argument.

[v1.7 gap] Producer and consumer have opposite roles here and the proposal
never said so.  The repository's own Makefile got it wrong once, passing
``--scope`` to the scorer, which has no such flag.
"""

CAUSAL_WINDOW_RULE: Final[Literal["chunk_ends_at_t_k"]] = "chunk_ends_at_t_k"
"""At ``t_k`` a causal system may read
``[max(0, t_k - CHUNK_LENGTH_SEC), t_k]`` and no sample whose index exceeds
``round(t_k * SAMPLE_RATE_HZ)``.

[v1.7 gap] The proposal fixed the chunk length and hop but not where a chunk
sits relative to the grid point it answers.  A chunk *centred* on ``t_k`` reads
half a chunk of future audio while still calling itself causal, which is
exactly the leakage the causal/windowed split exists to prevent.  This is also
the only form in which causality is testable: the unit test feeds a session
whose tail is replaced by noise and asserts that no prediction at or before
the splice changes.
"""

WINDOWED_REQUIRES_LOOKAHEAD: Final[bool] = True
"""A record declaring ``observation_mode == "windowed"`` must also report its
lookahead ``W`` in seconds, so the separate table can be split by ``W``.
Without it a windowed submission is unrankable even against other windowed
submissions. [v1.7 gap]
"""

OFFICIAL_METRICS: Final[tuple[str, ...]] = (
    "wer",
    "cpwer",
    "speaker_attribution_accuracy",
    "azimuth_mae_deg",
    "azimuth_acc_10deg",
    "azimuth_acc_20deg",
    "distance_mae_m",
    "pos_error_2d_m",
    "azimuth_nll",
    "distance_nll",
    "azimuth_coverage_90",
    "azimuth_interval_width_deg",
    "distance_coverage_90",
    "distance_interval_width_m",
    "rtf",
)
"""Metrics with a complete computation rule.

Dropped from the official set because the proposal never defined them:
``Adaptation Lag`` (no reference instant, no recovery criterion, no value for a
turn that never recovers), ``relative distance error`` (no denominator) and
``CER`` (no character unit for a mixed-script reference).  They may return as
diagnostics once defined.
"""

RTF_DEFINITION: Final[Literal["wall_clock_per_audio_second"]] = (
    "wall_clock_per_audio_second"
)
"""``rtf = wall-clock seconds spent on a case / audio seconds of that case``.

Measured per case and aggregated with :data:`SPATIAL_AGGREGATION`.  Manifest
discovery, file I/O and model loading are excluded; only inference counts.

[v1.7 gap] ``RTF`` was an official metric with no definition.  Per-case versus
whole-run, and whether model loading counts, change the number by more than
any modelling choice does.  It stays self-reported unless a submission is
rerun in the official environment -- see the proposal's "Verified-Run" label.
"""

MISSING_PROBS_REASON: Final[str] = "N/A (point-only submission)"
"""Reported in place of every distribution metric when a submission carries no
``azimuth_probs``/``distance_probs`` at all.

A point-only submission is valid and is ranked on the point metrics.  The
distribution columns are ``N/A`` by rule rather than scored with
:data:`PENALTY_NLL`, which would otherwise make "submit no distributions" look
like the worst possible calibration instead of an abstention. [v1.7 gap]
"""


TRACK_UNDEFINED_METRICS: Final[dict[int, tuple[str, ...]]] = {
    1: ("cpwer",),
    2: (),
}
"""Metrics that are ``N/A`` by rule rather than unmeasured.

``cpWER`` needs a speaker permutation to be meaningful, and Track 1 has a
single stationary speaker.
"""


# ---------------------------------------------------------------------------
# Rule identifiers
# ---------------------------------------------------------------------------

REJECTION_RULES: Final[dict[str, str]] = {
    "R-PARSE": "a record is unreadable, mistyped or missing a required field",
    "R-DUP-01": "two records declare the same case_id",
    "R-DUP-02": "two records declare the same (case_id, eval_time_sec)",
    "R-PROB-01": "a distribution has the wrong length",
    "R-PROB-02": "a distribution does not sum to 1 within PROB_SUM_TOLERANCE",
    "R-PROB-03": "a distribution holds a value outside [0, 1]",
    "R-SPK-01": "more speaker slots than the track or the permutation cap allows",
}
"""Ids a rejection message must start with, so a participant can look the cause
up instead of reading the evaluator source.

[v1.7 gap] The proposal summarised rule families in prose and the evaluator
invented its own ids.  Two places owning the same identifier set is how a
message and its documentation drift apart.
"""

SCORING_RULES: Final[dict[str, str]] = {
    "R-DUP": "duplicates reject the file (DUPLICATE_RECORD_POLICY)",
    "R-MISS": "a missing prediction draws the PENALTY_* values, never a drop",
    "R-PERM": "speaker metrics are scored after PERMUTATION_TIE_BREAK",
    "R-NLL": "no smoothing; only the true bin is clipped at NLL_EPSILON",
    "R-PT": "an explicit point wins; otherwise the MAP bin center is used",
    "R-TXT": "WER reads final_transcripts only, after TEXT_NORMALISATION",
}
"""The scoring rules the proposal named, each pointing at the constant that
realises it.  Unlike :data:`REJECTION_RULES` these never abort a run."""


# ---------------------------------------------------------------------------
# Internal consistency
# ---------------------------------------------------------------------------


def _self_check() -> None:
    """Fail at import time if two declarations in this file contradict.

    Each check covers a pair that is easy to edit apart and whose disagreement
    would otherwise surface as a wrong score rather than an error.

    >>> _self_check()
    """
    named = tuple(
        float(m.group(1))
        for m in (re.fullmatch(r"azimuth_acc_(\d+)deg", name) for name in OFFICIAL_METRICS)
        if m
    )
    if named != AZIMUTH_ACC_THRESHOLDS_DEG:
        raise RuntimeError(
            f"AZIMUTH_ACC_THRESHOLDS_DEG={AZIMUTH_ACC_THRESHOLDS_DEG} does not match "
            f"the thresholds named in OFFICIAL_METRICS ({named})"
        )
    if AZIMUTH_ACC_THRESHOLDS_DEG[0] != AZIMUTH_BIN_WIDTH_DEG:
        raise RuntimeError("the tightest azimuth accuracy threshold must be one bin wide")

    percent = f"{round(COVERAGE_LEVEL * 100)}"
    for name in OFFICIAL_METRICS:
        if "coverage" in name and not name.endswith(f"_{percent}"):
            raise RuntimeError(f"COVERAGE_LEVEL={COVERAGE_LEVEL} does not match metric {name!r}")

    if set(PROB_FIELD_LENGTHS.values()) != {K_AZIMUTH, K_DISTANCE}:
        raise RuntimeError("PROB_FIELD_LENGTHS disagrees with the bin counts")

    if PENALTY_NLL <= -math.log(1.0 / max(K_AZIMUTH, K_DISTANCE)):
        raise RuntimeError("PENALTY_NLL must exceed a uniform distribution's NLL")

    if set(RECORD_TYPE_ALIASES.values()) - set(RECORD_TYPES):
        raise RuntimeError("an alias points at a record type that does not exist")
    if set(RECORD_TYPE_ALIASES) & set(RECORD_TYPES):
        raise RuntimeError("a canonical record type is also listed as an alias")

    for canonical, spellings in FIELD_ALIASES.items():
        if spellings[0] != canonical:
            raise RuntimeError(f"FIELD_ALIASES[{canonical!r}] must resolve itself first")
        if len(set(spellings)) != len(spellings):
            raise RuntimeError(f"FIELD_ALIASES[{canonical!r}] repeats a spelling")

    if set(REQUIRED_FIELDS) != set(RECORD_TYPES):
        raise RuntimeError("REQUIRED_FIELDS must cover exactly the canonical record types")

    for mask in EVAL_MASKS:
        mask_to_channels(mask)

    if not DISTANCE_BIN_EDGES_M[0] == DISTANCE_RANGE_M[0] or not DISTANCE_BIN_EDGES_M[-1] == DISTANCE_RANGE_M[1]:
        raise RuntimeError("DISTANCE_BIN_EDGES_M must span DISTANCE_RANGE_M exactly")

    if N_GRID_POINTS * DELTA_T_SEC != SESSION_LENGTH_SEC:
        raise RuntimeError("SESSION_LENGTH_SEC is not a whole number of grid steps")
    if N_CHANNELS_MAX != len(EVAL_MASKS[0]):
        raise RuntimeError("EVAL_MASKS width disagrees with N_CHANNELS_MAX")
    if 2.0 * ARRAY_RADIUS_RANGE_M[1] != MAX_PAIR_SPACING_M:
        raise RuntimeError("MAX_PAIR_SPACING_M is not the diameter at the maximum radius")
    if VAD_REL_THRESHOLD_DB >= 0.0:
        raise RuntimeError("VAD_REL_THRESHOLD_DB is an offset below the reference, so it must be negative")
    if not 0.0 < VAD_REF_PERCENTILE <= 100.0:
        raise RuntimeError("VAD_REF_PERCENTILE must be a percentile")
    if set(REJECTION_RULES) & set(SCORING_RULES):
        raise RuntimeError("a rule id is listed as both a rejection and a scoring rule")
    for rule_id in (*REJECTION_RULES, *SCORING_RULES):
        if not rule_id.startswith("R-"):
            raise RuntimeError(f"malformed rule id: {rule_id!r}")
    if ALIASING_LIMIT_HZ >= SAMPLE_RATE_HZ / 2:
        raise RuntimeError("the aliasing limit must sit below Nyquist to be observable")


_self_check()



if __name__ == "__main__":
    import doctest

    failures, _ = doctest.testmod()
    if failures:
        raise SystemExit(1)
    print(f"spec {SPEC_VERSION}")
    print(f"  azimuth : K={K_AZIMUTH} width={AZIMUTH_BIN_WIDTH_DEG}deg")
    print(f"  distance: K={K_DISTANCE} edges={DISTANCE_BIN_EDGES_M}")
    print(f"  centers : {tuple(round(c, 4) for c in DISTANCE_BIN_CENTERS_M)}")
    print(f"  aliasing: {ALIASING_LIMIT_HZ:.1f} Hz")
    print(f"  penalty : az={PENALTY_AZIMUTH_DEG} r={PENALTY_DISTANCE_M} nll={PENALTY_NLL:.4f}")
    print(f"  masks   : {EVAL_MASKS}")
    print(f"  audio   : {SAMPLE_RATE_HZ} Hz {AUDIO_ENCODING}, max TDOA {MAX_TDOA_SAMPLES:.2f} samples")
    print(f"  records : {RECORD_TYPES} (+{len(RECORD_TYPE_ALIASES)} deprecated aliases)")
    print(f"  support : az={AZIMUTH_SUPPORT_WIDTH_DEG}deg r={DISTANCE_SUPPORT_WIDTH_M}m")
    print(f"  2d pen  : {PENALTY_POS_ERROR_2D_M} m")
    print(f"  grid    : {N_GRID_POINTS} instants, endpoint {GRID_ENDPOINT}")
    print(f"  azimuth0: {AZIMUTH_CONVENTION}, array {DEFAULT_ARRAY_GEOMETRY}")
