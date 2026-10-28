"""Official scorer for the Variable-Channel Spatial ASR benchmark.

Standard library only, with an *optional* numpy fast path.  No ``jiwer``, no
``torch``, no ``pandas``: the whole point of this benchmark is that anybody can
reproduce a number bit-for-bit without an account, a GPU or a network, so the
edit distance and the speaker permutation search are implemented here.

Every rule and every constant comes from :mod:`spec`.  Nothing in this file
re-declares a value that ``spec.py`` already fixes.

Submission format (JSON Lines, one object per line)
---------------------------------------------------
Two record types, distinguished by ``record_type`` (inferred when absent):

``record_type == "final_transcripts"`` -- at most one per ``case_id``::

    {"record_type": "final_transcripts",
     "case_id": "sess_001__m1010",
     "final_transcripts": {"spk0": "the quick brown fox"},
     "observation_mode": "causal",      # prediction files only, default causal
     "rtf": 0.42}                       # prediction files only, optional

``record_type == "timepoint"`` -- at most one per ``(case_id, eval_time_sec)``.
Ground truth::

    {"record_type": "timepoint", "case_id": "...", "eval_time_sec": 1.25,
     "azimuth_deg": 37.0, "distance_m": 1.6, "speaker": "spk0"}

Prediction::

    {"record_type": "timepoint", "case_id": "...", "eval_time_sec": 1.25,
     "pred_azimuth_deg": 35.0, "pred_distance_m": 1.63,
     "pred_x_m": null, "pred_y_m": null,
     "azimuth_probs": [...36 floats...], "distance_probs": [...8 floats...],
     "pred_speaker": "spk0"}

Either side may give a position as polar (``azimuth_deg``/``distance_m``,
``pred_azimuth_deg``/``pred_distance_m``) or as cartesian (``x_m``/``y_m``,
``pred_x_m``/``pred_y_m``); the two are interconverted.

Accepted aliases
----------------
Every name, every deprecated spelling and every resolution order below comes
from :data:`spec.RECORD_TYPE_ALIASES` and :data:`spec.FIELD_ALIASES`.  This file
declares none of them.  Nothing is relaxed: an unrecognised name is ``R-PARSE``.

* ``record_type``: ``"timestep"`` reads as ``"timepoint"`` and
  ``"final_transcript"`` as ``"final_transcripts"`` -- the spellings the
  baselines wrote before the contract owned a record schema.
* Speaker label: ``pred_speaker``, then ``speaker_id``, then ``speaker``.
* ``rtf``: ``rtf``, then ``reported_rtf``.
* Ground truth may instead carry **one row per** ``case_id`` -- the shape
  ``make_eval_dev.py`` writes -- with the reference transcript under
  ``final_transcripts_reference``, the label under ``speaker_id`` and the
  instants nested in a ``frames`` list of
  ``{eval_time_sec, azimuth_deg, distance_m, azimuth_bin, distance_bin}``.
  A declared ``azimuth_bin``/``distance_bin`` that disagrees with ``spec``'s
  own binning of the continuous label is rejected.

Rejection rules
---------------
Every rejection raises :class:`SubmissionError` (a ``ValueError``) whose message
starts with a rule id: ``R-PARSE``, ``R-DUP-01``, ``R-DUP-02``, ``R-PROB-01``,
``R-PROB-02``, ``R-PROB-03``, ``R-SPK-01``.  A duplicate key rejects the whole
file rather than picking a record arbitrarily.

CLI
---
``python evaluate.py --pred predictions.jsonl --gt ground_truth.jsonl
--track 1 [--out scores.json]``
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Final

_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:  # make ``import spec`` work from anywhere
    sys.path.insert(0, str(_MODULE_DIR))

import spec  # noqa: E402  (deliberately after the sys.path bootstrap)

try:  # optional fast path only; every code path has a pure-Python fallback
    import numpy as _np
except ImportError:  # pragma: no cover - exercised on numpy-less machines
    _np = None

HAS_NUMPY: Final[bool] = _np is not None

EVALUATOR_VERSION: Final[str] = "0.1.0"
"""Version of this scorer. Independent of :data:`spec.SPEC_VERSION`."""


# ---------------------------------------------------------------------------
# Rejection rule ids
# ---------------------------------------------------------------------------

def _rule(rule_id: str) -> str:
    """Return ``rule_id`` after checking :mod:`spec` declares it.

    >>> _rule("R-PARSE")
    'R-PARSE'
    >>> _rule("R-NOPE")
    Traceback (most recent call last):
        ...
    RuntimeError: R-NOPE is not in spec.REJECTION_RULES
    """
    if rule_id not in spec.REJECTION_RULES:
        raise RuntimeError(f"{rule_id} is not in spec.REJECTION_RULES")
    return rule_id


RULE_PARSE: Final[str] = _rule("R-PARSE")
RULE_DUPLICATE_CASE: Final[str] = _rule("R-DUP-01")
RULE_DUPLICATE_TIME: Final[str] = _rule("R-DUP-02")
RULE_PROB_LENGTH: Final[str] = _rule("R-PROB-01")
RULE_PROB_SUM: Final[str] = _rule("R-PROB-02")
RULE_PROB_VALUE: Final[str] = _rule("R-PROB-03")
RULE_SPEAKER_COUNT: Final[str] = _rule("R-SPK-01")


class SubmissionError(ValueError):
    """A submission or ground-truth file violates the frozen contract.

    Subclasses ``ValueError`` so callers may catch either.  The message always
    begins with the rule id that was violated.

    >>> raise SubmissionError(f"{RULE_DUPLICATE_CASE}: duplicate case_id 'x'")
    Traceback (most recent call last):
        ...
    evaluate.SubmissionError: R-DUP-01: duplicate case_id 'x'
    """


def _reject(rule: str, message: str) -> None:
    """Raise :class:`SubmissionError` tagged with ``rule``.

    >>> _reject(RULE_PARSE, "nope")
    Traceback (most recent call last):
        ...
    evaluate.SubmissionError: R-PARSE: nope
    """
    raise SubmissionError(f"{rule}: {message}")


# ---------------------------------------------------------------------------
# Constants derived from the contract (never re-declared)
# ---------------------------------------------------------------------------

AZIMUTH_ACC_METRICS: Final[tuple[tuple[str, float], ...]] = tuple(
    (f"azimuth_acc_{int(threshold)}deg", threshold)
    for threshold in spec.AZIMUTH_ACC_THRESHOLDS_DEG
)
"""``(metric_name, threshold_deg)`` pairs built from
:data:`spec.AZIMUTH_ACC_THRESHOLDS_DEG`.

``spec._self_check`` asserts that these names are exactly the ones
:data:`spec.OFFICIAL_METRICS` lists, so the two cannot drift.

>>> AZIMUTH_ACC_METRICS
(('azimuth_acc_10deg', 10.0), ('azimuth_acc_20deg', 20.0))
"""

AZIMUTH_SUPPORT_WIDTH_DEG: Final[float] = spec.AZIMUTH_SUPPORT_WIDTH_DEG
DISTANCE_SUPPORT_WIDTH_M: Final[float] = spec.DISTANCE_SUPPORT_WIDTH_M
PENALTY_POS_ERROR_2D_M: Final[float] = spec.PENALTY_POS_ERROR_2D_M
PENALTY_COVERAGE: Final[float] = spec.PENALTY_COVERAGE
PENALTY_AZIMUTH_INTERVAL_WIDTH_DEG: Final[float] = spec.PENALTY_AZIMUTH_INTERVAL_WIDTH_DEG
PENALTY_DISTANCE_INTERVAL_WIDTH_M: Final[float] = spec.PENALTY_DISTANCE_INTERVAL_WIDTH_M
PAD_LABEL_PREFIX: Final[str] = spec.PAD_LABEL_PREFIX
MISSING_PROBS_REASON: Final[str] = spec.MISSING_PROBS_REASON
"""Re-exports of values :mod:`spec` owns.

They were declared here while ``spec.py`` fixed no penalty for the 2-D error,
the coverage metrics or the interval widths, and no padding prefix.  It does
now, so these are aliases kept only so that call sites and tests need not
change; the values live in one place.
"""

_TIME_DECIMALS: Final[int] = spec.TIME_KEY_DECIMALS
_MAX_PERMUTATION_SLOTS: Final[int] = spec.MAX_PERMUTATION_SLOTS

METRIC_WER: Final[str] = "wer"
METRIC_CPWER: Final[str] = "cpwer"
METRIC_SPEAKER_ACC: Final[str] = "speaker_attribution_accuracy"
METRIC_AZ_MAE: Final[str] = "azimuth_mae_deg"
METRIC_R_MAE: Final[str] = "distance_mae_m"
METRIC_POS2D: Final[str] = "pos_error_2d_m"
METRIC_AZ_NLL: Final[str] = "azimuth_nll"
METRIC_R_NLL: Final[str] = "distance_nll"
METRIC_AZ_COVERAGE: Final[str] = "azimuth_coverage_90"
METRIC_AZ_WIDTH: Final[str] = "azimuth_interval_width_deg"
METRIC_R_COVERAGE: Final[str] = "distance_coverage_90"
METRIC_R_WIDTH: Final[str] = "distance_interval_width_m"
METRIC_RTF: Final[str] = "rtf"

AZIMUTH_DISTRIBUTION_METRICS: Final[tuple[str, ...]] = (
    METRIC_AZ_NLL,
    METRIC_AZ_COVERAGE,
    METRIC_AZ_WIDTH,
)
DISTANCE_DISTRIBUTION_METRICS: Final[tuple[str, ...]] = (
    METRIC_R_NLL,
    METRIC_R_COVERAGE,
    METRIC_R_WIDTH,
)

MACRO_METRICS: Final[tuple[str, ...]] = (
    (METRIC_AZ_MAE,)
    + tuple(name for name, _ in AZIMUTH_ACC_METRICS)
    + (METRIC_R_MAE, METRIC_POS2D, METRIC_SPEAKER_ACC)
    + AZIMUTH_DISTRIBUTION_METRICS
    + DISTANCE_DISTRIBUTION_METRICS
    + (METRIC_RTF,)
)
"""Metrics aggregated with :data:`spec.SPATIAL_AGGREGATION`."""

MICRO_METRICS: Final[tuple[str, ...]] = (METRIC_WER, METRIC_CPWER)
"""Metrics aggregated with :data:`spec.WER_AGGREGATION`."""


def _check_contract_alignment() -> None:
    """Fail fast if this file cannot honour the contract :mod:`spec` declares.

    Only the checks ``spec._self_check`` cannot make are here: whether this
    implementation actually supports the rule variants the contract selected,
    and whether the metric set it produces is the official one.

    >>> _check_contract_alignment()
    """
    produced = set(MACRO_METRICS) | set(MICRO_METRICS)
    official = set(spec.OFFICIAL_METRICS)
    if produced != official:
        raise RuntimeError(
            "evaluate.py and spec.OFFICIAL_METRICS disagree: "
            f"missing={sorted(official - produced)} extra={sorted(produced - official)}"
        )
    if spec.INTERVAL_RULE != "hpd_greedy":
        raise RuntimeError(f"unsupported spec.INTERVAL_RULE: {spec.INTERVAL_RULE!r}")
    if spec.NORMALISE_SUBMITTED_PROBS:
        raise RuntimeError("this evaluator never rescales submitted distributions")
    if spec.MULTI_SPEAKER_WER_CONCAT_ORDER != "ascending_label":
        raise RuntimeError(
            f"unsupported concatenation order: {spec.MULTI_SPEAKER_WER_CONCAT_ORDER!r}"
        )
    if spec.PERMUTATION_LEXICOGRAPHIC_KEY != "assigned_reference_slot_indices":
        raise RuntimeError(
            f"unsupported tie-break key: {spec.PERMUTATION_LEXICOGRAPHIC_KEY!r}"
        )
    if spec.NUMBER_WORDS_LANGUAGE != "en":
        raise RuntimeError("the number-word table in this file is English only")
    if not spec.SPEAKER_LABELS_ARE_OPAQUE:
        raise RuntimeError("this evaluator always scores speakers through a permutation")
    if _MAX_PERMUTATION_SLOTS < max(spec.MAX_SPEAKERS_PER_TRACK.values()):
        raise RuntimeError("the permutation cap is below a track's own speaker limit")


_check_contract_alignment()


# ---------------------------------------------------------------------------
# Text normalisation (spec.TEXT_NORMALISATION, in order)
# ---------------------------------------------------------------------------

_SMALL_NUMBER_WORDS: Final[tuple[str, ...]] = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS_WORDS: Final[tuple[str, ...]] = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)
_DIGIT_RUN: Final[re.Pattern[str]] = re.compile(r"\d+")
_PUNCT_TO_SPACE_CATEGORIES: Final[frozenset[str]] = frozenset(
    spec.PUNCTUATION_TO_SPACE_CATEGORIES
)
_PUNCT_TO_SPACE_CHARS: Final[frozenset[str]] = frozenset(spec.PUNCTUATION_TO_SPACE_CHARS)


def integer_to_words(value: int) -> list[str]:
    """Spell ``value`` in English words up to :data:`spec.DIGITS_TO_WORDS_MAX_VALUE`.

    Above that bound the value is read digit by digit, per the contract: no
    scale convention above 999 is uncontroversial, and reading the digits needs
    none.

    >>> integer_to_words(0)
    ['zero']
    >>> integer_to_words(42)
    ['forty', 'two']
    >>> integer_to_words(105)
    ['one', 'hundred', 'five']
    >>> integer_to_words(900)
    ['nine', 'hundred']
    >>> integer_to_words(1234)
    ['one', 'two', 'three', 'four']
    """
    if value < 0:
        return ["minus", *integer_to_words(-value)]
    if value > spec.DIGITS_TO_WORDS_MAX_VALUE:
        return [_SMALL_NUMBER_WORDS[int(d)] for d in str(value)]
    if value < 20:
        return [_SMALL_NUMBER_WORDS[value]]
    if value < 100:
        tens, ones = divmod(value, 10)
        return [_TENS_WORDS[tens]] + ([_SMALL_NUMBER_WORDS[ones]] if ones else [])
    hundreds, rest = divmod(value, 100)
    words = [_SMALL_NUMBER_WORDS[hundreds], "hundred"]
    if rest:
        words.extend(integer_to_words(rest))
    return words


def _strip_punctuation(text: str) -> str:
    """Drop punctuation, turning dashes, connectors and ``/`` into spaces.

    Unicode categories ``Pd`` (dashes) and ``Pc`` (connectors) plus ``/`` become
    a space so compounds split into words; every other ``P*`` character is
    deleted so contractions stay single words.  Symbols (``S*``) are kept.

    >>> _strip_punctuation("don't -- mic/array_pair.")
    'dont    mic array pair'
    """
    out: list[str] = []
    for char in text:
        if char in _PUNCT_TO_SPACE_CHARS:
            out.append(" ")
            continue
        category = unicodedata.category(char)
        if category in _PUNCT_TO_SPACE_CATEGORIES:
            out.append(" ")
        elif category.startswith("P"):
            continue
        else:
            out.append(char)
    return "".join(out)


def _digits_to_words(text: str) -> str:
    """Replace every digit run by its spelled-out form.

    A run with a significant leading zero is read digit by digit, since ``007``
    is an identifier rather than the number seven.

    >>> _digits_to_words("channel 2 of 4")
    'channel  two  of  four '
    >>> _digits_to_words("007")
    ' zero zero seven '
    """

    def replace(match: re.Match[str]) -> str:
        digits = match.group()
        if len(digits) > 1 and digits[0] == "0":
            words = [_SMALL_NUMBER_WORDS[int(d)] for d in digits]
        else:
            words = integer_to_words(int(digits))
        return " " + " ".join(words) + " "

    return _DIGIT_RUN.sub(replace, text)


def normalise_text(text: str) -> str:
    """Apply :data:`spec.TEXT_NORMALISATION` in order and return one string.

    The order is the contract's, which has a visible consequence: punctuation is
    stripped *before* digits are spelled, so ``"3.5"`` becomes ``"35"`` and then
    ``"thirty five"``.

    >>> normalise_text("Hello, World!  It's 42 degrees.")
    'hello world its forty two degrees'
    >>> normalise_text("well-known mic/array")
    'well known mic array'
    >>> normalise_text("")
    ''
    """
    result = text
    for step in spec.TEXT_NORMALISATION:
        if step == "unicode_nfkc":
            result = unicodedata.normalize("NFKC", result)
        elif step == "casefold":
            result = result.casefold()
        elif step == "strip_punctuation":
            result = _strip_punctuation(result)
        elif step == "digits_to_words":
            result = _digits_to_words(result)
        elif step == "collapse_whitespace":
            result = " ".join(result.split())
        else:  # pragma: no cover - guards a spec change
            raise RuntimeError(f"unknown normalisation step {step!r}")
    return result


def tokenise(text: str) -> list[str]:
    """Normalise ``text`` and split it into scoring tokens (words).

    >>> tokenise("Two mics.")
    ['two', 'mics']
    >>> tokenise("   ")
    []
    """
    return normalise_text(text).split()


def levenshtein(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return the unit-cost word-level edit distance.

    Substitution, insertion and deletion all cost 1, which is what ``WER`` and
    ``cpWER`` need.  Implemented here so the benchmark needs no ``jiwer``.

    >>> levenshtein(["a", "b", "c"], ["a", "c"])
    1
    >>> levenshtein([], ["x", "y"])
    2
    >>> levenshtein(["x", "y"], [])
    2
    >>> levenshtein([], [])
    0
    """
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(reference) + 1))
    for hyp_word in hypothesis:
        current = [previous[0] + 1]
        for index, ref_word in enumerate(reference, start=1):
            current.append(
                min(
                    previous[index] + 1,
                    current[index - 1] + 1,
                    previous[index - 1] + (0 if ref_word == hyp_word else 1),
                )
            )
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def wrap_azimuth_deg(azimuth_deg: float) -> float:
    """Wrap an angle into :data:`spec.AZIMUTH_RANGE_DEG`, i.e. ``[-180, 180)``.

    ``+180`` is the same bearing as ``-180`` and the contract keeps the range
    half-open, so ``+180`` canonicalises to ``-180`` -- the same place
    :func:`spec.azimuth_bin_index` sends it.

    >>> wrap_azimuth_deg(180.0)
    -180.0
    >>> wrap_azimuth_deg(-190.0)
    170.0
    >>> wrap_azimuth_deg(0.0)
    0.0
    >>> wrap_azimuth_deg(-180.0)
    -180.0
    """
    low, high = spec.AZIMUTH_RANGE_DEG
    span = high - low
    return ((float(azimuth_deg) - low) % span) + low


def polar_to_xy(distance_m: float, azimuth_deg: float) -> tuple[float, float]:
    """Convert ``(r, theta)`` to ``(x, y)`` on the plane at ``z = 1.5`` m.

    >>> polar_to_xy(1.0, 0.0)
    (1.0, 0.0)
    >>> tuple(round(v, 9) for v in polar_to_xy(2.0, 90.0))
    (0.0, 2.0)
    """
    radians = math.radians(float(azimuth_deg))
    return (float(distance_m) * math.cos(radians), float(distance_m) * math.sin(radians))


def xy_to_polar(x_m: float, y_m: float) -> tuple[float, float]:
    """Convert ``(x, y)`` to ``(r, theta)`` with ``theta`` wrapped.

    >>> xy_to_polar(0.0, 2.0)
    (2.0, 90.0)
    >>> xy_to_polar(1.0, 0.0)
    (1.0, 0.0)
    """
    return (
        math.hypot(float(x_m), float(y_m)),
        wrap_azimuth_deg(math.degrees(math.atan2(float(y_m), float(x_m)))),
    )


def euclidean_2d(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    """Return the planar distance between two ``(x, y)`` points.

    >>> euclidean_2d((0.0, 0.0), (3.0, 4.0))
    5.0
    """
    return math.hypot(first[0] - second[0], first[1] - second[1])


def time_key(eval_time_sec: float) -> float:
    """Canonicalise ``eval_time_sec`` for use as a dictionary key.

    :data:`spec.DELTA_T_SEC` is exactly representable in binary floating point,
    so grid instants pass through unchanged; the rounding only protects against
    an off-grid value whose text form carries noise.

    >>> time_key(0.25)
    0.25
    >>> time_key(2.0000000001)
    2.0
    """
    return round(float(eval_time_sec), _TIME_DECIMALS)


def format_time(eval_time_sec: float) -> str:
    """Render a time stamp for error messages without trailing float noise.

    >>> format_time(0.25)
    '0.25'
    >>> format_time(2.0)
    '2.0'
    """
    return repr(time_key(eval_time_sec))


# ---------------------------------------------------------------------------
# Probability-vector validation (R-PROB-01/02/03) and distributional metrics
# ---------------------------------------------------------------------------


def validate_prob_vector(
    values: Any, expected_length: int, *, where: str
) -> tuple[float, ...]:
    """Validate one submitted distribution and return it **unchanged**.

    Checks run in this order: length (``R-PROB-01``), element sanity
    (``R-PROB-03``: ``NaN``, ``inf``, ``p < 0`` or ``p > 1``) and only then the
    mass (``R-PROB-02``: ``|sum - 1| < spec.PROB_SUM_TOLERANCE``) -- summing is
    meaningless while a ``NaN`` is still in the vector.  Nothing is rescaled,
    because :data:`spec.NORMALISE_SUBMITTED_PROBS` is ``False``.

    >>> validate_prob_vector([0.5, 0.5], 2, where="t")
    (0.5, 0.5)
    >>> validate_prob_vector([0.5], 2, where="t")
    Traceback (most recent call last):
        ...
    evaluate.SubmissionError: R-PROB-01: t: expected 2 probabilities, got 1
    >>> validate_prob_vector([1.5, -0.5], 2, where="t")
    Traceback (most recent call last):
        ...
    evaluate.SubmissionError: R-PROB-03: t: probability out of [0, 1] at index 0: 1.5
    >>> validate_prob_vector([0.3, 0.3], 2, where="t")
    Traceback (most recent call last):
        ...
    evaluate.SubmissionError: R-PROB-02: t: probabilities sum to 0.6, tolerance is 0.0001
    """
    if _np is not None and isinstance(values, _np.ndarray):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _reject(RULE_PARSE, f"{where}: probability vector must be a list, got {type(values).__name__}")
    if len(values) != expected_length:
        _reject(
            RULE_PROB_LENGTH,
            f"{where}: expected {expected_length} probabilities, got {len(values)}",
        )
    probs: list[float] = []
    for index, raw in enumerate(values):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            _reject(
                RULE_PROB_VALUE,
                f"{where}: probability at index {index} is not a number: {raw!r}",
            )
        value = float(raw)
        if not math.isfinite(value):
            _reject(RULE_PROB_VALUE, f"{where}: non-finite probability at index {index}: {value}")
        if value < 0.0 or value > 1.0:
            _reject(
                RULE_PROB_VALUE,
                f"{where}: probability out of [0, 1] at index {index}: {value}",
            )
        probs.append(value)
    total = math.fsum(probs)
    if abs(total - 1.0) >= spec.PROB_SUM_TOLERANCE:
        _reject(
            RULE_PROB_SUM,
            f"{where}: probabilities sum to {total!r}, tolerance is {spec.PROB_SUM_TOLERANCE}",
        )
    return tuple(probs)


def argmax_bin(probs: Sequence[float]) -> int:
    """Return the MAP bin index, breaking ties toward the smallest index.

    >>> argmax_bin((0.1, 0.4, 0.4, 0.1))
    1
    >>> argmax_bin((0.0, 0.0, 1.0))
    2
    """
    best_index = 0
    best_value = probs[0]
    for index in range(1, len(probs)):
        if probs[index] > best_value:
            best_index, best_value = index, probs[index]
    return best_index


def negative_log_likelihood(probs: Sequence[float], true_bin: int) -> float:
    """Return ``-log(max(p[true_bin], spec.NLL_EPSILON))``.

    The clipping floor is applied to the true bin only and there is no
    smoothing, so the worst value a submitted distribution can score is exactly
    :data:`spec.PENALTY_NLL` -- which is why omitting a time step can never pay.

    >>> negative_log_likelihood((1.0, 0.0), 0)
    0.0
    >>> negative_log_likelihood((1.0, 0.0), 1) == spec.PENALTY_NLL
    True
    """
    log_p = math.log(max(float(probs[true_bin]), spec.NLL_EPSILON))
    return -log_p if log_p != 0.0 else 0.0  # avoid reporting "-0.0"


def hpd_set(probs: Sequence[float], level: float = spec.COVERAGE_LEVEL) -> tuple[int, ...]:
    """Build the greedy highest-posterior-density set (:data:`spec.INTERVAL_RULE`).

    Bins are taken in descending probability, ties toward the smaller index,
    until the accumulated mass reaches ``level``.  The returned indices are
    sorted but need not be contiguous -- that is the point for a bimodal
    two-microphone posterior.

    >>> hpd_set((0.25, 0.25, 0.25, 0.25), 0.9)
    (0, 1, 2, 3)
    >>> hpd_set((0.5, 0.25, 0.25), 0.7)
    (0, 1)
    >>> hpd_set((1.0, 0.0, 0.0), 0.9)
    (0,)
    """
    order = sorted(range(len(probs)), key=lambda index: (-probs[index], index))
    chosen: list[int] = []
    for index in order:
        chosen.append(index)
        if math.fsum(probs[i] for i in chosen) >= level:
            break
    return tuple(sorted(chosen))


def azimuth_interval_width_deg(bins: Sequence[int]) -> float:
    """Summed width of the azimuth bins in an HPD set, in degrees.

    >>> azimuth_interval_width_deg((0, 18))
    20.0
    >>> azimuth_interval_width_deg(())
    0.0
    """
    return float(len(bins)) * spec.AZIMUTH_BIN_WIDTH_DEG


def distance_interval_width_m(bins: Sequence[int]) -> float:
    """Summed width of the distance bins in an HPD set, in meters.

    >>> round(distance_interval_width_m((0, 7)), 10)
    1.2
    >>> distance_interval_width_m(())
    0.0
    """
    return math.fsum(spec.DISTANCE_BIN_WIDTHS_M[index] for index in bins)


# ---------------------------------------------------------------------------
# Record model
# ---------------------------------------------------------------------------

RECORD_TRANSCRIPTS: Final[str] = spec.RECORD_TYPE_FINAL_TRANSCRIPTS
RECORD_TIMEPOINT: Final[str] = spec.RECORD_TYPE_TIMEPOINT
RECORD_CASE_ROW: Final[str] = "case_row"
"""Internal record kind for a generator-style one-row-per-``case_id`` ground
truth.  Never a submitted ``record_type`` value; see :func:`_expand_case_row`."""

RECORD_TRANSCRIPTS_ALIASES: Final[tuple[str, ...]] = tuple(
    name
    for name in (RECORD_TRANSCRIPTS, *spec.RECORD_TYPE_ALIASES)
    if spec.canonical_record_type(name) == RECORD_TRANSCRIPTS
)
RECORD_TIMEPOINT_ALIASES: Final[tuple[str, ...]] = tuple(
    name
    for name in (RECORD_TIMEPOINT, *spec.RECORD_TYPE_ALIASES)
    if spec.canonical_record_type(name) == RECORD_TIMEPOINT
)
"""Accepted spellings of ``record_type``, derived from
:data:`spec.RECORD_TYPE_ALIASES`.

>>> RECORD_TIMEPOINT_ALIASES
('timepoint', 'timestep')
>>> RECORD_TRANSCRIPTS_ALIASES
('final_transcripts', 'final_transcript')
"""

FRAMES_KEY: Final[str] = spec.F_FRAMES
TRANSCRIPTS_REFERENCE_KEY: Final[str] = spec.F_TRANSCRIPTS_REFERENCE
SPEAKER_ID_KEY: Final[str] = spec.FIELD_ALIASES[spec.F_SPEAKER][1]
RTF_ALIAS_KEY: Final[str] = spec.FIELD_ALIASES[spec.F_RTF][1]
"""Field names :mod:`spec` owns, re-exported under this module's older names.

The alias order is the contract's: ``spec.FIELD_ALIASES[F_RTF]`` is
``("rtf", "reported_rtf")``, so index 1 is the deprecated spelling by
construction rather than by a literal repeated here.

>>> SPEAKER_ID_KEY, RTF_ALIAS_KEY
('speaker_id', 'reported_rtf')
"""


@dataclass(frozen=True)
class TruthPoint:
    """One ground-truth target instant. Defines membership of ``T_eval``."""

    case_id: str
    eval_time_sec: float
    azimuth_deg: float | None
    distance_m: float | None
    speaker: str | None

    @property
    def azimuth_bin(self) -> int | None:
        """MAP-free true azimuth bin, or ``None`` when unlabelled."""
        if self.azimuth_deg is None:
            return None
        return spec.azimuth_bin_index(self.azimuth_deg)

    @property
    def distance_bin(self) -> int | None:
        """True distance bin, or ``None`` when unlabelled."""
        if self.distance_m is None:
            return None
        return spec.distance_bin_index(self.distance_m)

    @property
    def xy(self) -> tuple[float, float] | None:
        """True ``(x, y)``, or ``None`` when either polar component is absent."""
        if self.azimuth_deg is None or self.distance_m is None:
            return None
        return polar_to_xy(self.distance_m, self.azimuth_deg)


@dataclass(frozen=True)
class PredPoint:
    """One submitted prediction instant, exactly as written by the model."""

    case_id: str
    eval_time_sec: float
    azimuth_deg: float | None
    distance_m: float | None
    x_m: float | None
    y_m: float | None
    azimuth_probs: tuple[float, ...] | None
    distance_probs: tuple[float, ...] | None
    speaker: str | None


@dataclass
class GroundTruth:
    """Parsed ground truth. ``eval_times`` is the authoritative ``T_eval``."""

    transcripts: dict[str, dict[str, str]] = field(default_factory=dict)
    points: dict[str, dict[float, TruthPoint]] = field(default_factory=dict)
    eval_times: dict[str, list[float]] = field(default_factory=dict)

    def case_ids(self) -> list[str]:
        """Every scored ``case_id``, sorted. Transcript-only cases are included."""
        return sorted(set(self.transcripts) | set(self.points))


@dataclass
class Predictions:
    """Parsed submission."""

    transcripts: dict[str, dict[str, str]] = field(default_factory=dict)
    observation_modes: dict[str, str] = field(default_factory=dict)
    rtf: dict[str, float] = field(default_factory=dict)
    points: dict[str, dict[float, PredPoint]] = field(default_factory=dict)

    def case_ids(self) -> list[str]:
        """Every ``case_id`` the submission mentions, sorted."""
        return sorted(set(self.transcripts) | set(self.points))


# ---------------------------------------------------------------------------
# Parsing (R-PARSE, R-DUP-01, R-DUP-02, R-SPK-01)
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, object)`` for every non-blank line of a JSONL file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        _reject(RULE_PARSE, f"cannot read {path}: {error}")
        raise  # pragma: no cover
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as error:
            _reject(RULE_PARSE, f"{path.name}:{line_number}: invalid JSON ({error.msg})")
            raise  # pragma: no cover
        if not isinstance(obj, dict):
            _reject(
                RULE_PARSE,
                f"{path.name}:{line_number}: each line must be a JSON object, got {type(obj).__name__}",
            )
        yield line_number, obj


def _infer_record_type(obj: Mapping[str, Any], where: str) -> str:
    """Return the canonical record type, inferring it from the fields if absent.

    A declared ``record_type`` wins and is canonicalised through
    :data:`RECORD_TIMEPOINT_ALIASES` / :data:`RECORD_TRANSCRIPTS_ALIASES`.
    Otherwise a nested :data:`FRAMES_KEY` list marks a generator-style per-case
    ground-truth row, a top-level ``eval_time_sec`` a single instant, and a
    ``final_transcripts`` mapping a transcript record.
    """
    declared = obj.get("record_type")
    if declared is not None:
        if declared in RECORD_TRANSCRIPTS_ALIASES:
            return RECORD_TRANSCRIPTS
        if declared in RECORD_TIMEPOINT_ALIASES:
            return RECORD_TIMEPOINT
        _reject(RULE_PARSE, f"{where}: unknown record_type {declared!r}")
    if isinstance(obj.get(FRAMES_KEY), list):
        return RECORD_CASE_ROW
    if "eval_time_sec" in obj:
        return RECORD_TIMEPOINT
    if RECORD_TRANSCRIPTS in obj:
        return RECORD_TRANSCRIPTS
    _reject(
        RULE_PARSE,
        f"{where}: cannot infer record_type (no {FRAMES_KEY!r}, no 'eval_time_sec' "
        "and no 'final_transcripts')",
    )
    raise AssertionError  # pragma: no cover


def _require_case_id(obj: Mapping[str, Any], where: str) -> str:
    """Return a validated ``case_id``."""
    case_id = obj.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        _reject(RULE_PARSE, f"{where}: 'case_id' must be a non-empty string, got {case_id!r}")
    try:
        spec.parse_case_id(str(case_id))
    except ValueError as error:
        _reject(RULE_PARSE, f"{where}: {error}")
    return str(case_id)


def _optional_float(obj: Mapping[str, Any], key: str, where: str) -> float | None:
    """Return ``obj[key]`` as a finite float, or ``None`` when absent/null."""
    if key not in obj or obj[key] is None:
        return None
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject(RULE_PARSE, f"{where}: {key!r} must be a number or null, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        _reject(RULE_PARSE, f"{where}: {key!r} is not finite: {number}")
    return number


def _optional_speaker(obj: Mapping[str, Any], key: str, where: str) -> str | None:
    """Return ``obj[key]`` as a validated speaker label, or ``None``."""
    if key not in obj or obj[key] is None:
        return None
    value = obj[key]
    if not isinstance(value, str) or not value:
        _reject(RULE_PARSE, f"{where}: {key!r} must be a non-empty string or null, got {value!r}")
    if str(value).startswith(PAD_LABEL_PREFIX):
        _reject(RULE_PARSE, f"{where}: speaker label must not start with {PAD_LABEL_PREFIX!r}")
    return str(value)


def _require_eval_time(obj: Mapping[str, Any], where: str) -> float:
    """Return a validated ``eval_time_sec`` on the frozen inference grid's span.

    The upper bound is *exclusive*, per :data:`spec.GRID_ENDPOINT`: the last
    instant of a 20 s session is 19.75 s, and an instant at exactly 20.0 s has no
    audio under it, so a causal system could only be charged a penalty for it.
    """
    value = _optional_float(obj, spec.F_EVAL_TIME, where)
    if value is None:
        _reject(RULE_PARSE, f"{where}: timepoint record needs {spec.F_EVAL_TIME!r}")
        raise AssertionError  # pragma: no cover
    if value < 0.0 or value >= spec.SESSION_LENGTH_SEC:
        _reject(
            RULE_PARSE,
            f"{where}: {spec.F_EVAL_TIME} {value} outside [0, {spec.SESSION_LENGTH_SEC})",
        )
    return value


def _parse_transcripts(obj: Mapping[str, Any], where: str, track: int) -> dict[str, str]:
    """Validate and return the ``final_transcripts`` mapping of one record."""
    mapping = obj.get(RECORD_TRANSCRIPTS)
    if not isinstance(mapping, Mapping):
        _reject(
            RULE_PARSE,
            f"{where}: 'final_transcripts' must be an object, got {type(mapping).__name__}",
        )
        raise AssertionError  # pragma: no cover
    result: dict[str, str] = {}
    for label, text in mapping.items():
        if not isinstance(label, str) or not label:
            _reject(RULE_PARSE, f"{where}: speaker key must be a non-empty string, got {label!r}")
        if str(label).startswith(PAD_LABEL_PREFIX):
            _reject(RULE_PARSE, f"{where}: speaker label must not start with {PAD_LABEL_PREFIX!r}")
        if text is None:
            text = ""
        if not isinstance(text, str):
            _reject(RULE_PARSE, f"{where}: transcript for {label!r} must be a string, got {text!r}")
        result[str(label)] = str(text)
    _check_speaker_budget(result.keys(), track, where)
    return result


def _check_speaker_budget(labels: Any, track: int, where: str) -> None:
    """Reject more speaker slots than :data:`spec.MAX_SPEAKERS_PER_TRACK` allows."""
    allowed = spec.MAX_SPEAKERS_PER_TRACK.get(track)
    if allowed is None:
        _reject(RULE_PARSE, f"{where}: unknown track {track!r}")
        raise AssertionError  # pragma: no cover
    count = len(set(labels))
    if count > allowed:
        _reject(
            RULE_SPEAKER_COUNT,
            f"{where}: {count} speaker slots but track {track} allows at most {allowed}",
        )


def _read_position(obj: Mapping[str, Any], where: str) -> tuple[float | None, float | None]:
    """Return ``(wrapped azimuth_deg, distance_m)`` from a polar or cartesian record.

    Polar wins; a cartesian pair is used only when neither polar component is
    present, so a record carrying both is never scored against a converted value
    that disagrees with its own polar fields.
    """
    azimuth = _optional_float(obj, "azimuth_deg", where)
    distance = _optional_float(obj, "distance_m", where)
    x_m = _optional_float(obj, "x_m", where)
    y_m = _optional_float(obj, "y_m", where)
    if azimuth is None and distance is None and x_m is not None and y_m is not None:
        distance, azimuth = xy_to_polar(x_m, y_m)
    return (None if azimuth is None else wrap_azimuth_deg(azimuth)), distance


def _check_declared_bins(obj: Mapping[str, Any], point: TruthPoint, where: str) -> None:
    """Reject a ground-truth record whose declared bin disagrees with :mod:`spec`.

    The generator writes ``azimuth_bin``/``distance_bin`` next to the continuous
    labels.  A disagreement would silently corrupt every NLL and coverage number,
    so it is a hard rejection rather than a warning.
    """
    for key, derived in (("azimuth_bin", point.azimuth_bin), ("distance_bin", point.distance_bin)):
        declared = obj.get(key)
        if declared is None or derived is None:
            continue
        if isinstance(declared, bool) or not isinstance(declared, int):
            _reject(RULE_PARSE, f"{where}: {key!r} must be an integer, got {declared!r}")
        if int(declared) != derived:
            _reject(
                RULE_PARSE,
                f"{where}: declared {key}={declared} but spec derives {derived} "
                "from the continuous label",
            )


def _add_truth_point(
    truth: GroundTruth,
    case_id: str,
    obj: Mapping[str, Any],
    where: str,
    default_speaker: str | None = None,
) -> None:
    """Validate one ground-truth instant and file it under ``case_id``.

    Shared by the flat one-row-per-instant form and the nested per-case form so
    both get identical duplicate, range and binning checks.
    """
    eval_time = _require_eval_time(obj, where)
    key = time_key(eval_time)
    bucket = truth.points.setdefault(case_id, {})
    if key in bucket:
        _reject(
            RULE_DUPLICATE_TIME,
            f"duplicate timepoint ({case_id!r}, {format_time(eval_time)})",
        )
    azimuth, distance = _read_position(obj, where)
    speaker = _optional_speaker(obj, "speaker", where)
    if speaker is None:
        speaker = _optional_speaker(obj, SPEAKER_ID_KEY, where)
    point = TruthPoint(
        case_id=case_id,
        eval_time_sec=eval_time,
        azimuth_deg=azimuth,
        distance_m=distance,
        speaker=speaker if speaker is not None else default_speaker,
    )
    _check_declared_bins(obj, point, where)
    bucket[key] = point
    truth.eval_times.setdefault(case_id, []).append(key)


def _expand_case_row(
    truth: GroundTruth, case_id: str, obj: Mapping[str, Any], where: str, track: int
) -> None:
    """Expand a generator-style one-row-per-``case_id`` ground-truth record.

    ``make_eval_dev.py`` writes the reference transcript under
    :data:`TRANSCRIPTS_REFERENCE_KEY`, the speaker label under
    :data:`SPEAKER_ID_KEY` and the per-instant labels nested in
    :data:`FRAMES_KEY`.  [contract gap] ``spec.py`` fixes none of those names,
    so the row is translated into the flat form instead of being rejected --
    which would make a whole eval-dev set unscorable.
    """
    speaker = _optional_speaker(obj, SPEAKER_ID_KEY, where)
    reference = obj.get(TRANSCRIPTS_REFERENCE_KEY)
    transcripts: dict[str, str] | None = None
    if isinstance(reference, Mapping):
        transcripts = _parse_transcripts({RECORD_TRANSCRIPTS: reference}, where, track)
    elif isinstance(obj.get("transcript"), str) and speaker is not None:
        transcripts = _parse_transcripts(
            {RECORD_TRANSCRIPTS: {speaker: obj["transcript"]}}, where, track
        )
    if transcripts is not None:
        if case_id in truth.transcripts:
            _reject(RULE_DUPLICATE_CASE, f"duplicate case_id {case_id!r}")
        truth.transcripts[case_id] = transcripts
    frames = obj.get(FRAMES_KEY)
    if not isinstance(frames, list):  # pragma: no cover - guarded by the caller
        _reject(RULE_PARSE, f"{where}: {FRAMES_KEY!r} must be a list")
        raise AssertionError  # pragma: no cover
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            _reject(RULE_PARSE, f"{where}: {FRAMES_KEY}[{index}] must be an object")
            raise AssertionError  # pragma: no cover
        _add_truth_point(truth, case_id, frame, f"{where} {FRAMES_KEY}[{index}]", speaker)


def load_ground_truth(path: Path, track: int) -> GroundTruth:
    """Parse and validate a ground-truth JSONL file.

    The ground truth is authoritative for ``T_eval``: the set of instants that
    get scored is exactly the set of timepoint records found here, in file
    order.  ``T_eval`` is never read from the submission.

    Both ground-truth shapes are accepted: one flat record per instant, and the
    generator's one row per ``case_id`` with the instants nested in
    :data:`FRAMES_KEY`.
    """
    truth = GroundTruth()
    for line_number, obj in _iter_jsonl(path):
        where = f"{path.name}:{line_number}"
        _check_declared_spec_version(obj, where)
        case_id = _require_case_id(obj, where)
        kind = _infer_record_type(obj, where)
        if kind == RECORD_TRANSCRIPTS:
            if case_id in truth.transcripts:
                _reject(RULE_DUPLICATE_CASE, f"duplicate case_id {case_id!r}")
            truth.transcripts[case_id] = _parse_transcripts(obj, where, track)
            continue
        if kind == RECORD_CASE_ROW:
            _expand_case_row(truth, case_id, obj, where, track)
            continue
        _add_truth_point(truth, case_id, obj, where)
    for case_id, bucket in truth.points.items():
        speakers = {point.speaker for point in bucket.values() if point.speaker}
        speakers |= set(truth.transcripts.get(case_id, {}))
        _check_speaker_budget(speakers, track, f"{path.name} case {case_id}")
    return truth


def _pred_speaker(obj: Mapping[str, Any], where: str) -> str | None:
    """Return the predicted speaker label under any of its accepted names.

    ``pred_speaker`` is this evaluator's name; both baselines write
    :data:`SPEAKER_ID_KEY` instead.  Reading only the first would silently score
    ``speaker_attribution_accuracy`` as zero for every submitted instant, which
    is worse than a rejection because it looks like a model result.
    """
    for key in ("pred_speaker", SPEAKER_ID_KEY, "speaker"):
        speaker = _optional_speaker(obj, key, where)
        if speaker is not None:
            return speaker
    return None


def load_predictions(path: Path, track: int) -> Predictions:
    """Parse and validate a submission JSONL file.

    Field aliases written by the baselines (:data:`SPEAKER_ID_KEY`,
    :data:`RTF_ALIAS_KEY`, ``record_type`` ``"timestep"`` /
    ``"final_transcript"``) are accepted and canonicalised.
    """
    preds = Predictions()
    for line_number, obj in _iter_jsonl(path):
        where = f"{path.name}:{line_number}"
        _check_declared_spec_version(obj, where)
        case_id = _require_case_id(obj, where)
        kind = _infer_record_type(obj, where)
        if kind == RECORD_TRANSCRIPTS:
            if case_id in preds.transcripts:
                _reject(RULE_DUPLICATE_CASE, f"duplicate case_id {case_id!r}")
            preds.transcripts[case_id] = _parse_transcripts(obj, where, track)
            mode = obj.get("observation_mode", spec.OFFICIAL_SCOPES[0])
            if mode not in spec.OBSERVATION_MODES:
                _reject(
                    RULE_PARSE,
                    f"{where}: observation_mode {mode!r} not in {spec.OBSERVATION_MODES}",
                )
            preds.observation_modes[case_id] = str(mode)
            rtf = _optional_float(obj, METRIC_RTF, where)
            if rtf is None:
                rtf = _optional_float(obj, RTF_ALIAS_KEY, where)
            if rtf is not None:
                if rtf < 0.0:
                    _reject(RULE_PARSE, f"{where}: rtf must be non-negative, got {rtf}")
                preds.rtf[case_id] = rtf
            continue
        if kind == RECORD_CASE_ROW:
            _reject(
                RULE_PARSE,
                f"{where}: a submission must list one record per timepoint, "
                f"not a nested {FRAMES_KEY!r} list",
            )
        eval_time = _require_eval_time(obj, where)
        key = time_key(eval_time)
        bucket = preds.points.setdefault(case_id, {})
        if key in bucket:
            _reject(
                RULE_DUPLICATE_TIME,
                f"duplicate timepoint ({case_id!r}, {format_time(eval_time)})",
            )
        azimuth_probs = None
        if obj.get("azimuth_probs") is not None:
            azimuth_probs = validate_prob_vector(
                obj["azimuth_probs"], spec.K_AZIMUTH, where=f"{where} azimuth_probs"
            )
        distance_probs = None
        if obj.get("distance_probs") is not None:
            distance_probs = validate_prob_vector(
                obj["distance_probs"], spec.K_DISTANCE, where=f"{where} distance_probs"
            )
        bucket[key] = PredPoint(
            case_id=case_id,
            eval_time_sec=eval_time,
            azimuth_deg=_optional_float(obj, "pred_azimuth_deg", where),
            distance_m=_optional_float(obj, "pred_distance_m", where),
            x_m=_optional_float(obj, "pred_x_m", where),
            y_m=_optional_float(obj, "pred_y_m", where),
            azimuth_probs=azimuth_probs,
            distance_probs=distance_probs,
            speaker=_pred_speaker(obj, where),
        )
    for case_id in preds.case_ids():
        speakers = {
            point.speaker
            for point in preds.points.get(case_id, {}).values()
            if point.speaker
        }
        speakers |= set(preds.transcripts.get(case_id, {}))
        _check_speaker_budget(speakers, track, f"{path.name} case {case_id}")
    return preds


def _check_declared_spec_version(obj: Mapping[str, Any], where: str) -> None:
    """Reject a record that declares a different contract version."""
    declared = obj.get("spec_version")
    if declared is not None and str(declared) != spec.SPEC_VERSION:
        _reject(
            RULE_PARSE,
            f"{where}: record declares spec_version {declared!r}, this evaluator is {spec.SPEC_VERSION!r}",
        )


# ---------------------------------------------------------------------------
# Point representative resolution (explicit field first, MAP bin center second)
# ---------------------------------------------------------------------------

SOURCE_EXPLICIT_POLAR: Final[str] = "explicit_polar"
SOURCE_EXPLICIT_CARTESIAN: Final[str] = "explicit_cartesian"
SOURCE_MAP_BIN: Final[str] = "map_bin_center"
SOURCE_MISSING: Final[str] = "missing"


def _cartesian(point: PredPoint) -> tuple[float, float] | None:
    """Return the submitted ``(x, y)`` when both components are non-null."""
    if point.x_m is None or point.y_m is None:
        return None
    return (point.x_m, point.y_m)


def resolve_azimuth_deg(point: PredPoint | None) -> tuple[float | None, str]:
    """Resolve the azimuth representative point and say where it came from.

    Priority: an explicit ``pred_azimuth_deg``, then an explicit
    ``(pred_x_m, pred_y_m)`` pair, then the MAP bin center of
    ``azimuth_probs`` (ties toward the smallest index), then nothing.
    """
    if point is None:
        return None, SOURCE_MISSING
    if point.azimuth_deg is not None:
        return wrap_azimuth_deg(point.azimuth_deg), SOURCE_EXPLICIT_POLAR
    cartesian = _cartesian(point)
    if cartesian is not None:
        return xy_to_polar(*cartesian)[1], SOURCE_EXPLICIT_CARTESIAN
    if point.azimuth_probs is not None:
        index = argmax_bin(point.azimuth_probs)
        return spec.AZIMUTH_BIN_CENTERS_DEG[index], SOURCE_MAP_BIN
    return None, SOURCE_MISSING


def resolve_distance_m(point: PredPoint | None) -> tuple[float | None, str]:
    """Resolve the distance representative point and say where it came from."""
    if point is None:
        return None, SOURCE_MISSING
    if point.distance_m is not None:
        return point.distance_m, SOURCE_EXPLICIT_POLAR
    cartesian = _cartesian(point)
    if cartesian is not None:
        return xy_to_polar(*cartesian)[0], SOURCE_EXPLICIT_CARTESIAN
    if point.distance_probs is not None:
        index = argmax_bin(point.distance_probs)
        return spec.DISTANCE_BIN_CENTERS_M[index], SOURCE_MAP_BIN
    return None, SOURCE_MISSING


def resolve_xy(point: PredPoint | None) -> tuple[tuple[float, float] | None, str]:
    """Resolve the 2-D representative point.

    An explicit cartesian pair is used verbatim; otherwise the resolved polar
    pair is converted, so a point-only polar submission and a distribution-only
    submission both get a well-defined 2-D error.
    """
    if point is None:
        return None, SOURCE_MISSING
    cartesian = _cartesian(point)
    if cartesian is not None:
        return cartesian, SOURCE_EXPLICIT_CARTESIAN
    azimuth, azimuth_source = resolve_azimuth_deg(point)
    distance, distance_source = resolve_distance_m(point)
    if azimuth is None or distance is None:
        return None, SOURCE_MISSING
    source = (
        SOURCE_MAP_BIN
        if SOURCE_MAP_BIN in (azimuth_source, distance_source)
        else SOURCE_EXPLICIT_POLAR
    )
    return polar_to_xy(distance, azimuth), source


# ---------------------------------------------------------------------------
# Speaker permutation pi* (spec.PERMUTATION_TIE_BREAK)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeakerAssignment:
    """The single ``pi*`` chosen for one ``case_id``.

    ``mapping`` sends each submitted speaker label to a reference label, or to
    ``None`` when it was matched against a :data:`spec.SPEAKER_PAD_TOKEN` slot
    (all of its words are then insertions).  ``unmatched_ref`` lists reference
    speakers with no submitted counterpart (all of their words are deletions).
    """

    mapping: dict[str, str | None]
    unmatched_ref: tuple[str, ...]
    cpwer_edits: int
    match_count: int


def solve_permutation(
    pred_labels: Sequence[str],
    ref_labels: Sequence[str],
    pred_texts: Mapping[str, str],
    ref_texts: Mapping[str, str],
    speaker_pairs: Sequence[tuple[str | None, str | None]],
) -> SpeakerAssignment:
    """Pick the one ``pi*`` used for a whole ``case_id``.

    Both label sets are padded to ``max(len(pred), len(ref))`` with
    :data:`spec.SPEAKER_PAD_TOKEN` slots, so an assignment always exists, and
    every bijection is enumerated (a track allows at most two speakers, so
    brute force is exact).  :data:`spec.PERMUTATION_TIE_BREAK` is applied in
    order:

    1. ``min_cpwer`` -- the reference word count is the same for every
       assignment, so minimising ``cpWER`` is minimising total edit distance;
    2. ``max_speaker_match_count`` -- most time steps whose mapped speaker
       equals the reference speaker;
    3. ``lexicographic`` -- smallest tuple of assigned reference-slot indices,
       taken over the submitted labels in ascending order with the padded slots
       last.  This is :data:`spec.PERMUTATION_LEXICOGRAPHIC_KEY`, which names the
       sequence being ordered.

    >>> a = solve_permutation(["s0"], ["spk0"], {"s0": "hi there"},
    ...                       {"spk0": "hi there"}, [])
    >>> a.mapping, a.cpwer_edits
    ({'s0': 'spk0'}, 0)
    """
    slots = max(len(pred_labels), len(ref_labels))
    if slots > _MAX_PERMUTATION_SLOTS:
        _reject(
            RULE_SPEAKER_COUNT,
            f"{slots} speaker slots exceeds the permutation search limit of {_MAX_PERMUTATION_SLOTS}",
        )
    if slots == 0:
        return SpeakerAssignment(mapping={}, unmatched_ref=(), cpwer_edits=0, match_count=0)
    pred_slots = list(pred_labels) + [
        f"{PAD_LABEL_PREFIX}p{i}" for i in range(slots - len(pred_labels))
    ]
    ref_slots = list(ref_labels) + [
        f"{PAD_LABEL_PREFIX}r{i}" for i in range(slots - len(ref_labels))
    ]
    ref_tokens = {
        label: tokenise(ref_texts.get(label, spec.SPEAKER_PAD_TOKEN)) for label in ref_slots
    }
    hyp_tokens = {
        label: tokenise(pred_texts.get(label, spec.SPEAKER_PAD_TOKEN)) for label in pred_slots
    }
    best_key: tuple[int, int, tuple[int, ...]] | None = None
    best_internal: dict[str, str] = {}
    best_edits = 0
    best_matches = 0
    for perm in permutations(range(slots)):
        internal = {pred_slots[i]: ref_slots[perm[i]] for i in range(slots)}
        edits = sum(
            levenshtein(ref_tokens[internal[pred_slots[i]]], hyp_tokens[pred_slots[i]])
            for i in range(slots)
        )
        matches = sum(
            1
            for truth_speaker, pred_speaker in speaker_pairs
            if truth_speaker is not None
            and pred_speaker is not None
            and internal.get(pred_speaker) == truth_speaker
        )
        key = (edits, -matches, perm)
        if best_key is None or key < best_key:
            best_key, best_internal, best_edits, best_matches = key, internal, edits, matches
    mapping: dict[str, str | None] = {
        label: (None if target.startswith(PAD_LABEL_PREFIX) else target)
        for label, target in best_internal.items()
        if not label.startswith(PAD_LABEL_PREFIX)
    }
    unmatched_ref = tuple(
        sorted(
            target
            for label, target in best_internal.items()
            if label.startswith(PAD_LABEL_PREFIX) and not target.startswith(PAD_LABEL_PREFIX)
        )
    )
    return SpeakerAssignment(
        mapping=mapping,
        unmatched_ref=unmatched_ref,
        cpwer_edits=best_edits,
        match_count=best_matches,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class MacroAccumulator:
    """``spec.SPATIAL_AGGREGATION == "macro_per_case"``.

    Time steps are averaged uniformly inside a case, then cases are averaged
    uniformly.  ``math.fsum`` makes the result independent of insertion order.

    >>> acc = MacroAccumulator()
    >>> acc.add("a", 0.0); acc.add("a", 2.0); acc.add("b", 3.0)
    >>> acc.result()
    2.0
    >>> MacroAccumulator().result() is None
    True
    """

    __slots__ = ("_per_case",)

    def __init__(self) -> None:
        self._per_case: dict[str, list[float]] = {}

    def add(self, case_id: str, value: float) -> None:
        """Record one within-case observation."""
        self._per_case.setdefault(case_id, []).append(float(value))

    def case_count(self) -> int:
        """Number of cases that contributed at least one observation."""
        return len(self._per_case)

    def result(self) -> float | None:
        """Return the grand mean, or ``None`` when nothing was recorded."""
        if not self._per_case:
            return None
        per_case = [math.fsum(values) / len(values) for values in self._per_case.values()]
        return math.fsum(per_case) / len(per_case)


def _micro_rate(edits: int, reference_words: int) -> float | None:
    """``spec.WER_AGGREGATION == "micro_global"``: total edits / total ref words.

    >>> _micro_rate(3, 12)
    0.25
    >>> _micro_rate(0, 0) is None
    True
    """
    if reference_words == 0:
        return None
    return edits / reference_words


def _concatenated_tokens(transcripts: Mapping[str, str]) -> list[str]:
    """Speaker-agnostic token stream, concatenated per
    :data:`spec.MULTI_SPEAKER_WER_CONCAT_ORDER`.

    ``"ascending_label"`` sorts the mapping keys as strings.  Track 1 has one
    slot, so the order is first visible in Track 2.

    >>> _concatenated_tokens({"b": "two", "a": "one"})
    ['one', 'two']
    """
    tokens: list[str] = []
    for label in sorted(transcripts):
        tokens.extend(tokenise(transcripts[label]))
    return tokens


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _scope_of_case(preds: Predictions, case_id: str) -> str:
    """Observation mode a case is reported under; the official one by default.

    A case with no ``final_transcripts`` record declares no mode, so it falls
    into the official scope -- omitting the record must not let a submission
    duck the official table.
    """
    return preds.observation_modes.get(case_id, spec.OFFICIAL_SCOPES[0])


def _score_scope(
    track: int,
    case_ids: Sequence[str],
    truth: GroundTruth,
    preds: Predictions,
) -> dict[str, Any]:
    """Score one group of cases and return its metric table."""
    macro = {name: MacroAccumulator() for name in MACRO_METRICS}
    wer_edits = 0
    wer_reference_words = 0
    cpwer_edits = 0
    cpwer_reference_words = 0
    scored_timepoints = 0
    missing_timepoints = 0
    per_case: dict[str, Any] = {}

    has_azimuth_probs = any(
        point.azimuth_probs is not None
        for case_id in case_ids
        for point in preds.points.get(case_id, {}).values()
    )
    has_distance_probs = any(
        point.distance_probs is not None
        for case_id in case_ids
        for point in preds.points.get(case_id, {}).values()
    )

    for case_id in case_ids:
        truth_points = truth.points.get(case_id, {})
        truth_order = truth.eval_times.get(case_id, [])
        pred_points = preds.points.get(case_id, {})
        reference_texts = truth.transcripts.get(case_id, {})
        hypothesis_texts = preds.transcripts.get(case_id, {})

        reference_labels = sorted(
            set(reference_texts)
            | {point.speaker for point in truth_points.values() if point.speaker}
        )
        predicted_labels = sorted(
            set(hypothesis_texts)
            | {point.speaker for point in pred_points.values() if point.speaker}
        )
        speaker_pairs = [
            (
                truth_points[key].speaker,
                pred_points[key].speaker if key in pred_points else None,
            )
            for key in truth_order
        ]
        assignment = solve_permutation(
            predicted_labels,
            reference_labels,
            hypothesis_texts,
            reference_texts,
            speaker_pairs,
        )

        # --- transcription (micro_global; a missing record is all deletions) --
        reference_tokens = _concatenated_tokens(reference_texts)
        hypothesis_tokens = _concatenated_tokens(hypothesis_texts)
        case_wer_edits = levenshtein(reference_tokens, hypothesis_tokens)
        wer_edits += case_wer_edits
        wer_reference_words += len(reference_tokens)
        case_cpwer_reference_words = sum(
            len(tokenise(reference_texts.get(label, spec.SPEAKER_PAD_TOKEN)))
            for label in reference_labels
        )
        cpwer_edits += assignment.cpwer_edits
        cpwer_reference_words += case_cpwer_reference_words

        if case_id in preds.rtf:
            macro[METRIC_RTF].add(case_id, preds.rtf[case_id])

        case_missing = 0
        for key in truth_order:
            target = truth_points[key]
            prediction = pred_points.get(key)
            scored_timepoints += 1
            if prediction is None:
                missing_timepoints += 1
                case_missing += 1

            if target.azimuth_deg is not None:
                azimuth, _ = resolve_azimuth_deg(prediction)
                error = (
                    spec.PENALTY_AZIMUTH_DEG
                    if azimuth is None
                    else spec.circular_error_deg(target.azimuth_deg, azimuth)
                )
                macro[METRIC_AZ_MAE].add(case_id, error)
                for name, threshold in AZIMUTH_ACC_METRICS:
                    macro[name].add(case_id, 1.0 if error <= threshold else 0.0)

            if target.distance_m is not None:
                distance, _ = resolve_distance_m(prediction)
                error = (
                    spec.PENALTY_DISTANCE_M
                    if distance is None
                    else abs(target.distance_m - distance)
                )
                macro[METRIC_R_MAE].add(case_id, error)

            target_xy = target.xy
            if target_xy is not None:
                predicted_xy, _ = resolve_xy(prediction)
                macro[METRIC_POS2D].add(
                    case_id,
                    PENALTY_POS_ERROR_2D_M
                    if predicted_xy is None
                    else euclidean_2d(target_xy, predicted_xy),
                )

            if has_azimuth_probs and target.azimuth_bin is not None:
                probs = None if prediction is None else prediction.azimuth_probs
                if probs is None:
                    macro[METRIC_AZ_NLL].add(case_id, spec.PENALTY_NLL)
                    macro[METRIC_AZ_COVERAGE].add(case_id, PENALTY_COVERAGE)
                    macro[METRIC_AZ_WIDTH].add(case_id, PENALTY_AZIMUTH_INTERVAL_WIDTH_DEG)
                else:
                    interval = hpd_set(probs)
                    macro[METRIC_AZ_NLL].add(
                        case_id, negative_log_likelihood(probs, target.azimuth_bin)
                    )
                    macro[METRIC_AZ_COVERAGE].add(
                        case_id, 1.0 if target.azimuth_bin in interval else 0.0
                    )
                    macro[METRIC_AZ_WIDTH].add(case_id, azimuth_interval_width_deg(interval))

            if has_distance_probs and target.distance_bin is not None:
                probs = None if prediction is None else prediction.distance_probs
                if probs is None:
                    macro[METRIC_R_NLL].add(case_id, spec.PENALTY_NLL)
                    macro[METRIC_R_COVERAGE].add(case_id, PENALTY_COVERAGE)
                    macro[METRIC_R_WIDTH].add(case_id, PENALTY_DISTANCE_INTERVAL_WIDTH_M)
                else:
                    interval = hpd_set(probs)
                    macro[METRIC_R_NLL].add(
                        case_id, negative_log_likelihood(probs, target.distance_bin)
                    )
                    macro[METRIC_R_COVERAGE].add(
                        case_id, 1.0 if target.distance_bin in interval else 0.0
                    )
                    macro[METRIC_R_WIDTH].add(case_id, distance_interval_width_m(interval))

            if target.speaker is not None:
                if prediction is None:
                    correct = bool(spec.PENALTY_SPEAKER_CORRECT)
                else:
                    correct = (
                        prediction.speaker is not None
                        and assignment.mapping.get(prediction.speaker) == target.speaker
                    )
                macro[METRIC_SPEAKER_ACC].add(case_id, 1.0 if correct else 0.0)

        per_case[case_id] = {
            "observation_mode": _scope_of_case(preds, case_id),
            "has_transcript_record": case_id in preds.transcripts,
            "n_eval_timepoints": len(truth_order),
            "n_missing_timepoints": case_missing,
            "speaker_permutation": dict(assignment.mapping),
            "unmatched_reference_speakers": list(assignment.unmatched_ref),
            "wer_edits": case_wer_edits,
            "wer_reference_words": len(reference_tokens),
            "cpwer_edits": assignment.cpwer_edits,
            "cpwer_reference_words": case_cpwer_reference_words,
            "speaker_match_count": assignment.match_count,
        }

    metrics: dict[str, float | None] = {}
    reasons: dict[str, str] = {}
    undefined = set(spec.TRACK_UNDEFINED_METRICS.get(track, ()))

    for name in spec.OFFICIAL_METRICS:
        if name in undefined:
            metrics[name] = None
            reasons[name] = (
                f"N/A by rule: spec.TRACK_UNDEFINED_METRICS[{track}] lists {name!r}"
            )
            continue
        if name in AZIMUTH_DISTRIBUTION_METRICS and not has_azimuth_probs:
            metrics[name] = None
            reasons[name] = f"{MISSING_PROBS_REASON}: no 'azimuth_probs' in any scored record"
            continue
        if name in DISTANCE_DISTRIBUTION_METRICS and not has_distance_probs:
            metrics[name] = None
            reasons[name] = f"{MISSING_PROBS_REASON}: no 'distance_probs' in any scored record"
            continue
        if name == METRIC_WER:
            value = _micro_rate(wer_edits, wer_reference_words)
            if value is None:
                reasons[name] = "N/A: the ground truth has no reference words in this scope"
            metrics[name] = value
            continue
        if name == METRIC_CPWER:
            value = _micro_rate(cpwer_edits, cpwer_reference_words)
            if value is None:
                reasons[name] = "N/A: the ground truth has no reference words in this scope"
            metrics[name] = value
            continue
        value = macro[name].result()
        if value is None:
            reasons[name] = (
                f"N/A: no scorable observation for {name!r} in this scope"
                if name != METRIC_RTF
                else "N/A: no case reported an 'rtf' value"
            )
        metrics[name] = value

    return {
        "n_cases": len(case_ids),
        "n_eval_timepoints": scored_timepoints,
        "n_missing_timepoints": missing_timepoints,
        "submitted_azimuth_distributions": has_azimuth_probs,
        "submitted_distance_distributions": has_distance_probs,
        "wer_edits": wer_edits,
        "wer_reference_words": wer_reference_words,
        "cpwer_edits": cpwer_edits,
        "cpwer_reference_words": cpwer_reference_words,
        "metrics": metrics,
        "na_reasons": reasons,
        "per_case": per_case,
    }


def score(truth: GroundTruth, preds: Predictions, track: int) -> dict[str, Any]:
    """Score a parsed submission against parsed ground truth.

    The official table covers every case whose declared observation mode is in
    :data:`spec.OFFICIAL_SCOPES`; the other modes get their own tables and are
    not mixed in, because :data:`spec.OFFICIAL_SCOPES` is ``("causal",)`` for
    this round.
    """
    if track not in spec.MAX_SPEAKERS_PER_TRACK:
        _reject(
            RULE_PARSE,
            f"unknown track {track!r}, expected one of {sorted(spec.MAX_SPEAKERS_PER_TRACK)}",
        )
    case_ids = truth.case_ids()
    by_scope: dict[str, list[str]] = {}
    for case_id in case_ids:
        by_scope.setdefault(_scope_of_case(preds, case_id), []).append(case_id)
    official_cases = [
        case_id for case_id in case_ids if _scope_of_case(preds, case_id) in spec.OFFICIAL_SCOPES
    ]
    official = _score_scope(track, official_cases, truth, preds)
    scopes = {
        mode: _score_scope(track, scope_cases, truth, preds)
        for mode, scope_cases in sorted(by_scope.items())
    }
    unknown_cases = sorted(set(preds.case_ids()) - set(case_ids))
    missing_transcripts = [
        case_id for case_id in case_ids if case_id not in preds.transcripts
    ]
    return {
        "spec_version": spec.SPEC_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "track": track,
        "official_scopes": list(spec.OFFICIAL_SCOPES),
        "rules": {
            "spatial_aggregation": spec.SPATIAL_AGGREGATION,
            "wer_aggregation": spec.WER_AGGREGATION,
            "missing_transcript_policy": spec.MISSING_TRANSCRIPT_POLICY,
            "interval_rule": spec.INTERVAL_RULE,
            "coverage_level": spec.COVERAGE_LEVEL,
            "permutation_tie_break": list(spec.PERMUTATION_TIE_BREAK),
            "text_normalisation": list(spec.TEXT_NORMALISATION),
            "normalise_submitted_probs": spec.NORMALISE_SUBMITTED_PROBS,
            "session_length_sec": spec.SESSION_LENGTH_SEC,
            "delta_t_sec": spec.DELTA_T_SEC,
        },
        "missing_penalties": {
            METRIC_AZ_MAE: spec.PENALTY_AZIMUTH_DEG,
            METRIC_R_MAE: spec.PENALTY_DISTANCE_M,
            METRIC_POS2D: PENALTY_POS_ERROR_2D_M,
            METRIC_AZ_NLL: spec.PENALTY_NLL,
            METRIC_R_NLL: spec.PENALTY_NLL,
            METRIC_AZ_COVERAGE: PENALTY_COVERAGE,
            METRIC_R_COVERAGE: PENALTY_COVERAGE,
            METRIC_AZ_WIDTH: PENALTY_AZIMUTH_INTERVAL_WIDTH_DEG,
            METRIC_R_WIDTH: PENALTY_DISTANCE_INTERVAL_WIDTH_M,
            METRIC_SPEAKER_ACC: bool(spec.PENALTY_SPEAKER_CORRECT),
        },
        "metrics": official["metrics"],
        "na_reasons": official["na_reasons"],
        "official": official,
        "scopes": scopes,
        "diagnostics": {
            "numpy_available": HAS_NUMPY,
            "n_ground_truth_cases": len(case_ids),
            "unknown_case_ids": unknown_cases,
            "cases_missing_transcripts": missing_transcripts,
        },
    }


def evaluate_files(
    pred_path: str | Path, gt_path: str | Path, track: int
) -> dict[str, Any]:
    """Load both files, validate them and return the score report.

    Raises :class:`SubmissionError` with a rule-id prefix on any violation.
    """
    if track not in spec.MAX_SPEAKERS_PER_TRACK:
        _reject(
            RULE_PARSE,
            f"unknown track {track!r}, expected one of {sorted(spec.MAX_SPEAKERS_PER_TRACK)}",
        )
    truth = load_ground_truth(Path(gt_path), track)
    preds = load_predictions(Path(pred_path), track)
    return score(truth, preds, track)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    >>> build_parser().prog
    'evaluate.py'
    """
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Official scorer for the Variable-Channel Spatial ASR benchmark.",
    )
    parser.add_argument("--pred", required=True, help="submission JSONL file")
    parser.add_argument("--gt", required=True, help="ground-truth JSONL file")
    parser.add_argument(
        "--track",
        required=True,
        type=int,
        choices=sorted(spec.MAX_SPEAKERS_PER_TRACK),
        help="benchmark track",
    )
    parser.add_argument("--out", default=None, help="also write the report JSON here")
    parser.add_argument(
        "--indent", type=int, default=2, help="JSON indentation of the report"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns 0 on success and 2 on a contract violation."""
    args = build_parser().parse_args(argv)
    try:
        report = evaluate_files(pred_path=args.pred, gt_path=args.gt, track=args.track)
    except SubmissionError as error:
        print(str(error), file=sys.stderr)
        return 2
    text = json.dumps(report, indent=args.indent, sort_keys=False, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
