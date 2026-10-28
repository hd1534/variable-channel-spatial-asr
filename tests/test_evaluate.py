"""Unit tests for the official evaluator.

Standard library only (``unittest`` + ``doctest``).  Every fixture is written
into a temporary directory inside the test, so the suite depends on no dataset
and no checkpoint and runs identically on any machine.

Run with::

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import doctest
import json
import math
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import evaluate  # noqa: E402
import spec  # noqa: E402

CASE = spec.make_case_id("sess_001", spec.EVAL_MASKS[0])
CASE_B = spec.make_case_id("sess_002", spec.EVAL_MASKS[0])
REF_SPEAKER = "spk0"
HYP_SPEAKER = "s0"

TRUE_AZIMUTH_DEG = 37.0
TRUE_DISTANCE_M = 1.6
TRUE_AZIMUTH_BIN = spec.azimuth_bin_index(TRUE_AZIMUTH_DEG)
TRUE_DISTANCE_BIN = spec.distance_bin_index(TRUE_DISTANCE_M)
TRUE_TEXT = "The quick brown fox, 42 times."
HYP_TEXT = "the quick brown fox 42 times"


def one_hot(length: int, index: int) -> list[float]:
    """Return a length-``length`` one-hot distribution with mass at ``index``.

    >>> one_hot(3, 1)
    [0.0, 1.0, 0.0]
    """
    vector = [0.0] * length
    vector[index] = 1.0
    return vector


def gt_timepoint(
    case_id: str = CASE,
    eval_time_sec: float = 0.0,
    azimuth_deg: float | None = TRUE_AZIMUTH_DEG,
    distance_m: float | None = TRUE_DISTANCE_M,
    speaker: str | None = REF_SPEAKER,
    **extra: Any,
) -> dict[str, Any]:
    """Build one ground-truth timepoint record."""
    record: dict[str, Any] = {
        "record_type": "timepoint",
        "case_id": case_id,
        "eval_time_sec": eval_time_sec,
        "azimuth_deg": azimuth_deg,
        "distance_m": distance_m,
        "speaker": speaker,
    }
    record.update(extra)
    return record


def gt_transcripts(
    case_id: str = CASE, transcripts: dict[str, str] | None = None
) -> dict[str, Any]:
    """Build one ground-truth ``final_transcripts`` record."""
    return {
        "record_type": "final_transcripts",
        "case_id": case_id,
        "final_transcripts": {REF_SPEAKER: TRUE_TEXT} if transcripts is None else transcripts,
    }


def pred_timepoint(
    case_id: str = CASE, eval_time_sec: float = 0.0, **fields: Any
) -> dict[str, Any]:
    """Build one prediction timepoint record from explicit field overrides."""
    record: dict[str, Any] = {
        "record_type": "timepoint",
        "case_id": case_id,
        "eval_time_sec": eval_time_sec,
        "pred_azimuth_deg": None,
        "pred_distance_m": None,
        "pred_x_m": None,
        "pred_y_m": None,
        "azimuth_probs": None,
        "distance_probs": None,
        "pred_speaker": HYP_SPEAKER,
    }
    record.update(fields)
    return record


def pred_transcripts(
    case_id: str = CASE, transcripts: dict[str, str] | None = None, **extra: Any
) -> dict[str, Any]:
    """Build one prediction ``final_transcripts`` record."""
    record: dict[str, Any] = {
        "record_type": "final_transcripts",
        "case_id": case_id,
        "final_transcripts": {HYP_SPEAKER: HYP_TEXT} if transcripts is None else transcripts,
    }
    record.update(extra)
    return record


def point_prediction(eval_time_sec: float = 0.0, **fields: Any) -> dict[str, Any]:
    """A perfect explicit-point prediction at ``eval_time_sec``."""
    return pred_timepoint(
        eval_time_sec=eval_time_sec,
        pred_azimuth_deg=TRUE_AZIMUTH_DEG,
        pred_distance_m=TRUE_DISTANCE_M,
        **fields,
    )


class EvaluatorTestCase(unittest.TestCase):
    """Base class that materialises JSONL fixtures and runs the evaluator."""

    def write_pair(
        self, gt_records: list[dict[str, Any]], pred_records: list[dict[str, Any]]
    ) -> tuple[Path, Path]:
        """Write both JSONL files into a fresh temporary directory."""
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        gt_path = root / "ground_truth.jsonl"
        pred_path = root / "predictions.jsonl"
        for path, records in ((gt_path, gt_records), (pred_path, pred_records)):
            path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
        return gt_path, pred_path

    def score(
        self,
        gt_records: list[dict[str, Any]],
        pred_records: list[dict[str, Any]],
        track: int = 1,
    ) -> dict[str, Any]:
        """Write the fixtures and return the official score report."""
        gt_path, pred_path = self.write_pair(gt_records, pred_records)
        return evaluate.evaluate_files(pred_path=pred_path, gt_path=gt_path, track=track)

    def metrics(
        self,
        gt_records: list[dict[str, Any]],
        pred_records: list[dict[str, Any]],
        track: int = 1,
    ) -> dict[str, Any]:
        """Shorthand for the official metric table."""
        return self.score(gt_records, pred_records, track=track)["metrics"]


class TestContractAlignment(EvaluatorTestCase):
    """The evaluator must cover exactly the contract's official metric set."""

    def test_every_official_metric_is_reported(self) -> None:
        report = self.score(
            [gt_transcripts(), gt_timepoint()],
            [pred_transcripts(), point_prediction()],
        )
        self.assertEqual(set(report["metrics"]), set(spec.OFFICIAL_METRICS))
        self.assertEqual(report["spec_version"], spec.SPEC_VERSION)

    def test_na_metrics_carry_a_reason(self) -> None:
        report = self.score(
            [gt_transcripts(), gt_timepoint()],
            [pred_transcripts(), point_prediction()],
        )
        for name, value in report["metrics"].items():
            if value is None:
                self.assertIn(name, report["na_reasons"], f"{name} is N/A without a reason")
                self.assertTrue(report["na_reasons"][name])

    def test_penalties_come_from_the_contract(self) -> None:
        self.assertEqual(spec.PENALTY_NLL, -math.log(spec.NLL_EPSILON))
        self.assertGreater(spec.PENALTY_NLL, math.log(spec.K_AZIMUTH))


class TestExplicitPointSubmission(EvaluatorTestCase):
    """(a) A perfect explicit-point submission scores exactly zero."""

    def setUp(self) -> None:
        self.report = self.score(
            [gt_transcripts(), gt_timepoint(eval_time_sec=0.0), gt_timepoint(eval_time_sec=0.25)],
            [
                pred_transcripts(),
                point_prediction(eval_time_sec=0.0),
                point_prediction(eval_time_sec=0.25),
            ],
        )

    def test_wer_is_zero(self) -> None:
        self.assertEqual(self.report["metrics"]["wer"], 0.0)
        self.assertEqual(self.report["official"]["wer_edits"], 0)
        self.assertGreater(self.report["official"]["wer_reference_words"], 0)

    def test_point_errors_are_zero(self) -> None:
        metrics = self.report["metrics"]
        self.assertEqual(metrics["azimuth_mae_deg"], 0.0)
        self.assertEqual(metrics["distance_mae_m"], 0.0)
        self.assertEqual(metrics["pos_error_2d_m"], 0.0)
        self.assertEqual(metrics["azimuth_acc_10deg"], 1.0)
        self.assertEqual(metrics["azimuth_acc_20deg"], 1.0)
        self.assertEqual(metrics["speaker_attribution_accuracy"], 1.0)

    def test_cpwer_is_na_by_rule_in_track_1(self) -> None:
        self.assertIsNone(self.report["metrics"]["cpwer"])
        self.assertIn("TRACK_UNDEFINED_METRICS", self.report["na_reasons"]["cpwer"])

    def test_cartesian_point_fields_are_accepted(self) -> None:
        x_m, y_m = evaluate.polar_to_xy(TRUE_DISTANCE_M, TRUE_AZIMUTH_DEG)
        metrics = self.metrics(
            [gt_transcripts(), gt_timepoint()],
            [pred_transcripts(), pred_timepoint(pred_x_m=x_m, pred_y_m=y_m)],
        )
        self.assertEqual(metrics["pos_error_2d_m"], 0.0)
        self.assertAlmostEqual(metrics["azimuth_mae_deg"], 0.0, places=12)
        self.assertAlmostEqual(metrics["distance_mae_m"], 0.0, places=12)


class TestMapRepresentativePoint(EvaluatorTestCase):
    """(b) One-hot distributions only: NLL is 0 and the point error is the
    quantisation error between the true coordinate and its bin center."""

    def setUp(self) -> None:
        self.report = self.score(
            [gt_transcripts(), gt_timepoint()],
            [
                pred_transcripts(),
                pred_timepoint(
                    azimuth_probs=one_hot(spec.K_AZIMUTH, TRUE_AZIMUTH_BIN),
                    distance_probs=one_hot(spec.K_DISTANCE, TRUE_DISTANCE_BIN),
                ),
            ],
        )
        self.center_azimuth = spec.AZIMUTH_BIN_CENTERS_DEG[TRUE_AZIMUTH_BIN]
        self.center_distance = spec.DISTANCE_BIN_CENTERS_M[TRUE_DISTANCE_BIN]

    def test_nll_is_zero_on_a_correct_one_hot(self) -> None:
        self.assertEqual(self.report["metrics"]["azimuth_nll"], 0.0)
        self.assertEqual(self.report["metrics"]["distance_nll"], 0.0)

    def test_azimuth_point_error_equals_quantisation_error(self) -> None:
        expected = spec.circular_error_deg(TRUE_AZIMUTH_DEG, self.center_azimuth)
        self.assertAlmostEqual(self.report["metrics"]["azimuth_mae_deg"], expected, places=12)
        self.assertLessEqual(expected, spec.AZIMUTH_BIN_WIDTH_DEG / 2.0)

    def test_distance_point_error_equals_quantisation_error(self) -> None:
        expected = abs(TRUE_DISTANCE_M - self.center_distance)
        self.assertAlmostEqual(self.report["metrics"]["distance_mae_m"], expected, places=12)
        self.assertLessEqual(expected, spec.DISTANCE_BIN_WIDTHS_M[TRUE_DISTANCE_BIN])

    def test_2d_error_equals_the_bin_center_offset(self) -> None:
        expected = evaluate.euclidean_2d(
            evaluate.polar_to_xy(TRUE_DISTANCE_M, TRUE_AZIMUTH_DEG),
            evaluate.polar_to_xy(self.center_distance, self.center_azimuth),
        )
        self.assertAlmostEqual(self.report["metrics"]["pos_error_2d_m"], expected, places=12)

    def test_one_hot_interval_is_a_single_bin(self) -> None:
        metrics = self.report["metrics"]
        self.assertEqual(metrics["azimuth_coverage_90"], 1.0)
        self.assertEqual(metrics["distance_coverage_90"], 1.0)
        self.assertAlmostEqual(
            metrics["azimuth_interval_width_deg"], spec.AZIMUTH_BIN_WIDTH_DEG, places=12
        )
        self.assertAlmostEqual(
            metrics["distance_interval_width_m"],
            spec.DISTANCE_BIN_WIDTHS_M[TRUE_DISTANCE_BIN],
            places=12,
        )


class TestAzimuthWraparound(EvaluatorTestCase):
    """(c) The circular boundary: -179 vs +179 is 2 degrees and +180 is -180."""

    def test_opposite_signs_near_the_seam(self) -> None:
        metrics = self.metrics(
            [gt_transcripts(), gt_timepoint(azimuth_deg=-179.0)],
            [pred_transcripts(), pred_timepoint(pred_azimuth_deg=179.0, pred_distance_m=TRUE_DISTANCE_M)],
        )
        self.assertEqual(metrics["azimuth_mae_deg"], 2.0)
        self.assertEqual(metrics["azimuth_acc_10deg"], 1.0)

    def test_plus_180_lands_in_the_minus_180_bin(self) -> None:
        self.assertEqual(spec.azimuth_bin_index(180.0), spec.azimuth_bin_index(-180.0))
        self.assertEqual(spec.azimuth_bin_index(180.0), 0)
        self.assertEqual(evaluate.wrap_azimuth_deg(180.0), -180.0)

    def test_plus_180_truth_is_scored_in_bin_zero(self) -> None:
        metrics = self.metrics(
            [gt_transcripts(), gt_timepoint(azimuth_deg=180.0)],
            [
                pred_transcripts(),
                pred_timepoint(
                    pred_azimuth_deg=-180.0,
                    pred_distance_m=TRUE_DISTANCE_M,
                    azimuth_probs=one_hot(spec.K_AZIMUTH, 0),
                ),
            ],
        )
        self.assertEqual(metrics["azimuth_mae_deg"], 0.0)
        self.assertEqual(metrics["azimuth_nll"], 0.0)
        self.assertEqual(metrics["azimuth_coverage_90"], 1.0)

    def test_error_is_symmetric_across_the_seam(self) -> None:
        self.assertEqual(spec.circular_error_deg(-179.0, 179.0), 2.0)
        self.assertEqual(spec.circular_error_deg(179.0, -179.0), 2.0)
        self.assertEqual(spec.circular_error_deg(180.0, -180.0), 0.0)


class TestDuplicateRejection(EvaluatorTestCase):
    """(d) A repeated key rejects the whole file, with a rule-id message."""

    def test_submission_error_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(evaluate.SubmissionError, ValueError))

    def test_duplicate_case_id_in_predictions(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.score(
                [gt_transcripts(), gt_timepoint()],
                [pred_transcripts(), pred_transcripts(), point_prediction()],
            )
        message = str(caught.exception)
        self.assertTrue(message.startswith(evaluate.RULE_DUPLICATE_CASE), message)
        self.assertEqual(message, f"R-DUP-01: duplicate case_id '{CASE}'")

    def test_duplicate_timepoint_in_predictions(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.score(
                [gt_transcripts(), gt_timepoint()],
                [pred_transcripts(), point_prediction(0.0), point_prediction(0.0)],
            )
        message = str(caught.exception)
        self.assertTrue(message.startswith(evaluate.RULE_DUPLICATE_TIME), message)
        self.assertEqual(message, f"R-DUP-02: duplicate timepoint ('{CASE}', 0.0)")

    def test_duplicate_case_id_in_ground_truth(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.score(
                [gt_transcripts(), gt_transcripts(), gt_timepoint()],
                [pred_transcripts(), point_prediction()],
            )
        self.assertTrue(str(caught.exception).startswith(evaluate.RULE_DUPLICATE_CASE))

    def test_duplicate_timepoint_in_ground_truth(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.score(
                [gt_transcripts(), gt_timepoint(eval_time_sec=0.25), gt_timepoint(eval_time_sec=0.25)],
                [pred_transcripts(), point_prediction(0.25)],
            )
        self.assertTrue(str(caught.exception).startswith(evaluate.RULE_DUPLICATE_TIME))

    def test_no_arbitrary_winner_is_picked(self) -> None:
        """Both duplicates are rejected rather than one being silently chosen."""
        first = pred_transcripts(transcripts={HYP_SPEAKER: HYP_TEXT})
        second = pred_transcripts(transcripts={HYP_SPEAKER: "completely different words"})
        with self.assertRaises(evaluate.SubmissionError):
            self.score([gt_transcripts(), gt_timepoint()], [first, second, point_prediction()])


class TestProbabilityValidation(EvaluatorTestCase):
    """The R-PROB family rejects malformed distributions without repairing them."""

    def _expect(self, rule: str, probs: Any) -> None:
        with self.assertRaises(evaluate.SubmissionError) as caught:
            self.score(
                [gt_transcripts(), gt_timepoint()],
                [pred_transcripts(), pred_timepoint(azimuth_probs=probs)],
            )
        self.assertTrue(str(caught.exception).startswith(rule), str(caught.exception))

    def test_wrong_length(self) -> None:
        self._expect(evaluate.RULE_PROB_LENGTH, one_hot(spec.K_AZIMUTH - 1, 0))

    def test_sum_outside_tolerance(self) -> None:
        self._expect(evaluate.RULE_PROB_SUM, [0.5] + [0.0] * (spec.K_AZIMUTH - 1))

    def test_negative_value(self) -> None:
        bad = one_hot(spec.K_AZIMUTH, 0)
        bad[1] = -0.25
        bad[2] = 0.25
        self._expect(evaluate.RULE_PROB_VALUE, bad)

    def test_value_above_one(self) -> None:
        bad = one_hot(spec.K_AZIMUTH, 0)
        bad[0] = 1.25
        bad[1] = -0.25
        self._expect(evaluate.RULE_PROB_VALUE, bad)

    def test_tolerance_is_the_contract_value(self) -> None:
        inside = one_hot(spec.K_AZIMUTH, TRUE_AZIMUTH_BIN)
        inside[TRUE_AZIMUTH_BIN] = 1.0 - spec.PROB_SUM_TOLERANCE / 2.0
        metrics = self.metrics(
            [gt_transcripts(), gt_timepoint()],
            [pred_transcripts(), pred_timepoint(azimuth_probs=inside)],
        )
        self.assertIsNotNone(metrics["azimuth_nll"])
        self.assertGreater(metrics["azimuth_nll"], 0.0)

    def test_submitted_mass_is_not_renormalised(self) -> None:
        """A slightly light vector keeps its own mass, so the NLL is not 0."""
        self.assertFalse(spec.NORMALISE_SUBMITTED_PROBS)
        light = one_hot(spec.K_AZIMUTH, TRUE_AZIMUTH_BIN)
        light[TRUE_AZIMUTH_BIN] = 1.0 - spec.PROB_SUM_TOLERANCE / 2.0
        metrics = self.metrics(
            [gt_transcripts(), gt_timepoint()],
            [pred_transcripts(), pred_timepoint(azimuth_probs=light)],
        )
        expected = -math.log(1.0 - spec.PROB_SUM_TOLERANCE / 2.0)
        self.assertAlmostEqual(metrics["azimuth_nll"], expected, places=15)


class TestMissingPredictionPenalty(EvaluatorTestCase):
    """(e) Omitted time steps take spec.PENALTY_*, and omission never pays."""

    GT = [
        gt_transcripts(),
        gt_timepoint(eval_time_sec=0.0),
        gt_timepoint(eval_time_sec=0.25),
    ]

    def _perfect_first_step(self) -> dict[str, Any]:
        return point_prediction(
            eval_time_sec=0.0,
            azimuth_probs=one_hot(spec.K_AZIMUTH, TRUE_AZIMUTH_BIN),
            distance_probs=one_hot(spec.K_DISTANCE, TRUE_DISTANCE_BIN),
        )

    def setUp(self) -> None:
        self.omitted = self.metrics(
            self.GT, [pred_transcripts(), self._perfect_first_step()]
        )

    def test_half_of_each_contract_penalty_is_charged(self) -> None:
        self.assertAlmostEqual(
            self.omitted["azimuth_mae_deg"], spec.PENALTY_AZIMUTH_DEG / 2.0, places=12
        )
        self.assertAlmostEqual(
            self.omitted["distance_mae_m"], spec.PENALTY_DISTANCE_M / 2.0, places=12
        )
        self.assertAlmostEqual(
            self.omitted["pos_error_2d_m"], evaluate.PENALTY_POS_ERROR_2D_M / 2.0, places=12
        )
        self.assertAlmostEqual(self.omitted["azimuth_nll"], spec.PENALTY_NLL / 2.0, places=12)
        self.assertAlmostEqual(self.omitted["distance_nll"], spec.PENALTY_NLL / 2.0, places=12)

    def test_missing_speaker_takes_the_contract_value(self) -> None:
        self.assertFalse(spec.PENALTY_SPEAKER_CORRECT)
        self.assertAlmostEqual(self.omitted["speaker_attribution_accuracy"], 0.5, places=12)

    def test_missing_coverage_and_width_are_worst_case(self) -> None:
        self.assertAlmostEqual(self.omitted["azimuth_coverage_90"], 0.5, places=12)
        self.assertAlmostEqual(
            self.omitted["azimuth_interval_width_deg"],
            (spec.AZIMUTH_BIN_WIDTH_DEG + evaluate.AZIMUTH_SUPPORT_WIDTH_DEG) / 2.0,
            places=12,
        )
        self.assertAlmostEqual(
            self.omitted["distance_interval_width_m"],
            (
                spec.DISTANCE_BIN_WIDTHS_M[TRUE_DISTANCE_BIN]
                + evaluate.DISTANCE_SUPPORT_WIDTH_M
            )
            / 2.0,
            places=12,
        )
        self.assertEqual(self.omitted["azimuth_acc_10deg"], 0.5)

    def test_report_counts_the_missing_step(self) -> None:
        report = self.score(self.GT, [pred_transcripts(), self._perfect_first_step()])
        self.assertEqual(report["official"]["n_eval_timepoints"], 2)
        self.assertEqual(report["official"]["n_missing_timepoints"], 1)

    def test_a_confidently_wrong_distribution_is_no_worse_than_omitting(self) -> None:
        """The core anti-gaming property of spec.PENALTY_NLL."""
        wrong_bin = (TRUE_AZIMUTH_BIN + spec.K_AZIMUTH // 2) % spec.K_AZIMUTH
        wrong_distance_bin = (TRUE_DISTANCE_BIN + 1) % spec.K_DISTANCE
        confident = self.metrics(
            self.GT,
            [
                pred_transcripts(),
                self._perfect_first_step(),
                pred_timepoint(
                    eval_time_sec=0.25,
                    pred_azimuth_deg=spec.AZIMUTH_BIN_CENTERS_DEG[wrong_bin],
                    pred_distance_m=spec.DISTANCE_RANGE_M[1],
                    azimuth_probs=one_hot(spec.K_AZIMUTH, wrong_bin),
                    distance_probs=one_hot(spec.K_DISTANCE, wrong_distance_bin),
                ),
            ],
        )
        self.assertLessEqual(confident["azimuth_nll"], self.omitted["azimuth_nll"])
        self.assertLessEqual(confident["distance_nll"], self.omitted["distance_nll"])
        self.assertLessEqual(confident["azimuth_mae_deg"], self.omitted["azimuth_mae_deg"])
        self.assertLessEqual(confident["distance_mae_m"], self.omitted["distance_mae_m"])
        self.assertLessEqual(confident["pos_error_2d_m"], self.omitted["pos_error_2d_m"])
        # a one-hot on the wrong bin hits the clipping floor exactly
        self.assertAlmostEqual(confident["azimuth_nll"], spec.PENALTY_NLL / 2.0, places=12)

    def test_a_hedged_wrong_distribution_beats_omitting(self) -> None:
        wrong_bin = (TRUE_AZIMUTH_BIN + spec.K_AZIMUTH // 2) % spec.K_AZIMUTH
        leak = 1e-9
        hedged = one_hot(spec.K_AZIMUTH, wrong_bin)
        hedged[wrong_bin] = 1.0 - leak
        hedged[TRUE_AZIMUTH_BIN] = leak
        metrics = self.metrics(
            self.GT,
            [
                pred_transcripts(),
                self._perfect_first_step(),
                pred_timepoint(eval_time_sec=0.25, azimuth_probs=hedged),
            ],
        )
        self.assertLess(metrics["azimuth_nll"], self.omitted["azimuth_nll"])
        self.assertAlmostEqual(metrics["azimuth_nll"], -math.log(leak) / 2.0, places=12)

    def test_a_case_with_no_predictions_at_all_is_still_scored(self) -> None:
        metrics = self.metrics(self.GT, [])
        self.assertAlmostEqual(metrics["azimuth_mae_deg"], spec.PENALTY_AZIMUTH_DEG, places=12)
        self.assertAlmostEqual(metrics["distance_mae_m"], spec.PENALTY_DISTANCE_M, places=12)
        self.assertEqual(metrics["wer"], 1.0)


class TestPointOnlySubmission(EvaluatorTestCase):
    """(f) Without distributions the distributional metrics are JSON null."""

    def setUp(self) -> None:
        self.report = self.score(
            [gt_transcripts(), gt_timepoint()],
            [pred_transcripts(), point_prediction()],
        )

    def test_distribution_metrics_are_none(self) -> None:
        for name in (
            evaluate.AZIMUTH_DISTRIBUTION_METRICS + evaluate.DISTANCE_DISTRIBUTION_METRICS
        ):
            self.assertIsNone(self.report["metrics"][name], name)

    def test_each_one_carries_the_point_only_reason(self) -> None:
        for name in (
            evaluate.AZIMUTH_DISTRIBUTION_METRICS + evaluate.DISTANCE_DISTRIBUTION_METRICS
        ):
            self.assertIn(evaluate.MISSING_PROBS_REASON, self.report["na_reasons"][name])

    def test_point_metrics_are_still_numbers(self) -> None:
        self.assertIsNotNone(self.report["metrics"]["azimuth_mae_deg"])
        self.assertIsNotNone(self.report["metrics"]["pos_error_2d_m"])

    def test_json_round_trip_keeps_null(self) -> None:
        reloaded = json.loads(json.dumps(self.report))
        self.assertIsNone(reloaded["metrics"]["azimuth_nll"])

    def test_one_sided_distributions_are_reported_independently(self) -> None:
        report = self.score(
            [gt_transcripts(), gt_timepoint()],
            [
                pred_transcripts(),
                point_prediction(azimuth_probs=one_hot(spec.K_AZIMUTH, TRUE_AZIMUTH_BIN)),
            ],
        )
        self.assertEqual(report["metrics"]["azimuth_nll"], 0.0)
        self.assertIsNone(report["metrics"]["distance_nll"])


def two_speaker_gt(
    reference: dict[str, str], speakers: list[str]
) -> list[dict[str, Any]]:
    """Track-2 ground truth: one transcript record plus one step per speaker."""
    records: list[dict[str, Any]] = [gt_transcripts(CASE, reference)]
    for index, speaker in enumerate(speakers):
        records.append(
            gt_timepoint(CASE, eval_time_sec=index * spec.DELTA_T_SEC, speaker=speaker)
        )
    return records


def two_speaker_pred(
    hypothesis: dict[str, str], speakers: list[str | None]
) -> list[dict[str, Any]]:
    """Track-2 submission mirroring :func:`two_speaker_gt`."""
    records: list[dict[str, Any]] = [pred_transcripts(CASE, hypothesis)]
    for index, speaker in enumerate(speakers):
        records.append(
            point_prediction(eval_time_sec=index * spec.DELTA_T_SEC, pred_speaker=speaker)
        )
    return records


class TestPermutationTieBreak(EvaluatorTestCase):
    """(g) Identical transcripts tie on cpWER; the 3-stage rule still decides."""

    REFERENCE = {"spkA": "hello world", "spkB": "hello world"}
    HYPOTHESIS = {"s0": "hello world", "s1": "hello world"}

    def _permutation(self, truth_speakers: list[str], pred_speakers: list[str | None]):
        report = self.score(
            two_speaker_gt(self.REFERENCE, truth_speakers),
            two_speaker_pred(self.HYPOTHESIS, pred_speakers),
            track=2,
        )
        return report, report["official"]["per_case"][CASE]

    def test_tie_break_stage_1_leaves_a_tie(self) -> None:
        """Both assignments give cpWER 0, so stage 1 cannot decide alone."""
        self.assertEqual(spec.PERMUTATION_TIE_BREAK[0], "min_cpwer")
        left = evaluate.solve_permutation(
            ["s0", "s1"], ["spkA", "spkB"], self.HYPOTHESIS, self.REFERENCE, []
        )
        swapped = evaluate.levenshtein(
            evaluate.tokenise(self.REFERENCE["spkB"]), evaluate.tokenise(self.HYPOTHESIS["s0"])
        )
        self.assertEqual(left.cpwer_edits, 0)
        self.assertEqual(swapped, 0)

    def test_stage_2_picks_the_assignment_with_more_matching_steps(self) -> None:
        report, case = self._permutation(["spkA", "spkA", "spkB"], ["s0", "s0", "s1"])
        self.assertEqual(case["speaker_permutation"], {"s0": "spkA", "s1": "spkB"})
        self.assertEqual(case["speaker_match_count"], 3)
        self.assertEqual(report["metrics"]["cpwer"], 0.0)
        self.assertEqual(report["metrics"]["speaker_attribution_accuracy"], 1.0)

    def test_stage_2_overrides_the_lexicographic_default(self) -> None:
        """Proof that stage 2 runs before stage 3: the winner is the swap."""
        _report, case = self._permutation(["spkB", "spkB", "spkA"], ["s0", "s0", "s1"])
        self.assertEqual(case["speaker_permutation"], {"s0": "spkB", "s1": "spkA"})
        self.assertEqual(case["speaker_match_count"], 3)

    def test_stage_3_decides_when_stages_1_and_2_tie(self) -> None:
        self.assertEqual(spec.PERMUTATION_TIE_BREAK[2], "lexicographic")
        _report, case = self._permutation(["spkA", "spkB"], [None, None])
        self.assertEqual(case["speaker_match_count"], 0)
        self.assertEqual(case["speaker_permutation"], {"s0": "spkA", "s1": "spkB"})

    def test_the_choice_is_reproducible(self) -> None:
        first = self._permutation(["spkA", "spkB"], [None, None])[1]
        second = self._permutation(["spkA", "spkB"], [None, None])[1]
        self.assertEqual(first["speaker_permutation"], second["speaker_permutation"])

    def test_one_permutation_per_case_not_per_timepoint(self) -> None:
        """A single pi* for the whole case, so a swap mid-session costs accuracy."""
        _report, case = self._permutation(["spkA", "spkB"], ["s1", "s0"])
        mapping = case["speaker_permutation"]
        self.assertEqual(len(mapping), 2)
        self.assertEqual(set(mapping.values()), {"spkA", "spkB"})


class TestSpeakerCountMismatch(EvaluatorTestCase):
    """(h) Padding keeps pi* defined when the counts differ, both ways."""

    def test_more_predicted_than_reference_counts_insertions(self) -> None:
        report = self.score(
            two_speaker_gt({"spkA": "hello world"}, ["spkA"]),
            two_speaker_pred({"s0": "hello world", "s1": "extra noise"}, ["s0"]),
            track=2,
        )
        case = report["official"]["per_case"][CASE]
        self.assertEqual(case["speaker_permutation"], {"s0": "spkA", "s1": None})
        self.assertEqual(case["unmatched_reference_speakers"], [])
        self.assertEqual(case["cpwer_reference_words"], 2)
        self.assertEqual(case["cpwer_edits"], 2)  # the two unpaired words
        self.assertEqual(report["metrics"]["cpwer"], 1.0)
        self.assertEqual(report["metrics"]["speaker_attribution_accuracy"], 1.0)

    def test_fewer_predicted_than_reference_counts_deletions(self) -> None:
        report = self.score(
            two_speaker_gt({"spkA": "hello world", "spkB": "good morning"}, ["spkA"]),
            two_speaker_pred({"s0": "hello world"}, ["s0"]),
            track=2,
        )
        case = report["official"]["per_case"][CASE]
        self.assertEqual(case["speaker_permutation"], {"s0": "spkA"})
        self.assertEqual(case["unmatched_reference_speakers"], ["spkB"])
        self.assertEqual(case["cpwer_reference_words"], 4)
        self.assertEqual(case["cpwer_edits"], 2)  # the two unpaired reference words
        self.assertEqual(report["metrics"]["cpwer"], 0.5)

    def test_pad_token_is_the_contract_value(self) -> None:
        self.assertEqual(spec.SPEAKER_PAD_TOKEN, "")
        self.assertEqual(evaluate.tokenise(spec.SPEAKER_PAD_TOKEN), [])

    def test_too_many_speaker_slots_are_rejected(self) -> None:
        allowed = spec.MAX_SPEAKERS_PER_TRACK[1]
        too_many = {f"s{index}": "hello" for index in range(allowed + 1)}
        with self.assertRaises(evaluate.SubmissionError) as caught:
            self.score(
                [gt_transcripts(), gt_timepoint()],
                [pred_transcripts(transcripts=too_many), point_prediction(pred_speaker="s0")],
            )
        self.assertTrue(str(caught.exception).startswith(evaluate.RULE_SPEAKER_COUNT))


class TestMissingTranscriptRecord(EvaluatorTestCase):
    """(i) A case with no transcript record charges every reference word."""

    def setUp(self) -> None:
        self.report = self.score([gt_transcripts(), gt_timepoint()], [point_prediction()])
        self.reference_words = len(evaluate.tokenise(TRUE_TEXT))

    def test_policy_is_the_contract_one(self) -> None:
        self.assertEqual(spec.MISSING_TRANSCRIPT_POLICY, "count_all_deletions")

    def test_every_reference_word_is_a_deletion(self) -> None:
        self.assertEqual(self.report["official"]["wer_edits"], self.reference_words)
        self.assertEqual(
            self.report["official"]["wer_reference_words"], self.reference_words
        )
        self.assertEqual(self.report["metrics"]["wer"], 1.0)

    def test_the_case_is_not_dropped(self) -> None:
        self.assertEqual(self.report["official"]["n_cases"], 1)
        self.assertEqual(self.report["diagnostics"]["cases_missing_transcripts"], [CASE])
        self.assertFalse(self.report["official"]["per_case"][CASE]["has_transcript_record"])

    def test_spatial_metrics_are_still_scored(self) -> None:
        self.assertEqual(self.report["metrics"]["azimuth_mae_deg"], 0.0)
        self.assertEqual(self.report["metrics"]["distance_mae_m"], 0.0)

    def test_a_case_without_the_record_stays_in_the_official_scope(self) -> None:
        self.assertEqual(
            self.report["official"]["per_case"][CASE]["observation_mode"],
            spec.OFFICIAL_SCOPES[0],
        )


def bimodal(length: int, first: int, second: int, peak: float = 0.46) -> list[float]:
    """Two equal peaks plus a flat floor over the remaining bins.

    >>> round(sum(bimodal(4, 0, 2)), 10)
    1.0
    """
    vector = [(1.0 - 2.0 * peak) / (length - 2)] * length
    vector[first] = peak
    vector[second] = peak
    return vector


class TestHpdInterval(EvaluatorTestCase):
    """(j) A bimodal posterior yields a non-contiguous HPD set."""

    AZIMUTH_PEAKS = (0, spec.K_AZIMUTH // 2)
    DISTANCE_PEAKS = (0, spec.K_DISTANCE - 1)

    def test_rule_is_the_contract_one(self) -> None:
        self.assertEqual(spec.INTERVAL_RULE, "hpd_greedy")
        self.assertEqual(spec.COVERAGE_LEVEL, 0.90)

    def test_set_is_exactly_the_two_peaks(self) -> None:
        probs = bimodal(spec.K_AZIMUTH, *self.AZIMUTH_PEAKS)
        self.assertEqual(evaluate.hpd_set(probs), self.AZIMUTH_PEAKS)

    def test_set_is_not_contiguous(self) -> None:
        chosen = evaluate.hpd_set(bimodal(spec.K_AZIMUTH, *self.AZIMUTH_PEAKS))
        span = chosen[-1] - chosen[0] + 1
        self.assertGreater(span, len(chosen))

    def test_width_is_the_sum_of_the_two_peak_bins(self) -> None:
        metrics = self.metrics(
            [gt_transcripts(), gt_timepoint(azimuth_deg=-175.0, distance_m=0.6)],
            [
                pred_transcripts(),
                pred_timepoint(
                    azimuth_probs=bimodal(spec.K_AZIMUTH, *self.AZIMUTH_PEAKS),
                    distance_probs=bimodal(spec.K_DISTANCE, *self.DISTANCE_PEAKS),
                ),
            ],
        )
        self.assertAlmostEqual(
            metrics["azimuth_interval_width_deg"], 2.0 * spec.AZIMUTH_BIN_WIDTH_DEG, places=12
        )
        self.assertAlmostEqual(
            metrics["distance_interval_width_m"],
            spec.DISTANCE_BIN_WIDTHS_M[self.DISTANCE_PEAKS[0]]
            + spec.DISTANCE_BIN_WIDTHS_M[self.DISTANCE_PEAKS[1]],
            places=12,
        )
        self.assertEqual(metrics["azimuth_coverage_90"], 1.0)
        self.assertEqual(metrics["distance_coverage_90"], 1.0)

    def test_coverage_is_zero_when_the_truth_misses_both_peaks(self) -> None:
        metrics = self.metrics(
            [gt_transcripts(), gt_timepoint(azimuth_deg=-5.0, distance_m=TRUE_DISTANCE_M)],
            [
                pred_transcripts(),
                pred_timepoint(azimuth_probs=bimodal(spec.K_AZIMUTH, *self.AZIMUTH_PEAKS)),
            ],
        )
        self.assertEqual(metrics["azimuth_coverage_90"], 0.0)

    def test_ties_resolve_toward_the_smaller_index(self) -> None:
        flat = [1.0 / spec.K_AZIMUTH] * spec.K_AZIMUTH
        chosen = evaluate.hpd_set(flat)
        self.assertEqual(chosen, tuple(range(len(chosen))))
        self.assertGreaterEqual(len(chosen) / spec.K_AZIMUTH, spec.COVERAGE_LEVEL)


class TestTextNormalisation(EvaluatorTestCase):
    """spec.TEXT_NORMALISATION is applied to both sides, in order."""

    def test_pipeline_is_the_contract_one(self) -> None:
        self.assertEqual(
            spec.TEXT_NORMALISATION,
            ("unicode_nfkc", "casefold", "strip_punctuation", "digits_to_words",
             "collapse_whitespace"),
        )

    def test_digits_and_words_match(self) -> None:
        metrics = self.metrics(
            [gt_transcripts(transcripts={REF_SPEAKER: "channel 2 of 4"}), gt_timepoint()],
            [
                pred_transcripts(transcripts={HYP_SPEAKER: "CHANNEL two, of FOUR!"}),
                point_prediction(),
            ],
        )
        self.assertEqual(metrics["wer"], 0.0)

    def test_punctuation_is_stripped_after_casefolding(self) -> None:
        self.assertEqual(evaluate.tokenise("It's -- A/B test."), ["its", "a", "b", "test"])

    def test_decimal_point_is_removed_before_digits_are_spelled(self) -> None:
        self.assertEqual(evaluate.normalise_text("3.5"), "thirty five")

    def test_numbers_up_to_999(self) -> None:
        self.assertEqual(evaluate.integer_to_words(999), ["nine", "hundred", "ninety", "nine"])
        self.assertEqual(evaluate.integer_to_words(100), ["one", "hundred"])
        self.assertEqual(evaluate.integer_to_words(20), ["twenty"])


class TestObservationScopes(EvaluatorTestCase):
    """Only causal is official; other modes get their own table."""

    def test_non_causal_cases_leave_the_official_table(self) -> None:
        report = self.score(
            [
                gt_transcripts(CASE),
                gt_timepoint(CASE),
                gt_transcripts(CASE_B),
                gt_timepoint(CASE_B),
            ],
            [
                pred_transcripts(CASE, observation_mode="causal", rtf=0.5),
                point_prediction(),
                pred_transcripts(CASE_B, observation_mode="windowed", rtf=1.5),
                point_prediction(case_id=CASE_B),
            ],
        )
        self.assertEqual(spec.OFFICIAL_SCOPES, ("causal",))
        self.assertEqual(report["official"]["n_cases"], 1)
        self.assertEqual(sorted(report["scopes"]), ["causal", "windowed"])
        self.assertEqual(report["scopes"]["windowed"]["n_cases"], 1)
        self.assertEqual(report["metrics"]["rtf"], 0.5)
        self.assertEqual(report["scopes"]["windowed"]["metrics"]["rtf"], 1.5)

    def test_unknown_observation_mode_is_rejected(self) -> None:
        with self.assertRaises(evaluate.SubmissionError) as caught:
            self.score(
                [gt_transcripts(), gt_timepoint()],
                [pred_transcripts(observation_mode="oracle"), point_prediction()],
            )
        self.assertTrue(str(caught.exception).startswith(evaluate.RULE_PARSE))

    def test_unknown_case_ids_are_reported_not_scored(self) -> None:
        report = self.score(
            [gt_transcripts(CASE), gt_timepoint(CASE)],
            [pred_transcripts(CASE), point_prediction(), point_prediction(case_id=CASE_B)],
        )
        self.assertEqual(report["diagnostics"]["unknown_case_ids"], [CASE_B])
        self.assertEqual(report["official"]["n_cases"], 1)

    def test_rtf_is_na_when_nobody_reports_it(self) -> None:
        report = self.score([gt_transcripts(), gt_timepoint()], [pred_transcripts(), point_prediction()])
        self.assertIsNone(report["metrics"]["rtf"])
        self.assertIn("rtf", report["na_reasons"])


class TestCommandLine(EvaluatorTestCase):
    """The documented CLI runs end to end and writes a JSON report."""

    def test_successful_run_writes_the_report(self) -> None:
        gt_path, pred_path = self.write_pair(
            [gt_transcripts(), gt_timepoint()], [pred_transcripts(), point_prediction()]
        )
        out_path = gt_path.parent / "scores.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "evaluate.py"),
                "--pred", str(pred_path),
                "--gt", str(gt_path),
                "--track", "1",
                "--out", str(out_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(report["spec_version"], spec.SPEC_VERSION)
        self.assertEqual(report["metrics"]["wer"], 0.0)
        self.assertEqual(json.loads(completed.stdout)["track"], 1)

    def test_violation_exits_with_code_2_and_a_rule_id(self) -> None:
        gt_path, pred_path = self.write_pair(
            [gt_transcripts(), gt_timepoint()],
            [pred_transcripts(), pred_transcripts(), point_prediction()],
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "evaluate.py"),
                "--pred", str(pred_path),
                "--gt", str(gt_path),
                "--track", "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertTrue(completed.stderr.startswith(evaluate.RULE_DUPLICATE_CASE), completed.stderr)


class TestSiblingModuleSchemas(EvaluatorTestCase):
    """The record shapes ``make_eval_dev.py`` and ``baselines/`` actually write.

    These are regression guards.  Before ``spec.py`` owned a record schema the
    three modules settled on different vocabularies and the evaluator read none
    of the records the baselines wrote.  The schema is now the contract's, but
    the older ``record_type`` spellings stay readable so that submission files
    generated then still score; that is what these tests hold in place.
    """

    def gt_case_row(self, **extra: Any) -> dict[str, Any]:
        """One generator-style ground-truth row, instants nested in ``frames``."""
        record: dict[str, Any] = {
            "case_id": CASE,
            "speaker_id": REF_SPEAKER,
            "final_transcripts_reference": {REF_SPEAKER: TRUE_TEXT},
            "t_eval_sec": [0.0, spec.DELTA_T_SEC],
            "frames": [
                {
                    "eval_time_sec": t,
                    "azimuth_deg": TRUE_AZIMUTH_DEG,
                    "distance_m": TRUE_DISTANCE_M,
                    "azimuth_bin": TRUE_AZIMUTH_BIN,
                    "distance_bin": TRUE_DISTANCE_BIN,
                }
                for t in (0.0, spec.DELTA_T_SEC)
            ],
        }
        record.update(extra)
        return record

    def test_generator_case_row_expands_into_t_eval(self) -> None:
        gt_path, _ = self.write_pair([self.gt_case_row()], [])
        truth = evaluate.load_ground_truth(gt_path, track=1)
        self.assertEqual(truth.case_ids(), [CASE])
        self.assertEqual(truth.eval_times[CASE], [0.0, spec.DELTA_T_SEC])
        self.assertEqual(truth.transcripts[CASE], {REF_SPEAKER: TRUE_TEXT})
        point = truth.points[CASE][evaluate.time_key(spec.DELTA_T_SEC)]
        self.assertEqual(point.speaker, REF_SPEAKER)
        self.assertEqual(point.azimuth_bin, TRUE_AZIMUTH_BIN)
        self.assertEqual(point.distance_bin, TRUE_DISTANCE_BIN)

    def test_case_row_rejects_a_declared_bin_that_contradicts_spec(self) -> None:
        bad = self.gt_case_row()
        bad["frames"][0]["azimuth_bin"] = (TRUE_AZIMUTH_BIN + 1) % spec.K_AZIMUTH
        gt_path, _ = self.write_pair([bad], [])
        with self.assertRaises(evaluate.SubmissionError) as caught:
            evaluate.load_ground_truth(gt_path, track=1)
        self.assertTrue(str(caught.exception).startswith(evaluate.RULE_PARSE))
        self.assertIn("azimuth_bin", str(caught.exception))

    def test_case_row_still_rejects_a_duplicate_instant(self) -> None:
        bad = self.gt_case_row()
        bad["frames"].append(dict(bad["frames"][0]))
        gt_path, _ = self.write_pair([bad], [])
        with self.assertRaises(evaluate.SubmissionError) as caught:
            evaluate.load_ground_truth(gt_path, track=1)
        self.assertTrue(str(caught.exception).startswith(evaluate.RULE_DUPLICATE_TIME))

    def test_case_row_still_enforces_the_speaker_budget(self) -> None:
        bad = self.gt_case_row(final_transcripts_reference={"a": "x", "b": "y"})
        gt_path, _ = self.write_pair([bad], [])
        with self.assertRaises(evaluate.SubmissionError) as caught:
            evaluate.load_ground_truth(gt_path, track=1)
        self.assertTrue(str(caught.exception).startswith(evaluate.RULE_SPEAKER_COUNT))

    def test_deprecated_record_type_spellings_are_still_accepted(self) -> None:
        """``timestep`` / ``final_transcript`` are what the baselines used to write."""
        preds = [
            {
                "record_type": "timestep",
                "case_id": CASE,
                "eval_time_sec": 0.0,
                "speaker_id": REF_SPEAKER,
                "pred_azimuth_deg": TRUE_AZIMUTH_DEG,
                "pred_distance_m": TRUE_DISTANCE_M,
            },
            {
                "record_type": "final_transcript",
                "case_id": CASE,
                "final_transcripts": {REF_SPEAKER: TRUE_TEXT},
                "observation_mode": spec.OFFICIAL_SCOPES[0],
                "reported_rtf": 0.5,
            },
        ]
        report = self.score([self.gt_case_row()], preds)
        table = report["metrics"]
        self.assertEqual(table[evaluate.METRIC_WER], 0.0)
        self.assertEqual(table[evaluate.METRIC_RTF], 0.5)
        # The first instant is exact and the second is absent, so the macro
        # azimuth error is half the missing-prediction penalty.
        self.assertEqual(table[evaluate.METRIC_AZ_MAE], spec.PENALTY_AZIMUTH_DEG / 2.0)
        self.assertEqual(report["official"]["n_missing_timepoints"], 1)

    def test_speaker_id_alias_makes_attribution_scorable(self) -> None:
        """Reading only ``pred_speaker`` would score every instant wrong in silence."""
        preds = [
            {
                "record_type": "timestep",
                "case_id": CASE,
                "eval_time_sec": t,
                "speaker_id": REF_SPEAKER,
            }
            for t in (0.0, spec.DELTA_T_SEC)
        ]
        table = self.metrics([self.gt_case_row()], preds)
        self.assertEqual(table[evaluate.METRIC_SPEAKER_ACC], 1.0)

    def test_pred_speaker_wins_over_the_alias(self) -> None:
        record = {
            "record_type": "timepoint",
            "case_id": CASE,
            "eval_time_sec": 0.0,
            "pred_speaker": REF_SPEAKER,
            "speaker_id": "someone_else",
        }
        _, pred_path = self.write_pair([], [record])
        preds = evaluate.load_predictions(pred_path, track=1)
        self.assertEqual(preds.points[CASE][0.0].speaker, REF_SPEAKER)

    def test_rtf_wins_over_its_alias(self) -> None:
        record = {
            "record_type": "final_transcript",
            "case_id": CASE,
            "final_transcripts": {REF_SPEAKER: TRUE_TEXT},
            "rtf": 0.25,
            "reported_rtf": 9.0,
        }
        _, pred_path = self.write_pair([], [record])
        preds = evaluate.load_predictions(pred_path, track=1)
        self.assertEqual(preds.rtf[CASE], 0.25)

    def test_an_unknown_record_type_is_still_rejected(self) -> None:
        _, pred_path = self.write_pair(
            [], [{"record_type": "nonsense", "case_id": CASE, "eval_time_sec": 0.0}]
        )
        with self.assertRaises(evaluate.SubmissionError) as caught:
            evaluate.load_predictions(pred_path, track=1)
        self.assertTrue(str(caught.exception).startswith(evaluate.RULE_PARSE))

    def test_a_nested_frames_submission_is_rejected(self) -> None:
        """A submission lists one record per instant, never a per-case row."""
        _, pred_path = self.write_pair([], [self.gt_case_row()])
        with self.assertRaises(evaluate.SubmissionError) as caught:
            evaluate.load_predictions(pred_path, track=1)
        self.assertTrue(str(caught.exception).startswith(evaluate.RULE_PARSE))
        self.assertIn(evaluate.FRAMES_KEY, str(caught.exception))

    def test_a_row_with_neither_shape_is_still_rejected(self) -> None:
        _, pred_path = self.write_pair([], [{"case_id": CASE, "note": "nothing usable"}])
        with self.assertRaises(evaluate.SubmissionError) as caught:
            evaluate.load_predictions(pred_path, track=1)
        self.assertTrue(str(caught.exception).startswith(evaluate.RULE_PARSE))


class TestRecordSchemaContract(EvaluatorTestCase):
    """``spec.py`` owns the record schema, and no module may re-declare it.

    The schema was the last thing the proposal left open, and the one that cost
    the most: three modules invented three vocabularies, the evaluator grew an
    alias table to bridge them, and a field-name mismatch made it read none of
    the records the baselines wrote.  These tests assert that the names, the
    deprecated spellings and the resolution order all have exactly one home.
    """

    def test_canonical_record_types_are_fixed_points(self) -> None:
        for name in spec.RECORD_TYPES:
            self.assertEqual(spec.canonical_record_type(name), name)

    def test_every_alias_resolves_to_a_canonical_type(self) -> None:
        self.assertTrue(spec.RECORD_TYPE_ALIASES, "the alias table must not be empty")
        for alias, canonical in spec.RECORD_TYPE_ALIASES.items():
            self.assertIn(canonical, spec.RECORD_TYPES)
            self.assertNotIn(alias, spec.RECORD_TYPES, f"{alias!r} is both canonical and an alias")
            self.assertEqual(spec.canonical_record_type(alias), canonical)

    def test_an_unknown_record_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            spec.canonical_record_type("frames")

    def test_the_evaluator_reads_exactly_the_declared_spellings(self) -> None:
        declared = set(spec.RECORD_TYPES) | set(spec.RECORD_TYPE_ALIASES)
        read = set(evaluate.RECORD_TIMEPOINT_ALIASES) | set(
            evaluate.RECORD_TRANSCRIPTS_ALIASES
        )
        self.assertEqual(read, declared)

    def test_field_alias_tables_resolve_their_own_name_first(self) -> None:
        for canonical, spellings in spec.FIELD_ALIASES.items():
            self.assertEqual(spellings[0], canonical)
            self.assertEqual(len(set(spellings)), len(spellings))
            self.assertEqual(spec.accepted_field_names(canonical), spellings)

    def test_a_field_with_no_alias_resolves_to_itself(self) -> None:
        self.assertEqual(spec.accepted_field_names(spec.F_CASE_ID), (spec.F_CASE_ID,))

    def test_resolve_field_follows_the_declared_order(self) -> None:
        both = {"rtf": 1.0, "reported_rtf": 9.0}
        self.assertEqual(spec.resolve_field(both, spec.F_RTF), 1.0)
        self.assertEqual(spec.present_field_name(both, spec.F_RTF), "rtf")
        self.assertEqual(
            spec.present_field_name({"reported_rtf": 9.0}, spec.F_RTF), "reported_rtf"
        )
        self.assertIsNone(spec.present_field_name({}, spec.F_RTF))
        self.assertEqual(spec.resolve_field({}, spec.F_RTF, default="?"), "?")

    def test_present_field_name_separates_absent_from_null(self) -> None:
        """An explicit null point estimate is an abstention, not a missing field."""
        self.assertEqual(spec.present_field_name({"rtf": None}, spec.F_RTF), "rtf")
        self.assertIsNone(spec.resolve_field({"rtf": None}, spec.F_RTF))

    def test_the_evaluator_takes_its_alias_names_from_the_contract(self) -> None:
        self.assertEqual(evaluate.SPEAKER_ID_KEY, spec.FIELD_ALIASES[spec.F_SPEAKER][1])
        self.assertEqual(evaluate.RTF_ALIAS_KEY, spec.FIELD_ALIASES[spec.F_RTF][1])
        self.assertEqual(evaluate.FRAMES_KEY, spec.F_FRAMES)
        self.assertEqual(evaluate.TRANSCRIPTS_REFERENCE_KEY, spec.F_TRANSCRIPTS_REFERENCE)

    def test_distribution_lengths_come_from_the_bin_counts(self) -> None:
        self.assertEqual(spec.PROB_FIELD_LENGTHS[spec.F_AZIMUTH_PROBS], spec.K_AZIMUTH)
        self.assertEqual(spec.PROB_FIELD_LENGTHS[spec.F_DISTANCE_PROBS], spec.K_DISTANCE)

    def test_required_fields_cover_both_record_types(self) -> None:
        self.assertEqual(set(spec.REQUIRED_FIELDS), set(spec.RECORD_TYPES))
        for fields in spec.REQUIRED_FIELDS.values():
            self.assertIn(spec.F_CASE_ID, fields)

    def test_a_timepoint_without_its_required_fields_is_rejected(self) -> None:
        for dropped in spec.REQUIRED_FIELDS[spec.RECORD_TYPE_TIMEPOINT]:
            record = {
                spec.F_RECORD_TYPE: spec.RECORD_TYPE_TIMEPOINT,
                spec.F_CASE_ID: CASE,
                spec.F_EVAL_TIME: 0.0,
            }
            del record[dropped]
            _, pred_path = self.write_pair([], [record])
            with self.assertRaises(evaluate.SubmissionError, msg=f"missing {dropped}"):
                evaluate.load_predictions(pred_path, track=1)

    def test_the_permutation_cap_is_the_contracts(self) -> None:
        self.assertEqual(evaluate._MAX_PERMUTATION_SLOTS, spec.MAX_PERMUTATION_SLOTS)
        self.assertGreaterEqual(
            spec.MAX_PERMUTATION_SLOTS, max(spec.MAX_SPEAKERS_PER_TRACK.values())
        )

    def test_the_contract_self_check_catches_a_contradiction(self) -> None:
        """A value edited out of step with its partner must fail at import, not at score time."""
        with unittest.mock.patch.object(spec, "AZIMUTH_ACC_THRESHOLDS_DEG", (5.0,)):
            with self.assertRaises(RuntimeError):
                spec._self_check()
        with unittest.mock.patch.object(spec, "NLL_EPSILON", 0.5):
            with unittest.mock.patch.object(spec, "PENALTY_NLL", 0.1):
                with self.assertRaises(RuntimeError):
                    spec._self_check()
        spec._self_check()  # unchanged values still agree


class TestInferenceGrid(EvaluatorTestCase):
    """:data:`spec.GRID_ENDPOINT` fixes how many instants a session has."""

    def test_the_grid_excludes_the_session_end(self) -> None:
        instants = [k * spec.DELTA_T_SEC for k in range(spec.N_GRID_POINTS)]
        self.assertEqual(len(instants), spec.N_GRID_POINTS)
        self.assertEqual(instants[0], 0.0)
        self.assertEqual(instants[-1], spec.SESSION_LENGTH_SEC - spec.DELTA_T_SEC)
        self.assertLess(instants[-1], spec.SESSION_LENGTH_SEC)

    def test_an_instant_at_the_session_end_is_rejected(self) -> None:
        """It is off-grid and has no audio under it, so it could only draw a penalty."""
        record = {
            spec.F_RECORD_TYPE: spec.RECORD_TYPE_TIMEPOINT,
            spec.F_CASE_ID: CASE,
            spec.F_EVAL_TIME: spec.SESSION_LENGTH_SEC,
        }
        _, pred_path = self.write_pair([], [record])
        with self.assertRaises(evaluate.SubmissionError) as caught:
            evaluate.load_predictions(pred_path, track=1)
        self.assertTrue(str(caught.exception).startswith(evaluate.RULE_PARSE))

    def test_the_last_grid_instant_is_accepted(self) -> None:
        record = {
            spec.F_RECORD_TYPE: spec.RECORD_TYPE_TIMEPOINT,
            spec.F_CASE_ID: CASE,
            spec.F_EVAL_TIME: spec.SESSION_LENGTH_SEC - spec.DELTA_T_SEC,
        }
        _, pred_path = self.write_pair([], [record])
        preds = evaluate.load_predictions(pred_path, track=1)
        self.assertEqual(len(preds.points[CASE]), 1)


class TestBaselineEmissionIsCanonical(EvaluatorTestCase):
    """What ``baselines/`` writes must need no alias path to score."""

    def setUp(self) -> None:
        import baselines

        self.baselines = baselines

    def timestep(self) -> dict[str, Any]:
        return self.baselines.build_timestep_record(
            CASE,
            0.0,
            REF_SPEAKER,
            self.baselines.uniform_distribution(spec.K_AZIMUTH),
            self.baselines.uniform_distribution(spec.K_DISTANCE),
        )

    def final(self) -> dict[str, Any]:
        return self.baselines.build_final_record(
            CASE, {REF_SPEAKER: HYP_TEXT}, spec.OFFICIAL_SCOPES[0], 0.5, "test"
        )

    def test_emitted_record_types_are_canonical(self) -> None:
        for record in (self.timestep(), self.final()):
            declared = record[spec.F_RECORD_TYPE]
            self.assertIn(declared, spec.RECORD_TYPES)
            self.assertNotIn(declared, spec.RECORD_TYPE_ALIASES)

    def test_every_emitted_scored_field_is_a_contract_name(self) -> None:
        """Metadata may be anything; a field the evaluator reads may not be invented."""
        scored = {
            spec.F_RECORD_TYPE,
            spec.F_CASE_ID,
            spec.F_EVAL_TIME,
            spec.F_AZIMUTH_PROBS,
            spec.F_DISTANCE_PROBS,
            spec.F_PRED_AZIMUTH_DEG,
            spec.F_PRED_DISTANCE_M,
            spec.F_PRED_X_M,
            spec.F_PRED_Y_M,
            spec.F_OBSERVATION_MODE,
            spec.F_FINAL_TRANSCRIPTS,
        }
        for canonical in (spec.F_PRED_SPEAKER, spec.F_RTF):
            scored.update(spec.accepted_field_names(canonical))
        metadata = {
            "hardware",
            "system",
            "track",
            "spec_version",
            "baseline_impl_version",
            "wer_reportable",
            "verification_tag",
            "meta",
        }
        for record in (self.timestep(), self.final()):
            unknown = set(record) - scored - metadata
            self.assertEqual(unknown, set(), f"invented field names: {sorted(unknown)}")

    def test_a_baseline_pair_scores_end_to_end(self) -> None:
        """The exact numbers a uniform-output baseline must earn, by construction."""
        report = self.score([gt_transcripts(), gt_timepoint()], [self.final(), self.timestep()])
        table = report["metrics"]
        self.assertEqual(table[evaluate.METRIC_WER], 0.0)
        self.assertEqual(table[evaluate.METRIC_RTF], 0.5)
        self.assertEqual(table[evaluate.METRIC_SPEAKER_ACC], 1.0)
        self.assertAlmostEqual(
            table[evaluate.METRIC_AZ_NLL], math.log(spec.K_AZIMUTH), places=9
        )
        self.assertAlmostEqual(
            table[evaluate.METRIC_R_NLL], math.log(spec.K_DISTANCE), places=9
        )
        self.assertEqual(report["official"]["n_missing_timepoints"], 0)

    def test_the_slot_helper_is_the_contracts(self) -> None:
        self.assertEqual(self.baselines.speaker_slot(0), spec.speaker_slot(0))
        self.assertEqual(self.baselines.SPEAKER_SLOT_FORMAT, spec.SPEAKER_SLOT_FORMAT)

    def test_the_local_register_no_longer_claims_contract_rules(self) -> None:
        """Every entry lifted into ``spec.py`` must be gone from the local register."""
        lifted = {
            "record_schema",
            "record_type_field",
            "speaker_slot_format",
            "azimuth_sign_convention",
            "inference_grid_endpoint",
            "causal_window_rule",
            "rtf_definition",
            "array_geometry_default",
        }
        self.assertEqual(lifted & set(self.baselines.LOCAL_CONVENTIONS), set())


class TestLiftedConventions(EvaluatorTestCase):
    """The rules that moved out of ``baselines/LOCAL_CONVENTIONS`` into the contract."""

    def test_the_azimuth_convention_inverts_to_cartesian(self) -> None:
        self.assertEqual(spec.AZIMUTH_CONVENTION, "atan2_y_x_ccw")
        for azimuth_deg, distance_m in ((0.0, 1.0), (90.0, 2.0), (-135.0, 3.3)):
            radians = math.radians(azimuth_deg)
            x, y = distance_m * math.cos(radians), distance_m * math.sin(radians)
            self.assertAlmostEqual(math.degrees(math.atan2(y, x)), azimuth_deg, places=9)
            self.assertAlmostEqual(math.hypot(x, y), distance_m, places=9)

    def test_the_default_geometry_justifies_the_mask_comments(self) -> None:
        """``1100`` must really be the shortest pair and ``1010`` the longest."""
        self.assertEqual(spec.DEFAULT_ARRAY_GEOMETRY, "uniform_circular")
        radius = spec.MAX_PAIR_SPACING_M / 2.0
        mics = [
            (
                radius * math.cos(2.0 * math.pi * i / spec.N_CHANNELS_MAX),
                radius * math.sin(2.0 * math.pi * i / spec.N_CHANNELS_MAX),
            )
            for i in range(spec.N_CHANNELS_MAX)
        ]

        def spacing(mask_bits: str) -> float:
            a, b = spec.mask_to_channels(mask_bits)
            return math.dist(mics[a], mics[b])

        self.assertLess(spacing("1100"), spacing("1010"))
        self.assertAlmostEqual(spacing("1010"), spec.MAX_PAIR_SPACING_M, places=9)

    def test_the_aliasing_limit_is_observable_below_nyquist(self) -> None:
        self.assertLess(spec.ALIASING_LIMIT_HZ, spec.SAMPLE_RATE_HZ / 2)
        self.assertAlmostEqual(
            spec.ALIASING_LIMIT_HZ,
            spec.SPEED_OF_SOUND_M_S / (2.0 * spec.MAX_PAIR_SPACING_M),
            places=9,
        )

    def test_the_tdoa_span_needs_sub_sample_interpolation(self) -> None:
        """One integer lag is coarser than one azimuth bin, so this is a requirement."""
        self.assertAlmostEqual(
            spec.MAX_TDOA_SAMPLES,
            spec.MAX_PAIR_SPACING_M / spec.SPEED_OF_SOUND_M_S * spec.SAMPLE_RATE_HZ,
            places=9,
        )
        lags_available = 2.0 * spec.MAX_TDOA_SAMPLES
        self.assertLess(lags_available, spec.K_AZIMUTH)

    def test_the_t_eval_rule_is_frozen(self) -> None:
        """|T_eval| is the denominator of every spatial metric, so it is contract."""
        self.assertEqual(spec.T_EVAL_RULE, "dry_source_rms_gate")
        self.assertLess(spec.VAD_REL_THRESHOLD_DB, 0.0)
        self.assertGreater(spec.VAD_REF_PERCENTILE, 0.0)
        self.assertLessEqual(spec.VAD_REF_PERCENTILE, 100.0)

    def test_the_generator_reads_the_gate_from_the_contract(self) -> None:
        import simulate

        self.assertEqual(simulate.VAD_REL_THRESHOLD_DB, spec.VAD_REL_THRESHOLD_DB)
        self.assertEqual(simulate.VAD_REF_PERCENTILE, spec.VAD_REF_PERCENTILE)
        self.assertEqual(simulate.SAMPLE_RATE_HZ, spec.SAMPLE_RATE_HZ)

    def test_rule_ids_have_exactly_one_home(self) -> None:
        self.assertEqual(set(spec.REJECTION_RULES) & set(spec.SCORING_RULES), set())
        for name in dir(evaluate):
            if name.startswith("RULE_"):
                self.assertIn(getattr(evaluate, name), spec.REJECTION_RULES)
        with self.assertRaises(RuntimeError):
            evaluate._rule("R-INVENTED")

    def test_the_rtf_definition_is_frozen(self) -> None:
        self.assertEqual(spec.RTF_DEFINITION, "wall_clock_per_audio_second")
        self.assertIn(evaluate.METRIC_RTF, spec.OFFICIAL_METRICS)

    def test_a_windowed_submission_must_declare_its_lookahead(self) -> None:
        self.assertTrue(spec.WINDOWED_REQUIRES_LOOKAHEAD)
        self.assertNotIn("windowed", spec.OFFICIAL_SCOPES)
        self.assertIn("windowed", spec.OBSERVATION_MODES)


class TestScoresCarryNoProvenance(EvaluatorTestCase):
    """A score file says what the numbers are, never where they came from.

    ``eval-dev`` ships its ground truth, so a prediction file built from the
    labels alone scores perfectly without opening a single audio file.  This is
    not about anyone cheating -- the challenge runs inside one team.  It is a
    property of offline scoring that two real situations depend on:

    * ``RTF`` is whatever the file claims, so comparing our own model variants
      only means something when the numbers come from one machine.
    * A model that accidentally reads future audio scores *better*, not worse.
      Causality has to be a test over a rerun; the leaderboard cannot catch it.

    Hence ``spec.OFFICIAL_SCOPES`` and the causality unit test, rather than
    trust in the number itself.
    """

    def label_only_submission(self) -> list[dict[str, Any]]:
        """A prediction file built from the labels, touching no audio."""
        azimuth_probs = one_hot(spec.K_AZIMUTH, TRUE_AZIMUTH_BIN)
        distance_probs = one_hot(spec.K_DISTANCE, TRUE_DISTANCE_BIN)
        records: list[dict[str, Any]] = [
            {
                spec.F_RECORD_TYPE: spec.RECORD_TYPE_FINAL_TRANSCRIPTS,
                spec.F_CASE_ID: CASE,
                spec.F_FINAL_TRANSCRIPTS: {REF_SPEAKER: TRUE_TEXT},
                spec.F_OBSERVATION_MODE: spec.OFFICIAL_SCOPES[0],
                spec.F_RTF: 0.001,
            }
        ]
        for t in (0.0, spec.DELTA_T_SEC):
            records.append(
                {
                    spec.F_RECORD_TYPE: spec.RECORD_TYPE_TIMEPOINT,
                    spec.F_CASE_ID: CASE,
                    spec.F_EVAL_TIME: t,
                    spec.F_PRED_AZIMUTH_DEG: TRUE_AZIMUTH_DEG,
                    spec.F_PRED_DISTANCE_M: TRUE_DISTANCE_M,
                    spec.F_PRED_SPEAKER: REF_SPEAKER,
                    spec.F_AZIMUTH_PROBS: azimuth_probs,
                    spec.F_DISTANCE_PROBS: distance_probs,
                }
            )
        return records

    def perfect_scores(self) -> dict[str, Any]:
        gt = [
            gt_transcripts(),
            gt_timepoint(eval_time_sec=0.0),
            gt_timepoint(eval_time_sec=spec.DELTA_T_SEC),
        ]
        return self.metrics(gt, self.label_only_submission())

    def test_label_only_submission_is_perfect_on_every_point_metric(self) -> None:
        table = self.perfect_scores()
        for name in (
            evaluate.METRIC_WER,
            evaluate.METRIC_AZ_MAE,
            evaluate.METRIC_R_MAE,
            evaluate.METRIC_POS2D,
        ):
            self.assertEqual(table[name], 0.0, f"{name} should be a perfect 0.0")
        self.assertEqual(table[evaluate.METRIC_SPEAKER_ACC], 1.0)
        for name, _ in evaluate.AZIMUTH_ACC_METRICS:
            self.assertEqual(table[name], 1.0)

    def test_a_one_hot_on_the_true_bin_gives_exactly_zero_nll(self) -> None:
        """No clipping floor is involved: -ln(1.0) is 0, so the NLL column is 0 too."""
        table = self.perfect_scores()
        self.assertEqual(table[evaluate.METRIC_AZ_NLL], 0.0)
        self.assertEqual(table[evaluate.METRIC_R_NLL], 0.0)

    def test_a_label_only_row_shows_the_tightest_possible_interval(self) -> None:
        """One-bin width at full coverage is the floor the interval metrics allow.

        Worth pinning because it is the lower bound a real model is measured
        against: an uncertainty column cannot go tighter than one bin.
        """
        table = self.perfect_scores()
        self.assertEqual(table[evaluate.METRIC_AZ_COVERAGE], 1.0)
        self.assertEqual(table[evaluate.METRIC_AZ_WIDTH], spec.AZIMUTH_BIN_WIDTH_DEG)
        self.assertEqual(
            table[evaluate.METRIC_R_WIDTH], spec.DISTANCE_BIN_WIDTHS_M[TRUE_DISTANCE_BIN]
        )

    def test_rtf_is_whatever_the_submission_claims(self) -> None:
        """So an RTF column only compares runs measured on the same machine."""
        table = self.perfect_scores()
        self.assertEqual(table[evaluate.METRIC_RTF], 0.001)
        self.assertEqual(spec.RTF_DEFINITION, "wall_clock_per_audio_second")


def load_tests(
    loader: unittest.TestLoader, tests: unittest.TestSuite, ignore: str
) -> unittest.TestSuite:
    """Add the evaluator's own doctests to the suite."""
    tests.addTests(doctest.DocTestSuite(evaluate))
    tests.addTests(doctest.DocTestSuite(sys.modules[__name__]))
    return tests
