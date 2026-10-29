"""Unit tests for the shared baseline helpers.

Standard library only, like ``test_evaluate.py``.  The point of this file is
the representative-point rule: which single number a baseline submits when it
has a distribution.  Getting that wrong does not fail anything loudly -- it
just reports a worse score than the model deserves, and the reported number
then gets read as a floor.

Run with::

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import doctest
import statistics
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import baselines  # noqa: E402
import spec  # noqa: E402

UNIFORM_DISTANCE = [1.0 / spec.K_DISTANCE] * spec.K_DISTANCE
UNIFORM_AZIMUTH = [1.0 / spec.K_AZIMUTH] * spec.K_AZIMUTH


class TestMedianPoint(unittest.TestCase):
    """The L1-optimal representative of an ordered distribution."""

    def test_uniform_distance_resolves_to_the_middle_of_the_grid(self) -> None:
        centers = spec.DISTANCE_BIN_CENTERS_M
        half = spec.K_DISTANCE // 2
        expected = (centers[half - 1] + centers[half]) / 2.0
        self.assertAlmostEqual(
            baselines.median_point(UNIFORM_DISTANCE, centers), expected, places=12
        )

    def test_map_sends_the_same_uniform_to_the_nearest_bin(self) -> None:
        """Documents the behaviour this rule exists to avoid."""
        centers = spec.DISTANCE_BIN_CENTERS_M
        self.assertEqual(baselines.map_point(UNIFORM_DISTANCE, centers), centers[0])
        self.assertNotAlmostEqual(
            baselines.map_point(UNIFORM_DISTANCE, centers),
            baselines.median_point(UNIFORM_DISTANCE, centers),
            places=3,
        )

    def test_one_hot_returns_that_bin(self) -> None:
        centers = spec.DISTANCE_BIN_CENTERS_M
        for index in range(spec.K_DISTANCE):
            probs = [0.0] * spec.K_DISTANCE
            probs[index] = 1.0
            self.assertEqual(baselines.median_point(probs, centers), centers[index])

    def test_mass_entirely_below_the_half_mark_still_returns_a_center(self) -> None:
        self.assertEqual(baselines.median_point([0.9, 0.1], (1.0, 2.0)), 1.0)
        self.assertEqual(baselines.median_point([0.1, 0.9], (1.0, 2.0)), 2.0)

    def test_unnormalised_input_is_normalised(self) -> None:
        self.assertEqual(
            baselines.median_point([2.0, 2.0], (1.0, 3.0)),
            baselines.median_point([0.5, 0.5], (1.0, 3.0)),
        )

    def test_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            baselines.median_point([0.5, 0.5], (1.0, 2.0, 3.0))

    def test_zero_mass_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            baselines.median_point([0.0, 0.0], (1.0, 2.0))


class TestUninformedDistanceFloor(unittest.TestCase):
    """A uniform answer should score the floor, and the floor is the median's.

    This is the claim the leaderboard footnote makes, pinned here so a later
    change to the representative rule cannot quietly move it.
    """

    def setUp(self) -> None:
        low, high = spec.DISTANCE_RANGE_M
        steps = 400
        self.truths = [
            low + (high - low) * index / (steps - 1) for index in range(steps)
        ]

    def mae(self, point: float) -> float:
        return statistics.fmean(abs(point - truth) for truth in self.truths)

    def test_median_point_beats_map_point_on_a_uniform_answer(self) -> None:
        centers = spec.DISTANCE_BIN_CENTERS_M
        median_mae = self.mae(baselines.median_point(UNIFORM_DISTANCE, centers))
        map_mae = self.mae(baselines.map_point(UNIFORM_DISTANCE, centers))
        self.assertLess(median_mae, map_mae)

    def test_median_point_is_within_a_bin_of_the_optimal_constant(self) -> None:
        """No constant can beat the median of the truths by much."""
        centers = spec.DISTANCE_BIN_CENTERS_M
        best_possible = self.mae(statistics.median(self.truths))
        submitted = self.mae(baselines.median_point(UNIFORM_DISTANCE, centers))
        self.assertLess(submitted - best_possible, spec.DISTANCE_BIN_WIDTHS_M[-1])


class TestAzimuthKeepsTheModeRule(unittest.TestCase):
    """Azimuth wraps and is bimodal at M=2, so it must not switch to a median."""

    def test_uniform_azimuth_still_uses_the_mode(self) -> None:
        centers = spec.AZIMUTH_BIN_CENTERS_DEG
        self.assertEqual(baselines.map_point(UNIFORM_AZIMUTH, centers), centers[0])

    def test_a_median_would_land_between_two_opposite_modes(self) -> None:
        """Why the azimuth axis keeps :func:`map_point`."""
        probs = [0.0] * spec.K_AZIMUTH
        front = spec.azimuth_bin_index(-90.0)
        back = spec.azimuth_bin_index(90.0)
        probs[front] = probs[back] = 0.5
        centers = spec.AZIMUTH_BIN_CENTERS_DEG
        between = baselines.median_point(probs, centers)
        self.assertEqual(probs[spec.azimuth_bin_index(between)], 0.0)
        self.assertIn(baselines.map_point(probs, centers), (centers[front], centers[back]))


def load_tests(
    loader: unittest.TestLoader, tests: unittest.TestSuite, ignore: str
) -> unittest.TestSuite:
    """Add this module's own doctests to the suite."""
    tests.addTests(doctest.DocTestSuite(sys.modules[__name__]))
    return tests
