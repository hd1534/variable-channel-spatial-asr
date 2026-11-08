"""Protect noise parity, checkpoint migration and ASR-specific result reuse."""

from __future__ import annotations

import copy
import json
import math
import random
import sys
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - minimal scoring installation
    raise unittest.SkipTest(f"torch/numpy not installed: {exc}") from exc

import eval_t2
from training.config import VAL_PROTOCOL_VERSION, config_fingerprint, load_config_dict
from training.engine import Checkpoints, Run, atomic_save
from training.loss import SessionCentroidLoss
from training.render import diffuse_noise
from training.scenes import NOISE_HOP, NOISE_N_FFT, diffuse_noise_numpy


class _FixedNoiseGenerator:
    """Supply identical real and imaginary spectra to the NumPy reference."""

    def __init__(self, parts: np.ndarray) -> None:
        """Keep the two independent random components and a read cursor."""
        self.parts: np.ndarray = parts
        self.index: int = 0

    def standard_normal(self, shape: tuple[int, ...]) -> np.ndarray:
        """Return the next component with its requested shape checked."""
        value: np.ndarray = self.parts[self.index]
        self.index += 1
        if value.shape != shape:
            raise ValueError(f"Unexpected noise shape: {shape}")
        return value


class TestNoiseParity(unittest.TestCase):
    """Training must use the committed evaluation set's noise synthesis law."""

    def test_identical_spectra_match_numpy_for_batches_and_partial_hops(self) -> None:
        """Compare both actual synthesizers before any overall RMS adjustment."""
        positions: np.ndarray = np.array([
            [[0.06, 0.0], [0.0, 0.05], [-0.07, 0.0], [0.0, -0.09]],
            [[0.08, 0.01], [-0.07, 0.02], [0.01, 0.09], [-0.02, -0.08]],
        ], dtype=np.float32)
        for n_samples in (2048, 2073):
            with self.subTest(n_samples=n_samples):
                n_frames: int = n_samples // NOISE_HOP + 2
                parts: np.ndarray = np.random.default_rng(19).standard_normal(
                    (2, 2, NOISE_N_FFT // 2 + 1, n_frames, 4)
                ).astype(np.float32)
                references: list[np.ndarray] = []
                for index in range(2):
                    with patch("numpy.random.default_rng", return_value=_FixedNoiseGenerator(parts[index])):
                        references.append(diffuse_noise_numpy(positions[index], n_samples, seed=index))
                draws: Iterator[np.ndarray] = iter(parts)

                def fixed_torch_parts(*shape: int, **options: Any) -> torch.Tensor:
                    """Feed the same spectra to each row of the Torch batch."""
                    value: np.ndarray = next(draws)
                    self.assertEqual(tuple(shape), value.shape)
                    return torch.from_numpy(value).to(options.get("device", "cpu"))

                with patch("torch.randn", side_effect=fixed_torch_parts):
                    actual: np.ndarray = diffuse_noise(
                        torch.from_numpy(positions), n_samples, torch.tensor([0, 1])
                    ).numpy()
                expected: np.ndarray = np.stack(references)
                self.assertEqual(actual.shape, (2, 4, n_samples))
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-6)
                self.assertLess(float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)), 1e-5)


class _MarkerModel(torch.nn.Linear):
    """Small trainable model whose single weight identifies a checkpoint."""

    trainable: bool = True

    def __init__(self) -> None:
        """Create one observable weight without a bias."""
        super().__init__(1, 1, bias=False)


def _migration_run(directory: Path, max_steps: int = 10) -> Run:
    """Build real persistence and optimizer machinery without rendering speech."""
    run: Run = Run.__new__(Run)
    run.cfg = load_config_dict({"name": "migration", "train": {"max_steps": max_steps}})
    run.run_dir = directory
    run.device = "cpu"
    run.log = lambda message: None
    run.fingerprint = config_fingerprint(run.cfg)
    run.commit = "test-commit"
    run.commits = [run.commit]
    run.allow_code_change = False
    run.amp_dtype = None
    run.system = _MarkerModel()
    run.loss_fn = SessionCentroidLoss()
    run.pos_loss_fn = SessionCentroidLoss()
    parameters: list[torch.nn.Parameter] = (
        list(run.system.parameters()) + list(run.loss_fn.parameters()) + list(run.pos_loss_fn.parameters())
    )
    run.opt = torch.optim.AdamW(parameters)
    run.system.weight.grad = torch.ones_like(run.system.weight)
    run.opt.step()
    run.opt.zero_grad(set_to_none=True)
    run.sched = torch.optim.lr_scheduler.LambdaLR(run.opt, lambda step: 1.0)
    run.scaler = torch.amp.GradScaler("cuda", enabled=False)
    run.ckpts = Checkpoints(directory, keep=3)
    run.step = 10
    run.best = 0.99
    with torch.no_grad():
        run.system.weight.fill_(1.0)
    atomic_save({
        "step": 5, "fingerprint": run.fingerprint, "val_protocol_version": VAL_PROTOCOL_VERSION - 1,
        "model": run.system.state_dict(), "loss": run.loss_fn.state_dict(), "val": {"select_score": 0.99},
    }, directory / "best.pt")
    with torch.no_grad():
        run.system.weight.fill_(2.0)
        run.loss_fn.bias.fill_(-3.0)
    run.ckpts.save(run._state(), run.step)
    return run


def _validation_result(score: float) -> dict[str, Any]:
    """Represent official two-slot validation independently of its model selection policy."""
    return {
        "spk_acc": score, "spk_acc_by_M": {"2": score},
        "spk_acc_n2_cap2": score, "spk_acc_n2_cap2_by_M": {"2": score},
        "theta_by_M": {"2": 0.5}, "theta_official_by_M": {"2": 0.5},
    }


class TestBestProtocolMigration(unittest.TestCase):
    """Refresh scores while preserving viable weights and resumed training state."""

    def test_both_candidates_are_rescored_and_the_better_weights_survive(self) -> None:
        """Cover old-best wins, latest wins, ties and an unfinished training run."""
        for old_score, latest_score, max_steps in ((0.9, 0.6, 10), (0.6, 0.9, 10),
                                                  (0.8, 0.8, 10), (0.9, 0.6, 20)):
            with self.subTest(old_score=old_score, latest_score=latest_score, max_steps=max_steps):
                with tempfile.TemporaryDirectory() as temporary:
                    run: Run = _migration_run(Path(temporary), max_steps=max_steps)
                    optimizer: dict[str, Any] = copy.deepcopy(run.opt.state_dict())
                    scheduler: dict[str, Any] = copy.deepcopy(run.sched.state_dict())
                    rng: torch.Tensor = torch.get_rng_state().clone()
                    python_rng: Any = random.getstate()
                    numpy_rng: Any = np.random.get_state()
                    evaluated: list[float] = []

                    def validate(quick: bool = False, cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
                        """Score model identities and deliberately consume training randomness."""
                        marker: float = float(run.system.weight.detach().item())
                        evaluated.append(marker)
                        run.system.eval()
                        random.random()
                        np.random.random()
                        torch.rand(1)
                        return _validation_result(old_score if marker == 1.0 else latest_score)

                    with patch.object(run, "validate", side_effect=validate):
                        if max_steps == run.step:
                            run.fit()
                        else:
                            run._resume()
                    best: dict[str, Any] | None = run._load_best("cpu")
                    self.assertIsNotNone(best)
                    assert best is not None
                    self.assertEqual(evaluated, [1.0, 2.0])
                    winner: float = 1.0 if old_score >= latest_score else 2.0
                    self.assertEqual(float(best["model"]["weight"].item()), winner)
                    self.assertEqual(best["step"], 5 if winner == 1.0 else 10)
                    self.assertEqual(best["val_protocol_version"], VAL_PROTOCOL_VERSION)
                    self.assertEqual(best["val"]["select_score"], max(old_score, latest_score))
                    self.assertEqual(float(run.system.weight.detach().item()), 2.0)
                    self.assertEqual(float(run.loss_fn.bias.detach()), -3.0)
                    self.assertEqual(run.step, 10)
                    self.assertTrue(run.system.training)
                    torch.testing.assert_close(run.opt.state_dict(), optimizer)
                    self.assertEqual(run.sched.state_dict(), scheduler)
                    self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                    self.assertEqual(random.getstate(), python_rng)
                    current_numpy_rng: Any = np.random.get_state()
                    np.testing.assert_array_equal(current_numpy_rng[1], numpy_rng[1])
                    self.assertEqual(current_numpy_rng[2:], numpy_rng[2:])

    def test_failed_revalidation_preserves_original_file_and_training_state(self) -> None:
        """A validation failure must not destroy the readable legacy best."""
        with tempfile.TemporaryDirectory() as temporary:
            run: Run = _migration_run(Path(temporary))
            original: bytes = (run.run_dir / "best.pt").read_bytes()
            with patch.object(run, "validate", side_effect=RuntimeError("validation interrupted")):
                with self.assertRaisesRegex(RuntimeError, "validation interrupted"):
                    run._resume()
            self.assertEqual((run.run_dir / "best.pt").read_bytes(), original)
            self.assertEqual(float(run.system.weight.detach().item()), 2.0)
            self.assertEqual(float(run.loss_fn.bias.detach()), -3.0)
            self.assertEqual(run.step, 10)
            self.assertTrue(run.system.training)

    def test_interrupted_migration_preserves_old_weights_and_retries_both_candidates(self) -> None:
        """An interrupted comparison preserves the original file and finishes on the next resume."""
        with tempfile.TemporaryDirectory() as temporary:
            run: Run = _migration_run(Path(temporary))
            original: bytes = (run.run_dir / "best.pt").read_bytes()
            with patch.object(run, "validate", side_effect=[_validation_result(0.8), RuntimeError("interrupted")]):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run._resume()
            self.assertEqual((run.run_dir / "best.pt").read_bytes(), original)
            self.assertIsNone(run._load_best("cpu"))
            self.assertEqual(float(run.system.weight.detach().item()), 2.0)
            self.assertEqual(run.step, 10)
            with patch.object(run, "validate", side_effect=[_validation_result(0.8), _validation_result(0.9)]) as validate:
                run.fit()
            self.assertEqual(validate.call_count, 2)
            best: dict[str, Any] | None = run._load_best("cpu")
            self.assertIsNotNone(best)
            assert best is not None
            self.assertEqual(float(best["model"]["weight"].item()), 2.0)
            self.assertEqual(best["val"]["select_score"], 0.9)
            self.assertEqual(float(run.system.weight.detach().item()), 2.0)
            self.assertEqual(run.step, 10)


class TestAsrCompletion(unittest.TestCase):
    """A score can be reused only for the requested transcript backend."""

    def test_backend_aliases_match_and_other_or_missing_backends_invalidate_scores(self) -> None:
        """Exercise the notebook's default and explicitly selected external ASR IDs."""
        cfg: Any = load_config_dict({"name": "space", "model": {"name": "space_only"}})
        summary: dict[str, Any] = {
            "run": "space", "config": cfg.name, "system": cfg.model.name, "best_step": None,
            "fingerprint": config_fingerprint(cfg), "val_protocol_version": VAL_PROTOCOL_VERSION,
            "theta_official_by_M": {"4": 0.5},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root: Path = Path(temporary)
            dataset: Path = root / "data" / "eval-t2"
            dataset.mkdir(parents=True)
            (dataset / "manifest.json").write_text('{"official": false}')
            output: Path = root / "eval" / "eval-t2"
            (output / "full").mkdir(parents=True)
            np.savez_compressed(
                output / "embeddings.npz", fingerprint=eval_t2.inputs_fingerprint(cfg, dataset, None, 0),
                case_id=np.array(["t2_0000__m1111"]), mask=np.array(["1111"]),
                session_id=np.array(["t2_0000"]), ref=np.array([0]),
                speech=np.ones((1, 80), dtype=bool), labels=np.zeros((1, 80), dtype=np.int64),
                space=np.full((1, 80, 36), 1.0 / math.sqrt(36)),
            )
            info: dict[str, Any] = {
                "summary": {key: summary.get(key) for key in
                            ("run", "config", "best_step", "fingerprint", "val_protocol_version")},
                "limit": 0, "theta_by_M": {"4": 0.5}, "weight_by_M": None,
                "asr": "external-asr",
            }
            score_path: Path = output / "full" / "score.json"
            score_path.write_text(json.dumps({"eval_t2_run": info, "metrics": {"cpwer": 0.0}}))
            with patch.object(eval_t2, "REPO", root):
                arguments: dict[str, Any] = {"sets": ("eval-t2",), "variants": ("full",)}
                self.assertFalse(eval_t2.eval_is_complete(root / "eval", cfg, summary, **arguments))
                self.assertTrue(eval_t2.eval_is_complete(root / "eval", cfg, summary, asr="external-asr", **arguments))
                info["asr"] = eval_t2.asr_backend_id("wav2vec2")
                score_path.write_text(json.dumps({"eval_t2_run": info, "metrics": {"cpwer": 0.0}}))
                self.assertTrue(eval_t2.eval_is_complete(root / "eval", cfg, summary, **arguments))
                self.assertTrue(eval_t2.eval_is_complete(
                    root / "eval", cfg, summary, asr=eval_t2.LOCAL_ASR["wav2vec2"], **arguments
                ))
                info.pop("asr")
                score_path.write_text(json.dumps({"eval_t2_run": info, "metrics": {"cpwer": 0.0}}))
                self.assertFalse(eval_t2.eval_is_complete(root / "eval", cfg, summary, **arguments))


if __name__ == "__main__":
    unittest.main()
