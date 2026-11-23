"""Track A (models/track_a.py) and the code paths it added to training and evaluation.

* the other systems' run fingerprints did not move when Track A's config section
  was added (a moved fingerprint would make a live run refuse to resume, or a
  finished run retrain from scratch);
* causal mode never reads the future, full mode does;
* one set of weights for any channel subset (order, inactive content, M = 1);
* training draws both modes and labels each on its own window;
* a whole smoke run trains, validates both modes and is recognised as complete.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.config import config_fingerprint, config_to_dict, load_config, load_config_dict  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None

needs_torch = unittest.skipIf(torch is None, "torch not installed")

FINGERPRINTS_BEFORE_TRACK_A = {
    "naive_fusion": "363c7f6b63d3e07b",
    "proposed_carry": "0776073f4e8d6641",
    "proposed_carry_nodrop": "86aaa681cab5cbd9",
    "proposed_reset": "6f295fab25c08744",
    "smoke": "2b60c3ae468f6ed9",
    "smoke_real": "6c48893c6803e612",
    "space_only": "f23149c2a203be63",
    "voice_only": "74301b9e7240afcb",
}
"""Fingerprints of the committed configs at 2fdfcfc, before Track A existed."""

S = 320000


class TestConfig(unittest.TestCase):
    def test_existing_run_fingerprints_are_unchanged(self) -> None:
        for name, expected in FINGERPRINTS_BEFORE_TRACK_A.items():
            with self.subTest(config=name):
                self.assertEqual(config_fingerprint(load_config(_REPO_ROOT / "configs" / f"{name}.json")), expected)

    def test_track_a_section_belongs_to_track_a_only(self) -> None:
        self.assertNotIn("track_a", config_to_dict(load_config_dict({"name": "p"})))
        with self.assertRaisesRegex(ValueError, "only applies"):
            load_config_dict({"name": "p", "track_a": {"use_mel": False}})
        a = load_config(_REPO_ROOT / "configs" / "track_a.json")
        b = load_config(_REPO_ROOT / "configs" / "track_a_spatial.json")
        self.assertNotEqual(config_fingerprint(a), config_fingerprint(b))
        self.assertEqual(a.data.m_weights, (1.0, 1.0, 1.0, 1.0))

    def test_track_a_needs_no_pretrained_voice_encoder(self) -> None:
        if torch is None:
            self.skipTest("torch not installed")
        from models import needs_voice

        self.assertFalse(needs_voice(load_config(_REPO_ROOT / "configs" / "track_a.json")))
        self.assertTrue(needs_voice(load_config(_REPO_ROOT / "configs" / "proposed_carry.json")))


def _model(use_mel: bool = True):
    from models import build_system

    cfg = load_config_dict({"name": "t", "model": {"name": "track_a"},
                            "track_a": {"use_mel": use_mel, "d_model": 32, "n_layers": 2, "n_heads": 2,
                                        "n_pair_tac_blocks": 1, "d_embed": 16, "n_mels": 20, "dropout": 0.0}})
    torch.manual_seed(0)
    return build_system(cfg).eval()


def _batch(mask_bits: str = "1111", seed: int = 0, B: int = 2) -> dict:
    g = torch.Generator().manual_seed(seed)
    mask = torch.tensor([c == "1" for c in mask_bits])
    audio = torch.randn(B, 4, S, generator=g) * 0.05 * mask[None, :, None]
    xy = torch.tensor([[0.06, 0.0], [0.0, 0.05], [-0.07, 0.0], [0.0, -0.09]])
    return {"audio": audio, "mask": mask[None].expand(B, -1).clone(), "mic_xy": xy[None].expand(B, -1, -1).clone()}


@needs_torch
class TestTrackA(unittest.TestCase):
    def test_causal_mode_never_reads_the_future(self) -> None:
        model, batch = _model(), _batch()
        later = {**batch, "audio": batch["audio"].clone()}
        later["audio"][..., 160000:] = torch.randn_like(later["audio"][..., 160000:]) * 0.05
        with torch.no_grad():
            a, b = model.embed(batch), model.embed(later)
        self.assertEqual(float((a[:, :41] - b[:, :41]).abs().max()), 0.0)  # t_40 = 10.0 s reads up to 10.0 s
        self.assertGreater(float((a[:, 41:] - b[:, 41:]).abs().max()), 0.0)

    def test_full_mode_reads_the_future(self) -> None:
        model, batch = _model(), _batch()
        model.context = "full"
        later = {**batch, "audio": batch["audio"].clone()}
        later["audio"][..., 160000:] = torch.randn_like(later["audio"][..., 160000:]) * 0.05
        with torch.no_grad():
            a, b = model.embed(batch), model.embed(later)
        self.assertGreater(float((a[:, :40] - b[:, :40]).abs().max()), 0.0)

    def test_spatial_part_ignores_channel_order(self) -> None:
        """Position-only Track A: permuting channels (audio, mask, coordinates) changes nothing."""
        model, batch = _model(use_mel=False), _batch("1101")
        perm = [2, 0, 3, 1]
        shuffled = {k: v[:, perm] for k, v in batch.items()}
        with torch.no_grad():
            a, b = model.embed(batch), model.embed(shuffled)
        torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)

    def test_inactive_channel_content_is_ignored(self) -> None:
        model, batch = _model(), _batch("1010")
        noisy = {**batch, "audio": batch["audio"].clone()}
        noisy["audio"][:, 1] = torch.randn(2, S)
        with torch.no_grad():
            torch.testing.assert_close(model.embed(batch), model.embed(noisy))

    def test_position_only_model_has_nothing_at_one_microphone(self) -> None:
        model = _model(use_mel=False)
        with torch.no_grad():
            a, b = model.embed(_batch("1000", seed=1)), model.embed(_batch("1000", seed=2))
        torch.testing.assert_close(a, b)

    def test_unit_rows_and_size(self) -> None:
        from models import build_system

        full = build_system(load_config(_REPO_ROOT / "configs" / "track_a.json"))
        self.assertLess(sum(p.numel() for p in full.parameters()), 3_000_000)
        model = _model()
        with torch.no_grad():
            z = model.embed(_batch("1100"))
        self.assertEqual(tuple(z.shape), (2, 80, 16))
        norms = z.norm(dim=-1)
        self.assertTrue(bool(((norms - 1).abs() < 1e-4)[:, 1:].all()))

    def test_training_draws_both_modes_with_matching_labels(self) -> None:
        model = _model().train()
        seen = set()
        torch.manual_seed(3)
        batch = _batch(B=1)
        with torch.no_grad():
            for _ in range(12):
                model.embed(batch)
                seen.add((model.last_context, model.train_label_key))
        self.assertEqual(seen, {("causal", "labels"), ("full", "labels_eval")})


@needs_torch
class TestTrackARun(unittest.TestCase):
    def test_smoke_run_trains_validates_both_modes_and_completes(self) -> None:
        import train
        from training.engine import run_is_complete

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            config = _REPO_ROOT / "configs" / "smoke_track_a.json"
            self.assertEqual(train.main(["--config", str(config), "--data-root", str(Path(tmp) / "data"),
                                         "--run-dir", str(run_dir), "--device", "cpu"]), 0)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["system"], "track_a")
            self.assertEqual(sorted(summary["full_context"]["theta_official_by_M"]), ["1", "2", "3", "4"])
            self.assertIn("colocated", summary["diagnostics"])
            self.assertTrue(run_is_complete(run_dir, load_config(config)))


if __name__ == "__main__":
    unittest.main()
