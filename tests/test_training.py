"""Invariants of the speaker-attribution systems and the training scaffold.

Needs torch; without it the whole module is skipped, so ``make test`` keeps
passing on the stdlib + numpy tier.  The checks are the properties the page
promises and a bug would silently break:

* causality -- nothing at ``t_k`` changes when audio after ``t_k`` changes;
* Chunk-Reset reads only its own chunk ``[t_k - 1, t_k)``;
* one set of weights for any channel subset: the embedding does not depend on
  channel order, and an inactive channel's content is ignored;
* one active channel gives the spatial branch nothing to work with.

Run with::

    python3 -m unittest tests.test_training -v
"""

from __future__ import annotations

import doctest
import random
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import numpy as np
    import torch
except ImportError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"torch/numpy not installed: {exc}") from exc

import spec  # noqa: E402
from models import build_system  # noqa: E402
from models.common import GRID, StubVoice, causal_vad  # noqa: E402
from training.config import load_config_dict  # noqa: E402

S = int(spec.SESSION_LENGTH_SEC * spec.SAMPLE_RATE_HZ)


def _batch(seed: int = 0, mask_bits: str = "1111", B: int = 2) -> dict:
    g = torch.Generator().manual_seed(seed)
    mask = torch.tensor([c == "1" for c in mask_bits])
    audio = torch.randn(B, 4, S, generator=g) * 0.05 * mask[None, :, None]
    xy = torch.tensor([[0.06, 0.0], [0.0, 0.05], [-0.07, 0.0], [0.0, -0.09]])
    return {"audio": audio, "mask": mask[None].expand(B, -1).clone(), "mic_xy": xy[None].expand(B, -1, -1).clone()}


def _proposed(state: str):
    cfg = load_config_dict({"name": "t", "model": {"name": "proposed", "state": state, "voice_encoder": "stub",
                                                    "d_model": 24, "n_tac_blocks": 1, "d_spatial": 8}})
    torch.manual_seed(0)
    return build_system(cfg, StubVoice()).eval()


class TestCausality(unittest.TestCase):
    def _assert_prefix_only(self, embed) -> None:
        a = _batch(1)
        b = {k: v.clone() for k, v in a.items()}
        k_cut = 41
        b["audio"][..., k_cut * GRID :] = torch.randn_like(b["audio"][..., k_cut * GRID :])
        with torch.no_grad():
            za, zb = embed(a), embed(b)
        torch.testing.assert_close(za[:, : k_cut + 1], zb[:, : k_cut + 1], rtol=0, atol=1e-5)
        self.assertGreater(float((za[:, k_cut + 1 :] - zb[:, k_cut + 1 :]).abs().max()), 1e-3)

    def test_carry_over_never_reads_the_future(self) -> None:
        self._assert_prefix_only(_proposed("carry").embed)

    def test_chunk_reset_never_reads_the_future(self) -> None:
        self._assert_prefix_only(_proposed("reset").embed)

    def test_vad_never_reads_the_future(self) -> None:
        a = _batch(1)["audio"][:, 0]
        b = a.clone()
        b[:, 41 * GRID :] = 0.0  # the future goes silent
        va, vb = causal_vad(a), causal_vad(b)
        self.assertTrue(torch.equal(va[:, :42], vb[:, :42]))
        self.assertFalse(bool(vb[:, 43:].any()))

    def test_chunk_reset_forgets_everything_before_its_chunk(self) -> None:
        model = _proposed("reset")
        a = _batch(2)
        b = {k: v.clone() for k, v in a.items()}
        b["audio"][..., : 30 * GRID] = torch.randn_like(b["audio"][..., : 30 * GRID])
        with torch.no_grad():
            za, zb = model.embed(a), model.embed(b)
        # chunk k covers [k*GRID - 16000, k*GRID): untouched once k*GRID - 16000 >= 30*GRID
        torch.testing.assert_close(za[:, 34:], zb[:, 34:], rtol=0, atol=1e-5)

    def test_carry_over_remembers_before_its_chunk(self) -> None:
        model = _proposed("carry")
        hidden = model.gru.hidden_size
        with torch.no_grad():  # update gate ~0.99: a slow integrator, so memory outlives one chunk
            for layer in range(model.gru.num_layers):
                getattr(model.gru, f"bias_ih_l{layer}")[hidden : 2 * hidden] = 4.6
        a = _batch(2)
        b = {k: v.clone() for k, v in a.items()}
        b["audio"][..., 29 * GRID : 30 * GRID] = torch.randn_like(b["audio"][..., 29 * GRID : 30 * GRID])
        with torch.no_grad():
            za, zb = model.embed(a), model.embed(b)
        self.assertGreater(float((za[:, 34] - zb[:, 34]).abs().max()), 1e-4)


class TestCausalityEverySystem(unittest.TestCase):
    """spec.CAUSAL_WINDOW_RULE's splice test, for every rung of the ladder."""

    def test_splicing_the_future_changes_nothing_before_it(self) -> None:
        for name in ("voice_only", "space_only", "naive_fusion", "proposed"):
            cfg = load_config_dict({"name": name, "model": {"name": name, "voice_encoder": "stub", "d_model": 16,
                                                            "n_tac_blocks": 1, "d_spatial": 8}})
            torch.manual_seed(0)
            system = build_system(cfg, StubVoice()).eval()
            base = _batch(9, "1110", B=1)
            with torch.no_grad():
                z0 = system.embed(base)
            for splice_sec in (0.10, 5.03, 10.0):
                cut = int(splice_sec * spec.SAMPLE_RATE_HZ)
                other = {k: v.clone() for k, v in base.items()}
                other["audio"][..., cut:] = torch.randn_like(other["audio"][..., cut:]) * other["mask"][:, :, None]
                with torch.no_grad():
                    z1 = system.embed(other)
                last = int(splice_sec // spec.DELTA_T_SEC)  # every t_k <= splice
                torch.testing.assert_close(z0[:, : last + 1], z1[:, : last + 1], rtol=0, atol=1e-5,
                                           msg=f"{name} read past {splice_sec} s")


class TestChannelInvariance(unittest.TestCase):
    def test_spatial_branch_ignores_channel_order(self) -> None:
        model = _proposed("carry")
        a = _batch(3)
        perm = [2, 0, 3, 1]
        b = {k: (v[:, perm] if k in ("audio", "mask", "mic_xy") else v) for k, v in a.items()}
        with torch.no_grad():
            ha = model.temporal(model.spatial_frames(a["audio"], a["mask"], a["mic_xy"]))
            hb = model.temporal(model.spatial_frames(b["audio"], b["mask"], b["mic_xy"]))
        torch.testing.assert_close(ha, hb, rtol=1e-4, atol=1e-4)

    def test_inactive_channel_content_is_ignored(self) -> None:
        model = _proposed("carry")
        a = _batch(4, "1101")
        b = {k: v.clone() for k, v in a.items()}
        b["audio"][:, 2] = torch.randn_like(b["audio"][:, 2])  # channel 2 is off
        with torch.no_grad():
            torch.testing.assert_close(model.embed(a), model.embed(b), rtol=0, atol=1e-6)

    def test_one_channel_leaves_the_spatial_branch_blind(self) -> None:
        model = _proposed("carry")
        a, b = _batch(5, "0100"), _batch(6, "0100")
        with torch.no_grad():
            ha = model.temporal(model.spatial_frames(a["audio"], a["mask"], a["mic_xy"]))
            hb = model.temporal(model.spatial_frames(b["audio"], b["mask"], b["mic_xy"]))
        torch.testing.assert_close(ha, hb, rtol=0, atol=1e-5)


class TestLadder(unittest.TestCase):
    def test_proposed_at_one_microphone_is_voice_only(self) -> None:
        from models.slots import online_slots

        batch = _batch(13, "0010", B=1)
        torch.manual_seed(0)
        proposed = _proposed("carry")
        voice = build_system(load_config_dict({"name": "v", "model": {"name": "voice_only", "voice_encoder": "stub"}}),
                             proposed.voice)
        with torch.no_grad():
            z_p, z_v = proposed.embed(batch)[0].numpy(), voice.embed(batch)[0].numpy()
        speech = np.ones(80, dtype=bool)
        np.testing.assert_array_equal(online_slots(z_p, speech, [0.3, 0.6, 0.9], 4), online_slots(z_v, speech, [0.3, 0.6, 0.9], 4))

    def test_every_system_embeds_unit_rows(self) -> None:
        batch = _batch(7, "1110", B=1)
        for name in ("voice_only", "space_only", "naive_fusion", "proposed"):
            cfg = load_config_dict({"name": name, "model": {"name": name, "voice_encoder": "stub", "d_model": 16,
                                                            "n_tac_blocks": 1, "d_spatial": 8}})
            with torch.no_grad():
                z = build_system(cfg, StubVoice()).eval().embed(batch)
            self.assertEqual(tuple(z.shape[:2]), (1, spec.N_GRID_POINTS), name)
            norms = z[:, 1:].norm(dim=-1)
            torch.testing.assert_close(norms, torch.ones_like(norms), rtol=0, atol=1e-4, msg=name)

    def test_space_only_cannot_separate_at_one_microphone(self) -> None:
        cfg = load_config_dict({"name": "s", "model": {"name": "space_only"}})
        with torch.no_grad():
            z = build_system(cfg).embed(_batch(8, "1000", B=1))
        self.assertTrue(torch.allclose(z, z[:, :1].expand_as(z)))


@unittest.skipUnless(__import__("os").environ.get("VCSASR_ECAPA_DIR"), "set VCSASR_ECAPA_DIR to a cached ECAPA checkpoint")
class TestFrozenECAPA(unittest.TestCase):
    """The voice cue must stay the voice-only system's, bit for bit, while the proposed model trains."""

    @classmethod
    def setUpClass(cls) -> None:
        import os

        from models.common import FrozenECAPA

        cls.voice = FrozenECAPA("speechbrain/spkrec-ecapa-voxceleb", os.environ["VCSASR_ECAPA_DIR"], "cpu", batch=64)

    def _cfg(self, name: str):
        return load_config_dict({"name": name, "model": {"name": name, "d_model": 16, "n_tac_blocks": 1, "d_spatial": 8}})

    def test_training_mode_does_not_reach_ecapa(self) -> None:
        model = build_system(self._cfg("proposed"), self.voice)
        model.train()
        self.assertFalse(self.voice.model.training)
        self.assertFalse(any(name.startswith("voice") for name, _ in model.named_parameters()))
        self.assertFalse(any(key.startswith("voice") for key in model.state_dict()))

    def test_a_chunk_does_not_depend_on_its_batch_mates(self) -> None:
        from models.common import voice_embeddings

        pair = _batch(10, "1111", B=2)
        alone = {k: v[:1] for k, v in pair.items()}
        torch.testing.assert_close(voice_embeddings(self.voice, pair)[:1], voice_embeddings(self.voice, alone), rtol=0, atol=1e-5)

    def test_mixed_precision_cannot_reach_the_encoder(self) -> None:
        from models.common import voice_embeddings

        batch = _batch(12, "1100", B=1)
        batch["audio"][..., : 3 * spec.SAMPLE_RATE_HZ // 4] = 0.0  # silent lead-in chunks
        plain = voice_embeddings(self.voice, batch)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            mixed = voice_embeddings(self.voice, batch)
        self.assertTrue(bool(torch.isfinite(mixed).all()))
        torch.testing.assert_close(mixed, plain, rtol=0, atol=1e-6)

    def test_proposed_voice_part_is_the_voice_only_embedding(self) -> None:
        import torch.nn.functional as F

        batch = _batch(11, "1010", B=1)
        torch.manual_seed(0)
        proposed = build_system(self._cfg("proposed"), self.voice)
        proposed.train()  # even in training mode
        with torch.no_grad():
            z = proposed.embed(batch)
            v = build_system(self._cfg("voice_only"), self.voice).embed(batch)
        voice_part = F.normalize(z[..., -self.voice.dim :], dim=-1)
        torch.testing.assert_close(voice_part[:, 1:], v[:, 1:], rtol=0, atol=1e-5)


class TestRobustness(unittest.TestCase):
    """What a dropped Colab runtime or a half-uploaded Drive file must not break."""

    def test_resume_falls_back_past_a_corrupt_checkpoint(self) -> None:
        import tempfile

        from training.engine import Checkpoints

        with tempfile.TemporaryDirectory() as tmp:
            ckpts = Checkpoints(Path(tmp), keep=3)
            for step in (10, 20, 30):
                ckpts.save({"step": step}, step)
            newest = Path(tmp) / "ckpt" / "step_0000030.pt"
            newest.write_bytes(newest.read_bytes()[:20])  # cut short, as an interrupted upload would be
            logs: list[str] = []
            state = ckpts.load_latest("cpu", logs.append)
            self.assertEqual(state["step"], 20)
            self.assertTrue(any("step_0000030" in m for m in logs))
            ckpts.save({"step": 40}, 40)
            self.assertEqual([e["step"] for e in ckpts.entries()], [40, 30, 20])
            self.assertFalse((Path(tmp) / "ckpt" / "step_0000010.pt").exists())

    def test_a_truncated_bank_shard_is_rebuilt_from_scratch(self) -> None:
        import tempfile

        from training.rirbank import BankSpec, _shard_ok, bank_ready, build_bank

        bank = BankSpec(5, 60, 0.1, (0.15, 0.3), "fallback_ism")
        with tempfile.TemporaryDirectory() as tmp:
            local, drive = Path(tmp) / "local", Path(tmp) / "drive"
            self.assertFalse(bank_ready(bank, local, drive))
            build_bank(bank, local, workers=1, log=lambda *_: None, mirror=drive)
            self.assertTrue(bank_ready(bank, local, drive))
            shard = drive / bank.dirname / "rirs_0001.npy"
            shard.write_bytes(shard.read_bytes()[:100])
            import shutil

            shutil.rmtree(local)
            self.assertFalse(bank_ready(bank, local, drive))
            build_bank(bank, local, workers=1, log=lambda *_: None, mirror=drive)
            self.assertTrue(all(_shard_ok(d / bank.dirname, bank, s) for d in (local, drive) for s in (0, 1)))

    def test_a_later_session_without_the_simulator_reuses_the_mirrored_bank(self) -> None:
        """Colab installs pyroomacoustics only when a bank is missing; a later session must not need it."""
        import json
        import shutil
        import tempfile

        try:
            import pyroomacoustics  # noqa: F401
        except ImportError:
            self.skipTest("pyroomacoustics not installed")
        from training.rirbank import BankSpec, RirBank, build_bank

        bank = BankSpec(9, 2, 0.1, (0.15, 0.2), "pyroomacoustics", variants=2)
        with tempfile.TemporaryDirectory() as tmp:
            local, drive = Path(tmp) / "local", Path(tmp) / "drive"
            build_bank(bank, local, workers=1, log=lambda *_: None, mirror=drive)
            version = json.loads((drive / bank.dirname / "bank.json").read_text())["engine_version"]
            shutil.rmtree(local)
            saved = sys.modules.get("pyroomacoustics")
            sys.modules["pyroomacoustics"] = None  # import now fails, as on a fresh runtime
            try:
                path = build_bank(bank, local, workers=1, log=lambda *_: None, mirror=drive)
            finally:
                sys.modules["pyroomacoustics"] = saved
            self.assertEqual(json.loads((path / "bank.json").read_text())["engine_version"], version)
            self.assertEqual(len(RirBank(path)), 2)

    def test_bank_shape_settings_reach_the_bank(self) -> None:
        from training.engine import needed_banks

        cfg = load_config_dict({"name": "b", "data": {"bank_variants": 2, "jitter_m": [0.2, 0.4]}})
        for bank in needed_banks(cfg).values():
            self.assertEqual((bank.variants, bank.jitter_m), (2, (0.2, 0.4)))

    def test_a_non_finite_embedding_outside_the_labels_cannot_poison_the_loss(self) -> None:
        from training.loss import SessionCentroidLoss

        z = torch.nn.functional.normalize(torch.randn(1, 6, 4), dim=-1)
        z[0, 0] = float("nan")  # an unlabelled instant, e.g. a silent lead-in chunk
        loss, _ = SessionCentroidLoss()(z, torch.tensor([[-1, 0, 0, 1, 1, 0]]))
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_mixed_engine_versions_are_refused_when_completing_a_bank(self) -> None:
        import json
        import tempfile
        import unittest.mock

        from training.rirbank import BankSpec, _build_shard, build_bank

        bank = BankSpec(13, 60, 0.1, (0.15, 0.2), "fallback_ism")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with unittest.mock.patch("training.rirbank._engine_version", return_value="0.8.5"):
                build_bank(bank, root, workers=1, log=lambda *_: None)
            self.assertEqual(json.loads((root / bank.dirname / "bank.json").read_text())["engine_version"], "0.8.5")
            (root / bank.dirname / "rirs_0001.npy").unlink()
            with unittest.mock.patch("training.rirbank._engine_version", return_value="0.10.1"):
                with self.assertRaises(ValueError):
                    build_bank(bank, root, workers=1, log=lambda *_: None)

        # Also when the very first build is interrupted after shard 0 before completion:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def crash_on_second(job):
                if job[1] == 1:
                    raise RuntimeError("simulated disconnect after shard 0")
                return _build_shard(job)

            with unittest.mock.patch("training.rirbank._engine_version", return_value="0.8.5"), \
                 unittest.mock.patch("training.rirbank._build_shard", side_effect=crash_on_second):
                with self.assertRaises(RuntimeError):
                    build_bank(bank, root, workers=1, log=lambda *_: None)
            self.assertTrue((root / bank.dirname / "bank.json").is_file())
            with unittest.mock.patch("training.rirbank._engine_version", return_value="0.10.1"):
                with self.assertRaises(ValueError):
                    build_bank(bank, root, workers=1, log=lambda *_: None)

    def test_corrupted_best_pt_is_recovered_on_resume(self) -> None:
        import tempfile
        import unittest.mock

        from training.engine import Run

        cfg = load_config_dict({"name": "s", "data": {"synthetic": True, "n_rooms_train": 2, "n_rooms_val": 2,
                                                      "n_val_sessions": 2, "num_workers": 0},
                                "model": {"name": "proposed", "voice_encoder": "stub", "d_model": 16,
                                          "n_tac_blocks": 1, "d_spatial": 8},
                                "train": {"max_steps": 4}})
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            data_dir = Path(tmp) / "data"
            run = Run(cfg, run_dir, data_dir, device="cpu", log=lambda *_: None)
            run.step = 2
            run.best = 0.90
            run.ckpts.save(run._state(), run.step)
            (run_dir / "best.pt").write_bytes(b"corrupted-half-upload")

            resumed = Run(cfg, run_dir, data_dir, device="cpu", log=lambda *_: None)
            resumed._resume()
            self.assertEqual(resumed.step, 2)
            self.assertEqual(resumed.best, float("-inf"))
            fake_val = {
                "spk_acc": 0.80,
                "spk_acc_by_M": {"1": 0.5, "2": 0.80, "3": 0.80, "4": 0.80},
                "spk_acc_n2_cap2": 0.80,
                "spk_acc_n2_cap2_by_M": {"1": 0.5, "2": 0.80, "3": 0.80, "4": 0.80},
                "theta_by_M": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
                "theta_official_by_M": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
            }
            with unittest.mock.patch.object(resumed, "validate", return_value=dict(fake_val)):
                resumed._validate_and_keep_best()
            self.assertAlmostEqual(resumed.best, 0.80, places=4)
            self.assertIsNotNone(resumed._load_best("cpu"))

    def test_fusion_and_fixed_gate_weights_tune_under_official_two_slot_cap(self) -> None:
        import tempfile

        from training.engine import Run

        # 10-frame 2-talker case:
        # - At w=0 (voice), frames 0 and 1 have opposite outliers (cos = -1 < -0.30) that consume
        #   both slots under cap=2 (acc = 0.60), whereas cap=4 opens slots 2 and 3 for talkers 0 and 1
        #   and achieves acc = 0.80.
        # - At w=1 (space), only two directions appear so cap=2 and cap=4 both achieve acc = 0.70.
        # Therefore cap=4 selects w=0.0 (0.80 > 0.70), while official cap=2 selects w=1.0 (0.70 > 0.60).
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
        speech = np.ones(10, dtype=bool)
        voice = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        space = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        case = {
            "index": 0, "mask": "1111", "M": 4, "N": 2,
            "speech": speech, "labels": labels, "labels_eval": labels, "pos_labels": labels,
            "voice": voice.copy(), "space": space.copy(), "spatial": space.copy(),
            "emb": space.copy(), "tdoa_sep": 1.0,
            "meta": {"speaker_ids": ["a", "b"], "source_azimuth_deg": [0.0, 90.0]},
        }
        cfg_nf = load_config_dict({"name": "nf", "data": {"synthetic": True, "n_rooms_train": 2, "n_rooms_val": 2,
                                                          "n_val_sessions": 1, "num_workers": 0},
                                   "model": {"name": "naive_fusion", "voice_encoder": "stub"},
                                   "slots": {"weight_grid": [0.0, 1.0, 1.0]}})
        with tempfile.TemporaryDirectory() as tmp:
            run_nf = Run(cfg_nf, Path(tmp) / "nf", Path(tmp) / "data", device="cpu", log=lambda *_: None)
            self.assertEqual(run_nf._tune_fusion_weights([dict(case)], cap=4)["best_by_M"]["4"], 0.0)
            self.assertEqual(run_nf._tune_fusion_weights([dict(case)], cap=2)["best_by_M"]["4"], 1.0)
            # Pad to 80 frames for official_speaker_check inside validate(quick=False)
            pad = 80 - len(labels)
            case80 = {
                **case,
                "speech": np.pad(speech, (0, pad), constant_values=False),
                "labels": np.pad(labels, (0, pad), constant_values=-1),
                "labels_eval": np.pad(labels, (0, pad), constant_values=-1),
                "pos_labels": np.pad(labels, (0, pad), constant_values=-1),
                "voice": np.pad(voice, ((0, pad), (0, 0)), mode="edge"),
                "space": np.pad(space, ((0, pad), (0, 0)), mode="edge"),
                "spatial": np.pad(space, ((0, pad), (0, 0)), mode="edge"),
                "emb": np.pad(space, ((0, pad), (0, 0)), mode="edge"),
            }
            val_res = run_nf.validate(quick=False, cases=[dict(case80)])
            self.assertEqual(val_res["fusion_weight_by_M"]["4"], 1.0)
            self.assertEqual(val_res["weight_search_cap4"]["best_by_M"]["4"], 0.0)

    def test_run_and_eval_completion_checks_respect_overrides(self) -> None:
        import json
        import tempfile

        import eval_t2
        from training.config import VAL_PROTOCOL_VERSION, config_fingerprint
        from training.engine import atomic_save, run_is_complete

        cfg10 = load_config_dict({"name": "p", "model": {"name": "proposed", "voice_encoder": "stub",
                                                         "d_model": 16, "n_tac_blocks": 1, "d_spatial": 8},
                                  "train": {"max_steps": 10}})
        cfg20 = load_config_dict({"name": "p", "model": {"name": "proposed", "voice_encoder": "stub",
                                                         "d_model": 16, "n_tac_blocks": 1, "d_spatial": 8},
                                  "train": {"max_steps": 20}})
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            fp = config_fingerprint(cfg10)
            atomic_save({"step": 10, "fingerprint": fp, "val_protocol_version": VAL_PROTOCOL_VERSION,
                         "val": {"select_score": 0.8}}, run_dir / "best.pt")
            summary = {
                "run": "run", "config": "p", "system": "proposed", "trained_steps": 10, "best_step": 10,
                "fingerprint": fp, "val_protocol_version": VAL_PROTOCOL_VERSION,
                "theta_official_by_M": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
                "diagnostics": {
                    "learned_space_only_theta_official_by_M": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
                    "fixed_gate_theta_official_by_M": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
                    "fixed_gate_weight_by_M": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
                },
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            self.assertTrue(run_is_complete(run_dir, cfg10))
            # Increasing max_steps via override must mark the run incomplete
            self.assertFalse(run_is_complete(run_dir, cfg20))

            # eval_is_complete checks both embeddings.npz fingerprint and score.json metadata/thresholds
            eval_dir = Path(tmp) / "eval"
            (eval_dir / "eval-t2" / "full").mkdir(parents=True)
            exp_fp = eval_t2.inputs_fingerprint(cfg10, eval_t2.REPO / "data" / "eval-t2", run_dir / "best.pt", 0)
            np.savez_compressed(eval_dir / "eval-t2" / "embeddings.npz", fingerprint=np.asarray(exp_fp))
            score = {
                "eval_t2_run": {
                    "limit": 0,
                    "theta_by_M": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
                    "weight_by_M": None,
                    "summary": {"run": "run", "config": "p", "best_step": 10, "fingerprint": fp,
                                "val_protocol_version": VAL_PROTOCOL_VERSION},
                }
            }
            (eval_dir / "eval-t2" / "full" / "score.json").write_text(json.dumps(score))
            self.assertTrue(eval_t2.eval_is_complete(
                eval_dir, cfg10, summary, best=run_dir / "best.pt", sets=("eval-t2",), variants=("full",)
            ))
            summary_new = {**summary, "best_step": 20}
            self.assertFalse(eval_t2.eval_is_complete(
                eval_dir, cfg20, summary_new, best=run_dir / "best.pt", sets=("eval-t2",), variants=("full",)
            ))
            # A score.json written before the protocol version existed must be re-scored
            old_score = json.loads(json.dumps(score))
            old_score["eval_t2_run"]["summary"].pop("val_protocol_version")
            (eval_dir / "eval-t2" / "full" / "score.json").write_text(json.dumps(old_score))
            self.assertFalse(eval_t2.eval_is_complete(
                eval_dir, cfg10, summary, best=run_dir / "best.pt", sets=("eval-t2",), variants=("full",)
            ))

    def test_completion_rejects_artifacts_from_an_older_validation_protocol(self) -> None:
        """A baseline summary from 1f71328 (4-slot weights, no protocol version) must not count as complete."""
        import argparse
        import json
        import tempfile

        import eval_t2
        from training.config import VAL_PROTOCOL_VERSION, config_fingerprint
        from training.engine import atomic_save, run_is_complete

        cfg_nf = load_config_dict({"name": "naive_fusion", "model": {"name": "naive_fusion"}})
        cfg_p = load_config_dict({"name": "p", "model": {"name": "proposed", "voice_encoder": "stub",
                                                         "d_model": 16, "n_tac_blocks": 1, "d_spatial": 8},
                                  "train": {"max_steps": 10}})
        theta = {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5}
        with tempfile.TemporaryDirectory() as tmp:
            # (1) untrained rung: an old summary has the right fingerprint but no protocol version
            nf_dir = Path(tmp) / "nf"
            nf_dir.mkdir()
            old = {"run": "nf", "config": "naive_fusion", "system": "naive_fusion",
                   "fingerprint": config_fingerprint(cfg_nf), "theta_official_by_M": theta,
                   "fusion_weight_by_M": {"1": 0.0, "2": 0.3, "3": 0.3, "4": 0.3}}
            (nf_dir / "summary.json").write_text(json.dumps(old))
            self.assertFalse(run_is_complete(nf_dir, cfg_nf))
            (nf_dir / "summary.json").write_text(json.dumps({**old, "val_protocol_version": VAL_PROTOCOL_VERSION - 1}))
            self.assertFalse(run_is_complete(nf_dir, cfg_nf))
            (nf_dir / "summary.json").write_text(json.dumps({**old, "val_protocol_version": VAL_PROTOCOL_VERSION}))
            self.assertTrue(run_is_complete(nf_dir, cfg_nf))

            # eval_t2 refuses an old-protocol summary instead of scoring with its 4-slot weights
            cfg_path = Path(tmp) / "naive_fusion.json"
            cfg_path.write_text(json.dumps({"name": "naive_fusion", "model": {"name": "naive_fusion"}}))
            old_path = Path(tmp) / "old_summary.json"
            old_path.write_text(json.dumps(old))
            ns = argparse.Namespace(run_dir=None, config=str(cfg_path), summary=str(old_path), tune_on_eval=False)
            with self.assertRaises(SystemExit):
                eval_t2.load_system_inputs(ns)
            with self.assertRaises(SystemExit):
                eval_t2.summary_settings(old, "naive_fusion", "full")

            # (2) trained rung: an old best.pt without the protocol version is neither complete nor resumed as best
            p_dir = Path(tmp) / "p"
            p_dir.mkdir()
            fp = config_fingerprint(cfg_p)
            atomic_save({"step": 10, "fingerprint": fp, "val": {"select_score": 0.8}}, p_dir / "best.pt")
            (p_dir / "summary.json").write_text(json.dumps({
                "run": "p", "config": "p", "system": "proposed", "trained_steps": 10, "best_step": 10,
                "fingerprint": fp, "val_protocol_version": VAL_PROTOCOL_VERSION, "theta_official_by_M": theta}))
            self.assertFalse(run_is_complete(p_dir, cfg_p))

    def test_best_checkpoint_selection_uses_official_two_slot_cap(self) -> None:
        """Selection score must follow the official 2-talker 2-slot cap (M>=2), not the 4-slot diagnostic."""
        import tempfile
        import unittest.mock

        from training.engine import Run

        cfg = load_config_dict({"name": "s", "data": {"synthetic": True, "n_rooms_train": 2, "n_rooms_val": 2,
                                                      "n_val_sessions": 2, "num_workers": 0},
                                "model": {"name": "proposed", "voice_encoder": "stub", "d_model": 16,
                                          "n_tac_blocks": 1, "d_spatial": 8},
                                "train": {"max_steps": 1}})
        with tempfile.TemporaryDirectory() as tmp:
            run = Run(cfg, Path(tmp) / "run", Path(tmp) / "data", device="cpu", log=lambda *_: None)
            fake_val = {
                "spk_acc": 0.90,
                "spk_acc_by_M": {"1": 0.5, "2": 0.95, "3": 0.95, "4": 0.95},
                "spk_acc_n2_cap2": 0.70,
                "spk_acc_n2_cap2_by_M": {"1": 0.5, "2": 0.70, "3": 0.72, "4": 0.74},
                "theta_by_M": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
                "theta_official_by_M": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5},
            }
            with unittest.mock.patch.object(run, "validate", return_value=dict(fake_val)):
                run._validate_and_keep_best()
            self.assertAlmostEqual(run.best, 0.72, places=4)

    def test_consecutive_non_finite_batches_abort_training(self) -> None:
        import tempfile
        import unittest.mock

        from training.engine import Run

        cfg = load_config_dict({"name": "s", "data": {"synthetic": True, "n_rooms_train": 2, "n_rooms_val": 2,
                                                      "n_val_sessions": 2, "num_workers": 0},
                                "model": {"name": "proposed", "voice_encoder": "stub", "d_model": 16,
                                          "n_tac_blocks": 1, "d_spatial": 8},
                                "train": {"max_steps": 10, "batch_size": 2, "max_consecutive_skips": 3}})
        with tempfile.TemporaryDirectory() as tmp:
            run = Run(cfg, Path(tmp) / "run", Path(tmp) / "data", device="cpu", log=lambda *_: None)
            bad = (torch.tensor(float("nan")), {"acc": 0.0})
            with unittest.mock.patch.object(run, "loss_fn", return_value=bad):
                with self.assertRaises(RuntimeError):
                    run.fit()
            self.assertEqual(run.step, 0)

    def test_eval_t2_guards_config_fingerprint_and_component_cache(self) -> None:
        import argparse
        import json
        import tempfile

        import eval_t2
        from training.config import config_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_v = load_config_dict({"name": "v", "model": {"name": "voice_only"}})
            cfg_s1 = load_config_dict({"name": "s1", "model": {"name": "space_only", "srp_temperature": 1.0}})
            cfg_s10 = load_config_dict({"name": "s10", "model": {"name": "space_only", "srp_temperature": 10.0}})
            cfg_nf = load_config_dict({"name": "nf", "model": {"name": "naive_fusion", "srp_temperature": 1.0}})

            # (1) --config + --summary rejects a summary from a different config fingerprint
            cfg_s10_path = root / "s10.json"
            cfg_s10_path.write_text(json.dumps({"name": "s10", "model": {"name": "space_only", "srp_temperature": 10.0}}))
            sum_s1_path = root / "summary_s1.json"
            sum_s1_path.write_text(json.dumps({"system": "space_only", "fingerprint": config_fingerprint(cfg_s1),
                                               "theta_official_by_M": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5}}))
            ns = argparse.Namespace(run_dir=None, config=str(cfg_s10_path), summary=str(sum_s1_path), tune_on_eval=False)
            with self.assertRaises(SystemExit):
                eval_t2.load_system_inputs(ns)

            # (2) --components-from invalidates cache when an upstream component changes
            set_dir = root / "eval-t2"
            set_dir.mkdir()
            (set_dir / "manifest.json").write_text('{"official": false}')
            v_dir, s_dir, nf_dir = root / "out_v", root / "out_s", root / "out_nf"
            for d in (v_dir, s_dir, nf_dir):
                (d / "eval-t2").mkdir(parents=True)

            base_fields = {
                "case_id": np.array(["c0"]), "session_id": np.array(["s0"]),
                "mask": np.array(["1111"]), "ref": np.array([0]),
                "speech": np.ones((1, 80), dtype=bool), "labels": np.zeros((1, 80), dtype=np.int64),
            }
            v1 = np.ones((1, 80, 4), dtype=np.float32)
            v2 = np.full((1, 80, 4), 2.0, dtype=np.float32)
            sp = np.ones((1, 80, 36), dtype=np.float32)
            fp_v = eval_t2.inputs_fingerprint(cfg_v, set_dir, None, 0)
            fp_s = eval_t2.inputs_fingerprint(cfg_s1, set_dir, None, 0)

            np.savez_compressed(v_dir / "eval-t2" / "embeddings.npz", fingerprint=np.asarray(fp_v), voice=v1, **base_fields)
            np.savez_compressed(s_dir / "eval-t2" / "embeddings.npz", fingerprint=np.asarray(fp_s), space=sp, **base_fields)

            args = argparse.Namespace(limit=0, components_from=[str(v_dir), str(s_dir)], device="cpu", model_cache=None)
            emb1 = eval_t2.load_or_embed(cfg_nf, set_dir, nf_dir / "eval-t2", args, None, [], {})
            np.testing.assert_array_equal(emb1["voice"], v1)

            # Replace voice component with new values and re-run: must not return stale v1
            np.savez_compressed(v_dir / "eval-t2" / "embeddings.npz", fingerprint=np.asarray(fp_v), voice=v2, **base_fields)
            emb2 = eval_t2.load_or_embed(cfg_nf, set_dir, nf_dir / "eval-t2", args, None, [], {})
            np.testing.assert_array_equal(emb2["voice"], v2)

            # Component with incompatible settings (e.g. srp_temperature=10 vs naive_fusion's 1) is rejected
            fp_s10 = eval_t2.inputs_fingerprint(cfg_s10, set_dir, None, 0)
            np.savez_compressed(s_dir / "eval-t2" / "embeddings.npz", fingerprint=np.asarray(fp_s10), space=sp, **base_fields)
            with self.assertRaises(SystemExit):
                eval_t2.load_or_embed(cfg_nf, set_dir, nf_dir / "eval-t2", args, None, [], {})


class TestOfficialPath(unittest.TestCase):
    def test_internal_spk_acc_matches_evaluate_py(self) -> None:
        """Per case and averaged over cases, across masks whose accuracy differs by M."""
        from models.slots import best_assignment_accuracy, online_slots
        from training.official import official_speaker_check

        rng = np.random.default_rng(0)
        cases = []
        for index in range(4):
            for mask in spec.EVAL_MASKS:
                labels = np.full(80, -1)
                labels[4:30], labels[33:50], labels[53:76] = 0, 1, 0
                emb = rng.standard_normal((80, 8))
                emb[labels == 1] += 0.5 * mask.count("1")  # easier with more microphones
                emb /= np.linalg.norm(emb, axis=1, keepdims=True)
                cases.append({"index": index, "mask": mask, "M": mask.count("1"), "N": 2, "emb": emb,
                              "speech": labels >= 0, "labels_eval": labels})
        theta_by_M = {1: 0.5, 2: 0.4, 3: 0.3, 4: 0.2}
        slots = lambda c: online_slots(c["emb"], c["speech"], [theta_by_M[c["M"]]], 2)[0]  # noqa: E731
        per_case = [best_assignment_accuracy(c["labels_eval"], slots(c)) for c in cases]
        self.assertGreater(len({round(a, 6) for a in per_case}), 1)  # accuracies really differ
        official = official_speaker_check(cases, slots)
        self.assertAlmostEqual(official["speaker_attribution_accuracy"], float(np.mean(per_case)), places=9)


class TestScenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import tempfile

        from training.corpus import synthetic_corpus
        from training.rirbank import BankSpec, open_bank

        cls._tmp = tempfile.TemporaryDirectory()
        cls.bank = open_bank(BankSpec(1, 2, 0.25, (0.15, 0.3), "fallback_ism"), cls._tmp.name, workers=1,
                             log=lambda *_: None)
        cls.corpus = synthetic_corpus(4, 2)
        cls.cfg = load_config_dict({"name": "t"}).data

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _recipe(self, **changes):  # noqa: ANN003, ANN202
        from dataclasses import replace

        from training.scenes import train_recipe

        return replace(train_recipe(self.cfg), **changes)

    def test_sessions_are_reproducible(self) -> None:
        from training.scenes import mix_session

        recipe = self._recipe(n_speakers=(2, 2), eval_layout_prob=0.0)
        a, audio_a = mix_session(random.Random("v:3"), self.corpus, self.bank, self.cfg, recipe)
        b, audio_b = mix_session(random.Random("v:3"), self.corpus, self.bank, self.cfg, recipe)
        np.testing.assert_array_equal(audio_a, audio_b)
        np.testing.assert_array_equal(a.labels, b.labels)
        self.assertEqual(a.n_speakers, 2)
        self.assertLessEqual(float(np.abs(audio_a).max()), 0.999 + 1e-6)
        # every talker returns at least once and is labelled somewhere
        self.assertEqual(sorted(set(a.labels[a.labels >= 0].tolist())), [0, 1])

    def test_validation_conditions_share_the_same_sessions(self) -> None:
        from training.scenes import draw_session, val_recipe

        cfg = self.cfg
        main = draw_session(random.Random("val:1"), self.corpus, self.bank, cfg, val_recipe(cfg, "main"))
        static = draw_session(random.Random("val:1"), self.corpus, self.bank, cfg, val_recipe(cfg, "static"))
        coloc = draw_session(random.Random("val:1"), self.corpus, self.bank, cfg, val_recipe(cfg, "colocated"))
        for other in (static, coloc):
            np.testing.assert_array_equal(main.dry, other.dry)
            np.testing.assert_array_equal(main.labels, other.labels)
            self.assertEqual(main.speaker_ids, other.speaker_ids)
        self.assertTrue(np.isnan(static.snr_db) and main.snr_db in cfg.val_snr_levels_db)
        self.assertEqual(int(static.variant_map.max()), 0)
        self.assertEqual(len(set(coloc.sources)), 1)
        self.assertEqual(main.layout, "eval")
        self.assertEqual(main.n_speakers, 2)

    def test_a_returning_talker_moves_to_another_copy(self) -> None:
        from training.scenes import draw_session, val_recipe

        returns = 0
        for i in range(20):
            session = draw_session(random.Random(f"val:{i}"), self.corpus, self.bank, self.cfg,
                                   val_recipe(self.cfg, "main"))
            for talker in range(session.n_speakers):
                copies = [t.variant for t in session.turns if t.speaker == talker]
                returns += len(copies) - 1
                self.assertTrue(all(a != b for a, b in zip(copies, copies[1:])), copies)
        self.assertGreater(returns, 0)

    def test_colocated_talkers_share_a_position_label(self) -> None:
        from training.scenes import draw_session

        recipe = self._recipe(n_speakers=(3, 3), eval_layout_prob=0.0, colocate_prob=1.0)
        session = draw_session(random.Random("c:2"), self.corpus, self.bank, self.cfg, recipe)
        self.assertLess(len(set(session.sources)), session.n_speakers)
        for k in np.flatnonzero(session.labels >= 0):
            self.assertEqual(session.pos_labels[k], session.sources[session.labels[k]])

    def test_device_renderer_matches_the_reference(self) -> None:
        from training.data import _item, collate
        from training.render import render_batch
        from training.scenes import draw_session, render_numpy

        sessions = [draw_session(random.Random(f"r:{i}"), self.corpus, self.bank, self.cfg,
                                 self._recipe(n_speakers=(n, n), noise_snr_db=None))
                    for i, n in enumerate((2, 3, 4))]
        self.assertTrue(any(int(s.variant_map.max()) > 0 for s in sessions))  # displaced turns are exercised
        batch = collate([_item(s, self.bank, "1101", dry_dtype=np.float32) for s in sessions])
        audio = render_batch(batch)["audio"].numpy()
        for b, session in enumerate(sessions):
            ref = render_numpy(session, self.bank) * np.array([1, 1, 0, 1], dtype=np.float32)[:, None]
            # identical up to float32 FFT rounding, which can move a sample by one 16-bit step
            self.assertLessEqual(float(np.abs(audio[b] - ref).max()), 2.0 / 32768.0)

    def test_noise_is_diffuse_and_at_the_requested_snr(self) -> None:
        from training.render import diffuse_noise

        xy = torch.tensor([[[0.0, 0.0], [0.02, 0.0], [0.24, 0.0], [0.0, 0.1]]])
        noise = diffuse_noise(xy, 4 * spec.SAMPLE_RATE_HZ, torch.tensor([7]))[0].numpy()

        def coherence(a: np.ndarray, b: np.ndarray, band: tuple[float, float]) -> float:
            n = 512
            frames = np.stack([noise[:, i : i + n] * np.hanning(n) for i in range(0, noise.shape[1] - n, n // 2)])
            X = np.fft.rfft(frames, axis=-1)
            f = np.fft.rfftfreq(n, 1 / spec.SAMPLE_RATE_HZ)
            sel = (f >= band[0]) & (f < band[1])
            cross = np.mean(X[:, a, sel] * np.conj(X[:, b, sel]), axis=0)
            pa, pb = np.mean(np.abs(X[:, a, sel]) ** 2, axis=0), np.mean(np.abs(X[:, b, sel]) ** 2, axis=0)
            return float(np.mean(np.real(cross) / np.sqrt(pa * pb)))

        # a 2 cm pair is nearly coherent at low frequencies, a 24 cm pair is not at 3-4 kHz
        self.assertGreater(coherence(0, 1, (100, 600)), 0.9)
        self.assertLess(abs(coherence(0, 2, (3000, 4000))), 0.2)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    import models.common
    import models.slots
    import models.tac
    import training.config
    import training.corpus
    import training.engine
    import training.loss
    import training.official
    import training.render
    import training.scenes

    for module in (models.common, models.slots, models.tac, training.config, training.corpus, training.engine,
                   training.loss, training.official, training.scenes):
        tests.addTests(doctest.DocTestSuite(module))
    return tests


if __name__ == "__main__":
    unittest.main()
