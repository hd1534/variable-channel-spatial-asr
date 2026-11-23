"""Track C: text and slot helpers, word-aligned sessions, and the unified model on a tiny random Qwen3-ASR.

The model tests need ``qwen-asr`` and ``peft`` (the Qwen venv) and are skipped
without them; they build a two-layer Qwen3-ASR from its own config classes and
a minimal processor, so nothing is downloaded.
"""

from __future__ import annotations

import doctest
import importlib.util
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None
HAS_QWEN = HAS_TORCH and all(importlib.util.find_spec(m) is not None for m in ("qwen_asr", "peft"))

from trackc import text as tt  # noqa: E402


class TestText(unittest.TestCase):
    def test_tags_and_slots_agree_on_first_appearance(self) -> None:
        turns = [SimpleNamespace(speaker=1, text="SO I WENT"), SimpleNamespace(speaker=0, text="DID YOU"),
                 SimpleNamespace(speaker=1, text="NO")]
        target = tt.target_text(turns)
        self.assertEqual(tt.parse_tagged(target), {"spk_0": "so i went no", "spk_1": "did you"})
        self.assertEqual(tt.slot_labels([-1, 1, 0, 1], [t.speaker for t in turns]), [-1, 0, 1, 0])

    def test_untagged_or_overflowing_output_still_scores(self) -> None:
        self.assertEqual(tt.parse_tagged("just words"), {"spk_0": "just words"})
        self.assertEqual(tt.parse_tagged(""), {})
        self.assertEqual(tt.grid_slots([0] * 80), [0] * 80)


@unittest.skipUnless(HAS_TORCH, "torch/numpy not installed")
class TestScenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from training.config import load_config
        from training.corpus import synthetic_corpus
        from training.engine import needed_banks
        from training.rirbank import RirBank, build_bank

        cls.cfg = load_config(_REPO_ROOT / "configs" / "smoke.json")
        cls.corpus = synthetic_corpus(tag="train")
        cls.align = {}
        for utts in cls.corpus.by_speaker.values():
            for u in utts:
                total = cls.corpus.n_samples(u) / 16000
                t, words, k = 0.3, [], 0
                while t + 0.35 < total - 0.2:
                    words.append((f"w{k}", t, t + 0.3))
                    t += 0.3 + (0.15 if k % 3 == 2 else 0.02)
                    k += 1
                cls.align[u.utt_id] = words
        import tempfile

        cls.tmp = tempfile.TemporaryDirectory()
        cls.bank = RirBank(str(build_bank(needed_banks(cls.cfg)["val"], Path(cls.tmp.name), log=lambda *_: None)))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_every_turn_is_whole_words_cut_in_a_real_gap(self) -> None:
        from trackc import scenes as sc

        reader = sc.AlignedReader(self.align)
        for seed in range(8):
            session = sc.draw_text_session(random.Random(seed), self.corpus, self.bank, self.cfg.data,
                                           sc.text_recipe(self.cfg.data), reader)
            self.assertEqual(len({t.speaker for t in session.turns}), 2)
            for turn in session.turns:
                self.assertTrue(turn.text)
                self.assertTrue(all(w.startswith("w") for w in turn.text.split()))
            labels = tt.slot_labels(session.labels_eval, [t.speaker for t in session.turns])
            self.assertEqual(next(x for x in labels if x >= 0), 0)  # the first talker heard is [S1]

    def test_cuts_only_in_gaps_long_enough(self) -> None:
        from trackc.scenes import MIN_CUT_GAP_SEC, AlignedReader

        cuts = AlignedReader.cut_times([("a", 0.0, 0.5), ("b", 0.5 + MIN_CUT_GAP_SEC / 2, 1.0), ("c", 1.2, 1.4)], 2.0)
        self.assertIsNone(cuts[1])
        self.assertAlmostEqual(cuts[2], 1.1)

    def test_items_collate_with_text(self) -> None:
        from trackc import scenes as sc

        session = sc.draw_text_session(random.Random(0), self.corpus, self.bank, self.cfg.data,
                                       sc.text_recipe(self.cfg.data), sc.AlignedReader(self.align))
        batch = sc.collate([sc.item(session, self.bank, "1100")] * 2)
        self.assertEqual(len(batch["text"]), 2)
        self.assertTrue(batch["text"][0].startswith("[S1] "))
        self.assertEqual(tuple(batch["slot_labels"].shape), (2, 80))


class TinyProcessor:
    """Enough of Qwen3-ASR's processor for the model: chat prompt, 128-bin features at 100 frames/s,
    audio placeholders expanded to the encoder's token count, right padding."""

    AUDIO = 150

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(eos_token="<eos>", pad_token_id=0)

    def apply_chat_template(self, conversations, add_generation_prompt=True, tokenize=False):  # noqa: ANN001, ANN201
        return ["<prompt>"] * len(conversations)

    def _ids(self, text: str) -> list[int]:
        return [10 + (ord(ch) % 100) for ch in text]

    def __call__(self, text, audio, return_tensors="pt", padding=True, truncation=False):  # noqa: ANN001, ANN201
        import torch

        from trackc.model import audio_token_count

        rows, feats = [], []
        for t, a in zip(text, audio):
            frames = len(a) // 160
            head = t.split("<asr_text>")[0] if "<asr_text>" in t else t
            tail = t[len(head):]
            rows.append([5, 6] + [self.AUDIO] * audio_token_count(frames) + self._ids(head)[:3] + self._ids(tail))
            feats.append(torch.randn(128, frames, generator=torch.Generator().manual_seed(len(a))))
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask, "input_features": torch.stack(feats),
                "feature_attention_mask": torch.ones(len(rows), feats[0].shape[1], dtype=torch.long)}

    def batch_decode(self, sequences, **_):  # noqa: ANN001, ANN201
        return [" ".join(str(int(x)) for x in row) for row in sequences]


def tiny_unified(use_spatial: bool = True):  # noqa: ANN201
    import torch
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (Qwen3ASRAudioEncoderConfig, Qwen3ASRConfig,
                                                                            Qwen3ASRTextConfig, Qwen3ASRThinkerConfig)
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

    from trackc.model import TrackCConfig, UnifiedASR

    torch.manual_seed(0)
    audio = Qwen3ASRAudioEncoderConfig(num_mel_bins=128, encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32,
                                       d_model=16, output_dim=24, downsample_hidden_size=8, n_window=50,
                                       n_window_infer=800)
    text = Qwen3ASRTextConfig(vocab_size=160, hidden_size=24, intermediate_size=48, num_hidden_layers=2,
                              num_attention_heads=2, num_key_value_heads=1, head_dim=12,
                              rope_scaling={"mrope_section": [2, 2, 2], "rope_type": "default", "interleaved": True})
    thinker = Qwen3ASRThinkerConfig(audio_config=audio.to_dict(), text_config=text.to_dict(),
                                    audio_token_id=TinyProcessor.AUDIO, audio_start_token_id=151)
    outer = Qwen3ASRForConditionalGeneration(Qwen3ASRConfig(thinker_config=thinker.to_dict()))
    cfg = TrackCConfig(use_spatial=use_spatial, lora_r=2, lora_alpha=4, lora_dropout=0.0, d_spatial=16,
                       n_pair_tac_blocks=1, max_new_tokens=3)
    return UnifiedASR(outer, TinyProcessor(), cfg)


def tiny_batch(seconds: float = 3.0, mask_bits: str = "1101"):  # noqa: ANN201
    import torch

    S = int(seconds * 16000)
    mask = torch.tensor([[c == "1" for c in mask_bits]] * 2)
    audio = torch.randn(2, 4, S, generator=torch.Generator().manual_seed(1)) * 0.05 * mask[:, :, None]
    xy = torch.tensor([[0.06, 0.0], [0.0, 0.05], [-0.07, 0.0], [0.0, -0.09]])[None].expand(2, -1, -1)
    return {"audio": audio, "mask": mask, "mic_xy": xy, "text": ["[S1] a b [S2] c", "[S1] d"],
            "slot_labels": torch.tensor([[-1] * 4 + [0] * 4 + [-1] * 72, [0] * 80])}


@unittest.skipUnless(HAS_QWEN, "qwen-asr and peft (the Qwen venv) not installed")
class TestUnifiedModel(unittest.TestCase):
    def test_training_reaches_lora_adapter_and_head_but_not_the_encoder(self) -> None:
        model = tiny_unified()
        loss, stats = model.loss(tiny_batch())
        loss.backward()
        grads = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        self.assertTrue(any("lora_" in n for n in grads))
        self.assertTrue(any(n.startswith("spatial.") for n in grads))
        self.assertTrue(any(n.startswith("head.") for n in grads))
        self.assertFalse(any("audio_tower" in n for n in grads))
        self.assertEqual(set(stats), {"loss_text", "loss_head", "acc_head"})

    def test_untrained_adapter_leaves_the_transcriber_unchanged(self) -> None:
        """Zero-initialised last layer: before training, space adds nothing."""
        import torch

        with_space, without = tiny_unified(True), tiny_unified(False)
        without.load_state_dict({k: v for k, v in with_space.state_dict().items() if not k.startswith("spatial.")})
        batch = tiny_batch()
        torch.manual_seed(5)
        a, _ = with_space.loss(batch)
        torch.manual_seed(5)
        b, _ = without.loss(batch)
        self.assertAlmostEqual(float(a), float(b), places=5)

    def test_one_microphone_adds_no_space(self) -> None:
        import torch

        model = tiny_unified()
        torch.nn.init.normal_(model.spatial.proj[-1].weight)
        batch = tiny_batch(mask_bits="0100")
        inputs = model.encode(batch["audio"], batch["mask"])
        model._set_space(batch, inputs)
        self.assertEqual(float(model._extra.abs().sum()), 0.0)

    def test_transcribe_gives_text_and_80_frame_classes(self) -> None:
        out = tiny_unified().transcribe(tiny_batch(seconds=20.0))
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]["classes"]), 80)
        self.assertIsInstance(out[0]["text"], str)

    def test_checkpoint_holds_only_trainable_tensors(self) -> None:
        model = tiny_unified()
        state = model.trainable_state()
        self.assertTrue(state)
        self.assertTrue(all(("lora_" in k) or k.startswith(("spatial.", "head.")) for k in state))
        fresh = tiny_unified()
        fresh.load_trainable(state)


@unittest.skipUnless(HAS_QWEN, "qwen-asr and peft (the Qwen venv) not installed")
class TestTrackCRun(unittest.TestCase):
    """The driver end to end on CPU: train, checkpoint, resume, validate, summarise, score the eval set."""

    def test_smoke_run(self) -> None:
        import argparse
        import json
        import tempfile

        from training.corpus import synthetic_corpus
        from trackc import train as tc
        from trackc.model import TrackCConfig

        class TinyRun(tc.TrackCRun):
            def _load_model(self):  # noqa: ANN202
                model = tiny_unified(self.cfg.use_spatial)
                model.cfg = self.cfg
                return model

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            align = root / "align.jsonl"
            with align.open("w") as fh:
                for tag in ("train", "val"):
                    corpus = synthetic_corpus(tag=tag)
                    for utts in corpus.by_speaker.values():
                        for u in utts:
                            total = corpus.n_samples(u) / 16000
                            t, words, k = 0.3, [], 0
                            while t + 0.35 < total - 0.2:
                                words.append([f"w{k}", t, t + 0.3])
                                t += 0.45
                                k += 1
                            fh.write(json.dumps({"utt_id": u.utt_id, "words": words}) + "\n")
            raw = json.loads((_REPO_ROOT / "configs" / "trackc" / "smoke.json").read_text())
            cfg = TrackCConfig.from_dict(raw)
            args = argparse.Namespace(run_dir=str(root / "run"), data_root=str(root / "data"), alignments=[str(align)],
                                      bank_mirror=None, device="cpu", allow_code_change=False)
            run = TinyRun(cfg, args)
            run.fit()
            summary = run.finalize(0.0)
            self.assertEqual(summary["trained_steps"], 6)
            self.assertIn("cpwer", summary["val"]["all"])
            self.assertEqual(sorted(summary["val"]["by_M"]), ["1", "2", "3", "4"])
            self.assertTrue((root / "run" / "best.pt").is_file())
            again = TinyRun(cfg, args)
            again._resume()
            self.assertEqual(again.step, 6)
            report = again.evaluate_sets(["eval-t2"], limit=1)
            self.assertEqual(report["eval-t2/full"]["n_cases"], 6)
            self.assertIn("eval-t2/no_space", report)
            self.assertEqual(sorted(report["eval-t2/full"]["by_M"]), ["1", "2", "3", "4"])


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    tests.addTests(doctest.DocTestSuite(tt))
    return tests


if __name__ == "__main__":
    unittest.main()
