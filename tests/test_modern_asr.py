"""Exercise cache contracts, official API adapters and mono evaluation without weights.

The integration fixture uses generated placeholder audio and fake model
responses. It tests inference plumbing and the real scorer, not ASR accuracy.
Real GPU/model installation is checked separately in the Colab smoke run.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np

import modern_asr as modern
import mono_baselines as mono
import spec
import track_b
import transcripts as tr

HAS_AUDIO: bool = importlib.util.find_spec("soundfile") is not None
HAS_TORCH: bool = importlib.util.find_spec("torch") is not None


class TestCacheContracts(unittest.TestCase):
    """Protect backend isolation and recovery after interrupted appends."""

    def test_aliases_preserve_backend_and_legacy_cache_identity(self) -> None:
        """All Qwen aliases identify the same pinned decoder; old IDs stay stable."""
        for name in ("qwen3-1.7b", "qwen3-asr-1.7b", modern.QWEN_MODEL, modern.QWEN_BACKEND):
            self.assertEqual(modern.canonical_asr(name), modern.QWEN_BACKEND)
        self.assertEqual(modern.cache_path("cache", "facebook/wav2vec2-base-960h").name,
                         "facebook__wav2vec2-base-960h.jsonl")
        self.assertNotEqual(modern.cache_path("cache", modern.QWEN_BACKEND),
                            modern.cache_path("cache", "facebook/wav2vec2-base-960h"))

    def test_transcript_tail_recovery_and_empty_text(self) -> None:
        """A partial final append is retried, while cached silence remains complete."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            cache = tr.SegmentCache(path, "test")
            cache.add("done", "")
            with path.open("ab") as handle:
                handle.write(b'{"key":"incomplete","text":"\xed')
            cache = tr.SegmentCache(path, "test")
            self.assertIn("done", cache)
            self.assertNotIn("incomplete", cache)
            cache.add("next", "hello")
            self.assertEqual(tr.SegmentCache(path, "test").text, {"done": "", "next": "hello"})
            with self.assertRaisesRegex(ValueError, "not 'different'"):
                tr.SegmentCache(path, "different")

    def test_complete_corruption_is_not_discarded(self) -> None:
        """Only an unterminated final fragment qualifies for recovery."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_bytes(b'broken\n{"key":"ok","backend":"test","text":"x"}\n')
            for reader in (lambda: tr.SegmentCache(path, "test"),
                           lambda: modern.jsonl_rows(path, repair_tail=True)):
                with self.assertRaises(json.JSONDecodeError):
                    reader()
            self.assertTrue(path.read_bytes().startswith(b"broken\n"))

    def test_valid_unterminated_json_is_not_lost_or_joined(self) -> None:
        """Both result formats preserve a valid final line lacking a newline."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            path.write_text(json.dumps({"key": "old", "backend": "test", "text": "first"}))
            cache = tr.SegmentCache(path, "test")
            cache.add("new", "second")
            self.assertEqual(len(modern.jsonl_rows(path)), 2)
            result = Path(directory) / "results.jsonl"
            result.write_text('{"job_id":"a"}')
            modern.append_row(result, {"job_id": "b"})
            with result.open("ab") as handle:
                handle.write(b'{"job_id":')
            self.assertEqual([r["job_id"] for r in modern.jsonl_rows(result, repair_tail=True)], ["a", "b"])

    def test_runtime_drift_requires_a_new_cache(self) -> None:
        """Changed resolved libraries cannot silently share an existing cache."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            modern.ensure_runtime(path, {"transformers": "one"})
            modern.ensure_runtime(path, {"transformers": "one"})
            with self.assertRaisesRegex(ValueError, "runtime/precision changed"):
                modern.ensure_runtime(path, {"transformers": "two"})
            self.assertEqual(json.loads(path.read_text()), {"transformers": "one"})

    def test_segment_validation_and_reference_selection(self) -> None:
        """Invalid timestamps/speakers fail; clipping does not invent speech."""
        row = {"start_time": -0.1, "end_time": 20.2, "speaker_id": "A", "text": "hello"}
        self.assertEqual(modern.normalize_segments([row], 20)[0],
                         {**row, "start_time": 0.0, "end_time": 20})
        self.assertEqual(modern.normalize_segments([{"start_time": None, "end_time": None,
                                                      "speaker_id": None, "text": "[Silence]"}], 20), [])
        for change in ({"start_time": float("nan")}, {"end_time": float("inf")},
                       {"end_time": -1}, {"speaker_id": None}, {"speaker_id": []}, {"text": 1}):
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                modern.normalize_segments([{**row, **change}], 20)
        with self.assertRaisesRegex(ValueError, "invalid set/session"):
            modern.audio_path(Path("data"), "../secret", "session")


class TestRuntimeProvisioning(unittest.TestCase):
    """Provisioning must isolate installs and only mark importable environments complete."""

    def test_only_the_isolated_interpreter_installs_packages(self) -> None:
        """Reusing a prepared environment checks imports without reinstalling packages."""
        import setup_asr_env

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def create(path: Path) -> None:
                """Make a stand-in interpreter for the mocked environment builder."""
                (path / "bin").mkdir(parents=True)
                (path / "bin" / "python").touch()

            with patch.object(setup_asr_env.venv, "EnvBuilder") as builder, \
                 patch.object(setup_asr_env.subprocess, "run") as execute, \
                 patch.object(setup_asr_env.subprocess, "check_output", return_value="qwen-asr==0.0.6\n"):
                builder.return_value.create.side_effect = create
                python = setup_asr_env.setup("qwen", root)
                self.assertTrue(all(call.args[0][0] == str(python) for call in execute.call_args_list))
                self.assertTrue((root / "qwen" / "vcsasr-runtime.json").exists())
                self.assertIn("qwen-asr==0.0.6", (root / "qwen" / "resolved-requirements.txt").read_text())
                execute.reset_mock()
                self.assertEqual(setup_asr_env.setup("qwen", root), python)
                self.assertEqual(execute.call_args.args[0][1], "-c")
                execute.assert_called_once()

    def test_failed_import_does_not_mark_environment_complete(self) -> None:
        """A dependency import failure remains retryable on the next Colab run."""
        import setup_asr_env

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qwen" / "bin").mkdir(parents=True)
            (root / "qwen" / "bin" / "python").touch()
            with patch.object(setup_asr_env.subprocess, "run", side_effect=[None, None, RuntimeError("import failed")]):
                with self.assertRaisesRegex(RuntimeError, "import failed"):
                    setup_asr_env.setup("qwen", root)
            self.assertFalse((root / "qwen" / "vcsasr-runtime.json").exists())


@unittest.skipUnless(HAS_TORCH, "official API shape tests require torch")
class TestOfficialAdapters(unittest.TestCase):
    """Validate calls to the documented Qwen, Nemotron and generative model APIs."""

    def test_qwen_uses_pinned_snapshot_and_fixed_language(self) -> None:
        """Qwen receives mono samples and English, without reference transcripts."""
        wrapper = Mock()
        wrapper.transcribe.return_value = [types.SimpleNamespace(text=" recognized ")]
        factory = Mock(return_value=wrapper)
        package = types.SimpleNamespace(Qwen3ASRModel=types.SimpleNamespace(from_pretrained=factory))
        with patch.dict("sys.modules", {"qwen_asr": package}), \
             patch.object(modern, "model_snapshot", return_value="/pinned") as snapshot:
            backend = modern.QwenASR("cpu")
            waveform = np.zeros(16000, dtype=np.float32)
            self.assertEqual(backend.transcribe(waveform, 16000), "recognized")
            snapshot.assert_called_once_with(modern.QWEN_MODEL, modern.QWEN_REVISION)
            self.assertEqual(factory.call_args.kwargs["max_new_tokens"], 512)
            self.assertEqual(wrapper.transcribe.call_args.kwargs["language"], "English")
            self.assertIs(wrapper.transcribe.call_args.kwargs["audio"][0], waveform)
            with self.assertRaises(ValueError):
                backend.transcribe(np.zeros((2, 16000)), 16000)

    def test_cuda_precision_does_not_require_bf16_on_t4(self) -> None:
        """The worker supports T4 FP16 and selects native BF16 on newer GPUs."""
        import torch

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
            self.assertEqual(modern.inference_dtype("cuda"), torch.float16)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            self.assertEqual(modern.inference_dtype("cuda"), torch.bfloat16)
            self.assertEqual(modern.qwen_dtype("cuda"), torch.float16)

    def test_nemotron_extracts_native_regions(self) -> None:
        """Frame probabilities go through the official timestamp/speaker extractor."""
        import torch

        adapter = object.__new__(modern.DiarizingModel)
        adapter.name, adapter.device, adapter.max_new_tokens = "nemotron", "cpu", 2048
        batch = Mock()
        batch.to.return_value = batch
        batch.attention_mask = torch.ones(1, 10)
        # **inputs requires an actual mapping; preserve the attribute interface too.
        class Batch(dict[str, Any]):
            """Small stand-in for a Transformers BatchFeature."""

            attention_mask: Any = torch.ones(1, 10)

            def to(self, device: str) -> Batch:
                """Return this CPU fixture unchanged."""
                return self

        adapter.processor = Mock(return_value=Batch(input_values=torch.zeros(1, 16000)))
        adapter.processor.extract_speaker_dict.return_value = [[{"Start": 0, "End": 1, "Speaker": 2}]]
        adapter.model = Mock(return_value=types.SimpleNamespace(logits=torch.zeros(1, 10, 8)))
        segments, raw = adapter.infer(np.zeros(16000, dtype=np.float32), Path("unused.wav"))
        self.assertEqual(segments, [{"start_time": 0.0, "end_time": 1.0, "speaker_id": 2, "text": ""}])
        self.assertEqual(raw, "")
        adapter.processor.extract_speaker_dict.assert_called_once()

    def test_vibevoice_parse_failure_is_not_successful_silence(self) -> None:
        """A nonempty unparseable output raises before the result is cached."""
        import torch

        adapter = object.__new__(modern.DiarizingModel)
        adapter.name, adapter.device, adapter.max_new_tokens = "vibevoice", "cpu", 10
        adapter.processor = Mock(return_value={"input_ids": torch.tensor([[1, 2]])})
        adapter.processor.tokenizer.eos_token_id = 9
        adapter.processor.pad_id = 0
        adapter.processor.decode.return_value = "invalid structured output"
        adapter.processor.post_process_transcription.return_value = []
        adapter.model = Mock()
        adapter.model.generate.return_value = torch.tensor([[1, 2, 3, 9]])
        with self.assertRaisesRegex(ValueError, "no parseable segments"):
            adapter.infer(np.zeros(16000, dtype=np.float32), Path("mono.wav"))
        adapter.processor.decode.return_value = "[]"
        self.assertEqual(adapter.infer(np.zeros(16000, dtype=np.float32), Path("mono.wav")), ([], "[]"))
        adapter.model.generate.return_value = torch.tensor([[1, 2] + [3] * 10])
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            adapter.infer(np.zeros(16000, dtype=np.float32), Path("mono.wav"))

    def test_moss_parser_and_truncation(self) -> None:
        """MOSS structured objects normalize correctly and capped output is rejected."""
        import torch

        adapter = object.__new__(modern.DiarizingModel)
        adapter.name, adapter.device, adapter.max_new_tokens, adapter.dtype = "moss", "cpu", 10, torch.float32
        adapter.model, adapter.processor = Mock(), Mock()
        generate = Mock(return_value={"text": "structured", "generated_tokens": 4})
        parser = Mock(return_value=[types.SimpleNamespace(start=0, end=1, speaker="S01", text="hello")])
        package = types.SimpleNamespace(parse_transcript=parser)
        utilities = types.SimpleNamespace(build_transcription_messages=Mock(return_value=[{"audio": "mono.wav"}]),
                                          generate_transcription=generate)
        with patch.dict("sys.modules", {"moss_transcribe_diarize": package,
                                        "moss_transcribe_diarize.inference_utils": utilities}):
            segments, raw = adapter.infer(np.zeros(16000, dtype=np.float32), Path("mono.wav"))
            self.assertEqual(segments[0]["text"], "hello")
            self.assertEqual(raw, "structured")
            generate.return_value["generated_tokens"] = 10
            with self.assertRaisesRegex(RuntimeError, "truncated"):
                adapter.infer(np.zeros(16000, dtype=np.float32), Path("mono.wav"))


@unittest.skipUnless(HAS_AUDIO, "audio integration requires soundfile")
class TestMonoIntegration(unittest.TestCase):
    """Run synthetic audio, resumable workers, transcript caches and the real scorer."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create paired placeholder sessions using the repository's actual generator."""
        import make_eval_t2

        cls.fixture: Any = tempfile.TemporaryDirectory()
        cls.data: Path = Path(cls.fixture.name)
        make_eval_t2.generate(cls.data / "eval-t2", cls.data / "eval-t2-static", 6, 11, None,
                              "fallback_ism", True, 1, log=lambda *_: None)
        cls.truth: dict[tuple[str, str], dict[str, Any]] = {}
        for set_name in ("eval-t2", "eval-t2-static"):
            for row in modern.jsonl_rows(cls.data / set_name / "ground_truth.jsonl"):
                cls.truth[(set_name, row["session_id"])] = row

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove generated audio after all integration scenarios."""
        cls.fixture.cleanup()

    def oracle_segments(self, path: Path, *, text: bool) -> list[dict[str, Any]]:
        """Supply a model fixture derived from truth only inside this test."""
        row = self.truth[(path.parent.parent.name, path.stem)]
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for turn in row["turns"]:
            speaker = turn["speaker"]
            result.append({"start_time": turn["start_sec"], "end_time": turn["end_sec"],
                           "speaker_id": speaker,
                           "text": row["final_transcripts_reference"][speaker] if text and speaker not in seen else ""})
            seen.add(speaker)
        return result

    def worker(self, command: list[str]) -> None:
        """Execute actual worker functions with fake model predictions, without weights."""
        self.assertIn("--data-root", command)
        arguments = command[2:]
        role = "diarize" if "diarize" in arguments else "transcribe"
        if role == "diarize":
            name = arguments[arguments.index("--backend") + 1]
            jobs = Path(arguments[arguments.index("--jobs") + 1])
            out = Path(arguments[arguments.index("--out") + 1])
            model = Mock()
            source_jobs = modern.jsonl_rows(jobs)
            job_index = iter([j for j in source_jobs if j["job_id"] not in
                              {r["job_id"] for r in modern.jsonl_rows(out, repair_tail=True)}])

            def predict(waveform: np.ndarray, wav_path: Path) -> tuple[list[dict[str, Any]], str]:
                """Return the next fixture's oracle regions through the real mono worker."""
                job = next(job_index)
                self.assertEqual(waveform.ndim, 1)
                self.assertTrue(wav_path.exists())
                source = modern.audio_path(self.data, job["set"], job["session_id"])
                return self.oracle_segments(source, text=name != "nemotron"), "fixture"

            model.infer.side_effect = predict
            with patch.object(modern, "DiarizingModel", return_value=model), \
                 patch.object(modern, "runtime_contract", return_value={"fixture": name}):
                modern.main(arguments)
        else:
            recognizer = Mock()
            recognizer.transcribe.return_value = "recognized words"
            with patch.object(modern, "QwenASR", return_value=recognizer), \
                 patch.object(modern, "runtime_contract", return_value={"fixture": "qwen"}):
                modern.main(arguments)

    def test_native_baseline_scores_and_resumes_paired_cases(self) -> None:
        """Oracle native responses score perfectly; reruns retain the same cached jobs."""
        for backend in ("vibevoice", "moss"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / backend
                with patch.object(mono, "run_command", side_effect=self.worker):
                    report = mono.run(backend, "worker", None, self.data, out, Path(directory) / "asr",
                                      ["eval-t2", "eval-t2-static"], "cpu", 2, 2048)
                    before = (out / "native_segments.jsonl").read_bytes()
                    mono.run(backend, "worker", None, self.data, out, Path(directory) / "asr",
                             ["eval-t2", "eval-t2-static"], "cpu", 2, 2048)
                self.assertEqual(before, (out / "native_segments.jsonl").read_bytes())
                self.assertEqual(len(modern.jsonl_rows(out / "native_segments.jsonl")), 8)
                self.assertEqual(report["baseline"]["input_channels"], 1)
                self.assertEqual(report["baseline"]["observation_mode"], "full_audio")
                for set_name in ("eval-t2", "eval-t2-static"):
                    self.assertEqual(report[set_name]["all"]["speaker_attribution_accuracy"], 1.0)
                    self.assertEqual(report[set_name]["all"]["cpwer"], 0.0)
                    self.assertEqual(set(report[set_name]["by_comparison_M"]), {"1", "2", "3", "4"})

    def test_nemotron_uses_shared_qwen_cache_and_preserves_regions(self) -> None:
        """The pipeline transcribes mono slot runs and scores original diarization regions."""
        with tempfile.TemporaryDirectory() as directory:
            out, cache_root = Path(directory) / "nemotron", Path(directory) / "asr"
            with patch.object(mono, "run_command", side_effect=self.worker):
                report = mono.run("nemotron", "diarizer", "qwen", self.data, out, cache_root,
                                  ["eval-t2"], "cpu", 1, 2048)
            self.assertEqual(report["baseline"]["asr"], modern.QWEN_BACKEND)
            self.assertEqual(report["eval-t2"]["all"]["speaker_attribution_accuracy"], 1.0)
            rows = modern.jsonl_rows(out / "segments.jsonl")
            self.assertTrue(all(row["final_transcripts"] for row in rows))
            self.assertTrue(all(not segment["text"] for row in rows for segment in row["segments"]))
            cache = tr.SegmentCache(modern.cache_path(cache_root, modern.QWEN_BACKEND), modern.QWEN_BACKEND)
            self.assertTrue(cache.text)
            requests = modern.jsonl_rows(out / "qwen_requests.jsonl")
            with patch.object(modern, "QwenASR") as factory, \
                 patch.object(modern, "runtime_contract", return_value={"fixture": "qwen"}):
                modern.fill_requests([out / "qwen_requests.jsonl"], self.data, cache_root, "cpu")
                factory.assert_not_called()
            self.assertEqual(set(cache.text), {r["key"] for r in requests})

    def test_changed_contract_fails_before_worker_runs(self) -> None:
        """Changing decoding settings cannot reuse the previous result directory."""
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "moss"
            with patch.object(mono, "run_command", side_effect=self.worker):
                mono.run("moss", "worker", None, self.data, out, Path(directory) / "asr", ["eval-t2"], "cpu", 1, 2048)
            with patch.object(mono, "run_command") as worker, self.assertRaisesRegex(ValueError, "different data/model/settings"):
                mono.run("moss", "worker", None, self.data, out, Path(directory) / "asr", ["eval-t2"], "cpu", 1, 1024)
            worker.assert_not_called()

    def test_failed_session_is_retried_after_interruption(self) -> None:
        """A failure retains completed jobs and retries the first incomplete job."""
        with tempfile.TemporaryDirectory() as directory:
            out, jobs_path = Path(directory) / "native.jsonl", Path(directory) / "jobs.jsonl"
            jobs = mono.build_jobs(self.data, ["eval-t2"], "moss", 1, 2048)
            jobs_path.write_text("".join(json.dumps(job) + "\n" for job in jobs))
            model = Mock()
            regions = self.oracle_segments(modern.audio_path(self.data, "eval-t2", jobs[0]["session_id"]), text=True)
            model.infer.side_effect = [(regions, "ok"), RuntimeError("interrupted")]
            with patch.object(modern, "runtime_contract", return_value={"fixture": "moss"}), \
                 patch.object(modern, "DiarizingModel", return_value=model):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    modern.run_diarization("moss", jobs_path, out, self.data, "cpu", 2048)
            first = out.read_bytes()
            self.assertEqual(len(modern.jsonl_rows(out)), 1)
            model.infer.side_effect = [(regions, "ok")]
            model.infer.reset_mock()
            with patch.object(modern, "runtime_contract", return_value={"fixture": "moss"}), \
                 patch.object(modern, "DiarizingModel", return_value=model):
                modern.run_diarization("moss", jobs_path, out, self.data, "cpu", 2048)
            model.infer.assert_called_once()
            self.assertTrue(out.read_bytes().startswith(first))
            self.assertEqual(len(modern.jsonl_rows(out)), 2)

    @unittest.skipUnless(HAS_TORCH, "eval_t2 requires torch")
    def test_eval_t2_fills_qwen_requests_and_scores(self) -> None:
        """The existing evaluation ladder can invoke the isolated Qwen worker end to end."""
        import eval_t2

        gt = modern.jsonl_rows(self.data / "eval-t2" / "ground_truth.jsonl")[0]
        metas = {row["case_id"]: row for row in modern.jsonl_rows(self.data / "eval-t2" / "meta.jsonl")}
        metadata = metas[gt["case_id"]]
        labels = eval_t2.frame_labels(gt)
        slots = np.maximum(labels, 0)[None, :]
        mask = "".join("1" if i in metadata["active_channels"] else "0" for i in range(4))
        emb = {"case_id": np.array([gt["case_id"]]), "session_id": np.array([gt["session_id"]]),
               "ref": np.array([min(metadata["active_channels"])]), "labels": labels[None, :], "mask": np.array([mask])}

        def worker(command: list[str], **kwargs: Any) -> None:
            """Check the environment boundary and execute the model-free worker fixture."""
            self.assertEqual(command[0], "/isolated path/bin/python")
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "0")
            recognizer = Mock()
            recognizer.transcribe.return_value = "test transcript"
            with patch.object(modern, "runtime_contract", return_value={"fixture": "qwen"}), \
                 patch.object(modern, "QwenASR", return_value=recognizer):
                modern.main(command[2:])

        with tempfile.TemporaryDirectory() as directory:
            cache = tr.SegmentCache(modern.cache_path(Path(directory) / "cache", modern.QWEN_BACKEND), modern.QWEN_BACKEND)
            with patch.object(eval_t2.subprocess, "run", side_effect=worker):
                report = eval_t2.run_variant("eval-t2", self.data / "eval-t2", Path(directory) / "full", emb,
                                             slots, cache, None, {"asr": modern.QWEN_BACKEND},
                                             "/isolated path/bin/python", "cpu")
            self.assertIsNotNone(report)
            self.assertEqual(report["eval_t2_run"]["asr_runtime"], {"fixture": "qwen"})
            self.assertTrue((Path(directory) / "full" / "score.json").exists())
            self.assertTrue(cache.text)

    def test_comparison_retains_scope_channels_and_case_population(self) -> None:
        """Mono M=4 rows remain 1ch and case hashes expose smoke/full differences."""
        import compare_asr
        import evaluate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "mono" / "moss"
            with patch.object(mono, "run_command", side_effect=self.worker):
                mono.run("moss", "worker", None, self.data, out, root / "cache", ["eval-t2"], "cpu", 1, 2048)
            pred_rows = modern.jsonl_rows(out / "eval-t2.predictions.jsonl")
            target = root / "eval" / "proposed_carry" / "eval-t2" / "full"
            target.mkdir(parents=True)
            predictions = target / "predictions.jsonl"
            predictions.write_text("".join(json.dumps({**row, "observation_mode": "causal"}) + "\n" for row in pred_rows))
            gt_rows = [row for row in modern.jsonl_rows(self.data / "eval-t2" / "ground_truth.jsonl")
                       if row["case_id"] in {pred["case_id"] for pred in pred_rows}]
            truth = root / "truth.jsonl"
            truth.write_text("".join(json.dumps(row) + "\n" for row in gt_rows))
            report = evaluate.evaluate_files(predictions, truth, 2)
            report["eval_t2_run"] = {"system": "proposed", "asr": modern.QWEN_BACKEND, "thresholds": "validation"}
            modern.atomic_json(target / "score.json", report)
            rows = compare_asr.comparison_rows(root / "eval", root / "mono", self.data, ["eval-t2"])
            m4 = [row for row in rows if row["comparison_M"] == 4]
            self.assertEqual({row["input_channels"] for row in m4}, {1, 4})
            self.assertEqual({row["observation_mode"] for row in m4}, {"causal", "full_audio"})
            self.assertEqual(len({row["case_sha256"] for row in m4}), 1)
            old_hash = next(row["case_sha256"] for row in rows if row["system"] == "moss")
            with patch.object(mono, "run_command", side_effect=self.worker):
                mono.run("moss", "worker", None, self.data, out, root / "cache", ["eval-t2"], "cpu", 2, 2048)
            updated = compare_asr.comparison_rows(root / "eval", root / "mono", self.data, ["eval-t2"])
            self.assertNotEqual(next(row["case_sha256"] for row in updated if row["system"] == "moss"), old_hash)

    def test_missing_duplicate_and_mixed_predictions_fail(self) -> None:
        """The scorer does not silently accept missing jobs or mixed backends."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "segments.jsonl"
            row = {"set": "eval-t2", "session_id": "t2_0000", "channel": 0,
                   "backend": "fixture", "segments": []}
            source.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(SystemExit, "have no segments"):
                track_b.score_segments(source, self.data, ["eval-t2"], Path(directory) / "out", limit=1)
            source.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                track_b.score_segments(source, self.data, ["eval-t2"], Path(directory) / "out", limit=1)
            source.write_text(json.dumps(row) + "\n" + json.dumps({**row, "channel": 2, "backend": "other"}) + "\n")
            with self.assertRaisesRegex(ValueError, "exactly one baseline"):
                track_b.score_segments(source, self.data, ["eval-t2"], Path(directory) / "out", limit=1)
            source.write_text(json.dumps({**row, "contract": {"revision": "old"}}) + "\n"
                              + json.dumps({**row, "channel": 2, "contract": {"revision": "new"}}) + "\n")
            with self.assertRaisesRegex(ValueError, "inconsistent inference contracts"):
                track_b.score_segments(source, self.data, ["eval-t2"], Path(directory) / "out", limit=1)

    def test_audio_change_and_invalid_requests_do_not_reuse_qwen(self) -> None:
        """Audio hashes and canonical request keys protect the shared transcript cache."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data" / "sample" / "audio" / "session.wav"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"original audio")
            request = {"key": tr.segment_key("sample", "session", 0, 0, 1), "session_id": "session",
                       "channel": 0, "start_sec": 0, "end_sec": 1}
            path = root / "requests.jsonl"
            path.write_text(json.dumps(request) + "\n")
            with patch.object(modern, "runtime_contract", return_value={"fixture": "qwen"}), \
                 patch.object(modern, "read_mono", return_value=np.zeros(16000, dtype=np.float32)), \
                 patch.object(modern, "QwenASR") as factory:
                factory.return_value.transcribe.return_value = "test"
                modern.fill_requests([path], root / "data", root / "cache", "cpu")
                source.write_bytes(b"different audio")
                with self.assertRaisesRegex(ValueError, "audio changed"):
                    modern.fill_requests([path], root / "data", root / "cache", "cpu")
                bad = {**request, "key": "sample/session/ch0/9.000-10.000"}
                path.write_text(json.dumps(bad) + "\n")
                with self.assertRaisesRegex(ValueError, "invalid segment request"):
                    modern.fill_requests([path], root / "data", root / "cache", "cpu")


if __name__ == "__main__":
    unittest.main()
