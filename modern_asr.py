"""Isolated GPU workers for frozen ASR and single-channel diarization.

Training never imports the model libraries here. Run this file with the
appropriate interpreter created by ``setup_asr_env.py``. Model/code revisions
and decoding settings form part of the cache identity, rather than a mutable
model alias. None of these workers reads speaker labels or reference text.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import spec
from transcripts import SegmentCache, segment_key

QWEN_MODEL: str = "Qwen/Qwen3-ASR-1.7B"
QWEN_REVISION: str = "7278e1e70fe206f11671096ffdd38061171dd6e5"
QWEN_BACKEND: str = f"{QWEN_MODEL}@{QWEN_REVISION}-en-t512-v1"
VIBE_TOKENIZER_REVISION: str = "d149729398750b98c0af14eb82c78cfe92750796"
MODEL_SPECS: dict[str, tuple[str, str]] = {
    "nemotron": ("nvidia/Nemotron-3-Diarization", "f667ed73aee57d40cc39428eb768b4fd87a0a29e"),
    "vibevoice": ("microsoft/VibeVoice-ASR", "d0c9efdb8d614685062c04425d91e01b6f37d944"),
    "moss": ("OpenMOSS-Team/MOSS-Transcribe-Diarize", "10304d926740cf5a5cee05a268086e3265197910"),
}
CODE_REVISIONS: dict[str, str] = {
    "qwen": "qwen-asr==0.0.6",
    "nemotron": "transformers@a6f5c8676b158c367dc3389f68dde7a97fb619d6",
    "vibevoice": "VibeVoice@16fb2cb1217c9934a886e1948ffb06120caa2df5",
    "moss": "MOSS-Transcribe-Diarize@79c675f0670cc2f824bcbcdd98263704f40f1f7e",
}


def canonical_asr(name: str) -> str:
    """Resolve Qwen aliases without changing existing external cache IDs."""
    if name in ("qwen3-1.7b", "qwen3-asr-1.7b", QWEN_MODEL):
        return QWEN_BACKEND
    return name


def cache_path(root: str | Path, backend: str) -> Path:
    """Use portable filenames while retaining the legacy wav2vec cache name."""
    return Path(root) / (re.sub(r"[^\w.\-]", "_", backend.replace("/", "__")) + ".jsonl")


def fingerprint(value: Any) -> str:
    """Hash a JSON contract independently of dictionary insertion order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Hash input bytes, including audio, before reusing inference outputs."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Replace a JSON file only after the complete new value has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def jsonl_rows(path: Path, *, repair_tail: bool = False) -> list[dict[str, Any]]:
    """Read JSON objects, optionally recovering a final interrupted append.

    Malformed complete lines always fail. Only an unterminated final fragment
    may be removed; a valid final object without a newline is retained.
    """
    if not path.exists():
        return []
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    offset = 0
    for line in payload.splitlines(keepends=True):
        if not line.strip():
            offset += len(line)
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: expected JSON object")
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not repair_tail or line.endswith(b"\n") or offset + len(line) != len(payload):
                raise
            with path.open("r+b") as handle:
                handle.truncate(offset)
            break
        rows.append(row)
        offset += len(line)
    return rows


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    """Durably append one result; callers use one writer per output file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab+") as handle:
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
        handle.write((json.dumps(dict(row), ensure_ascii=False) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def audio_path(data_root: Path, set_name: str, session_id: str) -> Path:
    """Resolve a benchmark audio file without allowing path traversal."""
    if not all(re.fullmatch(r"[\w-]+", value) for value in (set_name, session_id)):
        raise ValueError("invalid set/session identifier")
    return data_root / set_name / "audio" / f"{session_id}.wav"


def read_mono(path: Path, channel: int) -> np.ndarray:
    """Select one physical microphone; never average the multichannel input."""
    import soundfile as sf

    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != spec.SAMPLE_RATE_HZ or not 0 <= channel < waveform.shape[1]:
        raise ValueError(f"{path}: invalid sample rate/channel {rate}/{channel}")
    return np.ascontiguousarray(waveform[:, channel])


def model_snapshot(model: str, revision: str) -> str:
    """Download a pinned snapshot and use the same files for model/processor."""
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision, cache_dir=os.environ.get("VCSASR_MODEL_CACHE"),
                             ignore_patterns=["*.gguf", "*.nemo", "*.mp4", "*.gif", "*.png", "*.jpg"])


def inference_dtype(device: str) -> Any:
    """Use native BF16 on Ampere+, FP16 on older CUDA GPUs, FP32 on CPU."""
    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select a Colab GPU or pass --device cpu")
        return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return torch.float32


def qwen_dtype(device: str) -> Any:
    """Keep the shared decoder in FP16 on every CUDA GPU, including T4/A100."""
    import torch

    inference_dtype(device)  # validate CUDA availability before loading weights
    return torch.float16 if device.startswith("cuda") else torch.float32


def runtime_contract(name: str, device: str) -> dict[str, Any]:
    """Identify resolved libraries and numerical precision before cache reuse."""
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "numpy", "librosa", "soundfile", "scipy", "soxr",
                    "tokenizers", "huggingface-hub", "accelerate", "qwen-asr", "vibevoice", "moss-transcribe-diarize"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    precision = "torch.float32" if name == "nemotron" else str(qwen_dtype(device) if name == "qwen" else inference_dtype(device))
    return {"adapter": CODE_REVISIONS[name], "packages": packages, "precision": precision}


def ensure_runtime(path: Path, contract: dict[str, Any]) -> None:
    """Reject runtime drift instead of mixing outputs from different installations."""
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError(f"{path}: runtime/precision changed; choose a new output/cache directory")
    atomic_json(path, contract)


class QwenASR:
    """Frozen Qwen3-ASR-1.7B, fixed English decoding, shared by all rungs."""

    def __init__(self, device: str = "cuda") -> None:
        """Load the pinned model in its dedicated qwen-asr environment."""
        from qwen_asr import Qwen3ASRModel

        self.model: Any = Qwen3ASRModel.from_pretrained(
            model_snapshot(QWEN_MODEL, QWEN_REVISION), dtype=qwen_dtype(device),
            device_map=device, max_inference_batch_size=1, max_new_tokens=512,
            attn_implementation="sdpa",
        )
        self.model.model.eval()
        self.model.model.generation_config.do_sample = False
        self.model.model.generation_config.num_beams = 1

    def transcribe(self, mono: np.ndarray, sample_rate_hz: int) -> str:
        """Transcribe a mono segment with the fixed benchmark language hint."""
        if mono.ndim != 1 or not len(mono) or sample_rate_hz != spec.SAMPLE_RATE_HZ:
            raise ValueError("Qwen expects nonempty mono 16 kHz audio")
        results = self.model.transcribe(audio=(mono, sample_rate_hz), language="English")
        if len(results) != 1 or not isinstance(results[0].text, str):
            raise ValueError("unexpected Qwen transcription response")
        return results[0].text.strip()


def normalize_segments(rows: Sequence[Mapping[str, Any]], duration: float) -> list[dict[str, Any]]:
    """Validate native output strictly; empty speech output remains scoreable.

    Small timestamp overshoots are clipped to the recording. Malformed outputs
    fail instead of becoming successful empty cache entries.
    """
    result: list[dict[str, Any]] = []
    for row in rows:
        speaker, text = row["speaker_id"], row.get("text", "")
        if isinstance(text, str) and text.strip() and not re.sub(r"\[[^\]]*\]", " ", text).strip():
            continue  # explicit non-speech annotations do not require a speaker or timestamps
        start, end = float(row["start_time"]), float(row["end_time"])
        if (not math.isfinite(start) or not math.isfinite(end) or end <= start
                or not isinstance(speaker, (str, int)) or isinstance(speaker, bool)
                or not isinstance(text, str)):
            raise ValueError(f"invalid diarization segment: {row!r}")
        lo, hi = max(0.0, start), min(duration, end)
        if hi > lo:
            result.append({"start_time": lo, "end_time": hi, "speaker_id": speaker, "text": text})
    return sorted(result, key=lambda row: row["start_time"])


class DiarizingModel:
    """Lazy adapters for official Nemotron, VibeVoice and MOSS inference APIs."""

    def __init__(self, name: str, device: str, max_new_tokens: int) -> None:
        """Load just one baseline in its own runtime, releasing it on process exit."""
        self.name: str = name
        self.device: str = device
        self.max_new_tokens: int = max_new_tokens
        self.dtype: Any = inference_dtype(device)
        model_id, revision = MODEL_SPECS[name]
        local = model_snapshot(model_id, revision)
        if name == "vibevoice":
            from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
            from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

            self.processor: Any = VibeVoiceASRProcessor.from_pretrained(
                local, language_model_pretrained_name="Qwen/Qwen2.5-7B", revision=VIBE_TOKENIZER_REVISION,
                cache_dir=os.environ.get("VCSASR_MODEL_CACHE"))
            self.model: Any = VibeVoiceASRForConditionalGeneration.from_pretrained(
                local, dtype=self.dtype, attn_implementation="sdpa", trust_remote_code=True,
            ).to(device).eval()
        elif name == "nemotron":
            from transformers import AutoModelForAudioFrameClassification, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(local)
            # Keep this small diarizer in FP32, matching the official example.
            self.model = AutoModelForAudioFrameClassification.from_pretrained(local).to(device).eval()
        else:
            from transformers import AutoModelForCausalLM, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(local, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                local, trust_remote_code=True, dtype=self.dtype, attn_implementation="sdpa",
            ).to(device).eval()

    def infer(self, mono: np.ndarray, wav_path: Path) -> tuple[list[dict[str, Any]], str]:
        """Return native speaker segments and raw text without access to ground truth."""
        import torch

        raw = ""
        with torch.inference_mode():
            if self.name == "nemotron":
                inputs = self.processor(mono, sampling_rate=spec.SAMPLE_RATE_HZ).to(self.device)
                logits = self.model(**inputs).logits
                native = self.processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
                rows = [{"start_time": s["Start"], "end_time": s["End"],
                         "speaker_id": int(s["Speaker"]), "text": ""} for s in native]
            elif self.name == "vibevoice":
                inputs = self.processor(audio=[str(wav_path)], sampling_rate=None, return_tensors="pt",
                                        padding=True, add_generation_prompt=True)
                inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                          for key, value in inputs.items()}
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                             pad_token_id=self.processor.pad_id,
                                             eos_token_id=self.processor.tokenizer.eos_token_id)
                generated = output[0, inputs["input_ids"].shape[1]:]
                if len(generated) >= self.max_new_tokens and generated[-1].item() != self.processor.tokenizer.eos_token_id:
                    raise RuntimeError("VibeVoice output truncated; increase --max-new-tokens")
                raw = self.processor.decode(generated, skip_special_tokens=True)
                rows = self.processor.post_process_transcription(raw)
                if raw.strip() and raw.strip() != "[]" and not rows:
                    raise ValueError(f"VibeVoice produced no parseable segments: {raw[:300]}")
            else:
                from moss_transcribe_diarize import parse_transcript
                from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription

                response = generate_transcription(
                    self.model, self.processor, build_transcription_messages(str(wav_path)),
                    max_new_tokens=self.max_new_tokens, do_sample=False, device=torch.device(self.device),
                    dtype=self.dtype,
                )
                if response["generated_tokens"] >= self.max_new_tokens:
                    raise RuntimeError("MOSS output may be truncated; increase --max-new-tokens")
                raw = response["text"]
                rows = [{"start_time": s.start, "end_time": s.end, "speaker_id": s.speaker, "text": s.text}
                        for s in parse_transcript(raw)]
                if raw.strip() and not rows:
                    raise ValueError(f"MOSS produced no parseable segments: {raw[:300]}")
        return normalize_segments(rows, len(mono) / spec.SAMPLE_RATE_HZ), raw


def fill_requests(requests: Sequence[Path], data_root: Path, cache_root: Path, device: str) -> int:
    """Fill exported segment requests, checking audio provenance before reuse."""
    ensure_runtime(cache_root / "qwen-runtime.json", runtime_contract("qwen", device))
    cache = SegmentCache(cache_path(cache_root, QWEN_BACKEND), QWEN_BACKEND)
    contract_path = cache.path.with_suffix(".inputs.json")
    inputs: dict[str, str] = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    jobs: dict[str, dict[str, Any]] = {}
    sources: dict[Path, str] = {}
    for path in requests:
        for row in jsonl_rows(path):
            parts = row["key"].split("/")
            if len(parts) != 4:
                raise ValueError(f"invalid segment key: {row['key']}")
            set_name = parts[0]
            sid, ch = row["session_id"], row["channel"]
            lo, hi = float(row["start_sec"]), float(row["end_sec"])
            if (type(ch) is not int or not 0 <= ch < spec.N_CHANNELS_MAX
                    or not math.isfinite(lo) or not math.isfinite(hi)
                    or not 0 <= lo < hi <= spec.SESSION_LENGTH_SEC
                    or row["key"] != segment_key(set_name, sid, ch, lo, hi)):
                raise ValueError(f"invalid segment request: {row}")
            source = audio_path(data_root, set_name, sid)
            if source not in sources:
                sources[source] = file_hash(source)
            source_key = f"{set_name}/{sid}"
            if source_key in inputs and inputs[source_key] != sources[source]:
                raise ValueError(f"audio changed for {source_key}; use a new ASR cache directory")
            if source_key not in inputs and any(key.startswith(source_key + "/") for key in cache.text):
                raise ValueError(f"cache lacks audio provenance for {source_key}; use a new ASR cache directory")
            inputs[source_key] = sources[source]
            jobs[row["key"]] = {**row, "start_sec": lo, "end_sec": hi, "source": source}
    atomic_json(contract_path, inputs)
    pending = [row for key, row in jobs.items() if key not in cache]
    if not pending:
        print("Qwen cache complete", flush=True)
        return 0
    recognizer = QwenASR(device)
    previous_source: Path | None = None
    audio: dict[int, np.ndarray] = {}
    for index, row in enumerate(pending, 1):
        if row["source"] != previous_source:
            audio.clear()
            previous_source = row["source"]
        if row["channel"] not in audio:
            audio[row["channel"]] = read_mono(row["source"], row["channel"])
        mono = audio[row["channel"]]
        lo, hi = (round(row[key] * spec.SAMPLE_RATE_HZ) for key in ("start_sec", "end_sec"))
        if hi > len(mono):
            raise ValueError(f"request exceeds audio: {row['key']}")
        cache.add(row["key"], recognizer.transcribe(mono[lo:hi], spec.SAMPLE_RATE_HZ))
        if index % 20 == 0 or index == len(pending):
            print(f"Qwen {index}/{len(pending)} segments", flush=True)
    return 0


def run_diarization(name: str, jobs_path: Path, out: Path, data_root: Path,
                    device: str, max_new_tokens: int) -> int:
    """Execute pending mono jobs and append only valid results with their contracts."""
    import tempfile
    import soundfile as sf

    runtime = runtime_contract(name, device)
    ensure_runtime(out.with_suffix(".runtime.json"), runtime)
    jobs = jsonl_rows(jobs_path)
    existing = {row["job_id"]: row for row in jsonl_rows(out, repair_tail=True)}
    pending: list[dict[str, Any]] = []
    for job in jobs:
        model_id, revision = MODEL_SPECS[name]
        if (job["contract"]["model"] != model_id or job["contract"]["revision"] != revision
                or (name != "nemotron" and job["contract"]["max_new_tokens"] != max_new_tokens)):
            raise ValueError("worker model/settings differ from job contract")
        if job["job_id"] in existing:
            if (existing[job["job_id"]]["contract"] != job["contract"]
                    or existing[job["job_id"]].get("runtime") != runtime):
                raise ValueError(f"{out}: changed model/data/settings; choose a new output directory")
        else:
            pending.append(job)
    if not pending:
        return 0
    model = DiarizingModel(name, device, max_new_tokens)
    with tempfile.TemporaryDirectory(prefix="vcsasr-mono-") as temporary:
        wav = Path(temporary) / "mono.wav"
        for index, job in enumerate(pending, 1):
            source = audio_path(data_root, job["set"], job["session_id"])
            if file_hash(source) != job["audio_sha256"]:
                raise ValueError(f"audio changed during inference: {source}")
            mono = read_mono(source, job["channel"])
            sf.write(wav, mono, spec.SAMPLE_RATE_HZ, subtype="PCM_16")
            start = time.monotonic()
            segments, raw = model.infer(mono, wav)
            append_row(out, {**job, "segments": segments, "raw_text": raw,
                             "runtime": runtime,
                             "sec_per_session": time.monotonic() - start})
            print(f"{name} {index}/{len(pending)} sessions", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one worker role; installation and orchestration live outside this process."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    sub = parser.add_subparsers(dest="command", required=True)
    asr = sub.add_parser("transcribe")
    asr.add_argument("--requests", type=Path, nargs="+", required=True)
    asr.add_argument("--cache-dir", type=Path, required=True)
    diar = sub.add_parser("diarize")
    diar.add_argument("--backend", choices=MODEL_SPECS, required=True)
    diar.add_argument("--jobs", type=Path, required=True)
    diar.add_argument("--out", type=Path, required=True)
    diar.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.command == "transcribe":
        return fill_requests(args.requests, args.data_root, args.cache_dir, args.device)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    return run_diarization(args.backend, args.jobs, args.out, args.data_root, args.device, args.max_new_tokens)


if __name__ == "__main__":
    raise SystemExit(main())
