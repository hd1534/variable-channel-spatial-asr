"""Create isolated model runtimes without upgrading the training interpreter.

Colab's installed CUDA PyTorch is reused through system-site-packages. Each
model's Transformers dependencies are installed only inside its own venv.
The resolved package list is saved for reporting/reproduction.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path

from modern_asr import CODE_REVISIONS, atomic_json, fingerprint

PROFILES: dict[str, list[str]] = {
    "qwen": ["qwen-asr==0.0.6", "soundfile==0.13.1"],
    "nemotron": ["https://github.com/huggingface/transformers/archive/"
                 + CODE_REVISIONS["nemotron"].split("@")[1] + ".zip", "accelerate", "soundfile==0.13.1",
                 "librosa==0.11.0"],
    "vibevoice": ["https://github.com/microsoft/VibeVoice/archive/"
                  + CODE_REVISIONS["vibevoice"].split("@")[1] + ".zip", "soundfile==0.13.1"],
    "moss": ["https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/archive/"
             + CODE_REVISIONS["moss"].split("@")[1] + ".zip", "soundfile==0.13.1"],
    # Track C: the shared transcriber's runtime plus LoRA, for training (trackc/).
    "trackc": ["qwen-asr==0.0.6", "soundfile==0.13.1", "peft==0.17.1"],
}
IMPORT_CHECKS: dict[str, str] = {
    "qwen": "from qwen_asr import Qwen3ASRModel",
    "trackc": "from qwen_asr import Qwen3ASRModel; from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner; "
              "import peft",
    "nemotron": "from transformers import AutoModelForAudioFrameClassification, NemotronAsrStreamingFeatureExtractor; "
                "NemotronAsrStreamingFeatureExtractor()",
    "vibevoice": "from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration; "
                 "from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor",
    "moss": "from moss_transcribe_diarize import parse_transcript; "
            "from moss_transcribe_diarize.inference_utils import generate_transcription, build_transcription_messages",
}


def has_pip(python: Path) -> bool:
    """True when this interpreter can run pip (its own or the system one)."""
    return subprocess.run([str(python), "-m", "pip", "--version"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def ensure_pip(python: Path) -> None:
    """Give a venv a working pip without relying on ensurepip alone.

    With system_site_packages the system pip is importable and installs into
    the venv, so ensurepip is only the fallback.  A venv left half-made by an
    earlier failed attempt is repaired here instead of being reused broken.
    """
    if has_pip(python):
        return
    subprocess.run([str(python), "-m", "ensurepip", "--upgrade", "--default-pip"], check=False)
    if not has_pip(python):
        raise RuntimeError(f"{python} has no pip: neither the system pip nor ensurepip is available. "
                           "Install pip for the base interpreter (e.g. apt-get install python3-pip) and retry.")


def setup(profile: str, root: Path) -> Path:
    """Install a pinned adapter in a separate venv and return its interpreter."""
    directory = root / profile
    python = directory / "bin" / "python"
    contract = {"profile": profile, "requirements": PROFILES[profile],
                "base_python": sys.executable, "python_version": sys.version}
    marker = directory / "vcsasr-runtime.json"
    if marker.is_file():
        import json

        if json.loads(marker.read_text())["fingerprint"] == fingerprint(contract) and python.is_file():
            subprocess.run([str(python), "-c", IMPORT_CHECKS[profile]], check=True)
            return python
    if not python.is_file():
        # with_pip=True runs ensurepip, which Colab's Debian Python (3.13) does not ship.
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(directory)
    ensure_pip(python)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", *PROFILES[profile]], check=True)
    subprocess.run([str(python), "-c", IMPORT_CHECKS[profile]], check=True)
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (directory / "resolved-requirements.txt").write_text(freeze, encoding="utf-8")
    atomic_json(marker, {"fingerprint": fingerprint(contract), **contract})
    return python


def main(argv: list[str] | None = None) -> int:
    """Prepare one runtime; use the same command again after a Colab reconnect."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--root", type=Path, default=Path("/content/vcsasr-envs"))
    args = parser.parse_args(argv)
    print(setup(args.profile, args.root), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
