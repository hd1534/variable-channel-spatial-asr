"""Cut the two demo clips for the submission page's Examples section.

Clip A is one raw microphone -- what a single-channel system hears.
Clip B is the same instant with all four microphones delay-and-sum beamformed
toward the true direction. The point is that the difference is audible, and
that it lines up with the azimuth error the leaderboard already reports.

Both are cut from the same window of the same session so nothing else differs.
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import spec  # noqa: E402

SESSION = "sess_0008"
CLIP_SEC = 3.0
OUT_RATE = 8000          # 데모용. 원본 16 kHz 의 절반이라 용량이 절반이다.

# --- 세션의 기하와 정답 -----------------------------------------------------
meta = None
for line in open(REPO / "data/eval-dev/meta.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["case_id"] == spec.make_case_id(SESSION, "1111"):
        meta = r
        break
assert meta, SESSION

frames = None
for line in open(REPO / "data/eval-dev/ground_truth.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["case_id"] == spec.make_case_id(SESSION, "1111"):
        frames = r["frames"]
        break
assert frames

mics = np.asarray(meta["mic_positions_xy_m"], dtype=float)
az_deg = frames[len(frames) // 2]["azimuth_deg"]
dist_m = frames[len(frames) // 2]["distance_m"]
print(f"  {SESSION}: 방위 {az_deg:+.1f}°  거리 {dist_m:.2f} m  마이크 {len(mics)}개")

audio, sr = sf.read(REPO / f"data/eval-dev/audio/{SESSION}.wav", dtype="float64", always_2d=True)
print(f"  오디오 {audio.shape[0]/sr:.0f}초 × {audio.shape[1]}채널 @ {sr} Hz")

# --- 발화가 가장 촘촘한 3초 창을 고른다 --------------------------------------
times = sorted(f["eval_time_sec"] for f in frames)
best_start, best_n = times[0], 0
for t in times:
    n = sum(1 for u in times if t <= u < t + CLIP_SEC)
    if n > best_n:
        best_start, best_n = t, n
a = int(best_start * sr)
b = a + int(CLIP_SEC * sr)
seg = audio[a:b]
print(f"  창 {best_start:.2f}~{best_start + CLIP_SEC:.2f}초, 그 안의 발화 시점 {best_n}개")


def delay_and_sum(block: np.ndarray, mic_xy: np.ndarray, azimuth_deg: float, rate: int) -> np.ndarray:
    """Steer to ``azimuth_deg`` under a far-field plane-wave model.

    A plane wave from that direction reaches microphone ``m`` earlier by
    ``(x cos az + y sin az) / c`` than it reaches the array origin, so each
    channel is delayed by the negative of that before summing.  The shift is
    fractional, so it is applied as a phase ramp in the frequency domain
    rather than by rounding to whole samples -- at 0.16 m spacing the whole
    delay range is only about 7 samples at this rate, and rounding would throw
    away most of the steering.
    """
    rad = math.radians(azimuth_deg)
    direction = np.array([math.cos(rad), math.sin(rad)])
    tau = mic_xy @ direction / spec.SPEED_OF_SOUND_M_S      # seconds, + = arrives earlier

    n = block.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / rate)
    out = np.zeros(n)
    for m in range(block.shape[1]):
        spectrum = np.fft.rfft(block[:, m])
        shifted = spectrum * np.exp(-2j * np.pi * freqs * tau[m])
        out += np.fft.irfft(shifted, n=n)
    return out / block.shape[1]


single = seg[:, 0].copy()
beamed = delay_and_sum(seg, mics, az_deg, sr)


def resample_half(x: np.ndarray) -> np.ndarray:
    """16 kHz -> 8 kHz with an anti-alias filter, so the decimation is honest."""
    spectrum = np.fft.rfft(x)
    keep = len(spectrum) // 2
    spectrum[keep:] = 0.0
    filtered = np.fft.irfft(spectrum, n=len(x))
    return filtered[::2]


def to_wav_b64(x: np.ndarray, rate: int) -> tuple[str, int]:
    peak = np.max(np.abs(x))
    y = x / peak * 0.95 if peak > 0 else x
    pcm = np.clip(y * 32767.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), len(raw)


clips = {}
for label, x in (("single", single), ("beamed", beamed)):
    b64, raw_bytes = to_wav_b64(resample_half(x), OUT_RATE)
    clips[label] = b64
    print(f"  {label:8s} wav {raw_bytes/1024:6.1f} KB -> base64 {len(b64)/1024:6.1f} KB")

# 두 클립이 실제로 다른지 (같은 바이트를 두 번 넣는 사고 방지)
assert clips["single"] != clips["beamed"], "두 클립이 동일하다"
d = resample_half(beamed) - resample_half(single)
print(f"  두 파형의 차이 RMS: {np.sqrt((d**2).mean()):.5f}  (0 이면 빔포밍이 안 걸린 것)")

out = {
    "session": SESSION,
    "azimuth_deg": round(az_deg, 1),
    "distance_m": round(dist_m, 2),
    "window_sec": [round(best_start, 2), round(best_start + CLIP_SEC, 2)],
    "rate": OUT_RATE,
    "clips": clips,
}
dest = REPO / "runs" / "demo_clips.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out), encoding="utf-8")
print(f"  -> {dest}  ({dest.stat().st_size/1024:.0f} KB)")
