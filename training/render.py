"""Apply the room on the training device, a few sessions per batched FFT.

Mixing on the CPU was the bottleneck: four FFT convolutions of a 20 s session
per talker cost 0.2-0.5 s per session in float64 numpy, about 6 CPU-seconds per
batch of 16, while the GPU step takes a fraction of that.  So the DataLoader
workers only decode speech and lay out turns, and this module does what
:func:`training.scenes.render_numpy` does -- convolve every sample of a talker
with the response of the displaced copy of its position it was spoken from, add
diffuse noise, set the session level, per-channel mismatch, one anti-clip
scalar, 16-bit quantisation -- with ``torch.fft``.  Given the same float32
input and no noise the two agree to within two 16-bit steps
(``tests/test_training.py``); training ships the dry talkers as float16, which
moves a sample by up to about 15 steps (RMS under one).

Diffuse noise is the classic isotropic-field construction: per STFT bin, white
noise for the four microphones is mixed through the Cholesky factor of the
spherically isotropic coherence ``sinc(2 pi f d_ij / c)``, then shaped pink.  It
is generated for all four microphones of the array and masked with the speech,
so an inactive channel stays silent.

Expects a batch with ``dry`` ``[B, 4, S]`` and ``variant`` ``[B, 4, S]`` (rows
past the session's talkers are zero), ``rir`` ``[B, 4, K, 4, L]`` (talker,
displaced copy, microphone, tap), ``target_rms`` ``[B]``, ``channel_gain``
``[B, 4]``, ``snr_db`` ``[B]`` (``nan`` = no noise), ``noise_seed`` ``[B]``,
``mic_xy`` ``[B, 4, 2]`` and ``mask`` ``[B, 4]``; adds ``audio`` ``[B, 4, S]``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

import spec
from simulate import PEAK_CEILING

CHUNK_SESSIONS = 4
"""Sessions per FFT pass: bounds the transient memory (about 0.6 GB at K = 3)."""

NOISE_N_FFT = 512
NOISE_HOP = 256
PINK_KNEE_HZ = 100.0


def diffuse_noise(mic_xy: torch.Tensor, n_samples: int, seeds: torch.Tensor) -> torch.Tensor:
    """Spherically isotropic pink noise for every microphone, ``[B, 4, n_samples]``, unit-ish level."""
    B, C, _ = mic_xy.shape
    device = mic_xy.device
    n_bins = NOISE_N_FFT // 2 + 1
    n_frames = n_samples // NOISE_HOP + 2
    freqs = torch.arange(n_bins, device=device, dtype=torch.float32) * spec.SAMPLE_RATE_HZ / NOISE_N_FFT
    dist = torch.cdist(mic_xy.float(), mic_xy.float())  # [B, C, C]
    coherence = torch.sinc(2.0 * freqs[None, :, None, None] * dist[:, None] / spec.SPEED_OF_SOUND_M_S)
    coherence = coherence + 1e-4 * torch.eye(C, device=device)
    mixing = torch.linalg.cholesky(coherence)  # [B, F, C, C]
    white = torch.empty(B, n_bins, n_frames, C, dtype=torch.complex64, device=device)
    for b in range(B):
        g = torch.Generator(device=device).manual_seed(int(seeds[b]))
        parts = torch.randn(2, n_bins, n_frames, C, generator=g, device=device) / math.sqrt(2.0)
        white[b] = torch.complex(parts[0], parts[1])
    spectra = torch.einsum("bfij,bftj->bfti", mixing.to(torch.complex64), white)
    pink = (freqs.clamp_min(PINK_KNEE_HZ) / PINK_KNEE_HZ).rsqrt()
    spectra = spectra * pink[None, :, None, None]
    frames = spectra.permute(0, 3, 1, 2).reshape(B * C, n_bins, n_frames)
    window = torch.hann_window(NOISE_N_FFT, device=device)
    noise = torch.istft(frames, NOISE_N_FFT, NOISE_HOP, window=window, length=n_samples)
    return noise.reshape(B, C, n_samples)


def _render(batch: dict[str, Any], lo: int, hi: int) -> torch.Tensor:
    dry = batch["dry"][lo:hi].float()  # [b, T, S]
    variant = batch["variant"][lo:hi]
    rir = batch["rir"][lo:hi].float()  # [b, T, K, M, L]
    b, n_talkers, S = dry.shape
    K, L = rir.shape[2], rir.shape[-1]
    ids = torch.arange(K, device=dry.device)
    sources = dry[:, :, None, :] * (variant[:, :, None, :] == ids[None, None, :, None]).to(dry.dtype)
    sources = sources.reshape(b, n_talkers * K, S)
    rir = rir.reshape(b, n_talkers * K, rir.shape[3], L)
    size = 1 << (S + L - 2).bit_length()
    spectrum = torch.einsum("bsf,bsmf->bmf", torch.fft.rfft(sources, size), torch.fft.rfft(rir, size))
    wet = torch.fft.irfft(spectrum, size)[..., :S]

    snr = batch["snr_db"][lo:hi].float()
    noisy = torch.isfinite(snr)
    if bool(noisy.any()):
        noise = diffuse_noise(batch["mic_xy"][lo:hi], S, batch["noise_seed"][lo:hi])
        speech_power = wet.pow(2).mean(dim=(1, 2))
        noise_power = noise.pow(2).mean(dim=(1, 2)).clamp_min(1e-20)
        scale = torch.where(noisy, (speech_power / noise_power / 10 ** (snr / 10)).sqrt(), torch.zeros_like(snr))
        wet = wet + noise * torch.nan_to_num(scale)[:, None, None]

    level = wet.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-12)
    wet = wet * (batch["target_rms"][lo:hi].float() / level)[:, None, None]
    wet = wet * batch["channel_gain"][lo:hi].float()[:, :, None]
    peak = wet.abs().amax(dim=(1, 2))
    wet = wet * torch.where(peak > PEAK_CEILING, PEAK_CEILING / peak, torch.ones_like(peak))[:, None, None]
    return torch.round(wet.clamp(-1.0, 1.0) * 32767.0) / 32768.0


def render_batch(batch: dict[str, Any]) -> dict[str, Any]:
    with torch.autocast(device_type=batch["dry"].device.type, enabled=False):
        B = batch["dry"].shape[0]
        audio = torch.cat([_render(batch, lo, min(B, lo + CHUNK_SESSIONS)) for lo in range(0, B, CHUNK_SESSIONS)])
    out = dict(batch)
    out["audio"] = audio * batch["mask"][:, :, None].to(audio.dtype)
    return out
