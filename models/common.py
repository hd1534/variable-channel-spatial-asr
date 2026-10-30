"""Building blocks shared by every system of the comparison ladder.

Everything here obeys the causal rule of the page (1.4): the value at grid
instant ``t_k`` reads only samples before ``t_k``.  Concretely:

* chunks are ``[max(0, t_k - 1.0), t_k)`` (``spec.CHUNK_LENGTH_SEC``);
* the STFT is unpadded (``center=False``), so frame ``j`` covers samples
  ``[hop * j, hop * j + n_fft)`` and the frame read out for ``t_k`` is the last
  one that *ends* at or before ``t_k``;
* the VAD compares each grid window with the running maximum of the windows
  seen so far, never with a whole-session statistic.

Batches are dicts of tensors: ``audio`` ``[B, 4, S]`` (inactive channels
zeroed), ``mask`` ``[B, 4]`` bool, ``mic_xy`` ``[B, 4, 2]`` in the array frame,
and for training ``labels`` ``[B, 80]`` (talker index, ``-1`` = silence).
"""

from __future__ import annotations

import math
from typing import Any

import spec

SR = spec.SAMPLE_RATE_HZ
GRID = int(round(spec.DELTA_T_SEC * SR))
CHUNK = int(round(spec.CHUNK_LENGTH_SEC * SR))
N_GRID = spec.N_GRID_POINTS


def reference_channel(mask):  # noqa: ANN001, ANN201 - torch tensors
    """Lowest active channel per row, the positional rule of Baseline 1.

    >>> import torch
    >>> reference_channel(torch.tensor([[0, 0, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)).tolist()
    [2, 0]
    """
    import torch

    idx = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return torch.where(mask, idx, torch.full_like(idx, mask.shape[1])).min(dim=1).values


def reference_audio(batch: dict[str, Any]):  # noqa: ANN201
    import torch

    ref = reference_channel(batch["mask"])
    return batch["audio"][torch.arange(ref.shape[0], device=ref.device), ref]


def grid_chunks(signal):  # noqa: ANN001, ANN201
    """``[B, S]`` -> ``(chunks [B, 80, CHUNK], valid [B, 80])``.

    Chunk ``k`` holds ``[max(0, t_k - CHUNK), t_k)`` left-aligned (valid samples
    first, zeros after), the layout speechbrain's relative ``wav_lens`` expects.
    """
    import torch

    B, S = signal.shape
    padded = torch.cat([signal, signal.new_zeros(B, CHUNK)], dim=1)
    starts = [max(0, k * GRID - CHUNK) for k in range(N_GRID)]
    valid = torch.tensor([k * GRID - s for k, s in enumerate(starts)], device=signal.device)
    chunks = torch.stack([padded[:, s : s + CHUNK] for s in starts], dim=1)
    keep = torch.arange(CHUNK, device=signal.device)[None, :] < valid[:, None]
    chunks = chunks * keep[None].to(chunks.dtype)
    return chunks, valid.expand(B, N_GRID)


def causal_vad(signal, rel_db: float = -30.0, floor_dbfs: float = -60.0):  # noqa: ANN001, ANN201
    """Speech flag per grid instant from ``[B, S]`` audio, using only the past.

    ``t_k`` is speech when the RMS over ``[t_k - 0.25, t_k)`` is above
    ``floor_dbfs`` and within ``rel_db`` of the loudest window seen so far.

    >>> import torch
    >>> x = torch.zeros(1, 320000); x[0, 8000:24000] = 0.1
    >>> causal_vad(x)[0, :8].tolist()
    [False, False, False, True, True, True, True, False]
    """
    import torch

    B, S = signal.shape
    windows = signal[:, : (S // GRID) * GRID].reshape(B, -1, GRID)
    db = 10.0 * torch.log10(windows.pow(2).mean(-1) + 1e-12)
    db = torch.cat([db.new_full((B, 1), -200.0), db[:, : N_GRID - 1]], dim=1)  # window ending at t_k
    running = torch.cummax(db, dim=1).values
    return (db > floor_dbfs) & (db > running + rel_db)


def chunk_frame_ranges(n_fft: int, hop: int) -> list[tuple[int, int]]:
    """``[(first, last)]`` frame indices inside chunk ``k`` (``last = -1``: none).

    >>> r = chunk_frame_ranges(512, 160)
    >>> r[0], r[1], r[4], r[79][1] - r[79][0] + 1
    ((0, -1), (0, 21), (0, 96), 97)
    """
    out = []
    for k in range(N_GRID):
        end = k * GRID
        last = (end - n_fft) // hop if end >= n_fft else -1
        first = -(-max(0, end - CHUNK) // hop)
        out.append((first, last))
    return out


class FrozenECAPA:
    """speechbrain/spkrec-ecapa-voxceleb, frozen, outside every optimiser.

    Not an ``nn.Module`` on purpose: its weights never enter a checkpoint and
    never receive a gradient, which is what makes it the controlled voice cue
    of the page (2.2).  Chunks are fed in slices of ``batch`` to bound memory.
    """

    dim = 192

    def __init__(self, source: str, savedir: str | None, device: str, batch: int = 256) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        if device == "cpu":
            # The NNPACK convolution path is ~40x slower for ECAPA on Apple M1
            # (measured 640 ms vs 15 ms per 1 s segment); outputs are identical.
            torch.backends.nnpack.set_flags(False)
        # COPY, not speechbrain's default SYMLINK: Google Drive cannot hold
        # symlinks, and the checkpoint is cached there between Colab sessions.
        self.model = EncoderClassifier.from_hparams(source=source, savedir=savedir, run_opts={"device": device},
                                                    local_strategy=LocalStrategy.COPY)
        self.model.eval()
        self.batch = batch
        self.device = device

    def __call__(self, chunks, valid):  # noqa: ANN001, ANN204
        """``[N, CHUNK]`` and valid sample counts -> ``[N, 192]`` (zeros where empty)."""
        import torch

        out = chunks.new_zeros(chunks.shape[0], self.dim, dtype=torch.float32)
        nonempty = (valid > 0).nonzero().flatten()
        # Always float32: in float16 the filterbank's log of a silent chunk is
        # -inf, which turns into NaN after mean normalisation.
        with torch.no_grad(), torch.autocast(device_type=chunks.device.type, enabled=False):
            for i in range(0, nonempty.numel(), self.batch):
                idx = nonempty[i : i + self.batch]
                wav = chunks[idx].float()
                lens = valid[idx].float() / CHUNK
                emb = self.model.encode_batch(wav.to(self.device), lens.to(self.device)).squeeze(1)
                out[idx] = emb.to(out.dtype).to(out.device)
        return out


class StubVoice:
    """Offline stand-in for ECAPA: fixed random projection of log-spectrum statistics.

    Only for smoke runs without a checkpoint.  It carries some voice
    information (spectral envelope) but is not a speaker encoder; numbers from it
    mean nothing.
    """

    dim = 192

    def __init__(self, n_fft: int = 512, hop: int = 160, seed: int = 0) -> None:
        import torch

        g = torch.Generator().manual_seed(seed)
        self.n_fft, self.hop = n_fft, hop
        self.proj = torch.randn(2 * (n_fft // 2 + 1), self.dim, generator=g) / math.sqrt(n_fft)

    def __call__(self, chunks, valid):  # noqa: ANN001, ANN204
        import torch

        with torch.autocast(device_type=chunks.device.type, enabled=False):
            return self._embed(chunks.float(), valid)

    def _embed(self, chunks, valid):  # noqa: ANN001, ANN202
        import torch

        X = torch.stft(chunks, self.n_fft, self.hop, window=torch.hann_window(self.n_fft, device=chunks.device),
                       center=False, return_complex=True).abs().add(1e-6).log()  # [N, F, T]
        feats = torch.cat([X.mean(-1), X.std(-1)], dim=-1)
        out = feats @ self.proj.to(chunks.device)
        return torch.where((valid > 0)[:, None], out, torch.zeros_like(out))


def build_voice_encoder(model_cfg: Any, device: str, savedir: str | None = None):  # noqa: ANN201
    if model_cfg.voice_encoder == "ecapa":
        return FrozenECAPA(model_cfg.ecapa_source, savedir, device, batch=model_cfg.ecapa_batch)
    if model_cfg.voice_encoder == "stub":
        return StubVoice(model_cfg.n_fft, model_cfg.hop)
    raise ValueError(f"unknown voice_encoder {model_cfg.voice_encoder!r}")


def voice_embeddings(encoder: Any, batch: dict[str, Any], keep: Any = None):  # noqa: ANN201
    """Frozen voice embedding of the reference channel per grid instant, ``[B, 80, D]``.

    ``keep`` (``[B, 80]`` bool) skips the encoder where it is False and leaves a
    zero row there; training passes the labelled instants, the only ones the
    loss reads (about 65 % of them), to save about a third of the encoder's cost.
    """
    chunks, valid = grid_chunks(reference_audio(batch))
    if keep is not None:
        valid = valid * keep.to(valid.dtype)
    B = chunks.shape[0]
    emb = encoder(chunks.reshape(B * N_GRID, CHUNK), valid.reshape(-1))
    return emb.reshape(B, N_GRID, -1)


def srp_embeddings(batch: dict[str, Any], temperature: float = 1.0):  # noqa: ANN201
    """sqrt of Baseline 2's 36-bin SRP-PHAT distribution per chunk, ``[B, 80, 36]``.

    The square root makes the cosine of two embeddings the Bhattacharyya
    coefficient of the two distributions.  One active channel, an empty chunk
    or silence give the uniform distribution, so the embedding then carries no
    direction at all -- the space-only system cannot separate anyone at M = 1.
    """
    import numpy as np
    import torch

    from baselines.dsp_srp_phat import response_to_distribution, srp_phat_response

    audio = batch["audio"].detach().float().cpu().numpy()
    mask = batch["mask"].cpu().numpy()
    mic_xy = batch["mic_xy"].cpu().numpy().astype(np.float64)
    B = audio.shape[0]
    out = np.zeros((B, N_GRID, spec.K_AZIMUTH), dtype=np.float32)
    for b in range(B):
        channels = tuple(int(c) for c in np.flatnonzero(mask[b]))
        positions = [tuple(p) for p in mic_xy[b]]
        for k in range(N_GRID):
            lo, hi = max(0, k * GRID - CHUNK), k * GRID
            if hi - lo < GRID or len(channels) < 2:
                out[b, k] = 1.0 / spec.K_AZIMUTH
                continue
            resp = srp_phat_response(audio[b, list(channels), lo:hi], positions, channels, SR)
            out[b, k] = response_to_distribution(resp, temperature)
    return torch.from_numpy(np.sqrt(out)).to(batch["audio"].device)
