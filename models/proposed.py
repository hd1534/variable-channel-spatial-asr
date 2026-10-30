"""Proposed system: TAC spatial features + frozen ECAPA + causal GRU.

Spatial branch
    Per channel and STFT frame the input is *relative* only: each channel's
    unit phasor times the conjugate of the active channels' mean unit phasor
    (real and imaginary part: the phase against the array, scaled by how
    coherent the channels are in that bin) and its level against their mean
    log-magnitude (ILD), plus the channel's coordinates relative to the active
    microphones' centroid.  Absolute magnitude is withheld on purpose, to keep
    voice identity out of this branch.  The per-bin coherence can still carry
    some spectral pattern, so leakage is measured rather than assumed: the
    final validation re-runs every session with all talkers at one position
    (``Run._proposed_diagnostics``), where a branch that learned no voice
    separates nobody.  With one active channel every input is constant
    (phase 0, ILD 0, offset 0) and the branch carries no information, exactly
    as physics says it should.
    Shared-weight TAC blocks then mix the channels and a masked mean pools them,
    so the same weights take any M and any channel order.

Time
    A 2-layer GRU over the spatial frames.  ``carry`` (Carry-Over) runs once
    over the whole session and reads the hidden state at the last frame that
    ends by ``t_k``; ``reset`` (Chunk-Reset) restarts from zero for every chunk
    ``[t_k - 1, t_k)`` (page 2.1).  Both are causal.

Training split
    The spatial branch never sees the speaker loss.  It is trained to tell
    *positions* apart (talkers sharing a position share a label), because with
    position held fixed even phase and level differences between channels still
    identify a talker -- a branch trained on talker labels would learn voice
    and the gain over voice-only would no longer be the gain from space.  The
    speaker loss trains only the gate, which reads the spatial state (detached),
    M and the share of speech in the chunk, but not the voice.

Fusion
    ``z = [sin(a) * s, cos(a) * v]`` with ``s`` the unit spatial embedding, ``v``
    the unit frozen ECAPA embedding (exactly the voice-only system's) and
    ``a = pi/2 * sigmoid(gate(h, M, speech share))``.  Then
    ``cos(z_i, z_j) = sin a_i sin a_j cos_s + cos a_i cos a_j cos_v``: naive fusion
    is the special case of a constant gate and voice-only the case ``a = 0``.
    ``a`` is held at 0 for one active channel, so at M = 1 the model is exactly
    the voice-only system.  The gate sees one chunk at a time (its spatial
    state, M, the share of speech in the chunk): it can learn when space is
    unreliable in general, but not that two particular talkers share a position
    -- that pair-level evidence lives in the slot backend.  The angle form keeps ``|z| = 1`` and smooth gradients at both
    ends.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

import spec
from models.common import N_GRID, causal_vad, chunk_frame_ranges, reference_audio, voice_embeddings
from models.tac import TACBlock, masked_channel_mean

COORD_SCALE_M = spec.ARRAY_RADIUS_RANGE_M[1]
"""Largest array radius: brings microphone offsets to O(1)."""


class Proposed(nn.Module):
    trainable = True

    def __init__(self, cfg, voice) -> None:  # noqa: ANN001
        super().__init__()
        m = cfg.model
        self.voice = voice
        self.state = m.state
        self.vad = (cfg.slots.vad_rel_db, cfg.slots.vad_floor_dbfs)
        self.n_fft, self.hop = m.n_fft, m.hop
        n_bins = m.n_fft // 2 + 1
        d = m.d_model
        self.register_buffer("window", torch.hann_window(m.n_fft), persistent=False)
        self.inp = nn.Sequential(nn.Linear(3 * n_bins + 2, d), nn.PReLU(), nn.LayerNorm(d))
        self.tac = nn.ModuleList([TACBlock(d) for _ in range(m.n_tac_blocks)])
        self.gru = nn.GRU(d, d, num_layers=m.gru_layers, batch_first=True)
        self.proj = nn.Linear(d, m.d_spatial)
        self.m_embed = nn.Embedding(5, 16)
        self.gate = nn.Sequential(nn.Linear(d + 16 + 1, 128), nn.PReLU(), nn.Linear(128, 1))
        ranges = chunk_frame_ranges(m.n_fft, m.hop)
        self.chunk_len = max(last - first + 1 for first, last in ranges)
        last = torch.tensor([r[1] for r in ranges])
        idx = torch.full((N_GRID, self.chunk_len), -1, dtype=torch.long)
        for k, (first, end) in enumerate(ranges):
            n = end - first + 1
            if n > 0:
                idx[k, self.chunk_len - n :] = torch.arange(first, end + 1)
        self.register_buffer("readout_index", last, persistent=False)
        self.register_buffer("chunk_index", idx, persistent=False)
        self.last_gate: torch.Tensor | None = None
        self.last_spatial: torch.Tensor | None = None

    # -- spatial branch --------------------------------------------------------
    def spatial_frames(self, audio: torch.Tensor, mask: torch.Tensor, mic_xy: torch.Tensor) -> torch.Tensor:
        """``[B, 4, S]`` -> ``[B, T, d_model]``."""
        B, C, S = audio.shape
        with torch.autocast(device_type=audio.device.type, enabled=False):
            X = torch.stft(audio.float().reshape(B * C, S), self.n_fft, self.hop, window=self.window,
                           center=False, return_complex=True)
            X = X.reshape(B, C, X.shape[-2], X.shape[-1]).transpose(2, 3)  # [B, C, T, F]
            m = mask[:, :, None, None].to(X.real.dtype)
            mag = X.abs()
            unit = X / mag.clamp_min(1e-8)
            # Relative to the mean unit phasor, *not* re-normalised: where the
            # channels cancel (diffuse noise) the mean is ~0, so the feature fades
            # to 0 instead of taking an arbitrary direction from rounding error,
            # and its length is the inter-channel phase coherence.
            ref = (unit * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1.0)
            rel = unit * ref.conj()
            logmag = (mag + 1e-5).log()
            ild = logmag - masked_channel_mean(logmag, mask)[:, None]
            feats = torch.cat([rel.real, rel.imag, ild], dim=-1) * m
            centroid = masked_channel_mean(mic_xy.float(), mask)[:, None]
            coords = ((mic_xy.float() - centroid) / COORD_SCALE_M)[:, :, None].expand(B, C, feats.shape[2], 2)
            x = torch.cat([feats, coords * m], dim=-1)
        x = self.inp(x) * mask[:, :, None, None].to(x.dtype)
        for block in self.tac:
            x = block(x, mask)
        return masked_channel_mean(x, mask)

    def temporal(self, frames: torch.Tensor) -> torch.Tensor:
        """``[B, T, d]`` -> one hidden state per grid instant ``[B, 80, d]``.

        Runs in float32 even under mixed precision: the recurrence is a small
        share of the compute, and cuDNN's reduced-precision RNN support varies
        by GPU generation.
        """
        with torch.autocast(device_type=frames.device.type, enabled=False):
            return self._temporal(frames.float())

    def _temporal(self, frames: torch.Tensor) -> torch.Tensor:
        B, T, d = frames.shape
        if self.state == "carry":
            hidden, _ = self.gru(frames)
            idx = self.readout_index.clamp(min=0)
            out = hidden[:, idx]
            return out * (self.readout_index >= 0)[None, :, None].to(out.dtype)
        padded = torch.cat([frames.new_zeros(B, 1, d), frames], dim=1)
        chunks = padded[:, self.chunk_index + 1]  # [B, 80, L, d]; index 0 is the zero frame
        hidden, _ = self.gru(chunks.reshape(B * N_GRID, self.chunk_len, d))
        return hidden[:, -1].reshape(B, N_GRID, d)

    # -- fusion ----------------------------------------------------------------
    def embed(self, batch: dict, cache: dict | None = None) -> torch.Tensor:
        """Fused embedding ``[B, 80, d_spatial + 192]``; also leaves the spatial
        embedding (with its gradient) in ``last_spatial`` for the position loss."""
        mask = batch["mask"]
        with torch.no_grad():
            v = cache["voice"] if cache and "voice" in cache else voice_embeddings(self.voice, batch, batch.get("voice_keep"))
            # share of speech in the chunk's last second: tells the gate how much the spatial state has heard
            vad = causal_vad(reference_audio(batch), *self.vad).float()
            speech = torch.nn.functional.avg_pool1d(F.pad(vad[:, None], (3, 0)), 4, stride=1)[:, 0]
        v = F.normalize(v.float(), dim=-1)
        h = self.temporal(self.spatial_frames(batch["audio"], mask, batch["mic_xy"]))
        s = F.normalize(self.proj(h).float(), dim=-1)
        self.last_spatial = s
        n_active = mask.sum(1).clamp(max=4)
        m_emb = self.m_embed(n_active)[:, None].expand(-1, N_GRID, -1)
        # The gate is talker-agnostic (no voice input) and trains only on the speaker
        # loss: it reads the spatial state without sending gradient into it.
        gate_in = torch.cat([h.detach(), m_emb.to(h.dtype), speech[..., None].to(h.dtype)], dim=-1)
        angle = (math.pi / 2) * torch.sigmoid(self.gate(gate_in).float())
        # One microphone carries no direction at all: the spatial weight is zero by
        # construction there, so at M = 1 the model *is* the voice-only system.
        angle = angle * (n_active >= 2)[:, None, None].to(angle.dtype)
        self.last_gate = torch.sin(angle).pow(2).squeeze(-1).detach()
        return torch.cat([torch.sin(angle) * s.detach(), torch.cos(angle) * v], dim=-1)
