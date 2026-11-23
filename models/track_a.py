"""Track A: a lightweight Conformer trained from scratch, no pretrained encoder.

Inputs (per 10 ms STFT frame, ``n_fft`` 512 / ``hop`` 160 like the proposed model)
    * the log-mel spectrum of the reference channel (the lowest active one,
      the channel every other system's voice cue reads) -- optional
      (``track_a.use_mel``);
    * GCC-PHAT of every active microphone pair (both orderings), lags ``-L..L``
      samples, plus the pair's relative position.  Pairs share one encoder and are pooled by a
      masked mean over TAC blocks (:mod:`models.tac`), so one set of weights takes
      any M and any channel order.  At M = 1 there is no pair and the spatial
      input is zero.

``stack`` STFT frames make one model frame (40 ms by default).  A Conformer
(macaron feed-forward, self-attention, depthwise convolution) runs over the
session and the frame read out for each grid instant goes through a linear
layer to a unit embedding, handed to the shared slot backend like every other
system (``embed(batch) -> [B, 80, D]``).

One model, two observation modes
    ``causal``: self-attention sees ``left_context_sec`` of the past and nothing
    after; the frame read for ``t_k`` is the last one that *ends* by ``t_k``; the
    depthwise convolution is left-padded.  This is the official causal scope.
    ``full``: self-attention sees the whole session and the frame read is the one
    centred in the official label window ``[t_k, t_k + 0.25)``.  This is a
    ``full_audio`` reference row.
    Training draws the mode per batch (``track_a.full_context_prob``) and labels
    a causal batch with what the causal window holds (``labels``) and a full
    batch with the official window (``labels_eval``), the way dynamic-chunk
    training lets one ASR model both stream and see everything.  The
    convolution stays causal in both modes, so the modes differ only in what
    attention may read.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

import spec
from models.common import N_GRID, reference_channel
from models.tac import TACBlock, masked_channel_mean

SR = spec.SAMPLE_RATE_HZ
GRID = int(round(spec.DELTA_T_SEC * SR))
COORD_SCALE_M = spec.ARRAY_RADIUS_RANGE_M[1]
PAIRS: tuple[tuple[int, int], ...] = tuple((i, j) for i in range(spec.N_CHANNELS_MAX)
                                           for j in range(spec.N_CHANNELS_MAX) if i != j)
"""Both orderings of every microphone pair.  A channel permutation then only
reorders this set, so the masked mean over pairs does not depend on channel
order; with one ordering per pair a permutation would flip some pairs' lags
and geometry, which the shared pair encoder would read as different input."""
CONTEXTS: tuple[str, ...] = ("causal", "full")


def mel_filterbank(n_fft: int, n_mels: int, sample_rate: int = SR) -> torch.Tensor:
    """HTK-style triangular filters ``[n_mels, n_fft // 2 + 1]`` over 0..Nyquist.

    >>> fb = mel_filterbank(512, 80)
    >>> tuple(fb.shape), bool((fb.sum(1) > 0).all())
    ((80, 257), True)
    """
    def hz_to_mel(f: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    freqs = torch.linspace(0.0, sample_rate / 2, n_bins)
    edges = mel_to_hz(torch.linspace(0.0, float(hz_to_mel(torch.tensor(sample_rate / 2.0))), n_mels + 2))
    lower, centre, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rise = (freqs[None] - lower) / (centre - lower)
    fall = (upper - freqs[None]) / (upper - centre)
    return torch.clamp(torch.minimum(rise, fall), min=0.0)


def readout_frames(n_frames: int, n_fft: int, hop: int, stack: int, context: str) -> torch.Tensor:
    """Model frame read for each grid instant (``-1``: none ends by ``t_k`` yet).

    A model frame ``g`` stacks STFT frames ``stack*g .. stack*g + stack - 1`` and
    so covers samples ``[hop*stack*g, hop*(stack*g + stack - 1) + n_fft)``.

    >>> c = readout_frames(1997, 512, 160, 4, "causal")
    >>> int(c[0]), int(c[1]), int(c[79])
    (-1, 4, 492)
    >>> f = readout_frames(1997, 512, 160, 4, "full")
    >>> int(f[0]), int(f[79])
    (2, 496)
    """
    n_groups = n_frames // stack
    out = []
    for k in range(N_GRID):
        end = k * GRID
        if context == "causal":
            g = ((end - n_fft) // hop - (stack - 1)) // stack if end >= n_fft else -1
            out.append(min(g, n_groups - 1) if g >= 0 else -1)
        else:
            centre = end + GRID / 2
            g = round(((centre - n_fft / 2) / hop - (stack - 1) / 2) / stack)
            out.append(min(max(g, 0), n_groups - 1))
    return torch.tensor(out, dtype=torch.long)


def attention_mask(n: int, context: str, left: int, device: torch.device) -> torch.Tensor | None:
    """``[n, n]`` bool, True where query ``i`` may read key ``j`` (``None``: everything).

    >>> attention_mask(4, "causal", 1, torch.device("cpu")).int().tolist()
    [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
    """
    if context == "full":
        return None
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    return (j <= i) & (j >= i - left)


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(d * mult, d), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.heads = heads
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return F.dropout(self.out(y.transpose(1, 2).reshape(B, T, d)), self.dropout, self.training)


class CausalConv(nn.Module):
    """Conformer convolution module with a left-padded depthwise convolution and
    LayerNorm in place of BatchNorm (no statistics shared across a batch)."""

    def __init__(self, d: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw_in = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, kernel, groups=d)
        self.mid = nn.LayerNorm(d)
        self.pw_out = nn.Linear(d, d)
        self.kernel = kernel
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.glu(self.pw_in(self.norm(x)), dim=-1).transpose(1, 2)
        y = self.dw(F.pad(y, (self.kernel - 1, 0))).transpose(1, 2)
        y = self.pw_out(F.silu(self.mid(y)))
        return F.dropout(y, self.dropout, self.training)


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, mult: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.ff1 = FeedForward(d, mult, dropout)
        self.att = SelfAttention(d, heads, dropout)
        self.conv = CausalConv(d, kernel, dropout)
        self.ff2 = FeedForward(d, mult, dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.att(x, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


def sinusoid(n: int, d: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d // 2])
    return pe


class TrackA(nn.Module):
    trainable = True
    contexts = CONTEXTS
    select_min_M = 1
    """Checkpoints are chosen on M = 1..4: unlike the proposed model, Track A
    has to answer at M = 1 on its own."""

    def __init__(self, cfg, voice=None) -> None:  # noqa: ANN001 - voice is unused (no pretrained encoder)
        super().__init__()
        a = cfg.track_a
        self.n_fft, self.hop, self.stack = cfg.model.n_fft, cfg.model.hop, a.stack
        self.use_mel = a.use_mel
        self.lag = a.gcc_max_lag
        self.full_prob = a.full_context_prob
        d = a.d_model
        frame_ms = self.hop * self.stack / SR
        self.left = max(1, int(round(a.left_context_sec / frame_ms)))
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.register_buffer("mel_fb", mel_filterbank(self.n_fft, a.n_mels), persistent=False)
        self.register_buffer("pair_i", torch.tensor([p[0] for p in PAIRS]), persistent=False)
        self.register_buffer("pair_j", torch.tensor([p[1] for p in PAIRS]), persistent=False)
        n_lags = 2 * self.lag + 1
        self.pair_in = nn.Sequential(nn.Linear(self.stack * n_lags + 2, d), nn.PReLU(), nn.LayerNorm(d))
        self.pair_tac = nn.ModuleList([TACBlock(d) for _ in range(a.n_pair_tac_blocks)])
        if self.use_mel:
            self.mel_in = nn.Sequential(nn.LayerNorm(self.stack * a.n_mels), nn.Linear(self.stack * a.n_mels, d),
                                        nn.PReLU())
        self.fuse = nn.Linear(d * (2 if self.use_mel else 1), d)
        self.blocks = nn.ModuleList([ConformerBlock(d, a.n_heads, a.ff_mult, a.conv_kernel, a.dropout)
                                     for _ in range(a.n_layers)])
        self.proj = nn.Linear(d, a.d_embed)
        self.context = "causal"
        """Mode used outside training; training draws one per batch."""
        self.train_label_key = "labels"
        """Which label grid the batch just embedded should be trained on."""
        self.last_context = "causal"
        self.last_spatial: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None

    # -- features ----------------------------------------------------------------
    def _stack(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., T, F]`` -> ``[..., T // stack, stack * F]``."""
        G = x.shape[-2] // self.stack
        x = x[..., : G * self.stack, :]
        return x.reshape(*x.shape[:-2], G, self.stack * x.shape[-1])

    def features(self, batch: dict) -> torch.Tensor:
        """``[B, G, d_model]`` model frames before the Conformer."""
        audio, mask, mic_xy = batch["audio"], batch["mask"], batch["mic_xy"]
        B, C, S = audio.shape
        with torch.autocast(device_type=audio.device.type, enabled=False):
            X = torch.stft(audio.float().reshape(B * C, S), self.n_fft, self.hop, window=self.window,
                           center=False, return_complex=True)
            X = X.reshape(B, C, X.shape[-2], X.shape[-1]).transpose(2, 3)  # [B, C, T, F]
            cross = X[:, self.pair_i] * X[:, self.pair_j].conj()
            phat = cross / cross.abs().clamp_min(1e-8)
            gcc = torch.fft.irfft(phat, n=self.n_fft, dim=-1)
            gcc = torch.cat([gcc[..., -self.lag:], gcc[..., : self.lag + 1]], dim=-1)  # lags -L..L
            pair_mask = mask[:, self.pair_i] & mask[:, self.pair_j]  # [B, P]
            geom = (mic_xy[:, self.pair_j].float() - mic_xy[:, self.pair_i].float()) / COORD_SCALE_M
            g = self._stack(gcc)  # [B, P, G, stack * lags]
            pair_x = torch.cat([g, geom[:, :, None].expand(-1, -1, g.shape[2], -1)], dim=-1)
            pair_x = pair_x * pair_mask[:, :, None, None].to(pair_x.dtype)
            mel = None
            if self.use_mel:
                ref = reference_channel(mask)
                power = X[torch.arange(B, device=X.device), ref].abs().pow(2)  # [B, T, F]
                mel = self._stack(torch.log(power @ self.mel_fb.T + 1e-6))
        x = self.pair_in(pair_x) * pair_mask[:, :, None, None].to(pair_x.dtype)
        for block in self.pair_tac:
            x = block(x, pair_mask)
        spatial = masked_channel_mean(x, pair_mask)  # [B, G, d]; zero with no pair (M = 1)
        parts = [spatial] if mel is None else [self.mel_in(mel), spatial]
        return self.fuse(torch.cat(parts, dim=-1))

    # -- model ---------------------------------------------------------------------
    def embed(self, batch: dict, cache: dict | None = None) -> torch.Tensor:  # noqa: ARG002 - no cached cue
        """Unit embedding per grid instant ``[B, 80, d_embed]``."""
        if self.training:
            context = "full" if float(torch.rand(())) < self.full_prob else "causal"
        else:
            context = self.context
        self.last_context = context
        self.train_label_key = "labels_eval" if context == "full" else "labels"
        x = self.features(batch)
        x = x + sinusoid(x.shape[1], x.shape[2], x.device).to(x.dtype)
        mask = attention_mask(x.shape[1], context, self.left, x.device)
        for block in self.blocks:
            x = block(x, mask)
        n_frames = (batch["audio"].shape[-1] - self.n_fft) // self.hop + 1
        idx = readout_frames(n_frames, self.n_fft, self.hop, self.stack, context).to(x.device)
        z = self.proj(x[:, idx.clamp(min=0)]).float()
        z = z * (idx >= 0)[None, :, None].to(z.dtype)
        return F.normalize(z, dim=-1)
