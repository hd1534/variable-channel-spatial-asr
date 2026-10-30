"""Naive fusion: voice and space similarities added with one fixed weight.

``z = [sqrt(w) * s, sqrt(1 - w) * v]`` with unit ``s`` (space-only) and unit
``v`` (voice-only), so ``cos(z_i, z_j) = w * cos_s + (1 - w) * cos_v``: a
weighted sum of the two systems' scores, with ``w`` chosen per number of
microphones on validation and then frozen (at M = 1 the space term is a
constant, so the tuned weight there reproduces voice-only).  Whatever the proposed model gains over this rung is what learning
*when* to trust each cue buys (page 2.1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.common import srp_embeddings, voice_embeddings


class NaiveFusion(nn.Module):
    trainable = False

    def __init__(self, cfg, voice) -> None:  # noqa: ANN001
        super().__init__()
        self.voice = voice
        self.temperature = cfg.model.srp_temperature
        default = 0.5 if cfg.model.fusion_weight is None else float(cfg.model.fusion_weight)
        self.weights_by_M = {m: default for m in range(1, 5)}
        """Spatial weight per number of active microphones, tuned on validation."""

    def embed(self, batch: dict, cache: dict | None = None) -> torch.Tensor:
        v = cache["voice"] if cache and "voice" in cache else voice_embeddings(self.voice, batch)
        s = cache["space"] if cache and "space" in cache else srp_embeddings(batch, self.temperature)
        v = F.normalize(v.float(), dim=-1)
        w = torch.tensor([self.weights_by_M[int(m)] for m in batch["mask"].sum(1)], device=v.device)[:, None, None]
        return torch.cat([w.sqrt() * s.float(), (1.0 - w).sqrt() * v], dim=-1)
