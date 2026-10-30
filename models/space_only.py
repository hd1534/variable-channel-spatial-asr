"""Space-only: Baseline 2's SRP-PHAT direction distribution per chunk.

No voice information at all, so at M = 1 every chunk looks the same and the
system cannot separate anyone; how far it gets as M grows is the "space alone"
rung of the ladder (page 2.1).
"""

from __future__ import annotations

import torch
from torch import nn

from models.common import srp_embeddings


class SpaceOnly(nn.Module):
    trainable = False

    def __init__(self, cfg, voice=None) -> None:  # noqa: ANN001
        super().__init__()
        self.temperature = cfg.model.srp_temperature

    def embed(self, batch: dict, cache: dict | None = None) -> torch.Tensor:
        return cache["space"] if cache and "space" in cache else srp_embeddings(batch, self.temperature)
