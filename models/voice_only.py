"""Voice-only: frozen ECAPA embedding per chunk, nothing learned.

The baseline every hypothesis is measured against (page 2.3).  It reads only
the reference channel, so apart from which physical microphone that is, its
input does not change with M.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.common import voice_embeddings


class VoiceOnly(nn.Module):
    trainable = False

    def __init__(self, cfg, voice) -> None:  # noqa: ANN001
        super().__init__()
        self.voice = voice

    def embed(self, batch: dict, cache: dict | None = None) -> torch.Tensor:
        v = cache["voice"] if cache and "voice" in cache else voice_embeddings(self.voice, batch)
        return F.normalize(v.float(), dim=-1)
