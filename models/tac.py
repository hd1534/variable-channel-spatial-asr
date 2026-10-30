"""Transform-Average-Concatenate (Luo et al., ICASSP 2020, "FaSNet-TAC").

Every channel is transformed by the *same* weights, the transformed channels
are averaged over the **active** channels only, and the average is concatenated
back onto every channel.  Shared weights plus a mean make the block
permutation-equivariant and independent of the channel count, which is what
lets one set of weights serve M = 1, 2, 3 and 4 (the page's single-weight
premise).  Inactive channels are zeroed on the way out so they can never leak
into a later average.
"""

from __future__ import annotations

import torch
from torch import nn


def masked_channel_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over dim 1 (channels) of ``[B, C, ...]`` counting active channels only.

    >>> x = torch.tensor([[[1.0], [3.0], [100.0]]])
    >>> masked_channel_mean(x, torch.tensor([[True, True, False]])).tolist()
    [[2.0]]
    """
    m = mask.to(x.dtype).reshape(*mask.shape, *([1] * (x.dim() - 2)))
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class TACBlock(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or dim * 3 // 2
        self.transform = nn.Sequential(nn.Linear(dim, hidden), nn.PReLU())
        self.average = nn.Sequential(nn.Linear(hidden, hidden), nn.PReLU())
        self.concat = nn.Sequential(nn.Linear(2 * hidden, dim), nn.PReLU())
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x`` ``[B, C, T, D]``, ``mask`` ``[B, C]`` -> ``[B, C, T, D]``."""
        y = self.transform(x)
        z = self.average(masked_channel_mean(y, mask))[:, None].expand_as(y)
        out = x + self.norm(self.concat(torch.cat([y, z], dim=-1)))
        return out * mask[:, :, None, None].to(out.dtype)
