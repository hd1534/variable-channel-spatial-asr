"""Within-session centroid loss: the softmax variant of GE2E (Wan et al. 2018),
plus a calibrated same/different term.

Who-is-who only matters inside one session -- the same direction is a
different talker in another room -- so every comparison stays inside a session:
each labelled instant is scored against the centroid of every class *of its own
session* (its own class's centroid leaves the instant out), and the
cross-entropy asks for its own class.  The proposed model uses it twice: with
talker labels on the fused embedding (training only the gate) and with
position labels on the spatial embedding (training the spatial branch).

The softmax alone is blind to a shift of all of an anchor's scores, so it never
trains the absolute similarity scale -- yet "join this slot or open a new one"
is a decision on exactly that scale.  The calibrated term, a sigmoid
cross-entropy on ``scale * cos + bias`` against same/different (GE2E's
"contrast" form), trains it; ``-bias / scale`` is then a learned estimate of the
slot threshold.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SessionCentroidLoss(nn.Module):
    """``loss(z [B, K, D], labels [B, K] with -1 = unlabelled)``.

    >>> torch.manual_seed(0) and None
    >>> z = F.normalize(torch.tensor([[[1., 0.], [1., .1], [0., 1.], [.1, 1.]]]), dim=-1)
    >>> loss, stats = SessionCentroidLoss(contrast_weight=0.0)(z, torch.tensor([[0, 0, 1, 1]]))
    >>> round(stats["acc"], 2), loss.item() < 0.01
    (1.0, True)
    >>> loss_bad, _ = SessionCentroidLoss()(z, torch.tensor([[0, 1, 0, 1]]))
    >>> loss_bad.item() > 1.0
    True
    """

    def __init__(self, init_scale: float = 10.0, init_bias: float = -5.0, max_classes: int = 4,
                 contrast_weight: float = 0.5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))
        self.max_classes = max_classes
        self.contrast_weight = contrast_weight

    def threshold(self) -> float:
        """The cosine at which same and different are equally likely: ``-bias / scale``."""
        return float(-self.bias.detach() / self.scale.detach().clamp(min=0.1))

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
        labelled = labels >= 0
        # Zero the unlabelled rows first: a non-finite embedding where the loss
        # does not look (a silent chunk, say) must not leak into the sums below.
        z = torch.where(labelled[..., None], z.float(), torch.zeros_like(z, dtype=torch.float32))
        onehot = F.one_hot(labels.clamp(min=0), self.max_classes).to(z.dtype) * labelled[..., None].to(z.dtype)
        sums = torch.einsum("bks,bkd->bsd", onehot, z)
        counts = onehot.sum(1)
        cent = sums[:, None] - onehot[..., None] * z[:, :, None]  # [B, K, S, D], own instant left out
        n = counts[:, None] - onehot
        valid = n > 0
        cos = (F.normalize(cent, dim=-1) * z[:, :, None]).sum(-1)
        logits = self.scale.clamp(min=0.1) * cos + self.bias
        own_valid = valid.gather(-1, labels.clamp(min=0)[..., None]).squeeze(-1)
        anchors = labelled & own_valid & (valid.sum(-1) >= 2)
        if not anchors.any():
            zero = z.sum() * 0.0 + (self.scale + self.bias) * 0.0
            return zero, {"acc": float("nan"), "n": 0}
        sel_logits, sel_labels = logits[anchors], labels[anchors]
        softmax = F.cross_entropy(sel_logits.masked_fill(~valid[anchors], float("-inf")), sel_labels)
        loss = softmax
        if self.contrast_weight > 0:
            target = F.one_hot(sel_labels, self.max_classes).to(sel_logits.dtype)
            pair = valid[anchors]
            bce = F.binary_cross_entropy_with_logits(sel_logits[pair], target[pair])
            loss = loss + self.contrast_weight * bce
        with torch.no_grad():
            acc = (sel_logits.masked_fill(~valid[anchors], float("-inf")).argmax(-1) == sel_labels).float().mean().item()
        return loss, {"acc": acc, "n": int(anchors.sum())}
