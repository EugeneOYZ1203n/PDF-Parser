"""The paper's multi-task loss (Sec. 3.2, Eq. 2-4):

    L = 1/n_prim * sum_k [ L_cls(p_k, p^_k) + L_loc(theta_k, theta^_k) ]
    L_cls = binary cross-entropy of the confidence
    L_loc = (1 - lambda) * ||theta_k - theta^_k||_1 + lambda * ||theta_k - theta^_k||_2^2

Targets: the GT primitives sorted (`geometry.sort_prims`), then placeholder
rows with confidence 0 and every parameter 0 -- the loss is applied to every
row, placeholders included, exactly as the paper states."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import LAMBDA_LOC


def primitive_loss(pred: torch.Tensor, target: torch.Tensor, lam: float = LAMBDA_LOC) -> tuple[torch.Tensor, dict]:
    """`pred`, `target` `(B, n_prim, d_emb)` -- params first, confidence last
    (pred already through the sigmoid; pass it as fp32 -- BCE is not
    autocast-safe). Returns the batch-mean loss and its terms as detached
    0-d tensors (read back with `float()` -- no host sync here)."""
    p = pred[..., -1].clamp(1e-6, 1 - 1e-6)
    l_cls = F.binary_cross_entropy(p, target[..., -1], reduction="none")       # (B, n)
    diff = pred[..., :-1] - target[..., :-1]
    l1 = diff.abs().sum(-1)
    l2 = (diff * diff).sum(-1)
    l_loc = (1.0 - lam) * l1 + lam * l2
    loss = (l_cls + l_loc).mean()
    return loss, {"L_cls": l_cls.detach().mean(), "L_loc": l_loc.detach().mean()}
