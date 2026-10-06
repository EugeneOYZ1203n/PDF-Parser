"""The paper's loss terms (Eq. 3-10).

    L_e   = (1 - le) ||P - P^||_2^2 + le ||P - P^||_1                  (3)
    L_conf = BCE(p, p^)                                               (4)
    L_F   = 1/n_stroke * sum_i (beta * L_e + L_conf)                  (5)
    L_R   = (1 - lr) ||S - RS||_2^2 + lr ||S - RS||_1                 (6)
    L_eos = BCE(e, e^)                                                (7)
    L_c   = (1 - lc) ||theta - theta^||_2^2 + lc ||theta - theta^||_1 (8)
    L_P   = 1/n_primitive * sum (beta * L_c + L_eos)                  (9)
    L_total = L_F + 1/n_stroke * (lR * L_R + lP * L_P)                (10)

beta = min(W, H). Prediction slot i is supervised by the i-th ground-truth
stroke in sorted order (the batch builder sorts them); slots past the
number of GT strokes have target confidence 0. L_R is the per-pixel mean
over each stroke's raster; L_R and L_P are summed over the strokes the
raster/vector branches ran on and divided by that count (Eq. 10's
1/n_stroke)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import LAMBDA_C, LAMBDA_E, LAMBDA_PRIM, LAMBDA_R, LAMBDA_RECON


def _l2l1(diff: torch.Tensor, lam: float, dim=-1) -> torch.Tensor:
    return (1 - lam) * diff.pow(2).sum(dim) + lam * diff.abs().sum(dim)


def feature_loss(enc: dict, gt_endpoints: torch.Tensor, valid: torch.Tensor, beta: float) -> torch.Tensor:
    """Eq. 5, averaged over the batch. `gt_endpoints (B, N, 4)`, `valid (B, N)`."""
    le = _l2l1(enc["endpoints"] - gt_endpoints, LAMBDA_E)                       # (B, N)
    lconf = F.binary_cross_entropy_with_logits(enc["logit"], valid.float(), reduction="none")
    per_slot = beta * le * valid.float() + lconf
    return per_slot.mean(dim=1).mean()


def emb_loss(pred_emb: torch.Tensor, target_emb: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Bootstrap supervision of the encoder's emb by the learned per-stroke
    embeddings (paper: "Stroke Encoder is then directly supervised with
    (I, {P}, {emb})")."""
    if not bool(valid.any()):
        return pred_emb.sum() * 0.0
    return F.mse_loss(pred_emb[valid], target_emb[valid])


def recon_loss_sum(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Eq. 6 per stroke (pixel mean), summed over strokes. `(M, 1, H, W)`."""
    pred = torch.sigmoid(logits.float())
    diff = pred - target.float()
    per = (1 - LAMBDA_R) * diff.pow(2).flatten(1).mean(1) + LAMBDA_R * diff.abs().flatten(1).mean(1)
    return per.sum()


def primitive_loss_sum(pred_curves: torch.Tensor, eos_logits: torch.Tensor, gt_curves: torch.Tensor,
                       prim_mask: torch.Tensor, n_prims: torch.Tensor, beta: float) -> torch.Tensor:
    """Eq. 9 per stroke, summed over strokes. `pred/gt (M, T, 6)`,
    `prim_mask (M, T)` true for real primitives, `n_prims (M,)`."""
    lc = _l2l1(pred_curves.float() - gt_curves.float(), LAMBDA_C)               # (M, T)
    t = torch.arange(prim_mask.shape[1], device=prim_mask.device)[None]
    eos_target = (t == (n_prims[:, None] - 1)).float()
    leos = F.binary_cross_entropy_with_logits(eos_logits.float(), eos_target, reduction="none")
    m = prim_mask.float()
    per = ((beta * lc + leos) * m).sum(1) / m.sum(1).clamp_min(1.0)
    return per.sum()


def total_loss(l_f: torch.Tensor, l_r_sum: torch.Tensor, l_p_sum: torch.Tensor, n_stroke: int) -> torch.Tensor:
    """Eq. 10."""
    return l_f + (LAMBDA_RECON * l_r_sum + LAMBDA_PRIM * l_p_sum) / max(n_stroke, 1)
