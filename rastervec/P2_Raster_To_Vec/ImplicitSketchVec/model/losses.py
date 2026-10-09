"""The paper's losses.

L_line (Eq. 2), per UDF channel: masked L1 between the predicted and GT UDF,
the mask M being the GT centerline UDF below 4.5 px -- "to ensure that all
branches within the subsequent Line Reconstruction network ... receive
enough valid information" -- averaged over the mask (ours: the paper's
`average(... * M)` normalised by the mask instead of every lattice point).

L_rec (Eq. 4) = L_edge + 0.5 L_vertex + 0.01 L_skeleton: masked
cross-entropy over the 4 edge-flag classes (same mask, at cell centres), L2
on the vertex positions of cells the path passes through, BCE on the 1 px
skeleton.

Both return `(loss, terms)` with the terms as detached 0-d tensors (read back
with `float()` -- no host sync here). Pass fp32 predictions under AMP.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import LAMBDA_SKELETON, LAMBDA_VERTEX, UDF_TRUNC_PX


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def udf_loss(pred: torch.Tensor, gt_px: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """`pred (B, C, L, L)` normalised, `gt_px` the same in px, `mask (B, 1, L, L)`."""
    err = (pred * UDF_TRUNC_PX - gt_px).abs()
    per_ch = [_masked_mean(err[:, c:c + 1], mask) for c in range(err.shape[1])]
    loss = sum(per_ch)
    return loss, {"L_line": loss.detach(), "L_line_center_px": per_ch[0].detach()}


def rec_loss(out: dict, b: dict) -> tuple[torch.Tensor, dict]:
    """`out` from `LineReconstructionNet`; `b` holds `edge (B, n, n)` long,
    `emask (B, n, n)`, `vert (B, 2, n, n)`, `vmask (B, n, n)`, `skel (B, n, n)`."""
    ce = F.cross_entropy(out["edge"], b["edge"], reduction="none")
    l_edge = _masked_mean(ce, b["emask"])
    vm = b["vmask"][:, None].to(out["vertex"].dtype)
    l_vertex = ((out["vertex"] - b["vert"]) ** 2 * vm).sum() / vm.sum().clamp_min(1.0)
    l_skel = F.binary_cross_entropy_with_logits(out["skeleton"][:, 0], b["skel"])
    loss = l_edge + LAMBDA_VERTEX * l_vertex + LAMBDA_SKELETON * l_skel
    return loss, {"L_edge": l_edge.detach(), "L_vertex": l_vertex.detach(), "L_skel": l_skel.detach()}
