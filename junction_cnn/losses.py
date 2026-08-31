"""
Loss functions for JunctionCNN configured for high recall and angular accuracy:

1. junction_loss:
   - Modified CornerNet Focal Loss with an asymmetry bias towards higher recall.
   - Reduced positive exponent (alpha=1.0) punishes false negatives (missed junctions) 
     far more heavily than false positives, encouraging candidate predictions.
   - Spatial Distance Penalty: Background regions far from any target (determined via 
     max-pooling spatial expansion) receive a boosted negative loss multiplier. 
     This harshly penalizes false positives in true empty space ("no-junction zones") 
     while remaining lenient to slightly off-center peaks near ground-truth targets.

2. direction_loss:
   - Spatially-masked BCE weighted by circular angular distance between bins.
   - Evaluates predictions only in active spatial neighborhoods surrounding junctions.
   - Penalizes false positive direction predictions proportionally to their minimal 
     angular distance from true active direction targets. Near-miss angles incur small 
     penalties, whereas orthogonal or opposing direction errors face heavy scaling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def junction_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,  # Lowered from 2.0 to penalize missed junctions heavily
    beta: float = 4.0,
    soft_radius: int = 3,
    background_penalty_mult: float = 5.0,
    pos_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Numerically stable CornerNet focal loss.
    Penalizes far-away false positives heavily while allowing leniency near targets.
    """
    target = target.clamp(0.0, 1.0)
    probs = torch.sigmoid(logits)

    # Base BCE loss per pixel (unreduced)
    bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    # Positive vs Negative target masks
    pos_mask = target.ge(pos_threshold)
    neg_mask = ~pos_mask

    # Focal weights
    pos_weight = torch.pow(1.0 - probs, alpha) * pos_mask.float()
    neg_weight = torch.pow(probs, alpha) * torch.pow(1.0 - target, beta) * neg_mask.float()

    # Expand target region to find strict background pixels far from any target
    near_target_mask = F.max_pool2d(
        target,
        kernel_size=soft_radius * 2 + 1,
        stride=1,
        padding=soft_radius,
    )
    far_from_junction_mask = (near_target_mask < 0.1) & neg_mask

    # Apply boosted penalty only to false positives far from any target
    spatial_modifier = 1.0 + (far_from_junction_mask.float() * (background_penalty_mult - 1.0))
    neg_weight = neg_weight * spatial_modifier

    # Total weighted loss
    focal_loss = (bce_loss * pos_weight) + (bce_loss * neg_weight)

    num_pos = pos_mask.sum()
    if num_pos == 0:
        return focal_loss.mean()

    return focal_loss.sum() / num_pos.clamp_min(1.0)


def direction_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    junction_target: torch.Tensor,
    radius: int = 3,
    pos_weight_val: float = 5.0,
    pos_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Spatially-masked direction loss with angular error penalization for both
    False Positives and False Negatives.
    """
    num_bins = logits.shape[1]
    probs = torch.sigmoid(logits)

    if junction_target.ndim == 3:
        junction_target = junction_target.unsqueeze(1)

    # 1. Mask to local neighborhood around junctions
    junction_mask = (junction_target > 0.1).float()
    if radius > 0:
        junction_mask = F.max_pool2d(
            junction_mask,
            kernel_size=radius * 2 + 1,
            stride=1,
            padding=radius,
        )

    active_elements = junction_mask.sum() * num_bins
    if active_elements < 1.0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # 2. Circular angular distance matrix (num_bins x num_bins), normalized to [0, 1]
    bin_indices = torch.arange(num_bins, device=logits.device, dtype=torch.float32)
    angle_step = 2.0 * torch.pi / num_bins
    angles = bin_indices * angle_step
    diff = torch.abs(angles.unsqueeze(0) - angles.unsqueeze(1))
    angular_dist = torch.min(diff, 2.0 * torch.pi - diff) / torch.pi

    target_active = (target >= pos_threshold).float()

    # 3. Distance penalty calculation
    # False Positive Penalty: Distance from predicted active bins to ground truth
    dist_fp = torch.einsum("c t, b t h w -> b c h w", angular_dist, target_active)
    target_count = target_active.sum(dim=1, keepdim=True).clamp_min(1.0)
    dist_fp = dist_fp / target_count

    # False Negative Penalty: Distance from missed target bins to highest predicted bin
    best_pred_bin_dist = torch.einsum("c t, b c h w -> b t h w", angular_dist, probs)
    dist_fn = best_pred_bin_dist * target_active

    # 4. Numerically stable BCE with direction scaling
    pos_weight = torch.tensor([pos_weight_val], device=logits.device)
    bce_loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )

    # Apply FP penalty on neg targets and FN penalty on pos targets
    is_neg = (1.0 - target_active)
    angular_multiplier = 1.0 + 3.0 * (is_neg * dist_fp + target_active * dist_fn)
    angular_weighted_loss = bce_loss * angular_multiplier

    # Mask loss strictly to active spatial regions
    masked_loss = angular_weighted_loss * junction_mask
    return masked_loss.sum() / active_elements


def total_loss(
    outputs: dict[str, torch.Tensor],
    junction_target: torch.Tensor,
    direction_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:

    lj = junction_loss(outputs["junction_logits"], junction_target)
    ld = direction_loss(outputs["direction_logits"], direction_target, junction_target)

    loss = 8 * lj + ld ## Just to balance it so its roughly 1 to 1

    return loss, {
        "junction": float(lj.detach()),
        "direction": float(ld.detach()),
        "total": float(loss.detach()),
    }