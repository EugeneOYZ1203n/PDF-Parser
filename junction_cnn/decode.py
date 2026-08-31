from __future__ import annotations

import numpy as np
import torch

from junction_cnn.constants import (
    DIRECTION_NMS_DEGREES,
    DIRECTION_RADIUS,
    DIRECTION_THRESHOLD,
    JUNCTION_NMS_RADIUS,
    JUNCTION_THRESHOLD,
    NUM_DIRECTION_BINS,
)


def decode_junctions(
    heatmap: torch.Tensor,
    threshold: float = JUNCTION_THRESHOLD,
    nms_radius: int = JUNCTION_NMS_RADIUS,
) -> list[tuple[int, int, float]]:
    """Decode junctions from heatmap using local maximum pooling."""
    pooled = torch.nn.functional.max_pool2d(
        heatmap[None, None],
        kernel_size=nms_radius * 2 + 1,
        stride=1,
        padding=nms_radius,
    )[0, 0]

    maxima = (heatmap == pooled) & (heatmap >= threshold)
    ys, xs = torch.where(maxima)

    return [
        (int(x), int(y), float(heatmap[y, x]))
        for x, y in zip(xs.tolist(), ys.tolist())
    ]


def circular_difference(a: float, b: float) -> float:
    """Calculate minimum angular difference in degrees."""
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def decode_directions(
    direction_logits: torch.Tensor,
    junctions: list[tuple[int, int, float]],
    threshold: float = DIRECTION_THRESHOLD,
    radius: int = DIRECTION_RADIUS,
    nms_degrees: float = DIRECTION_NMS_DEGREES,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Decode directions for detected junctions using vectorized max pooling."""
    if not junctions:
        return {}

    probabilities = torch.sigmoid(direction_logits)
    c, h, w = probabilities.shape

    # Max pool over spatial neighborhood so each pixel holds local neighborhood maxima
    kernel_size = radius * 2 + 1
    pooled = torch.nn.functional.max_pool2d(
        probabilities[None],
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )[0]

    # Batch gather scores across all junction coordinates
    coords = torch.tensor([(x, y) for x, y, _ in junctions], device=probabilities.device, dtype=torch.long)
    xs = coords[:, 0].clamp(0, w - 1)
    ys = coords[:, 1].clamp(0, h - 1)

    # Junction direction scores shape: (num_junctions, BINS)
    junction_scores = pooled[:, ys, xs].T.cpu().numpy()
    bin_angles = np.arange(NUM_DIRECTION_BINS) * (360.0 / NUM_DIRECTION_BINS)
    result = {}

    for idx, (x, y, _) in enumerate(junctions):
        scores = junction_scores[idx]
        valid_indices = np.where(scores >= threshold)[0]

        if len(valid_indices) == 0:
            result[(x, y)] = []
            continue

        cand_angles = bin_angles[valid_indices]
        cand_scores = scores[valid_indices]

        # Sort by confidence descending
        order = np.argsort(-cand_scores)
        cand_angles = cand_angles[order]
        cand_scores = cand_scores[order]

        selected = []
        for angle, score in zip(cand_angles, cand_scores):
            if not any(
                circular_difference(angle, sel_angle) < nms_degrees
                for sel_angle, _ in selected
            ):
                selected.append((float(angle), float(score)))

        result[(x, y)] = selected

    return result