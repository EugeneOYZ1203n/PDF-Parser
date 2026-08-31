# targets.py

from __future__ import annotations

import math

import torch


NUM_BINS = 72
ANGLE_STEP = 360.0 / NUM_BINS


def angle_to_bin(angle_deg: float) -> int:
    angle_deg %= 360.0
    return int(round(angle_deg / ANGLE_STEP)) % NUM_BINS


def add_circular_angle_target(
    target: torch.Tensor,
    x: int,
    y: int,
    angle_deg: float,
    sigma_bins: float = 0.75,
) -> None:
    """
    target:
        [72, H, W]

    Adds a Gaussian-like circular target around an angle.
    """

    h, w = target.shape[1:]

    if not (0 <= x < w and 0 <= y < h):
        return

    center = angle_deg % 360.0

    for bin_idx in range(NUM_BINS):
        bin_angle = bin_idx * ANGLE_STEP

        diff = abs(bin_angle - center)
        diff = min(diff, 360.0 - diff)

        value = math.exp(
            -(diff * diff)
            / (2.0 * (sigma_bins * ANGLE_STEP) ** 2)
        )

        target[bin_idx, y, x] = max(
            target[bin_idx, y, x],
            value,
        )


def gaussian_heatmap(
    height: int,
    width: int,
    points: list[tuple[int, int]],
    sigma: float = 2.0,
) -> torch.Tensor:

    yy, xx = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij",
    )

    heatmap = torch.zeros(
        height,
        width,
        dtype=torch.float32,
    )

    for x, y in points:
        distance_sq = (
            (xx - x) ** 2
            + (yy - y) ** 2
        )

        blob = torch.exp(
            -distance_sq / (2.0 * sigma * sigma)
        )

        heatmap = torch.maximum(
            heatmap,
            blob,
        )

    return heatmap