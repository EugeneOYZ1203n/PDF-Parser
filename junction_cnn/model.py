# model.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_DIRECTION_BINS = 72


class JunctionCNN(nn.Module):
    """
    Input:
        [B, 1, H, W]

    Output:
        junction:
            [B, 1, H, W]

        direction:
            [B, 72, H, W]

    The network intentionally does not downsample. This keeps the output
    aligned pixel-for-pixel with the input image.
    """

    def __init__(
        self,
        num_direction_bins: int = NUM_DIRECTION_BINS,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
        )

        self.junction_head = nn.Conv2d(
            128,
            1,
            kernel_size=1,
        )

        self.direction_head = nn.Conv2d(
            128,
            num_direction_bins,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(x)

        junction_logits = self.junction_head(features)
        direction_logits = self.direction_head(features)

        return {
            "junction_logits": junction_logits,
            "direction_logits": direction_logits,
        }