"""Stroke Decoder head: X_s (the shared UNet's full-resolution output) ->
ResNet-based reconstructor -> raster stroke RS_i (a logit map; the loss
applies the sigmoid). Only used for training (pixel-level supervision)."""
from __future__ import annotations

import torch
from torch import nn

from .layers import ResBlock


class StrokeDecoderHead(nn.Module):
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(ResBlock(in_ch, in_ch), ResBlock(in_ch, in_ch))
        self.out = nn.Conv2d(in_ch, 1, 1)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(xs))
