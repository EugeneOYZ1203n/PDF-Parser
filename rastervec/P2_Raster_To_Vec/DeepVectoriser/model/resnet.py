"""ResNet18-style feature extractor for the Stroke Encoder (X_f = ResNet(I)).
Written in-repo -- torchvision isn't a dependency. Output stride 16."""
from __future__ import annotations

import torch
from torch import nn

from .layers import ResBlock, norm


class ResNetFeatures(nn.Module):
    def __init__(self, in_ch: int = 3, widths=(64, 128, 256, 256), blocks=(2, 2, 2, 2)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, widths[0], 3, 2, 1, bias=False), norm(widths[0]), nn.ReLU(inplace=True),
        )
        stages = []
        ch = widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            stride = 1 if i == 0 else 2
            layers = [ResBlock(ch, w, stride)] + [ResBlock(w, w) for _ in range(n - 1)]
            stages.append(nn.Sequential(*layers))
            ch = w
        self.stages = nn.Sequential(*stages)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))
