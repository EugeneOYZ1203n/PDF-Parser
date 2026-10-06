"""Small shared building blocks: norm, residual block, sine positional
encodings (DETR style), CoordConv input."""
from __future__ import annotations

import math

import torch
from torch import nn


def norm(ch: int) -> nn.GroupNorm:
    """GroupNorm instead of BatchNorm: batch statistics are meaningless at
    inference batch 1 and across the variable tile sizes we train on."""
    groups = 8 if ch % 8 == 0 else 1
    return nn.GroupNorm(groups, ch)


class ResBlock(nn.Module):
    """ResNet BasicBlock (two 3x3 convs + shortcut)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.n1 = norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.n2 = norm(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.short = None
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), norm(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        return self.act(y + (x if self.short is None else self.short(x)))


def pos1d(n: int, d: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """`(n, d)` sinusoidal encoding (Vaswani et al.)."""
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(n, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return pe.to(dtype)


def pos2d(h: int, w: int, d: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """`(h*w, d)` 2D sine encoding: half the channels encode y, half x
    (normalized to [0, 1] so it is resolution independent)."""
    half = d // 2
    ys = torch.linspace(0, 1, h, device=device)
    xs = torch.linspace(0, 1, w, device=device)
    i = torch.arange(0, half, 2, device=device, dtype=torch.float32)
    freq = torch.exp(-math.log(10000.0) * i / half) * 2 * math.pi * 10
    ey = torch.cat([torch.sin(ys[:, None] * freq), torch.cos(ys[:, None] * freq)], dim=1)  # (h, half)
    ex = torch.cat([torch.sin(xs[:, None] * freq), torch.cos(xs[:, None] * freq)], dim=1)  # (w, half)
    pe = torch.cat([ey[:, None, :].expand(h, w, -1), ex[None, :, :].expand(h, w, -1)], dim=2)
    pe = pe.reshape(h * w, -1)
    if pe.shape[1] < d:
        pe = torch.cat([pe, torch.zeros(h * w, d - pe.shape[1], device=device)], dim=1)
    return pe.to(dtype)


def prepare_input(gray: torch.Tensor) -> torch.Tensor:
    """uint8/float gray `(B, H, W)` (255 = paper) -> `(B, 3, H, W)`:
    ink intensity `1 - g/255` plus CoordConv x/y channels in (0, 1] (the
    paper's "mesh grid composed of the coordinates of each pixel")."""
    g = gray.float() / 255.0
    ink = 1.0 - g
    b, h, w = ink.shape
    xs = torch.arange(1, w + 1, device=ink.device, dtype=ink.dtype) / w
    ys = torch.arange(1, h + 1, device=ink.device, dtype=ink.dtype) / h
    gx = xs[None, None, :].expand(b, h, w)
    gy = ys[None, :, None].expand(b, h, w)
    return torch.stack([ink, gx, gy], dim=1)
