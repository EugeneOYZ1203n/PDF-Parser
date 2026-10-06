"""The UNet shared by the Stroke Decoder and the Stroke Vectorizer
(paper Fig. 3, "Shared Weight"): X_s = UNet(F_i, I).

The image half (`encode`) runs once per image; the stroke feature F_i is
concatenated into the innermost layer and the decoder half (`decode`) runs
once per stroke, gathering that stroke's image's skips by `index`.
`decode(..., stop_level=L)` stops at level L (resolution / 2**L): the
Stroke Decoder needs full resolution (L=0), the Vectorizer only reads a
coarse level, which keeps inference cheap."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .layers import norm


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False), norm(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), norm(out_ch), nn.ReLU(inplace=True),
    )


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = _conv_block(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, in_ch: int, cond_dim: int, base: int = 32, depth: int = 4, max_ch: int = 256) -> None:
        super().__init__()
        self.depth = depth
        self.chs = [min(base * 2 ** i, max_ch) for i in range(depth + 1)]
        self.downs = nn.ModuleList([_conv_block(in_ch, self.chs[0])])
        for i in range(1, depth + 1):
            self.downs.append(_conv_block(self.chs[i - 1], self.chs[i]))
        bottom = self.chs[depth]
        self.fuse = nn.Sequential(nn.Conv2d(bottom + cond_dim, bottom, 1), norm(bottom), nn.ReLU(inplace=True),
                                  _conv_block(bottom, bottom))
        # ups[i] produces level i from level i+1
        self.ups = nn.ModuleList([_Up(self.chs[i + 1], self.chs[i], self.chs[i]) for i in range(depth)])
        self.grad_checkpoint = False

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips = []
        for i, block in enumerate(self.downs):
            if i > 0:
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        return skips

    def decode(
        self, skips: list[torch.Tensor], cond: torch.Tensor, index: torch.Tensor,
        stop_level: int = 0, return_levels: bool = False,
    ):
        """`cond (M, cond_dim)` per stroke, `index (M,)` -> its image row in
        `skips`. Returns the level-`stop_level` features `(M, C, h, w)`, or
        `{level: features}` for every level down to it with `return_levels`."""
        bottom = skips[-1][index]
        c = cond[:, :, None, None].expand(-1, -1, bottom.shape[2], bottom.shape[3]).to(bottom.dtype)
        x = self.fuse(torch.cat([bottom, c], dim=1))
        levels = {self.depth: x}
        for level in range(self.depth - 1, stop_level - 1, -1):
            up = self.ups[level]
            skip = skips[level][index]
            if self.grad_checkpoint and self.training:
                x = checkpoint(up, x, skip, use_reentrant=False)
            else:
                x = up(x, skip)
            levels[level] = x
        return levels if return_levels else x
