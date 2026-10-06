"""Stroke Encoder (paper Eq. 1-2): TransEmb = Transformer(PosEmb_1d, X_f),
{F_i} = MLP(TransEmb), F_i = (P0, P1, emb, p).

DETR-style parallel decoding: the `n_stroke` queries are the fixed 1D
sinusoidal positional embedding, cross-attending to the ResNet feature map
(+ 2D sine positions). Endpoints are normalized to the tile, (x, y) in
[0, 1] (sigmoid); `p` is returned as a logit."""
from __future__ import annotations

import torch
from torch import nn

from .layers import pos1d, pos2d
from .resnet import ResNetFeatures


class StrokeEncoder(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.n_stroke = cfg["n_stroke"]
        self.d_emb = cfg["d_emb"]
        self.backbone = ResNetFeatures(3)
        self.proj = nn.Conv2d(self.backbone.out_ch, d, 1)
        layer = nn.TransformerDecoderLayer(
            d, cfg["n_heads"], cfg["ff_dim"], cfg["dropout"], batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, cfg["enc_layers"])
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, 4 + self.d_emb + 1))
        self.d = d

    def forward(self, x: torch.Tensor) -> dict:
        f = self.proj(self.backbone(x))
        b, d, h, w = f.shape
        mem = f.flatten(2).transpose(1, 2) + pos2d(h, w, d, f.device, f.dtype)[None]
        tgt = pos1d(self.n_stroke, d, f.device, f.dtype)[None].expand(b, -1, -1)
        out = self.head(self.transformer(tgt, mem))
        return {
            "endpoints": torch.sigmoid(out[..., :4]),       # (B, N, 4): x0, y0, x1, y1
            "emb": out[..., 4:4 + self.d_emb],              # (B, N, d_emb)
            "logit": out[..., -1],                          # (B, N)
        }
