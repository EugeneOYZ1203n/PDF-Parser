"""Egiazarian et al.'s primitive extraction network (paper Sec. 3.2,
Appendix A, Fig. 10), plus checkpoint save/load (a checkpoint stores its own
config, so a `.pth` always rebuilds the architecture it was trained with).

    patch I_p (B, 1, 64, 64) in [0, 1]
      -> ResNet: stem conv + n_res ResNet18 BasicBlocks, c channels   X_im
      -> n_dec Transformer decoder blocks over n_prim query rows,
         X_0 = sine positional table (n_prim, d_emb)                 X_pr
      -> Linear -> sigmoid: n_prim x (params..., width, confidence)

d_emb equals the number of primitive parameters incl. confidence (paper:
6 for lines, 8 for curves). With 4 heads that isn't divisible, so every
head attends at the full d_emb width (Q/K/V projected d_emb -> heads * d_emb,
concatenated, projected back) -- our reading of "d_emb = 6, 4 heads".
The stem's stride 2 (32 x 32 = 1024 feature tokens) is ours."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from ..config import N_PARAMS


def norm(ch: int) -> nn.GroupNorm:
    """GroupNorm instead of BatchNorm: inference runs arbitrary batch sizes."""
    groups = 8 if ch % 8 == 0 else 1
    return nn.GroupNorm(groups, ch)


class BasicBlock(nn.Module):
    """ResNet18 BasicBlock (two 3x3 convs + identity shortcut)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.n1 = norm(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.n2 = norm(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.n1(self.conv1(x)))
        return self.act(self.n2(self.conv2(y)) + x)


def sine_table(n: int, d: int) -> torch.Tensor:
    """`(n, d)` Vaswani sinusoidal table (X_0 in the paper)."""
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float32)
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
    return pe


def pos2d(h: int, w: int, d: int) -> torch.Tensor:
    """`(h*w, d)` 2D sine encoding for the feature tokens (half y, half x)."""
    half = d // 2
    ys = torch.linspace(0, 1, h)
    xs = torch.linspace(0, 1, w)
    i = torch.arange(0, half, 2, dtype=torch.float32)
    freq = torch.exp(-math.log(10000.0) * i / max(half, 1)) * 2 * math.pi
    ey = torch.cat([torch.sin(ys[:, None] * freq), torch.cos(ys[:, None] * freq)], dim=1)
    ex = torch.cat([torch.sin(xs[:, None] * freq), torch.cos(xs[:, None] * freq)], dim=1)
    pe = torch.cat([ey[:, None, :].expand(h, w, -1), ex[None, :, :].expand(h, w, -1)], dim=2).reshape(h * w, -1)
    if pe.shape[1] < d:
        pe = torch.cat([pe, torch.zeros(h * w, d - pe.shape[1])], dim=1)
    return pe[:, :d]


class FullWidthAttention(nn.Module):
    """Multi-head attention where every head is `d_q` wide."""

    def __init__(self, d_q: int, d_kv: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.h, self.d = heads, d_q
        self.q = nn.Linear(d_q, heads * d_q)
        self.k = nn.Linear(d_kv, heads * d_q)
        self.v = nn.Linear(d_kv, heads * d_q)
        self.out = nn.Linear(heads * d_q, d_q)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, mem_pos: "torch.Tensor | None" = None) -> torch.Tensor:
        b, n, _ = x.shape
        m = mem.shape[1]
        key_in = mem if mem_pos is None else mem + mem_pos
        q = self.q(x).view(b, n, self.h, self.d).transpose(1, 2)
        k = self.k(key_in).view(b, m, self.h, self.d).transpose(1, 2)
        v = self.v(mem).view(b, m, self.h, self.d).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.d), dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(b, n, self.h * self.d)
        return self.out(y)


class DecoderBlock(nn.Module):
    """Fig. 10: self-attention, cross-attention over the image features,
    position-wise feed-forward, each followed by Add & Norm."""

    def __init__(self, d: int, d_mem: int, heads: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.self_att = FullWidthAttention(d, d, heads, dropout)
        self.cross_att = FullWidthAttention(d, d_mem, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, mem_pos: torch.Tensor) -> torch.Tensor:
        x = self.n1(x + self.drop(self.self_att(x, x)))
        x = self.n2(x + self.drop(self.cross_att(x, mem, mem_pos)))
        return self.n3(x + self.drop(self.ff(x)))


class PrimitiveNet(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.kind = cfg["prim_kind"]
        self.n_params = N_PARAMS[self.kind]
        self.d_emb = self.n_params + 1
        c = cfg["res_ch"]
        self.stem = nn.Sequential(nn.Conv2d(1, c, 3, 2, 1, bias=False), norm(c), nn.ReLU(inplace=True))
        self.res = nn.Sequential(*[BasicBlock(c) for _ in range(cfg["n_res"])])
        self.blocks = nn.ModuleList(
            DecoderBlock(self.d_emb, c, cfg["n_heads"], cfg["ff_dim"], cfg["dropout"]) for _ in range(cfg["n_dec"])
        )
        self.head = nn.Linear(self.d_emb, self.d_emb)
        self.register_buffer("x0", sine_table(cfg["n_prim"], self.d_emb), persistent=False)
        self._pos_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _mem_pos(self, h: int, w: int, c: int, device) -> torch.Tensor:
        key = (h, w)
        if key not in self._pos_cache:
            self._pos_cache[key] = pos2d(h, w, c)
        return self._pos_cache[key].to(device)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """`patch` (B, 1, S, S) ink intensity in [0, 1] -> `(B, n_prim, d_emb)`
        in [0, 1]: normalised coordinates (/ S), normalised width (/ S),
        confidence."""
        feat = self.res(self.stem(patch))
        b, c, h, w = feat.shape
        mem = feat.flatten(2).transpose(1, 2)  # (B, h*w, c)
        mem_pos = self._mem_pos(h, w, c, patch.device)[None]
        x = self.x0[None].expand(b, -1, -1)
        for blk in self.blocks:
            x = blk(x, mem, mem_pos)
        return torch.sigmoid(self.head(x))


def prepare_input(gray: torch.Tensor) -> torch.Tensor:
    """uint8/float gray `(B, S, S)` (255 = paper) -> ink `(B, 1, S, S)` in [0, 1]."""
    return (1.0 - gray.float() / 255.0)[:, None]


def build_model(cfg: dict) -> PrimitiveNet:
    return PrimitiveNet(cfg)


def save_weights(path: "str | Path", model: PrimitiveNet, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": model.cfg, "model": model.state_dict(), "meta": meta or {}}, tmp)
    tmp.replace(path)


def load_weights(path: "str | Path", map_location="cpu") -> PrimitiveNet:
    blob = torch.load(str(path), map_location=map_location, weights_only=False)
    model = build_model(blob["config"])
    model.load_state_dict(blob["model"])
    return model
