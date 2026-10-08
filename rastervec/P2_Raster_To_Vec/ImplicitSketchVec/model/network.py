"""Yan et al.'s two sub-networks (paper Sec. 3, 5, 6, Appendix A/B), plus
checkpoint save/load (a checkpoint stores its own config).

Distance Field Prediction (DFP, Sec. 5): raster `(B, 1, H, W)` ink ->
six unsigned distance fields at 2x super-resolution, `(B, 6, 2H+1, 2W+1)`
(centerline, under-sampling map, end / sharp / junction keypoints, all
keypoints), each predicted as udf / UDF_TRUNC_PX in [0, 1]. Modelled on
Puhachov et al. 2021 (the paper's base): a fully convolutional ResNeXt
network with cardinality 8 and gradually increasing dilation; the heavy
blocks run at half resolution, a light head at 2x (layout ours). It ends in
the paper's raster -> UDF domain-conversion layer (Appendix B: a 2 x 2
convolution with padding 1 turns the 2H x 2W sub-pixel grid into its
(2H+1) x (2W+1) corner lattice).

Line Reconstruction (NDC, Sec. 6, Fig. 22): centerline UDF lattice
`(B, 1, 2H+1, 2W+1)` -> per 0.5 px cell `(2H, 2W)`: edge-flag logits over 4
classes (none / right edge / bottom edge / both), the vertex position
inside the cell (sigmoid, 2), and a 1 px sketch-skeleton logit. A 2 x 2
convolution without padding is the UDF -> cell domain conversion; then
three multi-resolution branches of three residual 3x3 convolutions with
dilation 1 / 2 / 3 (3x3 / 5x5 / 7x7), summed; a residual trunk; 1x1 heads.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def norm(ch: int) -> nn.GroupNorm:
    groups = 8 if ch % 8 == 0 else 1
    return nn.GroupNorm(groups, ch)


class ResNeXtBlock(nn.Module):
    """1x1 -> grouped dilated 3x3 (cardinality) -> 1x1, plus identity."""

    def __init__(self, ch: int, cardinality: int, dilation: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False), norm(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, groups=cardinality, bias=False),
            norm(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1, bias=False), norm(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class DistanceFieldNet(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        ch, hr = cfg["dfp_ch"], cfg["dfp_hr_ch"]
        self.stem = nn.Sequential(nn.Conv2d(1, hr, 3, padding=1, bias=False), norm(hr), nn.ReLU(inplace=True))
        self.down = nn.Sequential(nn.Conv2d(hr, ch, 3, stride=2, padding=1, bias=False), norm(ch),
                                  nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResNeXtBlock(ch, cfg["dfp_cardinality"], d) for d in cfg["dfp_dilations"]])
        self.up = nn.ConvTranspose2d(ch, hr, 4, stride=4)  # half res -> 2x super-resolution
        self.skip_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.head = nn.Sequential(
            nn.Conv2d(2 * hr, hr, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hr, hr, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.to_lattice = nn.Conv2d(hr, cfg["n_udf"], 2, padding=1)  # Appendix B domain conversion

    def forward(self, ink: torch.Tensor) -> torch.Tensor:
        """`ink (B, 1, H, W)` in [0, 1] -> `(B, n_udf, 2H+1, 2W+1)` in [0, 1]."""
        _, _, h, w = ink.shape
        s = self.stem(ink)
        x = self.up(self.blocks(self.down(s)))[..., :2 * h, :2 * w]
        x = self.head(torch.cat([x, self.skip_up(s)], dim=1))
        return torch.sigmoid(self.to_lattice(x))


class ResConv(nn.Module):
    """Residual 3x3 block (two convolutions) at a given dilation."""

    def __init__(self, ch: int, dilation: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False), norm(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False), norm(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class LineReconstructionNet(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        c = cfg["ndc_ch"]
        self.to_cells = nn.Sequential(nn.Conv2d(1, c, 2), nn.ReLU(inplace=True))  # lattice -> cells
        self.branches = nn.ModuleList(
            nn.Sequential(*[ResConv(c, d) for _ in range(cfg["ndc_branch_layers"])]) for d in cfg["ndc_dilations"]
        )
        self.trunk = nn.Sequential(*[ResConv(c) for _ in range(cfg["ndc_trunk_layers"])])
        self.edge = nn.Conv2d(c, 4, 1)
        self.vertex = nn.Conv2d(c, 2, 1)
        self.skeleton = nn.Conv2d(c, 1, 1)

    def forward(self, udf: torch.Tensor) -> dict[str, torch.Tensor]:
        """`udf (B, 1, 2H+1, 2W+1)` (normalised) -> cell maps `(B, ., 2H, 2W)`."""
        x = self.to_cells(udf)
        x = sum(b(x) for b in self.branches)
        x = self.trunk(x)
        return {"edge": self.edge(x), "vertex": torch.sigmoid(self.vertex(x)), "skeleton": self.skeleton(x)}


class ImplicitSketchNet(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.dfp = DistanceFieldNet(cfg)
        self.ndc = LineReconstructionNet(cfg)

    def forward(self, ink: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        udf = self.dfp(ink)
        return udf, self.ndc(udf[:, :1])


def prepare_input(gray: torch.Tensor) -> torch.Tensor:
    """uint8/float gray `(B, H, W)` (255 = paper) -> ink `(B, 1, H, W)` in [0, 1]."""
    return (1.0 - gray.float() / 255.0)[:, None]


def build_model(cfg: dict) -> ImplicitSketchNet:
    return ImplicitSketchNet(cfg)


def save_weights(path: "str | Path", model: ImplicitSketchNet, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": model.cfg, "model": model.state_dict(), "meta": meta or {}}, tmp)
    tmp.replace(path)


def load_weights(path: "str | Path", map_location="cpu") -> ImplicitSketchNet:
    blob = torch.load(str(path), map_location=map_location, weights_only=False)
    model = build_model(blob["config"])
    model.load_state_dict(blob["model"])
    return model


def load_submodule(model: ImplicitSketchNet, path: "str | Path", name: str) -> dict:
    """Copy one sub-network (`"dfp"` / `"ndc"`) out of another checkpoint
    (the separate-stage files the joint stage starts from); returns its meta."""
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    prefix = name + "."
    state = {k[len(prefix):]: v for k, v in blob["model"].items() if k.startswith(prefix)}
    getattr(model, name).load_state_dict(state)
    return blob.get("meta", {})
