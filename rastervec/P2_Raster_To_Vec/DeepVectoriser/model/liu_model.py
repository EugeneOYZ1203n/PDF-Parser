"""The full Liu et al. (AAAI-22) framework: Stroke Encoder + shared UNet +
Stroke Decoder head + Stroke Vectorizer, plus checkpoint save/load (a
checkpoint stores its own model config, so a `.pth` always rebuilds the
architecture it was trained with)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .decoder import StrokeDecoderHead
from .encoder import StrokeEncoder
from .unet import UNet
from .vectorizer import StrokeVectorizer

# The vectorizer reads the shared UNet at resolution / 2**VEC_LEVEL.
VEC_LEVEL = 3


class LiuVectorizer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.cond_dim = 5 + cfg["d_emb"]  # F_i = (P0, P1, emb, p)
        self.encoder = StrokeEncoder(cfg)
        self.unet = UNet(3, self.cond_dim, cfg["unet_base"], cfg["unet_depth"])
        self.vec_level = min(VEC_LEVEL, cfg["unet_depth"])
        self.decoder_head = StrokeDecoderHead(self.unet.chs[0])
        self.vectorizer = StrokeVectorizer(cfg, self.unet.chs[self.vec_level], self.cond_dim)

    @staticmethod
    def make_cond(endpoints: torch.Tensor, emb: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """F_i as one flat vector per stroke: `(M, 4)`, `(M, d_emb)`, `(M,)`
        (p as a probability) -> `(M, 5 + d_emb)`."""
        return torch.cat([endpoints, emb, p[:, None]], dim=1)

    def module_groups(self) -> dict[str, list[nn.Parameter]]:
        """Parameters per optimizer (the paper's three Adam optimizers): the
        shared UNet trains with the Stroke Decoder."""
        return {
            "encoder": list(self.encoder.parameters()),
            "decoder": list(self.unet.parameters()) + list(self.decoder_head.parameters()),
            "vectorizer": list(self.vectorizer.parameters()),
        }


def build_model(cfg: dict) -> LiuVectorizer:
    return LiuVectorizer(cfg)


def save_weights(path: "str | Path", model: LiuVectorizer, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": model.cfg, "model": model.state_dict(), "meta": meta or {}}, tmp)
    tmp.replace(path)


def load_weights(path: "str | Path", map_location="cpu") -> LiuVectorizer:
    blob = torch.load(str(path), map_location=map_location, weights_only=False)
    model = build_model(blob["config"])
    model.load_state_dict(blob["model"])
    return model
