"""Loading the authors' pretrained checkpoints straight into the vendored
networks. Both are plain `torch.save` dicts that load with
`weights_only=True`:

  centerline_checkpoint.pt    {"model_class": "ResNextUNetLarge", "state_dict", ...}
  intersection_checkpoint.pt  a Lightning 2.2 checkpoint, state_dict keys
                              prefixed "model." (the LightningModule wrapper)

The intersection model's width (`d_dim`, 32 in the released checkpoint) is
read off the state_dict rather than assumed."""
from __future__ import annotations

from pathlib import Path

import torch

from .centerline import ResNextUNetLarge
from .intersection import SceneGraphVectorModel

_LIGHTNING_PREFIX = "model."


def _load_dict(path: "str | Path") -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"{path}: not an InvolutionStroke checkpoint (no 'state_dict')")
    return ckpt


def load_centerline(path: "str | Path") -> ResNextUNetLarge:
    ckpt = _load_dict(path)
    model_class = ckpt.get("model_class", "ResNextUNetLarge")
    if model_class != "ResNextUNetLarge":
        raise ValueError(f"{path}: centerline checkpoint is a {model_class!r}, only ResNextUNetLarge is vendored")
    model = ResNextUNetLarge()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


def load_intersection(path: "str | Path") -> SceneGraphVectorModel:
    sd = _load_dict(path)["state_dict"]
    sd = {k[len(_LIGHTNING_PREFIX):] if k.startswith(_LIGHTNING_PREFIX) else k: v for k, v in sd.items()}
    d_dim = int(sd["backbone.mlp.5.weight"].shape[0])
    model = SceneGraphVectorModel(d_dim=d_dim)
    model.load_state_dict(sd, strict=True)
    return model
