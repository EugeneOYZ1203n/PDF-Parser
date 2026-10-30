"""Training plumbing shared by the trainable raster backends (`deep_vectoriser`,
`implicit_sketch_vec`): device / mixed-precision setup, the data loader, metric accumulation and
one prepared layer of the shared dataset (`LayerData`). Needs torch (the `ml` extra)."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from raster_parser.models.raster_text import LayerData


def setup_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # crop sizes repeat: autotuned kernels pay off
        torch.set_float32_matmul_precision("high")  # TF32 for the fp32 parts


def autocast_dtype(device: torch.device, enabled: bool):
    """bf16 where the GPU has it (no loss scaling needed), else fp16; None
    = full precision (always on CPU)."""
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def make_loader(stream, workers: int, device: torch.device) -> torch.utils.data.DataLoader:
    # persistent_workers stays off: each epoch's workers must see the new `stream.epoch`
    extra = {"prefetch_factor": 4} if workers > 0 else {}
    return torch.utils.data.DataLoader(
        stream,
        batch_size=None,
        num_workers=workers,
        persistent_workers=False,
        pin_memory=device.type == "cuda",
        **extra,
    )


def accumulate(sums: dict, terms: dict) -> None:
    """Sum loss terms without a host sync (tensors stay on the device)."""
    for k, v in terms.items():
        v = v.detach().float() if torch.is_tensor(v) else float(v)
        sums[k] = sums[k] + v if k in sums else v


def default_cache_dir(layers_dir: Path) -> Path:
    return Path(layers_dir).parent / "cache"


def gray_path(layers_dir: Path, key: str, cache_dir: "Path | None" = None) -> Path:
    """The memory-mappable `.npy` of a layer's mask: an old-format
    `layers/<key>.gray.npy` as is, else the PNG decoded once into
    `cache_dir` (default `<data>/cache/`, shared with every other trainer --
    same file names, same content). Written to a temp file and renamed; if
    the rename fails because another trainer has the target memory-mapped
    (Windows), the existing file is used."""
    layers_dir = Path(layers_dir)
    legacy = layers_dir / f"{key}.gray.npy"
    if legacy.is_file():
        return legacy
    png = layers_dir / f"{key}.gray.png"
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir(layers_dir)
    target = cache / f"{key}.gray.npy"
    if target.is_file() and target.stat().st_mtime >= png.stat().st_mtime:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    gray = cv2.imdecode(np.fromfile(str(png), np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"could not decode {png}")
    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, gray)
    try:
        os.replace(tmp, target)
    except PermissionError:
        if not target.is_file():
            raise
        tmp.unlink(missing_ok=True)  # another trainer holds it open: theirs is identical
    return target


def load_layer(layers_dir: Path, key: str, cache_dir: "Path | None" = None) -> LayerData:
    gray = np.load(gray_path(layers_dir, key, cache_dir), mmap_mode="r")
    z = np.load(Path(layers_dir) / f"{key}.strokes.npz")
    pieces, offsets = z["pieces"].astype(np.float64), z["offsets"]
    strokes = [pieces[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    return LayerData(key, gray, strokes, z["widths"], z["bboxes"], z["ink_pts"])
