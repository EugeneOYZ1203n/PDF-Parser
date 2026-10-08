"""Training data for `train.py` (numpy/cv2 only; never imported by the
pipeline itself).

**Dataset: the one `DeepVectoriser/prep_dataset.py` writes** -- this backend
has no prep script of its own. The on-disk format is owned by
`P2_Raster_To_Vec/DeepVectoriser/train_data.py` + `prep_dataset.py`; the
readers below are *copies* (sibling backends share no code), so a format
change there must be mirrored here:

    <data>/layers/<key>.gray.png      uint8 layer mask at 300 dpi (ink 0)
    <data>/layers/<key>.strokes.npz   pieces (P,4,2) cubic Beziers, offsets (n+1,),
                                      widths (n,) px, bboxes (n,4), ink_pts (Q,2)
    <data>/index.json                 train / val layer keys + n_strokes
    <cache>/<key>.gray.npy            the PNG decoded once (default <data>/cache/,
                                      shared with every other trainer)

Targets are derived from those vector strokes at crop time (paper Sec. 3.2
and Appendix B): a random 64 x 64 window with a random rotation and scale
(paper: "random 64 x 64 crops, with random rotation and scaling") is cut
from the mask, the same affine is applied to the GT Bezier control points
(exact), the strokes are clipped to the window and turned into this
model's primitives -- lines (cubics flattened, collinear chords merged) or
quadratic Beziers (least-squares fit, split until within tolerance) -- and
sorted the paper's way (endpoints within a primitive, then primitives
lexicographically), padded with zero placeholder rows.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import geometry as geo
from .config import (
    AUG_SCALE_RANGE, INK_GRAY_THRESHOLD, LINE_FLAT_TOL_PX, LINE_MERGE_ANGLE_DEG, MIN_PRIM_LEN_PX, N_PARAMS,
    OVERFLOW_RETRIES, PATCH_PX, QUAD_FIT_TOL_PX,
)


# ---------------------------------------------------------------------------
# Dataset readers (copied from DeepVectoriser/train_data.py)
# ---------------------------------------------------------------------------
@dataclass
class LayerData:
    key: str
    gray: np.ndarray
    strokes: list[np.ndarray]
    widths: np.ndarray
    bboxes: np.ndarray
    ink_pts: np.ndarray


def default_cache_dir(layers_dir: Path) -> Path:
    return Path(layers_dir).parent / "cache"


def gray_path(layers_dir: Path, key: str, cache_dir: "Path | None" = None) -> Path:
    """The memory-mappable `.npy` of a layer's mask: an old-format
    `layers/<key>.gray.npy` as is, else the PNG decoded once into
    `cache_dir` (default `<data>/cache/`, shared with DeepVectoriser's and
    every other trainer's cache -- same file names, same content).
    Written to a temp file and renamed; if the rename fails because another
    trainer has the target memory-mapped (Windows), the existing file is
    used."""
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
    strokes = [pieces[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    return LayerData(key, gray, strokes, z["widths"], z["bboxes"], z["ink_pts"])


def load_index(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "index.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------
@dataclass
class Patch:
    gray: np.ndarray                                    # (S, S) uint8, 255 = paper
    prims: list[geo.Prim] = field(default_factory=list)  # sorted, patch px
    overflow: bool = False                              # more GT primitives than n_prim


def stroke_prims(stroke: np.ndarray, width: float, kind: str) -> list[geo.Prim]:
    """One clipped GT stroke -> this model's primitives."""
    if kind == "line":
        pts = geo.stroke_to_lines(stroke, LINE_FLAT_TOL_PX, LINE_MERGE_ANGLE_DEG)
    else:
        pts = geo.stroke_to_quads(stroke, QUAD_FIT_TOL_PX)
    prims = [geo.Prim(p, float(width)) for p in pts]
    return [p for p in prims if p.length() >= MIN_PRIM_LEN_PX]


def _affine(center, angle: float, scale: float, size: int) -> np.ndarray:
    """2x3 layer px -> patch px: rotate by `angle` and scale about `center`,
    which lands on the patch centre."""
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    cx, cy = float(center[0]), float(center[1])
    half = size / 2.0
    return np.array([[c, -s, half - (c * cx - s * cy)],
                     [s, c, half - (s * cx + c * cy)]], float)


def cut_patch(layer: LayerData, center, angle: float, scale: float, kind: str, size: int = PATCH_PX) -> Patch:
    """The window around `center`, rotated/scaled into a `size` px patch, with
    its GT primitives (un-truncated, sorted)."""
    m = _affine(center, angle, scale, size)
    h, w = layer.gray.shape
    # source window that covers the patch after the inverse transform
    r = size / scale * 0.75 + 2
    x0, y0 = max(0, int(center[0] - r)), max(0, int(center[1] - r))
    x1, y1 = min(w, int(math.ceil(center[0] + r))), min(h, int(math.ceil(center[1] + r)))
    if x1 > x0 and y1 > y0:
        src = np.ascontiguousarray(layer.gray[y0:y1, x0:x1])
        m_src = m.copy()
        m_src[:, 2] += m[:, :2] @ np.array([x0, y0], float)
        gray = cv2.warpAffine(src, m_src, (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    else:
        gray = np.full((size, size), 255, np.uint8)

    prims: list[geo.Prim] = []
    b = layer.bboxes
    if len(b):
        hit = np.nonzero((b[:, 2] >= center[0] - r) & (b[:, 0] <= center[0] + r)
                         & (b[:, 3] >= center[1] - r) & (b[:, 1] <= center[1] + r))[0]
        for i in hit:
            st = layer.strokes[i] @ m[:, :2].T + m[:, 2]
            for part in geo.clip_stroke(st, (0.0, 0.0, float(size), float(size)), min_len=MIN_PRIM_LEN_PX):
                prims.extend(stroke_prims(part, float(layer.widths[i]) * scale, kind))
    return Patch(gray, geo.sort_prims(prims))


def sample_patch(layer: LayerData, rng: np.random.Generator, n_prim: int, kind: str, *, augment: bool = True,
                 size: int = PATCH_PX, ink_bias: float = 0.9) -> Patch:
    """A random patch (centred near a random ink pixel with probability
    `ink_bias`); with `augment`, a random rotation and scale. More GT
    primitives than `n_prim` -> re-drawn up to `OVERFLOW_RETRIES` times, then
    the `n_prim` longest are kept (`overflow` set)."""
    h, w = layer.gray.shape
    patch = None
    for _ in range(OVERFLOW_RETRIES + 1):
        if len(layer.ink_pts) and rng.random() < ink_bias:
            cx, cy = layer.ink_pts[rng.integers(len(layer.ink_pts))]
            center = (cx + 0.5 + rng.uniform(-size / 3, size / 3), cy + 0.5 + rng.uniform(-size / 3, size / 3))
        else:
            center = (rng.uniform(0, w), rng.uniform(0, h))
        angle = rng.uniform(0, 2 * math.pi) if augment else 0.0
        scale = rng.uniform(*AUG_SCALE_RANGE) if augment else 1.0
        patch = cut_patch(layer, center, angle, scale, kind, size)
        if len(patch.prims) <= n_prim:
            return patch
    longest = sorted(patch.prims, key=lambda p: -p.length())[:n_prim]
    return Patch(patch.gray, geo.sort_prims(longest), overflow=True)


# ---------------------------------------------------------------------------
# Targets <-> primitives
# ---------------------------------------------------------------------------
def encode_targets(prims: list[geo.Prim], n_prim: int, kind: str, size: int = PATCH_PX) -> np.ndarray:
    """`(n_prim, n_params + 1)`: sorted primitives, coordinates and width
    normalised by `size`, confidence 1; zero placeholder rows after."""
    d = N_PARAMS[kind] + 1
    out = np.zeros((n_prim, d), np.float32)
    for k, p in enumerate(geo.sort_prims(prims)[:n_prim]):
        out[k, :-2] = np.clip(np.asarray(p.pts, float).reshape(-1) / size, 0.0, 1.0)
        out[k, -2] = min(1.0, p.width / size)
        out[k, -1] = 1.0
    return out


def decode_output(row: np.ndarray, kind: str, size: int = PATCH_PX, conf_thresh: float = 0.5) -> list[geo.Prim]:
    """One patch's network output `(n_prim, d_emb)` -> primitives in patch px
    (confidence below `conf_thresh` discarded)."""
    n_pts = 2 if kind == "line" else 3
    out = []
    for r in np.asarray(row, float):
        if r[-1] < conf_thresh:
            continue
        pts = r[:2 * n_pts].reshape(n_pts, 2) * size
        out.append(geo.Prim(pts, float(r[-2] * size), float(r[-1])))
    return out


def build_batch(patches: list[Patch], n_prim: int, kind: str) -> dict:
    size = patches[0].gray.shape[0]
    return {
        "gray": np.stack([p.gray for p in patches]),
        "target": np.stack([encode_targets(p.prims, n_prim, kind, size) for p in patches]),
        "overflow": float(np.mean([p.overflow for p in patches])),
    }


# ---------------------------------------------------------------------------
# Rasterising primitives (validation IoU / Chamfer)
# ---------------------------------------------------------------------------
def prim_mask(prims: list[geo.Prim], size: int, width: "float | None" = None) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    for p in prims:
        pts = np.round((geo.prim_points(p, 16) - 0.5) * 16).astype(np.int32).reshape(-1, 1, 2)
        thick = max(1, int(round(width if width is not None else p.width)))
        cv2.polylines(m, [pts], False, 1, thick, cv2.LINE_8, shift=4)
    return m > 0


def ink_mask(gray: np.ndarray) -> np.ndarray:
    return gray < INK_GRAY_THRESHOLD
