"""Training data for `train.py` (numpy/scipy/cv2 only; never imported by the
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

Every target is generated on the fly from the vector strokes, after the
augmentation -- as the paper does ("we apply the random augmentation to the
vector sketch and generate the UDF on the fly", Sec. 5): a crop is cut with
a random rotation (and flip) applied to both the mask (`cv2.warpAffine`)
and the GT Bezier control points, then, for the crop's (2S+1)^2 UDF lattice
and (2S)^2 grid of 0.5 px cells (paper: "sampling rate 0.5-pixel"):

  UDFs (Sec. 5)      centerline: distance from each lattice point to the
                     densely sampled strokes (a KD-tree query -- the paper's
                     "closest point on the cropped vector paths"); USM:
                     distance to the under-sampled cells; end / sharp /
                     junc: distance to those keypoints; all: their minimum.
                     Clamped at UDF_TRUNC_PX.
  edge flags E       a grid edge crossed by a stroke (dense samples changing
                     cell), per cell: its right / bottom edge (Fig. 4 b);
  vertex map V       per cell the stroke passes through, the stroke point
                     closest to the cell centre (Fig. 4 a), as an offset in
                     the cell;
  USM                both cells beside an edge crossed twice or more, and a
                     cell whose four edges are all crossed (Fig. 4, Fig. 7 II);
  skeleton S         the occupied cells thinned to 1 cell (Zhang-Suen).

Keypoints (after Puhachov et al. 2021, ours in detail): an *end* is a stroke
end no other stroke touches; two stroke ends meeting at an angle sharper
than SHARP_DEG, or a joint inside a stroke turning that sharply, is *sharp*;
three or more ends meeting, a stroke end on another stroke's interior (T),
or two strokes crossing (X) is a *junction*.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import geometry as geo
from .config import (
    GT_MARGIN_PX, GT_SAMPLE_STEP_PX, KEYPOINT_MERGE_PX, LOSS_MASK_PX, SHARP_DEG, SR, UDF_TRUNC_PX,
)

SIZES = (128, 160, 192, 224, 256)  # training crop sides (px, ours)
# Cell assignment nudge: a stroke lying exactly on a grid line (axis-aligned
# CAD lines at integer coordinates) must not flicker between two cells on
# float noise, which would read as repeated crossings.
_EPS = 1e-6


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
    strokes = [pieces[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    return LayerData(key, gray, strokes, z["widths"], z["bboxes"], z["ink_pts"])


def load_index(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "index.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------
@dataclass
class Crop:
    gray: np.ndarray            # (S, S) uint8
    strokes: list[np.ndarray]   # crop px, clipped to the crop + GT_MARGIN_PX


def _affine(center, angle: float, flip: bool, size: int) -> np.ndarray:
    """2x3 layer px -> crop px: (mirror x,) rotate about `center`, which lands
    on the crop centre."""
    c, s = math.cos(angle), math.sin(angle)
    a = np.array([[c, -s], [s, c]])
    if flip:
        a = a @ np.array([[-1.0, 0.0], [0.0, 1.0]])
    half = size / 2.0
    t = np.array([half, half]) - a @ np.asarray(center, float)
    return np.concatenate([a, t[:, None]], axis=1)


def cut_crop(layer: LayerData, center, angle: float, flip: bool, size: int) -> Crop:
    m = _affine(center, angle, flip, size)
    h, w = layer.gray.shape
    r = size * 0.75 + GT_MARGIN_PX + 2
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
    strokes: list[np.ndarray] = []
    b = layer.bboxes
    box = (-GT_MARGIN_PX, -GT_MARGIN_PX, size + GT_MARGIN_PX, size + GT_MARGIN_PX)
    if len(b):
        hit = np.nonzero((b[:, 2] >= center[0] - r) & (b[:, 0] <= center[0] + r)
                         & (b[:, 3] >= center[1] - r) & (b[:, 1] <= center[1] + r))[0]
        for i in hit:
            st = layer.strokes[i] @ m[:, :2].T + m[:, 2]
            strokes.extend(geo.clip_stroke(st, box))
    return Crop(gray, strokes)


def sample_crop(layer: LayerData, size: int, rng: np.random.Generator, *, augment: bool = True,
                ink_bias: float = 0.9) -> Crop:
    h, w = layer.gray.shape
    if len(layer.ink_pts) and rng.random() < ink_bias:
        cx, cy = layer.ink_pts[rng.integers(len(layer.ink_pts))]
        center = (cx + 0.5 + rng.uniform(-size / 4, size / 4), cy + 0.5 + rng.uniform(-size / 4, size / 4))
    else:
        center = (rng.uniform(0, w), rng.uniform(0, h))
    angle = rng.uniform(0, 2 * math.pi) if augment else 0.0
    flip = bool(rng.random() < 0.5) if augment else False
    return cut_crop(layer, center, angle, flip, size)


# ---------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(*v))
    return v / n if n > 1e-12 else np.zeros(2)


def _end_tangents(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outgoing unit directions at the start and end of a sampled stroke,
    measured ~1.5 px in."""
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))])
    k0 = min(int(np.searchsorted(d, 1.5)), len(samples) - 1)
    k1 = max(int(np.searchsorted(d, d[-1] - 1.5)), 0)
    return _unit(samples[k0] - samples[0]), _unit(samples[k1] - samples[-1])


def _turn_deg(t_in: np.ndarray, t_out: np.ndarray) -> float:
    """Direction change (deg) from travelling along `t_in` to `t_out`."""
    c = float(np.clip(np.dot(_unit(t_in), _unit(t_out)), -1.0, 1.0))
    return math.degrees(math.acos(c))


def _polyline_crossings(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Proper intersection points of two polylines `(N, 2)`, `(M, 2)`."""
    p, r = a[:-1], np.diff(a, axis=0)
    q, s = b[:-1], np.diff(b, axis=0)
    rxs = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    qp = q[None] - p[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / rxs
        u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / rxs
    hit = (np.abs(rxs) > 1e-12) & (t > 0) & (t <= 1) & (u > 0) & (u <= 1)
    i, j = np.nonzero(hit)
    return p[i] + t[i, j][:, None] * r[i]


def keypoints(strokes: list[np.ndarray], box) -> dict[str, np.ndarray]:
    """`{"end", "sharp", "junc"}` -> `(K, 2)` points (crop px). Stroke ends on
    the margin `box` boundary are clipping artefacts, not endpoints."""
    out: dict[str, list] = {"end": [], "sharp": [], "junc": []}
    samples = [geo.dense_samples(s, 0.5) for s in strokes]
    ok = [len(sm) >= 2 for sm in samples]
    strokes = [s for s, o in zip(strokes, ok) if o]
    samples = [sm for sm, o in zip(samples, ok) if o]
    if not strokes:
        return {k: np.zeros((0, 2)) for k in out}
    x0, y0, x1, y1 = box

    def on_box(p) -> bool:
        return min(abs(p[0] - x0), abs(p[0] - x1), abs(p[1] - y0), abs(p[1] - y1)) < 1e-3

    # stroke ends: (point, outgoing tangent, stroke index)
    ends = []
    for k, sm in enumerate(samples):
        t0, t1 = _end_tangents(sm)
        for p, t in ((sm[0], t0), (sm[-1], t1)):
            if not on_box(p):
                ends.append((p, t, k))
    # all samples, for "end on another stroke's interior" (T) tests
    all_pts = np.concatenate(samples)
    owner = np.concatenate([np.full(len(sm), k) for k, sm in enumerate(samples)])
    tree_all = cKDTree(all_pts)

    if ends:
        pts = np.array([e[0] for e in ends])
        parent = list(range(len(ends)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in cKDTree(pts).query_pairs(KEYPOINT_MERGE_PX):
            parent[find(i)] = find(j)
        groups: dict[int, list[int]] = {}
        for i in range(len(ends)):
            groups.setdefault(find(i), []).append(i)
        for members in groups.values():
            p = pts[members].mean(axis=0)
            mine = {ends[i][2] for i in members}
            near = tree_all.query_ball_point(p, KEYPOINT_MERGE_PX)
            # another stroke passes through here (not just ending here): a T
            foreign = [n for n in near if owner[n] not in mine]
            if foreign or len(members) >= 3:
                out["junc"].append(p)
            elif len(members) == 2:
                ta, tb = ends[members[0]][1], ends[members[1]][1]
                if 180.0 - _turn_deg(ta, tb) > SHARP_DEG:  # straight continuation: tangents opposite
                    out["sharp"].append(p)
            else:
                out["end"].append(p)
    # sharp joints inside a stroke
    for s in strokes:
        for k in range(1, len(s)):
            t_in = s[k - 1][3] - s[k - 1][2] if np.hypot(*(s[k - 1][3] - s[k - 1][2])) > 1e-9 else s[k - 1][3] - s[k - 1][0]
            t_out = s[k][1] - s[k][0] if np.hypot(*(s[k][1] - s[k][0])) > 1e-9 else s[k][3] - s[k][0]
            if _turn_deg(t_in, t_out) > SHARP_DEG:
                out["sharp"].append(np.asarray(s[k][0], float))
    # X crossings between different strokes (away from both strokes' ends)
    coarse = [geo.rdp(sm, 0.1) for sm in samples]
    bbs = np.array([[*sm.min(0), *sm.max(0)] for sm in samples])
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if bbs[i, 0] > bbs[j, 2] or bbs[j, 0] > bbs[i, 2] or bbs[i, 1] > bbs[j, 3] or bbs[j, 1] > bbs[i, 3]:
                continue
            for x in _polyline_crossings(coarse[i], coarse[j]):
                ends_ij = np.stack([samples[i][0], samples[i][-1], samples[j][0], samples[j][-1]])
                if np.min(np.linalg.norm(ends_ij - x, axis=1)) > KEYPOINT_MERGE_PX:
                    out["junc"].append(x)
    return {k: (np.array(v, float).reshape(-1, 2)) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def _udf(tree_pts: np.ndarray, query: np.ndarray, where: "np.ndarray | None" = None) -> np.ndarray:
    """Distance (px, clamped at UDF_TRUNC_PX) from every `query` point to the
    nearest of `tree_pts`; with `where` (bool per query point), only those
    points are computed -- the rest are UDF_TRUNC_PX."""
    out = np.full(len(query), UDF_TRUNC_PX, np.float32)
    if len(tree_pts) == 0:
        return out
    sel = np.arange(len(query)) if where is None else np.nonzero(where)[0]
    if len(sel):
        d, _ = cKDTree(tree_pts).query(query[sel], distance_upper_bound=UDF_TRUNC_PX)
        out[sel] = np.minimum(d, UDF_TRUNC_PX)
    return out


def _near(pts: np.ndarray, n_pts: int, step: float, radius: float) -> np.ndarray:
    """Bool `(n_pts, n_pts)` grid of spacing `step` (origin 0): every grid
    point within ~`radius` px of `pts` (rasterised + dilated -- a superset)."""
    m = np.zeros((n_pts, n_pts), np.uint8)
    if len(pts):
        ij = np.round(pts / step).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < n_pts) & (ij[:, 1] >= 0) & (ij[:, 1] < n_pts)
        m[ij[ok, 1], ij[ok, 0]] = 1
        r = int(math.ceil(radius / step)) + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return m.astype(bool)


def _cell_moves(samples: np.ndarray, n: int):
    """Cell transitions of consecutive samples on the 1/SR px grid: returns
    `(right_cells, bottom_cells)` -- one row (a, b) per crossing of that
    cell's right / bottom edge, inside the n x n grid."""
    a = np.floor(samples[:, 0] * SR + _EPS).astype(np.int64)
    b = np.floor(samples[:, 1] * SR + _EPS).astype(np.int64)
    a0, a1, b0, b1 = a[:-1], a[1:], b[:-1], b[1:]
    da, db = a1 - a0, b1 - b0
    right, bottom = [], []
    # straight moves
    h = (da != 0) & (db == 0)
    right.append(np.stack([np.minimum(a0[h], a1[h]), b0[h]], 1))
    v = (da == 0) & (db != 0)
    bottom.append(np.stack([a0[v], np.minimum(b0[v], b1[v])], 1))
    # diagonal moves: split into two straight ones, in the order the grid
    # lines are crossed
    d = (da != 0) & (db != 0)
    if d.any():
        p0, p1 = samples[:-1][d], samples[1:][d]
        gx = np.maximum(a0[d], a1[d]) / SR
        gy = np.maximum(b0[d], b1[d]) / SR
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = (gx - p0[:, 0]) / (p1[:, 0] - p0[:, 0])
            ty = (gy - p0[:, 1]) / (p1[:, 1] - p0[:, 1])
        x_first = tx < ty
        aa0, aa1, bb0, bb1 = a0[d], a1[d], b0[d], b1[d]
        # x first: (a0,b0)->(a1,b0) [right edge], then (a1,b0)->(a1,b1) [bottom edge]
        right.append(np.stack([np.minimum(aa0, aa1), bb0], 1)[x_first])
        bottom.append(np.stack([aa1, np.minimum(bb0, bb1)], 1)[x_first])
        # y first: (a0,b0)->(a0,b1), then (a0,b1)->(a1,b1)
        bottom.append(np.stack([aa0, np.minimum(bb0, bb1)], 1)[~x_first])
        right.append(np.stack([np.minimum(aa0, aa1), bb1], 1)[~x_first])
    right = np.concatenate(right) if right else np.zeros((0, 2), np.int64)
    bottom = np.concatenate(bottom) if bottom else np.zeros((0, 2), np.int64)
    # an edge counts only between two cells of the grid
    right = right[(right[:, 0] >= 0) & (right[:, 0] < n - 1) & (right[:, 1] >= 0) & (right[:, 1] < n)]
    bottom = bottom[(bottom[:, 0] >= 0) & (bottom[:, 0] < n) & (bottom[:, 1] >= 0) & (bottom[:, 1] < n - 1)]
    return right, bottom


def make_targets(strokes: list[np.ndarray], size: int) -> dict[str, np.ndarray]:
    """Every training target for a `size` px crop whose GT strokes (crop px)
    are `strokes`."""
    from skimage.morphology import skeletonize

    n = SR * size
    lat = np.arange(n + 1) / SR
    lx, ly = np.meshgrid(lat, lat)                                # (L, L) [row = y]
    lattice = np.stack([lx.ravel(), ly.ravel()], 1)

    samples = [geo.dense_samples(s, GT_SAMPLE_STEP_PX) for s in strokes]
    samples = [sm for sm in samples if len(sm)]
    all_pts = np.concatenate(samples) if samples else np.zeros((0, 2))
    # (strokes reaching into the margin still shape the field near the border)
    shifted = all_pts + GT_MARGIN_PX
    band = _near(shifted, n + 1 + 2 * SR * int(GT_MARGIN_PX), 1.0 / SR, UDF_TRUNC_PX)
    m0 = SR * int(GT_MARGIN_PX)
    band = band[m0:m0 + n + 1, m0:m0 + n + 1]
    u_center = _udf(all_pts, lattice, band.ravel()).reshape(n + 1, n + 1)
    # at the cell centres, only for the edge-loss mask: the mean of the four corners
    cell_udf = 0.25 * (u_center[:-1, :-1] + u_center[1:, :-1] + u_center[:-1, 1:] + u_center[1:, 1:])
    # the other fields are only ever read inside the loss mask
    near = (u_center < LOSS_MASK_PX).ravel()

    # edge flags + crossing counts
    cnt_r = np.zeros((n, n), np.int32)
    cnt_b = np.zeros((n, n), np.int32)
    for sm in samples:
        right, bottom = _cell_moves(sm, n)
        np.add.at(cnt_r, (right[:, 1], right[:, 0]), 1)
        np.add.at(cnt_b, (bottom[:, 1], bottom[:, 0]), 1)
    er, eb = cnt_r > 0, cnt_b > 0
    edge = er.astype(np.int64) + 2 * eb.astype(np.int64)

    # USM: edges crossed twice or more -> both cells; cells with all four edges crossed
    usm = np.zeros((n, n), bool)
    ys, xs = np.nonzero(cnt_r >= 2)
    usm[ys, xs] = True
    usm[ys, xs + 1] = True
    ys, xs = np.nonzero(cnt_b >= 2)
    usm[ys, xs] = True
    usm[ys + 1, xs] = True
    left = np.zeros_like(er)
    left[:, 1:] = er[:, :-1]
    top = np.zeros_like(eb)
    top[1:, :] = eb[:-1, :]
    usm |= er & eb & left & top

    # vertex map: per occupied cell, the stroke point nearest the cell centre
    vert = np.zeros((2, n, n), np.float32)
    vmask = np.zeros((n, n), bool)
    if len(all_pts):
        a = np.floor(all_pts[:, 0] * SR + _EPS).astype(np.int64)
        b = np.floor(all_pts[:, 1] * SR + _EPS).astype(np.int64)
        inside = (a >= 0) & (a < n) & (b >= 0) & (b < n)
        p, a, b = all_pts[inside], a[inside], b[inside]
        if len(p):
            cell = b * n + a
            dist = np.hypot(p[:, 0] * SR - (a + 0.5), p[:, 1] * SR - (b + 0.5))
            order = np.lexsort((dist, cell))
            first = order[np.concatenate([[True], np.diff(cell[order]) != 0])]
            vmask[b[first], a[first]] = True
            vert[0, b[first], a[first]] = p[first, 0] * SR - a[first]
            vert[1, b[first], a[first]] = p[first, 1] * SR - b[first]
    skel = skeletonize(vmask).astype(np.float32) if vmask.any() else np.zeros((n, n), np.float32)

    # keypoint / USM fields
    box = (-GT_MARGIN_PX, -GT_MARGIN_PX, size + GT_MARGIN_PX, size + GT_MARGIN_PX)
    kp = keypoints(strokes, box)
    ys, xs = np.nonzero(usm)
    usm_pts = np.stack([(xs + 0.5) / SR, (ys + 0.5) / SR], 1) if len(xs) else np.zeros((0, 2))
    shape = (n + 1, n + 1)
    u_usm = _udf(usm_pts, lattice, near).reshape(shape)
    u_end = _udf(kp["end"], lattice, near).reshape(shape)
    u_sharp = _udf(kp["sharp"], lattice, near).reshape(shape)
    u_junc = _udf(kp["junc"], lattice, near).reshape(shape)
    u_all = np.minimum(np.minimum(u_end, u_sharp), u_junc)
    udf = np.stack([u_center, u_usm, u_end, u_sharp, u_junc, u_all]).astype(np.float32)
    return {
        "udf": udf,
        "mask": (u_center < LOSS_MASK_PX).astype(np.float32)[None],
        "edge": edge,
        "emask": (cell_udf < LOSS_MASK_PX).astype(np.float32),
        "vert": vert,
        "vmask": vmask,
        "skel": skel,
        "usm": usm,
        "keypoints": kp,
    }


def build_batch(crops: list[Crop]) -> dict:
    size = crops[0].gray.shape[0]
    tg = [make_targets(c.strokes, size) for c in crops]
    return {
        "gray": np.stack([c.gray for c in crops]),
        **{k: np.stack([t[k] for t in tg]) for k in ("udf", "mask", "edge", "emask", "vert", "vmask", "skel")},
        "size": size,
    }
