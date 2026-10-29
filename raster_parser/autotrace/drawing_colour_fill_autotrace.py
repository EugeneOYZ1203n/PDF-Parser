"""Colour-cluster fills + per-colour mask autotrace.

1. Inventory non-background RGB; reserve black + white clusters for near-B/W ink.
2. Greedy-group the rest in HSV space; exclusive nearest-centroid masks.
3. Outline-autotrace each non-white mask; drop solid page washes (high cover +
   high solidity/Q); keep thin page-frame paths and small white hole covers.
   Stamp remaining ink.
4. Write artefacts under ``{stem}_colours/``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from raster_parser.rendering.pdf_render.dto_reconstruct_pipeline import draw_vectors
from raster_parser.models import Vector
from raster_parser.serialisation.deserialization import deserialize_vector
from raster_parser.models import VectorDTO
from raster_parser.autotrace.drawing_autotrace import autotrace_drawing
from raster_parser.helpers.geometry import (
    PDF_POINTS_PER_INCH,
    _area,
    _split_loops,
    isoperimetric_quotient,
)

# A pixel with every RGB channel at or above this (0-255) is paper background, not ink.
_DEFAULT_BG_THRESH = 250
# Two colours join one group when their distance in the `rgb_to_hsv` embedding is at most this.
_DEFAULT_GROUP_DIST = 150.0
# Colour groups covering fewer pixels than this are anti-aliasing / noise and are dropped.
_DEFAULT_MIN_GROUP_PIXELS = 5000
# A colour whose max - min RGB channel spread (0-255) is at most this is grey (achromatic);
# `rgb_to_hsv` also zeroes OpenCV saturations below it, so greys don't carry a hue.
_ACHROMA_S_MAX = 40
# Saturation is weighted up against value in the HSV embedding, so two inks of different hue
# separate more than two shades of one ink.
_SATURATION_WEIGHT = 1.5
# OpenCV stores 8-bit hue as 0-180 (half degrees); this turns it into radians.
_OPENCV_HUE_TO_RAD = 2.0 * np.pi / 180.0
_BLACK_RGB = (0, 0, 0)
_WHITE_RGB = (255, 255, 255)
# Grey pixels with mean RGB at or below this are black ink.
_BLACK_MEAN_MAX = 80
_WHITE_MEAN_MIN = 220  # below bg_thresh so page BG stays out of inventory


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB (..., 3) 0..255 → HSV distance embedding ``(S cos H, S sin H, V)``."""
    arr = np.asarray(rgb, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, 1, 3)
    elif arr.ndim == 2 and arr.shape[-1] == 3:
        arr = arr.reshape(1, -1, 3)
    u8 = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    ang = hsv[..., 0] * _OPENCV_HUE_TO_RAD
    s_raw = hsv[..., 1]
    v = hsv[..., 2]
    s = np.where(s_raw < _ACHROMA_S_MAX, 0.0, s_raw * _SATURATION_WEIGHT)
    out = np.stack([s * np.cos(ang), s * np.sin(ang), v], axis=-1)
    if single:
        return out.reshape(3)
    if np.asarray(rgb).ndim == 2:
        return out.reshape(-1, 3)
    return out


def _is_achromatic(rgb: tuple[int, int, int]) -> bool:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return (max(r, g, b) - min(r, g, b)) <= _ACHROMA_S_MAX


def _is_near_black(rgb: tuple[int, int, int]) -> bool:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return _is_achromatic((r, g, b)) and (r + g + b) / 3.0 <= _BLACK_MEAN_MAX


def _is_near_white(rgb: tuple[int, int, int]) -> bool:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return _is_achromatic((r, g, b)) and (r + g + b) / 3.0 >= _WHITE_MEAN_MIN


@dataclass
class ColourGroup:
    """One colour cluster (centroid RGB + pixel count; optional black/white kind)."""

    label: int
    rgb: tuple[int, int, int]
    n_pixels: int
    kind: str | None = None  # "black" | "white" | None

    @property
    def tag(self) -> str:
        """Stable folder/file tag: ``c{label:02d}_{rrggbb}``."""
        r, g, b = self.rgb
        return f"c{self.label:02d}_{r:02x}{g:02x}{b:02x}"


def _stamp_rgb(group: ColourGroup) -> tuple[int, int, int]:
    if group.kind == "black":
        return _BLACK_RGB
    if group.kind == "white":
        return _WHITE_RGB
    return group.rgb


@dataclass
class ColourFillTraceResult:
    """Artefacts from ``colour_fill_autotrace`` (groups, vectors, QA paths)."""

    groups: list[ColourGroup]
    group_vectors: dict[int, list[Vector]]
    combined_vectors: list[Vector]
    mask_paths: dict[int, Path]
    isolate_paths: dict[int, Path]
    autotrace_svg_paths: dict[int, Path]
    autotrace_pdf_paths: dict[int, Path]
    palette_path: Path | None
    summary_path: Path | None
    colours_dir: Path | None = None
    combined_pdf_path: Path | None = None
    vectors_txt_path: Path | None = None
    vectors_json_path: Path | None = None
    centroid_map_path: Path | None = None
    page_width: float | None = None
    page_height: float | None = None


def dump_colour_vectors_json(
    vectors: list[Vector],
    out_path: str | Path,
    *,
    page_width: float,
    page_height: float,
    source: str | Path | None = None,
) -> Path:
    """Serialize ``vectors`` as VectorDTO JSON for stroke/fill post-processing."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(source) if source is not None else None,
        "page_width": float(page_width),
        "page_height": float(page_height),
        "vectors": [VectorDTO.from_vector(v).model_dump(mode="json") for v in vectors],
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return out_path.resolve()


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def unique_colours(
    bgr: np.ndarray,
    *,
    bg_thresh: int = _DEFAULT_BG_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """Unique non-background RGB (N×3) and counts, sorted by descending count."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    bg = (rgb[:, 0] >= bg_thresh) & (rgb[:, 1] >= bg_thresh) & (rgb[:, 2] >= bg_thresh)
    ink = rgb[~bg]
    if ink.size == 0:
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int64)
    packed = (
        ink[:, 0].astype(np.uint32) << 16
        | ink[:, 1].astype(np.uint32) << 8
        | ink[:, 2].astype(np.uint32)
    )
    uniq, counts = np.unique(packed, return_counts=True)
    order = np.argsort(-counts)
    uniq, counts = uniq[order], counts[order]
    rgb_u = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    return rgb_u, counts.astype(np.int64)


def group_colours(
    unique_rgb: np.ndarray,
    counts: np.ndarray,
    *,
    dist_thresh: float = _DEFAULT_GROUP_DIST,
    min_pixels: int = _DEFAULT_MIN_GROUP_PIXELS,
) -> list[ColourGroup]:
    """Reserve black/white; greedy-cluster everything else in HSV space."""
    if len(unique_rgb) == 0:
        return []

    black_i, white_i, rest = [], [], []
    for i, rgb in enumerate(unique_rgb):
        t = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        if _is_near_black(t):
            black_i.append(i)
        elif _is_near_white(t):
            white_i.append(i)
        else:
            rest.append(i)

    hsv_u = rgb_to_hsv(unique_rgb)
    free: list[tuple[np.ndarray, list[int], int]] = []  # cent_rgb, members, npix
    for i in rest:
        hsv_i = hsv_u[i]
        n = int(counts[i])
        best_j, best_d = -1, float("inf")
        for j, (cent, _m, _n) in enumerate(free):
            d = float(np.linalg.norm(hsv_i - rgb_to_hsv(cent)))
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= dist_thresh:
            _c, members, npix = free[best_j]
            members.append(i)
            mem = unique_rgb[members].astype(np.float64)
            w = counts[members].astype(np.float64)
            free[best_j] = ((mem * w[:, None]).sum(0) / w.sum(), members, npix + n)
        else:
            free.append((unique_rgb[i].astype(np.float64).copy(), [i], n))

    out: list[ColourGroup] = []
    for members, rgb, kind in (
        (black_i, _BLACK_RGB, "black"),
        (white_i, _WHITE_RGB, "white"),
    ):
        if members:
            out.append(ColourGroup(len(out), rgb, int(counts[members].sum()), kind))
    for cent, _members, npix in free:
        if npix < min_pixels:
            continue
        rgb = tuple(int(round(x)) for x in cent.tolist())
        out.append(ColourGroup(len(out), rgb, npix, None))
    return out


def _achroma_bw_masks(slab_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chroma = slab_rgb.max(2).astype(np.int16) - slab_rgb.min(2).astype(np.int16)
    lum = slab_rgb.astype(np.float32).mean(2)
    achro = chroma <= _ACHROMA_S_MAX
    return achro & (lum <= _BLACK_MEAN_MAX), achro & (lum >= _WHITE_MEAN_MIN)


def _reserved_indices(groups: list[ColourGroup]) -> tuple[int | None, int | None]:
    black_i = white_i = None
    for i, g in enumerate(groups):
        if g.kind == "black":
            black_i = i
        elif g.kind == "white":
            white_i = i
    return black_i, white_i


def _label_map(
    bgr: np.ndarray,
    groups: list[ColourGroup],
    *,
    dist_thresh: float,
    bg_thresh: int,
    chunk_rows: int = 512,
) -> np.ndarray:
    """HxW int32 labels (−1 = background / unassigned). Forces near-B/W to reserved.

    Nearest-centroid search is done **per group** (running min), not via a
    ``(H, W, N, 3)`` broadcast — that temporary alone can exceed 10GB on A3@300
    DPI when N is large.
    """
    h, w = bgr.shape[:2]
    labels = np.full((h, w), -1, dtype=np.int32)
    if not groups:
        return labels

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cents = rgb_to_hsv(np.array([g.rgb for g in groups], dtype=np.uint8)).astype(
        np.float32, copy=False
    )
    black_i, white_i = _reserved_indices(groups)
    n_groups = len(groups)

    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        slab = rgb[y0:y1]
        bg = (
            (slab[:, :, 0] >= bg_thresh)
            & (slab[:, :, 1] >= bg_thresh)
            & (slab[:, :, 2] >= bg_thresh)
        )
        near_b, near_w = _achroma_bw_masks(slab)
        hsv = rgb_to_hsv(slab).astype(np.float32, copy=False)

        best_d = np.full(hsv.shape[:2], np.inf, dtype=np.float32)
        arg = np.zeros(hsv.shape[:2], dtype=np.int32)
        for gi in range(n_groups):
            # (sh, sw) only — never materialize (sh, sw, N, 3).
            d = np.linalg.norm(hsv - cents[gi], axis=2).astype(np.float32, copy=False)
            closer = d < best_d
            best_d = np.where(closer, d, best_d)
            arg = np.where(closer, gi, arg)

        assign = (best_d <= dist_thresh) & ~bg
        if black_i is not None:
            hit = near_b & ~bg
            arg = np.where(hit, black_i, arg)
            assign |= hit
        if white_i is not None:
            hit = near_w & ~bg & ~near_b
            arg = np.where(hit, white_i, arg)
            assign |= hit
        slab_lab = np.full(assign.shape, -1, dtype=np.int32)
        slab_lab[assign] = arg[assign]
        labels[y0:y1] = slab_lab
        del hsv, best_d, arg, assign, slab_lab, near_b, near_w, bg
    return labels


def colour_masks_exclusive(
    bgr: np.ndarray,
    groups: list[ColourGroup],
    *,
    dist_thresh: float = _DEFAULT_GROUP_DIST,
    bg_thresh: int = _DEFAULT_BG_THRESH,
    chunk_rows: int = 512,
) -> dict[int, np.ndarray]:
    """Exclusive bool masks keyed by ``ColourGroup.label`` (−1 background omitted)."""
    lab = _label_map(
        bgr, groups, dist_thresh=dist_thresh, bg_thresh=bg_thresh, chunk_rows=chunk_rows
    )
    # Build one mask at a time so peak RAM stays ~1 page of bools, not N pages.
    out: dict[int, np.ndarray] = {}
    for i, g in enumerate(groups):
        out[g.label] = lab == i
    return out


def mask_to_bw_png(mask: np.ndarray, out: Path) -> Path:
    """Write a bool mask as black-ink / white-bg PNG."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.where(mask, 0, 255).astype(np.uint8))
    return out


def save_palette_preview(groups: list[ColourGroup], out: Path, *, swatch: int = 64) -> Path:
    """Write a horizontal RGB swatch strip for ``groups``."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not groups:
        img = np.full((swatch, swatch, 3), 255, dtype=np.uint8)
    else:
        img = np.full((swatch, swatch * len(groups), 3), 255, dtype=np.uint8)
        for i, g in enumerate(groups):
            r, gv, b = _stamp_rgb(g)
            img[:, i * swatch : (i + 1) * swatch] = (b, gv, r)
    cv2.imwrite(str(out), img)
    return out


def save_centroid_map_png(
    bgr: np.ndarray,
    groups: list[ColourGroup],
    out: Path,
    *,
    dist_thresh: float = _DEFAULT_GROUP_DIST,
    bg_thresh: int = _DEFAULT_BG_THRESH,
) -> Path:
    """Paint each assigned pixel with its group stamp RGB (white elsewhere)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = np.full(bgr.shape, 255, dtype=np.uint8)
    if groups:
        lab = _label_map(bgr, groups, dist_thresh=dist_thresh, bg_thresh=bg_thresh)
        stamps = np.array(
            [(_stamp_rgb(g)[2], _stamp_rgb(g)[1], _stamp_rgb(g)[0]) for g in groups],
            dtype=np.uint8,
        )
        hit = lab >= 0
        result[hit] = stamps[lab[hit]]
    cv2.imwrite(str(out), result)
    return out


def _is_white_fill(v: Vector, *, thresh: float = 0.98) -> bool:
    return v.fill is not None and all(c >= thresh for c in v.fill)


def _covers_most_of_page(
    v: Vector,
    page_w: float,
    page_h: float,
    *,
    cover: float = 0.95,
) -> bool:
    x0, y0, x1, y1 = v.rect
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) >= cover * page_w * page_h


# Full-page wash vs border frame (tightened to keep side grids).
_WASH_COVER = 0.95  # was 0.85 — tall side strips less often match
_WASH_MAX_SIMPLE_ITEMS = 8
_WASH_Q_MIN = 0.20  # compact solid
_WASH_MIN_SOLIDITY = 0.35  # path_area / bbox_area; below → thin frame, keep
_WASH_SOLID_SOLIDITY = 0.85  # very filled bbox → solid wash even if many items


def _path_fill_area(v: Vector) -> float:
    """Shoelace area of path loops (fallback: bbox area)."""
    loops = _split_loops(v.items or [])
    if not loops:
        x0, y0, x1, y1 = v.rect
        return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))
    return float(sum(_area(pts) for pts in loops))


def _is_page_wash(
    v: Vector,
    page_w: float,
    page_h: float,
    *,
    stamp_rgb: tuple[int, int, int],
    cover: float = _WASH_COVER,
) -> bool:
    """Drop solid page washes; keep thin page-frame / side-grid paths.

    Achromatic near-page paths are dropped only when they look *solid*:
    high fill/bbox solidity and high Q, optionally with few items.
    Low-solidity rings (border grids) are kept even if the bbox is page-sized.
    """
    if not _covers_most_of_page(v, page_w, page_h, cover=cover):
        return False
    if _is_white_fill(v):
        return True  # full-page white clear

    r, g, b = stamp_rgb
    chroma = max(r, g, b) - min(r, g, b)
    lum = (r + g + b) / 3.0
    if chroma > _ACHROMA_S_MAX or lum > _DEFAULT_BG_THRESH:
        return False

    x0, y0, x1, y1 = v.rect
    bbox_a = max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))
    if bbox_a <= 0:
        return False
    solidity = _path_fill_area(v) / bbox_a
    if solidity < _WASH_MIN_SOLIDITY:
        return False  # thin frame / side grids

    q = isoperimetric_quotient(v)
    n_items = len(v.items or [])
    # Simple solid clear: few items AND compact Q.
    if n_items <= _WASH_MAX_SIMPLE_ITEMS and q >= _WASH_Q_MIN:
        return True
    # Complex outline but still a solid page plate.
    return solidity >= _WASH_SOLID_SOLIDITY and q >= _WASH_Q_MIN


def _stamp_group_vectors(
    vecs: list[Vector],
    fill_rgb: tuple[int, int, int],
    *,
    seq_base: int,
    layer: str,
    page_w: float,
    page_h: float,
) -> list[Vector]:
    """Stamp ink; drop page washes; keep smaller white fills as hole covers."""
    fill = (fill_rgb[0] / 255.0, fill_rgb[1] / 255.0, fill_rgb[2] / 255.0)
    out: list[Vector] = []
    for v in vecs:
        if _is_page_wash(v, page_w, page_h, stamp_rgb=fill_rgb):
            continue
        seqno = seq_base + len(out)
        if _is_white_fill(v):
            # Hole cover: stay white so hollow shapes don't become solid black.
            out.append(
                replace(
                    v,
                    fill=(1.0, 1.0, 1.0),
                    color=None,
                    width=0.0,
                    fill_opacity=1.0,
                    seqno=seqno,
                    layer=layer,
                )
            )
        else:
            out.append(
                replace(
                    v,
                    # PyMuPDF path type: "f" fill only, "s" stroke only, "fs" both. A
                    # stamped fill keeps no stroke.
                    type="f" if v.type in ("f", "fs") else v.type,
                    fill=fill,
                    color=None,
                    width=0.0,
                    fill_opacity=1.0,
                    seqno=seqno,
                    layer=layer,
                )
            )
    return out


def _colour_trace_worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _colour_trace_job(job: dict) -> dict:
    t0 = time.perf_counter()
    # Each mask is already a clean two-colour (ink / paper) image, so autotrace traces outlines
    # with 2 colours and none of its own noise removal or despeckling.
    vecs = autotrace_drawing(
        job["mask_path"],
        centerline=False,
        color_count=2,
        noise_removal=0.0,
        filter_iterations=0,
        despeckle_level=0,
        dpi=job.get("dpi"),
        svg_out=job["svg_path"],
        pdf_out=job["pdf_path"],
        scratch_dir=job["scratch_dir"],
        split_strokes=False,
    )
    return {
        "label": int(job["label"]),
        "tag": str(job["tag"]),
        "t_trace": time.perf_counter() - t0,
        "n_raw": len(vecs),
        "vectors": [VectorDTO.from_vector(v).model_dump(mode="json") for v in vecs],
    }


def colour_fill_autotrace(
    drawing_png: str | Path,
    *,
    out_dir: str | Path | None = None,
    stem: str | None = None,
    dpi: int | None = 300,
    bg_thresh: int = _DEFAULT_BG_THRESH,
    group_dist: float = _DEFAULT_GROUP_DIST,
    min_group_pixels: int = _DEFAULT_MIN_GROUP_PIXELS,
    keep_intermediates: bool = True,
    workers: int = 1,
) -> ColourFillTraceResult:
    """Cluster colours, autotrace each non-white mask, combine stamped vectors."""
    src = Path(drawing_png)
    if not src.is_file():
        raise FileNotFoundError(f"PNG not found: {src}")

    t_all = time.perf_counter()
    parent = Path(out_dir) if out_dir is not None else src.parent
    parent.mkdir(parents=True, exist_ok=True)
    out_stem = stem if stem else src.stem
    colour_dir = parent / f"{out_stem}_colours"
    colour_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="colourfill_", dir=str(colour_dir)))
    n_workers = max(1, int(workers))

    t0 = time.perf_counter()
    bgr = _load_bgr(src)
    h, w = bgr.shape[:2]
    print(
        f"[colour] load {src.name} {w}x{h} ({time.perf_counter() - t0:.2f}s) -> {colour_dir}",
        flush=True,
    )

    t0 = time.perf_counter()
    uniq, counts = unique_colours(bgr, bg_thresh=bg_thresh)
    groups = group_colours(uniq, counts, dist_thresh=group_dist, min_pixels=min_group_pixels)
    n_uniq = len(uniq)
    del uniq, counts
    print(
        f"[colour] {n_uniq} RGB -> {len(groups)} groups "
        f"(dist={group_dist}, min_px={min_group_pixels}) in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    for g in groups:
        print(f"  [{g.tag}] kind={g.kind} stamp={_stamp_rgb(g)} px={g.n_pixels}", flush=True)

    palette_path = save_palette_preview(groups, colour_dir / f"{out_stem}_colour_palette.png")
    t0 = time.perf_counter()
    lab = _label_map(bgr, groups, dist_thresh=group_dist, bg_thresh=bg_thresh)
    stamps = (
        np.array(
            [(_stamp_rgb(g)[2], _stamp_rgb(g)[1], _stamp_rgb(g)[0]) for g in groups],
            dtype=np.uint8,
        )
        if groups
        else np.zeros((0, 3), dtype=np.uint8)
    )
    centroid = np.full_like(bgr, 255)
    if groups:
        hit = lab >= 0
        centroid[hit] = stamps[lab[hit]]
        del hit
    centroid_map_path = colour_dir / f"{out_stem}_colour_centroid_map.png"
    cv2.imwrite(str(centroid_map_path), centroid)
    del centroid, stamps
    print(f"[colour] masks+centroid map {time.perf_counter() - t0:.2f}s", flush=True)

    page_w = w * PDF_POINTS_PER_INCH / float(dpi) if dpi and dpi > 0 else float(w)
    page_h = h * PDF_POINTS_PER_INCH / float(dpi) if dpi and dpi > 0 else float(h)

    group_vectors: dict[int, list[Vector]] = {g.label: [] for g in groups}
    mask_paths: dict[int, Path] = {}
    isolate_paths: dict[int, Path] = {}
    autotrace_svg_paths: dict[int, Path] = {}
    autotrace_pdf_paths: dict[int, Path] = {}
    group_timings: list[tuple[str, float, float, int]] = []
    combined: list[Vector] = []

    jobs: list[dict] = []
    prep_meta: dict[int, dict] = {}
    for gi, group in enumerate(groups):
        i = gi + 1
        # One bool mask at a time — never hold N full-page masks.
        mask = lab == gi
        if group.kind == "white":
            print(f"[colour] ({i}/{len(groups)}) {group.tag}: white — skip autotrace", flush=True)
            del mask
            continue
        if not mask.any():
            print(f"[colour] ({i}/{len(groups)}) {group.tag}: empty — skip", flush=True)
            del mask
            continue

        tag = group.tag
        mask_path = colour_dir / f"{out_stem}_{tag}_mask.png"
        isolate_path = colour_dir / f"{out_stem}_{tag}_isolate.png"
        svg_path = colour_dir / f"{out_stem}_{tag}_autotrace.svg"
        pdf_path = colour_dir / f"{out_stem}_{tag}_autotrace.pdf"

        t0 = time.perf_counter()
        mask_to_bw_png(mask, mask_path)
        if keep_intermediates:
            iso = np.full_like(bgr, 255)
            iso[mask] = bgr[mask]
            cv2.imwrite(str(isolate_path), iso)
            del iso
        t_iso = time.perf_counter() - t0
        del mask

        jobs.append(
            {
                "label": group.label,
                "tag": tag,
                "mask_path": str(mask_path),
                "svg_path": str(svg_path),
                "pdf_path": str(pdf_path),
                "scratch_dir": str(scratch / tag),
                "dpi": dpi,
            }
        )
        prep_meta[group.label] = {
            "tag": tag,
            "stamp_rgb": _stamp_rgb(group),
            "mask_path": mask_path,
            "isolate_path": isolate_path,
            "svg_path": svg_path,
            "pdf_path": pdf_path,
            "t_iso": t_iso,
            "index": i,
        }

    # Page rasters no longer needed after mask PNGs are on disk.
    del lab, bgr

    trace_by_label: dict[int, dict] = {}
    t_trace_all = time.perf_counter()
    if jobs and (n_workers == 1 or len(jobs) == 1):
        for job in jobs:
            trace_by_label[int(job["label"])] = _colour_trace_job(job)
    elif jobs:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(n_workers, len(jobs)),
            mp_context=ctx,
            initializer=_colour_trace_worker_init,
        ) as pool:
            futs = {pool.submit(_colour_trace_job, j): int(j["label"]) for j in jobs}
            for fut in as_completed(futs):
                label = futs[fut]
                trace_by_label[label] = fut.result()
                print(
                    f"[colour] autotrace {prep_meta[label]['tag']} "
                    f"paths={trace_by_label[label]['n_raw']} "
                    f"{trace_by_label[label]['t_trace']:.1f}s",
                    flush=True,
                )
    if jobs:
        print(
            f"[colour] autotraces done in {time.perf_counter() - t_trace_all:.2f}s "
            f"(workers={n_workers})",
            flush=True,
        )

    seq_base = 0
    for group in groups:
        meta = prep_meta.get(group.label)
        if meta is None:
            continue
        trace = trace_by_label.pop(group.label)
        vecs = [deserialize_vector(VectorDTO.model_validate(row)) for row in trace["vectors"]]
        n_raw = int(trace["n_raw"])
        t_trace = float(trace["t_trace"])
        del trace
        vecs.sort(key=lambda v: v.seqno)
        stamped = _stamp_group_vectors(
            vecs,
            meta["stamp_rgb"],
            seq_base=seq_base,
            layer=meta["tag"],
            page_w=page_w,
            page_h=page_h,
        )
        del vecs
        seq_base += len(stamped)
        group_vectors[group.label] = stamped
        combined.extend(stamped)
        autotrace_svg_paths[group.label] = meta["svg_path"]
        autotrace_pdf_paths[group.label] = meta["pdf_path"]
        if keep_intermediates:
            mask_paths[group.label] = meta["mask_path"]
            isolate_paths[group.label] = meta["isolate_path"]
        else:
            meta["mask_path"].unlink(missing_ok=True)
            meta["isolate_path"].unlink(missing_ok=True)

        group_timings.append((meta["tag"], meta["t_iso"], t_trace, len(stamped)))
        print(
            f"[colour] ({meta['index']}/{len(groups)}) {meta['tag']}: "
            f"autotrace={t_trace:.1f}s raw={n_raw} -> {len(stamped)}",
            flush=True,
        )

    combined_pdf_path = vectors_txt_path = vectors_json_path = None
    if combined:
        t0 = time.perf_counter()
        combined_pdf_path = colour_dir / f"{out_stem}_colour_combined.pdf"
        draw_vectors(combined, combined_pdf_path, page_width=page_w, page_height=page_h)
        vectors_txt_path = colour_dir / f"{out_stem}_colour_vectors.txt"
        vectors_txt_path.write_text(
            "\n".join(f"Vector {i}: {v}" for i, v in enumerate(combined)) + "\n",
            encoding="utf-8",
        )
        vectors_json_path = dump_colour_vectors_json(
            combined,
            colour_dir / f"{out_stem}_colour_vectors.json",
            page_width=page_w,
            page_height=page_h,
            source=src,
        )
        print(
            f"[colour] combined {len(combined)} vectors -> {combined_pdf_path.name} "
            f"({time.perf_counter() - t0:.2f}s)",
            flush=True,
        )

    t_elapsed = time.perf_counter() - t_all
    print(
        f"[colour] DONE {src.name}: {len(groups)} groups, {len(combined)} vectors, "
        f"{t_elapsed:.2f}s",
        flush=True,
    )

    summary_path = colour_dir / f"{out_stem}_colour_summary.txt"
    lines = [
        f"source: {src}",
        f"colours_dir: {colour_dir}",
        f"workers: {n_workers}",
        f"unique RGB (non-bg): {n_uniq}",
        f"colour groups: {len(groups)} (group_dist={group_dist}, min_group_pixels={min_group_pixels})",
        f"combined vectors: {len(combined)}",
        f"combined_pdf: {combined_pdf_path}",
        f"vectors_json: {vectors_json_path}",
        f"centroid_map: {centroid_map_path}",
        f"page_width: {page_w}",
        f"page_height: {page_h}",
        f"total_seconds: {t_elapsed:.2f}",
        "",
        "groups:",
    ]
    for g in groups:
        lines.append(
            f"  [{g.tag}] kind={g.kind} stamp={_stamp_rgb(g)} "
            f"pixels={g.n_pixels} vectors={len(group_vectors.get(g.label, []))}"
        )
    if group_timings:
        lines += ["", "timings (seconds):"]
        for tag, t_iso, t_trace, n_vec in group_timings:
            lines.append(f"  [{tag}] prep={t_iso:.2f} autotrace={t_trace:.2f} vectors={n_vec}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return ColourFillTraceResult(
        groups=groups,
        group_vectors=group_vectors,
        combined_vectors=combined,
        mask_paths=mask_paths,
        isolate_paths=isolate_paths,
        autotrace_svg_paths=autotrace_svg_paths,
        autotrace_pdf_paths=autotrace_pdf_paths,
        palette_path=palette_path,
        summary_path=summary_path,
        colours_dir=colour_dir,
        combined_pdf_path=combined_pdf_path,
        vectors_txt_path=vectors_txt_path,
        vectors_json_path=vectors_json_path,
        centroid_map_path=centroid_map_path,
        page_width=page_w,
        page_height=page_h,
    )
