"""Running the pretrained InvolutionStroke chain over one ink layer.

`vectorize_layer(gray, ...)` -- `gray` is one color layer's mask (ink 0,
everything else 255) at the canonical scale (`config.TARGET_PX_PER_PT`):

  1. tile into `TILE_PX` tiles overlapping by `TILE_OVERLAP_PX` (tiles with
     fewer than `MIN_TILE_INK_PX` ink pixels skipped). Upstream runs on
     whole 512-1024 px images; a page layer is ~10k px and graph extraction
     is pure Python, so the *whole* upstream chain runs per tile, at the
     scale it was built for:
       a. invert to white-on-black (upstream `imgpreprocess.preprocess`;
          its normalization only with `PREPROCESS`), pad to a multiple of 16,
          centerline network -> 8-bit centerline image (upstream
          `centerline.predict_centerline`);
       b. `build_linegraph.process` + `compute_polylines` -> polylines between
          graph nodes (endpoints / junctions);
       c. each polyline split into two half-edges, the constraint matrix R
          (half-edges meeting at the same node), the pairing network's
          symmetric scores, `involutive_assignment2` (max-weight matching)
          and `get_matched_lines` -> strokes running straight through the
          junctions they should (upstream `main.process_image`);
     A tile whose graph extraction fails is logged and skipped (upstream
     likewise moves on to the next image).
  2. clip every stroke to its tile's *core* (the part of the layer it owns;
     neighbouring tiles split their overlap at its middle) and join stroke
     ends from different tiles that meet on a core border
     (`MERGE_SNAP_PX`);
  3. smooth each stroke: split at sharp corners, then Schneider Bezier fit
     or RDP (`config.FIT_MODE`).

Models load lazily from `raster_parser/weights/` (`download_weights.py` fetches
them; env vars override each path) and are cached per process; a missing
file raises `FileNotFoundError` naming the download command."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

from raster_parser.models.involution_stroke import TileResult, StrokePiece, LayerResult
from raster_parser.commons.logging_setup import get_logger

from . import geometry as geo
from .config import (
    CENTERLINE_WEIGHTS_ENV_VAR,
    CENTERLINE_WEIGHTS_FILENAME,
    CORNER_DEG,
    CORNER_EPS_PX,
    FIT_MODE,
    FIT_PRESMOOTH_PX,
    FIT_TOL_PX,
    GRAPH_DEG3,
    GRAPH_DEG4,
    GRAPH_SHARP,
    INFER_THREADS,
    INK_GRAY_THRESHOLD,
    INTERSECTION_WEIGHTS_ENV_VAR,
    INTERSECTION_WEIGHTS_FILENAME,
    MERGE_SNAP_PX,
    MIN_STROKE_PX,
    MIN_TILE_INK_PX,
    PREPROCESS,
    SMOOTH_D,
    SMOOTH_LINES,
    TILE_OVERLAP_PX,
    TILE_PX,
)

_LOG = get_logger("involution_stroke.infer")

_MODEL_CACHE: dict[str, object] = {}

# white-on-black uint8 tile (H, W) -> uint8 centerline image (H, W)
CenterlineFn = Callable[[np.ndarray], np.ndarray]
# (lines (N, 100, 2) float32, R (N, N) float32) -> scores (N, N) float32
PairingFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

_DOWNSAMPLE = 16  # model.centerline.DOWNSAMPLE (kept here so tests/fakes don't import torch)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def _weights_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "weights"


def default_centerline_path() -> Path:
    env = os.environ.get(CENTERLINE_WEIGHTS_ENV_VAR)
    return Path(env) if env else _weights_dir() / CENTERLINE_WEIGHTS_FILENAME


def default_intersection_path() -> Path:
    env = os.environ.get(INTERSECTION_WEIGHTS_ENV_VAR)
    return Path(env) if env else _weights_dir() / INTERSECTION_WEIGHTS_FILENAME


def _missing(path: Path, env_var: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"InvolutionStroke weights not found at {path}. Fetch the authors' pretrained checkpoints with "
        f"`python -m raster_parser.phase_raster_vectorize.involution_stroke.download_weights`, or point ${env_var} "
        f"at an existing file."
    )


def _for_cpu_inference(model):
    import torch

    if INFER_THREADS:
        torch.set_num_threads(INFER_THREADS)
    model.eval()
    model.requires_grad_(False)
    return model


def load_centerline_model(path: "str | Path | None" = None):
    path = Path(path) if path is not None else default_centerline_path()
    key = f"centerline:{path}"
    if key not in _MODEL_CACHE:
        if not path.is_file():
            raise _missing(path, CENTERLINE_WEIGHTS_ENV_VAR)
        from .model.weights import load_centerline

        _MODEL_CACHE[key] = _for_cpu_inference(load_centerline(path))
        _LOG.info("InvolutionStroke centerline model loaded from %s", path)
    return _MODEL_CACHE[key]


def load_intersection_model(path: "str | Path | None" = None):
    path = Path(path) if path is not None else default_intersection_path()
    key = f"intersection:{path}"
    if key not in _MODEL_CACHE:
        if not path.is_file():
            raise _missing(path, INTERSECTION_WEIGHTS_ENV_VAR)
        from .model.weights import load_intersection

        _MODEL_CACHE[key] = _for_cpu_inference(load_intersection(path))
        _LOG.info("InvolutionStroke intersection model loaded from %s", path)
    return _MODEL_CACHE[key]


def warmup() -> None:
    """Best-effort model load (missing weights are skipped)."""
    for load in (load_centerline_model, load_intersection_model):
        try:
            load()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# The two networks (plain arrays in / out, so they can run on Pool 2)
# ---------------------------------------------------------------------------
def predict_centerline(model, tile: np.ndarray) -> np.ndarray:
    """White-on-black uint8 tile -> uint8 centerline image of the same size
    (upstream `centerline.predict_centerline`: ToTensor, zero-pad, sigmoid
    output, `to_pil_image`'s `mul(255).byte()`). Padding is to a multiple of
    the encoder stride instead of upstream's next power of two -- the network
    is fully convolutional, so only the stride matters."""
    import torch

    h, w = tile.shape
    ph, pw = -h % _DOWNSAMPLE, -w % _DOWNSAMPLE
    x = torch.from_numpy(tile.astype(np.float32) / 255.0)[None, None]
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)
    with torch.inference_mode():
        out = model(x)[0, 0, :h, :w]
    return out.mul(255).byte().numpy()


def predict_pairing(model, lines: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Upstream `intersections.predict` minus the unused raster inputs."""
    import torch

    n = R.shape[0]
    batch = {
        "lines": torch.from_numpy(np.ascontiguousarray(lines, np.float32))[None],
        "R": torch.from_numpy(np.ascontiguousarray(R, np.float32))[None],
        "n_objects": torch.tensor([n], dtype=torch.int64),
    }
    with torch.inference_mode():
        return model(batch)[0, :n, :n].numpy()


def _centerline_job(tile: np.ndarray, path: str) -> np.ndarray:
    """Pool-2 job: plain data in, plain data out (no pymupdf)."""
    return predict_centerline(load_centerline_model(path), tile)


def _pairing_job(lines: np.ndarray, R: np.ndarray, path: str) -> np.ndarray:
    return predict_pairing(load_intersection_model(path), lines, R)


def make_model_fns(
    model_fns: "dict | None", compute, centerline_path=None, intersection_path=None
) -> tuple[CenterlineFn, PairingFn]:
    """`model_fns` (tests) may override `"centerline"` / `"pairing"`; the
    rest run in-process, or on Pool 2 when `compute` is given."""
    model_fns = model_fns or {}
    cpath = str(centerline_path or default_centerline_path())
    ipath = str(intersection_path or default_intersection_path())
    centerline = model_fns.get("centerline")
    pairing = model_fns.get("pairing")
    if compute is not None:
        if centerline is None:
            if not Path(cpath).is_file():
                load_centerline_model(cpath)  # raises the FileNotFoundError with its hint, locally
            centerline = lambda tile: compute.apply(_centerline_job, (tile, cpath))  # noqa: E731
        if pairing is None:
            if not Path(ipath).is_file():
                load_intersection_model(ipath)
            pairing = lambda lines, R: compute.apply(_pairing_job, (lines, R, ipath))  # noqa: E731
    else:
        if centerline is None:
            centerline = lambda tile: predict_centerline(load_centerline_model(cpath), tile)  # noqa: E731
        if pairing is None:
            pairing = lambda lines, R: predict_pairing(load_intersection_model(ipath), lines, R)  # noqa: E731
    return centerline, pairing


# ---------------------------------------------------------------------------
# One tile through the upstream chain
# ---------------------------------------------------------------------------
# upstream `imgpreprocess.preprocess`'s white top-hat ellipse and background-blur sizes (px)
_TOPHAT_PX = 21
_BACKGROUND_BLUR_PX = 51


def preprocess(gray: np.ndarray, normalize: bool = PREPROCESS) -> np.ndarray:
    """Layer mask (ink 0) -> the centerline network's white-on-black input
    (upstream `imgpreprocess.preprocess(invert=True, skip_norm=not normalize)`:
    white top-hat with a 21 px ellipse, then subtract a 51 px Gaussian blur)."""
    img = 255 - gray
    if not normalize:
        return img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_TOPHAT_PX, _TOPHAT_PX))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    return cv2.subtract(
        tophat, cv2.GaussianBlur(tophat, (_BACKGROUND_BLUR_PX, _BACKGROUND_BLUR_PX), 0)
    )


def relation_matrix(half_edges: dict, node_half_edges: dict) -> np.ndarray:
    """Upstream `build_intersection_segmentation` step 1: R[i, j] = 1 when
    half-edges i and j end at the same graph node (row/col 0 = background)."""
    n = len(half_edges) + 1
    R = np.zeros((n, n), dtype=np.int32)
    for row, (node, _pts) in half_edges.items():
        for col in node_half_edges[node]:
            R[row, col] = 1
    R[0, 0] = 1
    return R


def run_tile(tile_gray: np.ndarray, centerline: CenterlineFn, pairing: PairingFn) -> TileResult:
    """Upstream `main.process_image` on one layer-mask tile (ink 0)."""
    from ._upstream import intersections as up_int
    from ._upstream.graph_extraction import build_linegraph

    cl = centerline(preprocess(tile_gray))
    res = TileResult(centerline=cl)
    g, _ok = build_linegraph.process(
        cl, invert=False, debug_dir="", sharp=GRAPH_SHARP, deg3=GRAPH_DEG3, deg4=GRAPH_DEG4
    )
    lines, connections = build_linegraph.compute_polylines(g)
    if SMOOTH_LINES:
        lines = build_linegraph.get_smoothed_lines(lines, d=SMOOTH_D)
    lines = [np.asarray(pl, float) for pl in lines]
    res.polylines = lines
    pos = dict(g.nodes(data="pos"))
    res.nodes = sorted({tuple(map(float, pos[n])) for c in connections for n in c})
    if not lines:
        return res
    node_half_edges, half_edges = up_int.split_in_half(lines, connections)
    R = relation_matrix(half_edges, node_half_edges)
    line_tensor = up_int.get_line_tensor(half_edges).numpy()
    scores = pairing(line_tensor, R.astype(np.float32))
    matched = up_int.get_matched_lines(scores, half_edges)
    res.strokes = [_dedupe(np.concatenate(sublines)) for sublines in matched if sublines]
    return res


def _dedupe(pts: np.ndarray) -> np.ndarray:
    """Drop consecutive duplicate points (half-edges share their joints)."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return pts
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
    return pts[keep]


# ---------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------


def _on_core_border(p: np.ndarray, core, size: tuple[int, int], tol: float) -> bool:
    """`p` within `tol` of a side of `core` that is not the layer's edge."""
    x0, y0, x1, y1 = core
    w, h = size
    return (
        (x0 > 0 and abs(p[0] - x0) <= tol)
        or (x1 < w and abs(p[0] - x1) <= tol)
        or (y0 > 0 and abs(p[1] - y0) <= tol)
        or (y1 < h and abs(p[1] - y1) <= tol)
    )


def join_seams(
    pieces: list[StrokePiece], size: tuple[int, int], snap: float = MERGE_SNAP_PX
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Join piece ends from *different* tiles lying within `snap` of each
    other on their cores' shared border (greedy nearest first, one join per
    end) and walk the joined chains (closed loops included) into polylines.
    Returns (polylines, join points)."""
    from scipy.spatial import cKDTree

    ends = []  # (piece index, which end 0|1)
    pts = []
    for i, pc in enumerate(pieces):
        for e, p in ((0, pc.pts[0]), (1, pc.pts[-1])):
            if _on_core_border(p, pc.core, size, snap):
                ends.append((i, e))
                pts.append(p)
    link: dict[tuple[int, int], tuple[int, int]] = {}
    joins: list[tuple[float, float]] = []
    if len(pts) >= 2:
        arr = np.asarray(pts, float)
        pairs = sorted(
            cKDTree(arr).query_pairs(snap),
            key=lambda ab: float(np.hypot(*(arr[ab[0]] - arr[ab[1]]))),
        )
        for a, b in pairs:
            ea, eb = ends[a], ends[b]
            if ea in link or eb in link or pieces[ea[0]].tile == pieces[eb[0]].tile:
                continue
            link[ea] = eb
            link[eb] = ea
            mid = 0.5 * (arr[a] + arr[b])
            joins.append((float(mid[0]), float(mid[1])))

    used = [False] * len(pieces)

    def walk(start: int, enter: int) -> np.ndarray:
        out: list[np.ndarray] = []
        i, e = start, enter
        while not used[i]:
            used[i] = True
            seg = pieces[i].pts if e == 0 else pieces[i].pts[::-1]
            out.append(seg if not out else seg[1:])
            nxt = link.get((i, 1 - e))
            if nxt is None:
                break
            i, e = nxt
        return np.concatenate(out)

    lines = []
    for i in range(len(pieces)):  # open chains: start at an unlinked end
        if used[i]:
            continue
        if (i, 0) not in link:
            lines.append(walk(i, 0))
        elif (i, 1) not in link:
            lines.append(walk(i, 1))
    for i in range(len(pieces)):  # what's left is closed loops
        if not used[i]:
            lines.append(walk(i, 0))
    return lines, joins


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------


def smooth(
    polyline: np.ndarray,
    mode: str = FIT_MODE,
    tol: float = FIT_TOL_PX,
    corner_deg: float = CORNER_DEG,
) -> np.ndarray:
    """A joined stroke polyline -> cubic pieces `(K, 4, 2)`."""
    return geo.smooth_stroke(
        polyline, mode, tol, corner_deg, corner_eps=CORNER_EPS_PX, presmooth_px=FIT_PRESMOOTH_PX
    )


def vectorize_layer(
    gray: np.ndarray,
    *,
    model_fns: "dict | None" = None,
    compute=None,
    centerline_path=None,
    intersection_path=None,
    tile: int = TILE_PX,
    overlap: int = TILE_OVERLAP_PX,
    debug: bool = False,
    desc: str = "InvolutionStroke tiles",
) -> LayerResult:
    centerline, pairing = make_model_fns(model_fns, compute, centerline_path, intersection_path)
    h, w = gray.shape
    ink = gray < INK_GRAY_THRESHOLD
    cores_x = geo.core_spans(w, tile, overlap)
    cores_y = geo.core_spans(h, tile, overlap)
    jobs = [
        (x0, y0)
        for x0, y0 in geo.tile_grid(h, w, tile, overlap)
        if int(ink[y0 : y0 + tile, x0 : x0 + tile].sum()) >= MIN_TILE_INK_PX
    ]

    pieces: list[StrokePiece] = []
    tiles, failed = [], []
    graph_polylines, graph_nodes, matched = [], [], []
    n_graph = 0
    cl_canvas = np.zeros((h, w), np.uint8) if debug else None
    for t_idx, (x0, y0) in enumerate(
        tqdm(jobs, desc=desc, unit="tile", leave=False, disable=not jobs)
    ):
        sub = np.ascontiguousarray(gray[y0 : y0 + tile, x0 : x0 + tile])
        th, tw = sub.shape
        box = (x0, y0, tw, th)
        off = np.array([x0, y0], float)
        try:
            tr = run_tile(sub, centerline, pairing)
        except Exception:  # noqa: BLE001 -- upstream graph code has many failure modes; skip the tile
            _LOG.warning(
                "InvolutionStroke: tile at (%d, %d) failed, skipped", x0, y0, exc_info=True
            )
            failed.append(box)
            continue
        tiles.append(box)
        n_graph += len(tr.polylines)
        (ax, bx), (ay, by) = cores_x[x0], cores_y[y0]
        core = (float(ax), float(ay), float(bx), float(by))
        for s in tr.strokes:
            for part in geo.clip_polyline(s + off, core):
                pieces.append(StrokePiece(part, t_idx, core))
        if debug:
            graph_polylines += [pl + off for pl in tr.polylines]
            graph_nodes += [(x + x0, y + y0) for x, y in tr.nodes]
            matched += [s + off for s in tr.strokes]
            if tr.centerline is not None:
                cl_canvas[ay:by, ax:bx] = tr.centerline[ay - y0 : by - y0, ax - x0 : bx - x0]

    polylines, joins = join_seams(pieces, (w, h))
    polylines = [p for p in polylines if geo.polyline_length(p) >= MIN_STROKE_PX]
    strokes = [s for s in (smooth(p) for p in polylines) if len(s)]
    return LayerResult(
        strokes=strokes,
        polylines=polylines,
        tiles=tiles,
        failed_tiles=failed,
        graph_polylines=graph_polylines,
        graph_nodes=graph_nodes,
        matched=matched,
        joins=joins,
        n_graph_polylines=n_graph,
        centerline=cl_canvas,
    )
