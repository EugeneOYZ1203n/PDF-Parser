"""Running the trained Yan et al. pipeline over one ink layer.

`vectorize_layer(gray, ...)` -- `gray` is one color layer's mask (ink 0,
everything else 255) at the canonical scale (`config.TARGET_PX_PER_PT`):

  1. tile into `TILE_PX` tiles overlapping by `TILE_OVERLAP_PX` (tiles with
     fewer than `MIN_TILE_INK_PX` ink pixels skipped) and run both networks
     on each: DFP -> six UDFs at 2x, NDC on the centerline UDF -> edge
     flags + vertex map (in-process, `model_fn`, or Pool 2 via `_infer_job`);
  2. harvest each tile's *core* (the part it owns, so the overlap is never
     counted twice) into sparse page-wide data -- refined edge flags
     (`postprocess.refine_flags`, run on the whole tile first), vertex
     positions, keypoints (thresholded local minima of the end / sharp /
     junction UDFs), under-sampled cells -- since a page-sized dense cell
     grid at 2x would not fit in memory;
  3. dual contouring + post-processing on the page-wide sparse graph
     (`postprocess.py`): build, repair breaks, topology refinement, DC
     downsampling, split crossings, line grouping;
  4. smoothing: Schneider Bezier fitting or RDP (`config.FIT_MODE`).

The model is loaded lazily from `rastervec/weights/implicit_sketch_vec.pth`
(or `$IMPLICITSKETCHVEC_WEIGHTS_PATH`) and cached per process; a missing file
raises `FileNotFoundError` naming where to put it."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from rastervec.commons.logging_setup import get_logger

from . import geometry as geo
from . import postprocess as pp
from .config import (
    DC_DOWNSAMPLE, FIT_MODE, FIT_TOL_PX, INFER_BATCH_TILES, INK_GRAY_THRESHOLD, KEYPOINT_NMS_PX, KEYPOINT_UDF_PX,
    MIN_STROKE_PX, MIN_TILE_INK_PX, SR, TILE_OVERLAP_PX, TILE_PX, UDF_CHANNELS, UDF_TRUNC_PX, USM_UDF_PX,
    VERTEX_UDF_PX, WEIGHTS_ENV_VAR, WEIGHTS_FILENAME,
)

_LOG = get_logger("P2.ImplicitSketchVec.infer")

_MODEL_CACHE: dict[str, object] = {}

# tiles -> per tile {"udf": (6, 2S+1, 2S+1) px, "edge": (2S, 2S) class 0-3,
#                    "vertex": (2, 2S, 2S) offsets in the cell}
ModelFn = Callable[[list[np.ndarray]], list[dict]]

KEYPOINT_TYPES = ("end", "sharp", "junc")


def default_weights_path() -> Path:
    env = os.environ.get(WEIGHTS_ENV_VAR)
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "weights" / WEIGHTS_FILENAME


def load_model(path: "str | Path | None" = None):
    path = Path(path) if path is not None else default_weights_path()
    key = str(path)
    if key not in _MODEL_CACHE:
        if not path.is_file():
            raise FileNotFoundError(
                f"ImplicitSketchVec weights not found at {path}. Train them with "
                f"`python -m rastervec.P2_Raster_To_Vec.ImplicitSketchVec.train --data <DeepVectoriser prep dir> "
                f"--stage dfp|ndc|joint` (the joint stage writes {path.name}), or point ${WEIGHTS_ENV_VAR} "
                f"at an existing .pth."
            )
        import torch  # noqa: F401  (lazy: keep torch off the import path of the pipeline)

        from .model.network import load_weights

        model = load_weights(path, map_location="cpu")
        model.eval()
        _MODEL_CACHE[key] = model
        _LOG.info("ImplicitSketchVec model loaded from %s", path)
    return _MODEL_CACHE[key]


def warmup() -> None:
    """Best-effort model load (missing weights are skipped)."""
    try:
        load_model()
    except FileNotFoundError:
        pass


def predict_tiles(model, tiles: list[np.ndarray]) -> list[dict]:
    """Same-size uint8 gray tiles -> network fields per tile (numpy)."""
    import torch

    from .model.network import prepare_input

    if not tiles:
        return []
    device = next(model.parameters()).device
    with torch.no_grad():
        udf, out = model(prepare_input(torch.from_numpy(np.stack(tiles)).to(device)))
        udf = (udf * UDF_TRUNC_PX).cpu().numpy().astype(np.float32)
        edge = out["edge"].argmax(dim=1).cpu().numpy().astype(np.uint8)
        vert = out["vertex"].cpu().numpy().astype(np.float32)
    return [{"udf": udf[i], "edge": edge[i], "vertex": vert[i]} for i in range(len(tiles))]


def _infer_job(tiles: list[np.ndarray], weights_path: str) -> list[dict]:
    """Pool-2 job: plain data in, plain data out (no fitz)."""
    return predict_tiles(load_model(weights_path), tiles)


def _make_predict(model_fn: "ModelFn | None", compute, weights_path) -> ModelFn:
    if model_fn is not None:
        return model_fn
    path = str(weights_path or default_weights_path())
    if compute is not None:
        if not Path(path).is_file():
            load_model(path)  # raises the FileNotFoundError with its hint, locally
        return lambda tiles: compute.apply(_infer_job, (tiles, path))
    return lambda tiles: predict_tiles(load_model(path), tiles)


# ---------------------------------------------------------------------------
# Harvesting a tile's core
# ---------------------------------------------------------------------------
@dataclass
class _Sparse:
    right: list = field(default_factory=list)          # (N, 2) global cells, refined flags
    bottom: list = field(default_factory=list)
    raw_right: list = field(default_factory=list)      # before flag refinement (debug)
    raw_bottom: list = field(default_factory=list)
    vertex: dict = field(default_factory=dict)         # global cell -> px
    keypoints: dict = field(default_factory=lambda: {k: [] for k in KEYPOINT_TYPES})
    usm: set = field(default_factory=set)


def _cell_mean(lattice: np.ndarray) -> np.ndarray:
    return 0.25 * (lattice[:-1, :-1] + lattice[1:, :-1] + lattice[:-1, 1:] + lattice[1:, 1:])


def _harvest(f: dict, x0: int, y0: int, core_x: tuple[int, int], core_y: tuple[int, int], sp: _Sparse,
             heat: "np.ndarray | None", debug: bool) -> None:
    from scipy.ndimage import minimum_filter

    ax, bx = core_x
    ay, by = core_y
    c0x, c1x = SR * (ax - x0), SR * (bx - x0)        # core cells (local)
    c0y, c1y = SR * (ay - y0), SR * (by - y0)
    off = np.array([SR * x0, SR * y0])
    udf, edge, vert = f["udf"], f["edge"], f["vertex"]
    er = (edge == 1) | (edge == 3)
    eb = (edge == 2) | (edge == 3)
    if debug:
        sp.raw_right.append(np.argwhere(er[c0y:c1y, c0x:c1x])[:, ::-1] + off + [c0x, c0y])
        sp.raw_bottom.append(np.argwhere(eb[c0y:c1y, c0x:c1x])[:, ::-1] + off + [c0x, c0y])
    er, eb = pp.refine_flags(er, eb)
    sp.right.append(np.argwhere(er[c0y:c1y, c0x:c1x])[:, ::-1] + off + [c0x, c0y])
    sp.bottom.append(np.argwhere(eb[c0y:c1y, c0x:c1x])[:, ::-1] + off + [c0x, c0y])

    center = _cell_mean(udf[0])
    ys, xs = np.nonzero(center[c0y:c1y, c0x:c1x] < VERTEX_UDF_PX)
    ys, xs = ys + c0y, xs + c0x
    px = (xs + vert[0, ys, xs]) / SR + x0
    py = (ys + vert[1, ys, xs]) / SR + y0
    for a, b, p, q in zip(xs + SR * x0, ys + SR * y0, px, py):
        sp.vertex[(int(a), int(b))] = np.array([p, q])

    usm = _cell_mean(udf[UDF_CHANNELS.index("usm")])
    ys, xs = np.nonzero(usm[c0y:c1y, c0x:c1x] < USM_UDF_PX)
    sp.usm.update(zip((xs + c0x + SR * x0).tolist(), (ys + c0y + SR * y0).tolist()))

    k = 2 * int(round(KEYPOINT_NMS_PX * SR)) + 1
    for name in KEYPOINT_TYPES:
        u = udf[UDF_CHANNELS.index(name)]
        cand = (u < KEYPOINT_UDF_PX) & (u <= minimum_filter(u, size=k, mode="nearest"))
        ys, xs = np.nonzero(cand[c0y:c1y, c0x:c1x])
        for y, x in zip(ys + c0y, xs + c0x):
            sp.keypoints[name].append((x / SR + x0, y / SR + y0))

    if heat is not None:
        native = udf[0][::SR, ::SR][ay - y0:by - y0, ax - x0:bx - x0]
        heat[ay:by, ax:bx] = (255.0 * (1.0 - np.clip(native / UDF_TRUNC_PX, 0.0, 1.0))).astype(np.uint8)


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------
@dataclass
class LayerResult:
    strokes: list[np.ndarray]                     # (K, 4, 2) cubic pieces, layer px
    polylines: list[np.ndarray]                   # grouped lines before smoothing
    tiles: list[tuple[int, int, int]]             # (x0, y0, size) of every tile run
    keypoints: dict                               # type -> (K, 2) px
    usm_cells: set                                # global 1/SR px cells
    regions: list                                 # cell sets refined by topology surgery
    raw_segments: list                            # DC edges straight from the network flags
    graph_segments: list                          # DC edges after all graph post-processing
    heat: "np.ndarray | None" = None              # centerline UDF heat map (H, W) uint8, debug only


def _segments(right: np.ndarray, bottom: np.ndarray) -> list:
    out = []
    for (a, b) in np.asarray(right).reshape(-1, 2):
        out.append((pp.cell_centre((a, b)), pp.cell_centre((a + 1, b))))
    for (a, b) in np.asarray(bottom).reshape(-1, 2):
        out.append((pp.cell_centre((a, b)), pp.cell_centre((a, b + 1))))
    return out


def smooth(polyline: np.ndarray, mode: str = FIT_MODE, tol: float = FIT_TOL_PX) -> np.ndarray:
    """A grouped polyline -> cubic pieces `(K, 4, 2)`."""
    if mode == "rdp":
        pts = geo.rdp(polyline, tol)
        return np.stack([geo.line_to_cubic(a, b) for a, b in zip(pts, pts[1:])])
    pieces = geo.fit_cubics(polyline, tol)
    return np.stack(pieces) if pieces else np.zeros((0, 4, 2))


def vectorize_layer(
    gray: np.ndarray, *, model_fn: "ModelFn | None" = None, compute=None, weights_path=None,
    tile: int = TILE_PX, overlap: int = TILE_OVERLAP_PX, debug: bool = False,
    desc: str = "ImplicitSketchVec tiles",
) -> LayerResult:
    predict = _make_predict(model_fn, compute, weights_path)
    h0, w0 = gray.shape
    h, w = max(h0, tile), max(w0, tile)
    if (h, w) != (h0, w0):
        padded = np.full((h, w), 255, np.uint8)
        padded[:h0, :w0] = gray
        gray = padded
    ink = gray < INK_GRAY_THRESHOLD
    jobs = [(x0, y0) for x0, y0 in geo.tile_grid(h, w, tile, overlap)
            if int(ink[y0:y0 + tile, x0:x0 + tile].sum()) >= MIN_TILE_INK_PX]

    cores_x = geo.core_spans(w, tile, overlap)
    cores_y = geo.core_spans(h, tile, overlap)
    sp = _Sparse()
    heat = np.zeros((h, w), np.uint8) if debug else None
    with tqdm(total=len(jobs), desc=desc, unit="tile", leave=False, disable=not jobs) as bar:
        for b0 in range(0, len(jobs), INFER_BATCH_TILES):
            batch = jobs[b0:b0 + INFER_BATCH_TILES]
            fields = predict([np.ascontiguousarray(gray[y0:y0 + tile, x0:x0 + tile]) for x0, y0 in batch])
            for (x0, y0), f in zip(batch, fields):
                _harvest(f, x0, y0, cores_x[x0], cores_y[y0], sp, heat, debug)
            bar.update(len(batch))

    def cat(parts):
        return np.concatenate(parts) if parts else np.zeros((0, 2), np.int64)

    right, bottom = cat(sp.right), cat(sp.bottom)
    keypoints = {k: np.array(v, float).reshape(-1, 2) for k, v in sp.keypoints.items()}
    g = pp.build_graph(right, bottom, sp.vertex)
    pp.repair_breaks(g)
    regions = pp.refine_topology(g, sp.usm, np.concatenate(list(keypoints.values())))
    g = pp.downsample(g, DC_DOWNSAMPLE, sp.usm)
    pp.split_crossings(g)
    polylines = pp.group_lines(g, MIN_STROKE_PX)
    strokes = [s for s in (smooth(p) for p in polylines) if len(s)]
    return LayerResult(
        strokes=strokes, polylines=polylines, tiles=[(x0, y0, tile) for x0, y0 in jobs], keypoints=keypoints,
        usm_cells=sp.usm, regions=regions,
        raw_segments=_segments(cat(sp.raw_right), cat(sp.raw_bottom)) if debug else [],
        graph_segments=g.segments() if debug else [],
        heat=heat[:h0, :w0] if heat is not None else None,
    )
