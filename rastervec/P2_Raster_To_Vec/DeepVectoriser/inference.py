"""Running the trained Liu et al. model over one ink layer.

`vectorize_layer(gray, ...)` -- `gray` is one color layer's mask (ink 0,
everything else 255) at the canonical scale (`config.TARGET_PX_PER_PT`):

  1. tile into `TILE_PX` tiles overlapping by `TILE_OVERLAP_PX`, skipping
     tiles with fewer than `MIN_TILE_INK_PX` ink pixels;
  2. predict every tile in batches (`model_fn`, or the local model, or Pool 2
     via `_infer_job` when `compute` is given);
  3. a *saturated* tile (>= `SATURATION_FRAC` of the encoder's queries
     confident -- more strokes than the model can list) is split into
     half-size sub-tiles and re-run, once;
  4. `geometry.merge_tiles` -- core ownership, seam snapping, joining.

The model is loaded lazily from `rastervec/weights/deep_vectoriser.pth`
(or `$DEEPVEC_WEIGHTS_PATH`) and cached per process; a missing file raises
`FileNotFoundError` naming where to put it (same convention as
`OCR/fast_detect.py::FastDetector._model`)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from rastervec.commons.logging_setup import get_logger

from . import geometry as geo
from .config import (
    CONFIDENCE_THRESHOLD, INFER_BATCH_TILES, INK_GRAY_THRESHOLD, MERGE_SNAP_PX, MIN_TILE_INK_PX,
    SATURATION_FRAC, TILE_OVERLAP_PX, TILE_PX, WEIGHTS_ENV_VAR, WEIGHTS_FILENAME,
)

_LOG = get_logger("P2.DeepVectoriser.infer")

_MODEL_CACHE: dict[str, object] = {}
# strokes per chunk through the UNet decode + vectorizer (bounds memory)
_STROKE_CHUNK = 256


@dataclass
class TilePrediction:
    """One tile's output: strokes `(K, 4, 2)` in the tile's own pixel frame."""
    strokes: list[np.ndarray] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    n_confident: int = 0
    n_queries: int = 1

    @property
    def saturated(self) -> bool:
        return self.n_confident >= SATURATION_FRAC * self.n_queries


ModelFn = Callable[[list[np.ndarray]], list[TilePrediction]]


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
                f"DeepVectoriser weights not found at {path}. Train them with "
                f"`python -m rastervec.P2_Raster_To_Vec.DeepVectoriser.train --data <prep dir> --out {path}` "
                f"(or point ${WEIGHTS_ENV_VAR} at an existing .pth)."
            )
        import torch  # noqa: F401  (lazy: keep torch off the import path of the pipeline)

        from .model.liu_model import load_weights

        model = load_weights(path, map_location="cpu")
        model.eval()
        _MODEL_CACHE[key] = model
        _LOG.info("DeepVectoriser model loaded from %s", path)
    return _MODEL_CACHE[key]


def warmup() -> None:
    """Best-effort model load (missing weights are skipped)."""
    try:
        load_model()
    except FileNotFoundError:
        pass


def predict_tiles(model, tiles: list[np.ndarray], conf_thresh: float = CONFIDENCE_THRESHOLD,
                  max_prims: int | None = None, enc: "dict | None" = None) -> list[TilePrediction]:
    """Same-size uint8 gray tiles -> one `TilePrediction` each. `enc` is the
    encoder's output for exactly these tiles when the caller already has it
    (training validation), saving a second encoder pass."""
    import torch

    from .model.layers import prepare_input

    if not tiles:
        return []
    device = next(model.parameters()).device
    with torch.no_grad():
        size_h, size_w = tiles[0].shape
        x = prepare_input(torch.from_numpy(np.stack(tiles)).to(device))
        if enc is None:
            enc = model.encoder(x)
        p = torch.sigmoid(enc["logit"])
        keep = p >= conf_thresh
        n_queries = int(p.shape[1])
        preds = [TilePrediction(n_confident=int(keep[i].sum()), n_queries=n_queries) for i in range(len(tiles))]
        img_idx, _slot = torch.nonzero(keep, as_tuple=True)
        if len(img_idx) == 0:
            return preds
        endpoints = enc["endpoints"][keep]
        cond = model.make_cond(endpoints, enc["emb"][keep], p[keep])
        skips = model.unet.encode(x)
        scale = torch.tensor([size_w, size_h] * 3, dtype=torch.float32, device=device)
        for c0 in range(0, len(img_idx), _STROKE_CHUNK):
            sl = slice(c0, c0 + _STROKE_CHUNK)
            feats = model.unet.decode(skips, cond[sl], img_idx[sl], stop_level=model.vec_level)
            mem = model.vectorizer.memory(feats)
            start = endpoints[sl, :2]
            curves = model.vectorizer.generate(mem, cond[sl], start, max_prims)
            for k, cv in enumerate(curves):
                i = int(img_idx[c0 + k])
                cv_px = (cv.float() * scale).cpu().numpy()            # (K, 6)
                s0 = (start[k].float() * scale[:2]).cpu().numpy()
                pieces = []
                prev = s0
                for row in cv_px:
                    c1, c2, e = row[0:2], row[2:4], row[4:6]
                    pieces.append(np.stack([prev, c1, c2, e]))
                    prev = e
                preds[i].strokes.append(np.stack(pieces))
                preds[i].confidences.append(float(p[keep][c0 + k]))
    return preds


def _infer_job(tiles: list[np.ndarray], weights_path: str) -> list[TilePrediction]:
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


@dataclass
class LayerResult:
    strokes: list[np.ndarray]                     # merged, layer pixel frame
    tiles: list[tuple[int, int, int]]             # (x0, y0, size) of every tile run
    resplit: list[tuple[int, int, int]]           # saturated tiles that were split
    raw: list[np.ndarray]                         # every tile's strokes before merging


def vectorize_layer(
    gray: np.ndarray, *, model_fn: "ModelFn | None" = None, compute=None, weights_path=None,
    tile: int = TILE_PX, overlap: int = TILE_OVERLAP_PX, desc: str = "DeepVectoriser tiles",
) -> LayerResult:
    predict = _make_predict(model_fn, compute, weights_path)
    h0, w0 = gray.shape
    # pad up to one tile so every tile is full size (white = no ink)
    h, w = max(h0, tile), max(w0, tile)
    if (h, w) != (h0, w0):
        padded = np.full((h, w), 255, np.uint8)
        padded[:h0, :w0] = gray
        gray = padded
    ink = gray < INK_GRAY_THRESHOLD

    jobs = []  # (x0, y0, size, core)
    for x0, y0 in geo.tile_grid(h, w, tile, overlap):
        if int(ink[y0:y0 + tile, x0:x0 + tile].sum()) >= MIN_TILE_INK_PX:
            jobs.append((x0, y0, tile, geo.core_rect(x0, y0, tile, h, w, overlap)))

    results: list[geo.TileStrokes] = []
    ran: list[tuple[int, int, int]] = []
    resplit: list[tuple[int, int, int]] = []
    raw: list[np.ndarray] = []
    sub_jobs = []

    def run(batch_jobs, allow_split: bool, bar) -> None:
        for b0 in range(0, len(batch_jobs), INFER_BATCH_TILES):
            batch = batch_jobs[b0:b0 + INFER_BATCH_TILES]
            preds = predict([gray[y0:y0 + s, x0:x0 + s] for x0, y0, s, _c in batch])
            for (x0, y0, s, core), pred in zip(batch, preds):
                if allow_split and pred.saturated and s // 2 >= 32:
                    resplit.append((x0, y0, s))
                    sub, sub_ov = s // 2, overlap // 2
                    for sx, sy in geo.tile_grid(s, s, sub, sub_ov):
                        sc = geo.core_rect(sx, sy, sub, s, s, sub_ov)
                        sc = (max(sc[0] + x0, core[0]), max(sc[1] + y0, core[1]),
                              min(sc[2] + x0, core[2]), min(sc[3] + y0, core[3]))
                        sub_jobs.append((x0 + sx, y0 + sy, sub, sc))
                    continue
                ran.append((x0, y0, s))
                strokes = [st + np.array([x0, y0], float) for st in pred.strokes]
                raw.extend(strokes)
                results.append(geo.TileStrokes(len(results), core, strokes))
            bar.update(len(batch))

    with tqdm(total=len(jobs), desc=desc, unit="tile", leave=False, disable=not jobs) as bar:
        run(jobs, True, bar)
    if sub_jobs:
        _LOG.debug("  %d saturated tile(s) re-split into %d sub-tiles", len(resplit), len(sub_jobs))
        with tqdm(total=len(sub_jobs), desc=desc + " (re-split)", unit="tile", leave=False) as bar:
            run(sub_jobs, False, bar)

    merged = geo.merge_tiles(results, MERGE_SNAP_PX)
    return LayerResult(strokes=merged, tiles=ran, resplit=resplit, raw=raw)
