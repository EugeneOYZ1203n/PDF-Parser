"""Running the trained Egiazarian et al. pipeline over one ink layer.

`vectorize_layer(gray, ...)` -- `gray` is one color layer's mask (ink 0,
everything else 255) at the canonical scale (`config.TARGET_PX_PER_PT`):

  1. split into `PATCH_PX` patches overlapping by `PATCH_OVERLAP_PX`
     (paper: 64 x 64), skipping patches with fewer than `MIN_PATCH_INK_PX`
     ink pixels;
  2. estimate every patch's primitives with the network, in batches
     (paper Sec. 3.2), keeping those with confidence >= 0.5;
  3. refine them against the patch raster (`refine.py`, Sec. 3.3);
  4. move them into the layer frame, clip each patch's primitives to the
     part of the patch it owns (its core rect) so the overlap isn't counted
     twice;
  5. merge (`merge.py`, Sec. 3.4) when `config.MERGE`.

Steps 2-3 run in-process, or on Pool 2 via `_infer_job` when `compute` is
given. The model is loaded lazily from `rastervec/weights/
deep_tech_vec_line.pth` (or `$DEEPTECHVEC_WEIGHTS_PATH`) and cached per
process; a missing file raises `FileNotFoundError` naming where to put it.
A checkpoint knows its own primitive kind (line / curve)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from rastervec.commons.logging_setup import get_logger

from . import geometry as geo
from .config import (
    CONFIDENCE_THRESHOLD, DEFAULT_PRIM_KIND, INFER_BATCH_PATCHES, INFER_THREADS, INK_GRAY_THRESHOLD, MERGE,
    MIN_PATCH_INK_PX, MIN_PRIM_LEN_PX, PATCH_OVERLAP_PX, PATCH_PX, REFINE_ITERS, WEIGHTS_ENV_VAR, WEIGHTS_FILENAMES,
)

_LOG = get_logger("P2.DeepTechVec.infer")

_MODEL_CACHE: dict[str, object] = {}

# patches -> one list of primitives (patch px) per patch
ModelFn = Callable[[list[np.ndarray]], list[list[geo.Prim]]]


def default_weights_path(kind: str = DEFAULT_PRIM_KIND) -> Path:
    env = os.environ.get(WEIGHTS_ENV_VAR)
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "weights" / WEIGHTS_FILENAMES[kind]


def _for_cpu_inference(model):
    """Eval mode with frozen parameters (no autograd bookkeeping), plus the
    `INFER_THREADS` setting for this process."""
    import torch

    if INFER_THREADS:
        torch.set_num_threads(INFER_THREADS)
    model.eval()
    model.requires_grad_(False)
    return model


def load_model(path: "str | Path | None" = None):
    path = Path(path) if path is not None else default_weights_path()
    key = str(path)
    if key not in _MODEL_CACHE:
        if not path.is_file():
            raise FileNotFoundError(
                f"DeepTechVec weights not found at {path}. Train them with "
                f"`python -m rastervec.P2_Raster_To_Vec.DeepTechVec.train --data <DeepVectoriser prep dir> "
                f"--prim line --out {path}` (or point ${WEIGHTS_ENV_VAR} at an existing .pth)."
            )
        import torch  # noqa: F401  (lazy: keep torch off the import path of the pipeline)

        from .model.network import load_weights

        model = _for_cpu_inference(load_weights(path, map_location="cpu"))
        _MODEL_CACHE[key] = model
        _LOG.info("DeepTechVec model (%s) loaded from %s", model.kind, path)
    return _MODEL_CACHE[key]


def warmup() -> None:
    """Best-effort model load (missing weights are skipped)."""
    try:
        load_model()
    except FileNotFoundError:
        pass


def predict_patches(model, patches: list[np.ndarray], conf_thresh: float = CONFIDENCE_THRESHOLD) -> list[list[geo.Prim]]:
    """Same-size uint8 gray patches -> primitives (patch px) per patch."""
    import torch

    from .model.network import prepare_input
    from .train_data import decode_output

    if not patches:
        return []
    size = patches[0].shape[0]
    device = next(model.parameters()).device
    with torch.inference_mode():
        out = model(prepare_input(torch.from_numpy(np.stack(patches)).to(device))).cpu().numpy()
    return [decode_output(row, model.kind, size, conf_thresh) for row in out]


def _infer_job(patches: list[np.ndarray], weights_path: str, refine_iters: int):
    """Pool-2 job: plain data in, plain data out (no fitz). Returns
    `(raw, refined)` primitive lists per patch."""
    from .refine import refine_patches

    raw = predict_patches(load_model(weights_path), patches)
    return raw, refine_patches(patches, raw, iters=refine_iters)


def _make_run(model_fn: "ModelFn | None", compute, weights_path, refine_iters: int):
    """`patches -> (raw, refined)`."""
    from .refine import refine_patches

    if model_fn is not None:
        def run_fake(patches):
            raw = model_fn(patches)
            return raw, refine_patches(patches, raw, iters=refine_iters)
        return run_fake
    path = str(weights_path or default_weights_path())
    if compute is not None:
        if not Path(path).is_file():
            load_model(path)  # raises the FileNotFoundError with its hint, locally
        return lambda patches: compute.apply(_infer_job, (patches, path, refine_iters))
    return lambda patches: _infer_job(patches, path, refine_iters)


@dataclass
class LayerResult:
    prims: list[geo.Prim]                         # final, layer pixel frame
    patches: list[tuple[int, int, int]]           # (x0, y0, size) of every patch run
    raw: list[geo.Prim]                           # network output, layer frame, before refinement
    refined: list[geo.Prim]                       # after refinement + core clipping, before merging


def vectorize_layer(
    gray: np.ndarray, *, model_fn: "ModelFn | None" = None, compute=None, weights_path=None,
    patch: int = PATCH_PX, overlap: int = PATCH_OVERLAP_PX, refine_iters: int = REFINE_ITERS,
    merge: bool = MERGE, desc: str = "DeepTechVec patches",
) -> LayerResult:
    from .merge import merge_prims

    run = _make_run(model_fn, compute, weights_path, refine_iters)
    h0, w0 = gray.shape
    # pad up to one patch so every patch is full size (white = no ink)
    h, w = max(h0, patch), max(w0, patch)
    if (h, w) != (h0, w0):
        padded = np.full((h, w), 255, np.uint8)
        padded[:h0, :w0] = gray
        gray = padded
    ink = gray < INK_GRAY_THRESHOLD
    jobs = [(x0, y0) for x0, y0 in geo.tile_grid(h, w, patch, overlap)
            if int(ink[y0:y0 + patch, x0:x0 + patch].sum()) >= MIN_PATCH_INK_PX]

    raw_all: list[geo.Prim] = []
    refined_all: list[geo.Prim] = []
    with tqdm(total=len(jobs), desc=desc, unit="patch", leave=False, disable=not jobs) as bar:
        for b0 in range(0, len(jobs), INFER_BATCH_PATCHES):
            batch = jobs[b0:b0 + INFER_BATCH_PATCHES]
            raw, refined = run([np.ascontiguousarray(gray[y0:y0 + patch, x0:x0 + patch]) for x0, y0 in batch])
            for (x0, y0), r_ps, f_ps in zip(batch, raw, refined):
                off = np.array([x0, y0], float)
                core = geo.core_rect(x0, y0, patch, h, w, overlap)
                raw_all.extend(geo.Prim(np.asarray(p.pts, float) + off, p.width, p.conf) for p in r_ps)
                for p in f_ps:
                    moved = geo.Prim(np.asarray(p.pts, float) + off, p.width, p.conf)
                    refined_all.extend(geo.clip_prim(moved, core, min_len=MIN_PRIM_LEN_PX))
            bar.update(len(batch))
    final = merge_prims(refined_all) if merge else list(refined_all)
    return LayerResult(prims=final, patches=[(x0, y0, patch) for x0, y0 in jobs], raw=raw_all,
                       refined=refined_all)
