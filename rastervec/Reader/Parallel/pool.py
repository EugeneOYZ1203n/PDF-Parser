"""Generic `spawn` process-pool map with a serial fallback, plus the
per-worker environment pinning and the model-cache warmup that make the
pool safe to use on the very first run.

`run_parallel(items, fn, workers=N)` returns `[fn(x) for x in items]` in
input order -- serially when `workers <= 1` (byte-for-byte the old
behaviour), otherwise across a `ProcessPoolExecutor`. `fn` must be a
top-level importable callable (spawn pickles it by qualified name) and
should catch its own per-item errors.
"""
from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

from tqdm import tqdm

from rastervec.logging_setup import get_logger

_LOG = get_logger("reader.parallel")

_T = TypeVar("_T")
_R = TypeVar("_R")

# Pinned in every worker so N workers x all-cores-each doesn't oversubscribe
# the CPU. `setdefault` so an explicit outer value still wins.
_WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT": "False",
}


def worker_init() -> None:
    """`ProcessPoolExecutor` initializer -- pin BLAS/OMP threads to 1."""
    for key, value in _WORKER_ENV.items():
        os.environ.setdefault(key, value)


def default_worker_count() -> int:
    """A conservative default: half the CPUs, capped at 4 (each worker
    holds its own PaddleOCR + torch model, so memory, not cores, is the
    limit). Advisory -- callers pass their own `workers`."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def warmup() -> None:
    """Build the shared model caches once, here, in the calling process --
    so a pool spawned next finds PaddleOCR's models on disk (no worker
    races the first-run download) and every worker's own engine build is
    just a load. Cheap no-op once the caches / on-disk models exist."""
    try:
        from rastervec.OCR.Paddle_OCR.ocr_backend import PaddleOcrBackend

        PaddleOcrBackend.warmup()
    except Exception as exc:  # noqa: BLE001 -- warmup is best-effort
        _LOG.warning("PaddleOCR warmup skipped: %s", exc)
    try:
        from rastervec.OCR.FAST_Text_Detect.fast_detect import FastDetector

        FastDetector().warmup()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("FAST warmup skipped: %s", exc)


def run_parallel(
    items: Iterable[_T],
    fn: Callable[[_T], _R],
    *,
    workers: int = 1,
    desc: str = "",
    warmup_first: bool = True,
) -> list[_R]:
    """Map `fn` over `items`, returning results in input order.

    `workers <= 1`: plain serial loop. Otherwise a spawn `ProcessPoolExecutor`
    of `workers` processes (BLAS threads pinned to 1 each); `warmup()` runs
    once in the parent first when `warmup_first`."""
    work = list(items)
    if workers <= 1 or len(work) <= 1:
        return [fn(x) for x in tqdm(work, desc=desc)]

    if warmup_first:
        _LOG.info("warming model caches before spawning %d workers", workers)
        warmup()

    ctx = multiprocessing.get_context("spawn")
    results: list[_R | None] = [None] * len(work)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=worker_init,
    ) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(work)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            results[futures[future]] = future.result()
    return results  # type: ignore[return-value]
