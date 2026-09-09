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

import contextlib
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, TypeVar

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
}


def worker_init() -> None:
    """`ProcessPoolExecutor`/`Pool` initializer -- pin BLAS/OMP threads to 1,
    then warm this worker's own model caches immediately, before it's handed
    any job. Without this, a worker builds its models lazily on whichever
    job reaches it first -- harmless for Pool 1 (a page task always runs
    `fast` before `ocr`), but a real bug for Pool 2, where FAST and OCR jobs
    from many concurrent pages interleave arbitrarily across workers, so a
    worker can receive an OCR job first and `import paddle` before `import
    torch` (the exact Windows DLL-clash `warmup()` exists to prevent)."""
    for key, value in _WORKER_ENV.items():
        os.environ.setdefault(key, value)
    warmup()


def default_worker_count() -> int:
    """A conservative default: half the CPUs, capped at 4 (each worker
    holds its own PaddleOCR + torch model, so memory, not cores, is the
    limit). Advisory -- callers pass their own `workers`."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def warmup() -> None:
    """Build this process's model caches -- called once in the parent before
    a pool spawns (so a pool spawned next finds PaddleOCR's models on disk
    and no worker races another over the first-run download), and again in
    every worker via `worker_init` (so a worker's own in-memory engine is
    built at startup, in the safe torch-then-paddle order, instead of
    lazily on whichever job happens to reach it first). Cheap no-op once the
    caches / on-disk models exist."""
    # FAST (torch) first, then PaddleOCR (paddle): on Windows a paddle-first
    # process fails torch's later DLL load (clashing OpenMP runtimes).
    try:
        from rastervec.OCR.fast_detect import FastDetector

        FastDetector().warmup()
    except Exception as exc:  # noqa: BLE001 -- warmup is best-effort
        _LOG.warning("FAST warmup skipped: %s", exc)
    try:
        from rastervec.OCR.Paddle_OCR.ocr_backend import PaddleRecBackend

        PaddleRecBackend.warmup()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("PaddleOCR rec warmup skipped: %s", exc)


_PROGRESS_POLL_SECONDS = 0.5


@contextlib.contextmanager
def _progress_postfix(pbar, counter) -> Iterator[None]:
    """While the `with` block runs, polls `counter.value` (a
    `multiprocessing.managers.ValueProxy` or similar) every
    `_PROGRESS_POLL_SECONDS` and shows it as `pbar`'s postfix -- one shared
    counter surfaced on the single outer bar, instead of each worker
    opening its own `tqdm` bar and garbling shared stdout (or, once a Pool-2
    compute proxy is involved, no progress being visible at all -- see
    `OCR/fast_detect.py::detect_tiled`'s own docstring). A no-op when
    `counter` is `None` (the default -- callers that don't opt in see
    exactly today's behavior)."""
    if counter is None:
        yield
        return
    stop = threading.Event()

    def _poll() -> None:
        while not stop.wait(_PROGRESS_POLL_SECONDS):
            pbar.set_postfix_str(f"{counter.value} compute job(s) done", refresh=True)

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()
    try:
        yield
    finally:
        stop.set()
        poller.join()
        pbar.set_postfix_str(f"{counter.value} compute job(s) done", refresh=True)


def run_parallel(
    items: Iterable[_T],
    fn: Callable[[_T], _R],
    *,
    workers: int = 1,
    desc: str = "",
    warmup_first: bool = True,
    progress_counter=None,
) -> list[_R]:
    """Map `fn` over `items`, returning results in input order.

    `workers <= 1`: plain serial loop. Otherwise a spawn `ProcessPoolExecutor`
    of `workers` processes (BLAS threads pinned to 1 each); `warmup()` runs
    once in the parent first when `warmup_first`.

    `progress_counter`, when given a shared counter (e.g. a
    `multiprocessing.Manager().Value` -- see `benchmark_jobs.run_benchmark`),
    is polled and shown as this function's own `tqdm` bar's postfix (see
    `_progress_postfix`) -- `fn` itself is responsible for incrementing it as
    it does within-item work. `None` (the default) preserves today's
    behavior exactly."""
    work = list(items)
    if workers <= 1 or len(work) <= 1:
        with tqdm(work, desc=desc) as pbar, _progress_postfix(pbar, progress_counter):
            return [fn(x) for x in pbar]

    if warmup_first:
        _LOG.info("warming model caches before spawning %d workers", workers)
        warmup()

    ctx = multiprocessing.get_context("spawn")
    results: list[_R | None] = [None] * len(work)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=worker_init,
    ) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(work)}
        with (
            tqdm(as_completed(futures), total=len(futures), desc=desc) as pbar,
            _progress_postfix(pbar, progress_counter),
        ):
            for future in pbar:
                results[futures[future]] = future.result()
    return results  # type: ignore[return-value]


@contextlib.contextmanager
def compute_pool(workers: int) -> Iterator[object | None]:
    """Yield a `multiprocessing.Manager`-hosted `Pool` proxy (Pool 2) of
    `workers` processes for FAST tile detection + OCR crop recognition, or
    `None` when `workers <= 0` (fully-local, today's default behaviour).

    The proxy is picklable, so it can be threaded into Pool-1 spawn workers
    and shared by every page job (see `benchmark_jobs.run_benchmark`); a
    single-process caller (e.g. a notebook running one page) can use it too.
    `warmup()` runs first in this (parent) process so no worker races
    another over the first-run model download, and again in every worker via
    `initializer=worker_init` -- otherwise a Pool-2 worker builds its models
    lazily on whichever job reaches it first, and since FAST/OCR jobs from
    many concurrent pages interleave arbitrarily across workers, a worker
    could receive an OCR job before ever seeing a FAST one and `import
    paddle` before `import torch` (the Windows DLL-clash `warmup()` guards
    against)."""
    if workers <= 0:
        yield None
        return

    warmup()
    manager = multiprocessing.Manager()
    pool = manager.Pool(processes=workers, initializer=worker_init)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()
        manager.shutdown()
