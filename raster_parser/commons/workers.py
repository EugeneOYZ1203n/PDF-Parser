"""Process-pool plumbing shared by the steps that fan work out to worker processes.

Workers are always spawned (never forked: PaddleOCR and OpenCV hold threads and locks a fork
would copy half-initialised), and each one pins its BLAS / OpenMP pools to one thread so N
workers don't each start N threads.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def pin_blas_threads(*, paddle: bool = False) -> None:
    """Default every BLAS / OpenMP thread count to 1 (values already set are kept). `paddle`
    also turns OneDNN off: its multi-threaded inference has been seen to SIGSEGV."""
    for name in BLAS_THREAD_VARS:
        os.environ.setdefault(name, "1")
    if paddle:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")


def spawn_pool(
    max_workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
) -> ProcessPoolExecutor:
    """A `ProcessPoolExecutor` whose workers are spawned processes."""
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )
