"""Parallelism for the benchmarking suite: a generic `spawn` process-pool
map (`pool.py`) and the benchmark's picklable per-page job
(`benchmark_jobs.py`).

Why processes, never threads: `PaddleOcrBackend._ENGINE_CACHE` and
`FastDetector._MODEL_CACHE` are unlocked module-level singletons holding
engines that are not safe to call from multiple threads, and PyMuPDF is
not reentrant. Each worker process gets its own copies. Every worker pins
OMP / MKL / OpenBLAS to one thread so N workers do not oversubscribe the
CPU (mirrors `archive/raster_parser`'s `*_worker_init`).
"""
from rastervec.Reader.Parallel.benchmark_jobs import (
    PageResult,
    PageTask,
    ShowcaseSample,
    run_benchmark,
    run_page_task,
)
from rastervec.Reader.Parallel.pool import (
    default_worker_count,
    run_parallel,
    warmup,
    worker_init,
)

__all__ = [
    "PageResult",
    "PageTask",
    "ShowcaseSample",
    "run_benchmark",
    "run_page_task",
    "default_worker_count",
    "run_parallel",
    "warmup",
    "worker_init",
]
