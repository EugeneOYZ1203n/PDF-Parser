"""Parallelism for the benchmarking suite -- two pools:

- Pool 1 (`pool.py::run_parallel`, `workers`): a generic `spawn`
  process-pool map, one page's whole pipeline job
  (`benchmark_jobs.py::run_page_task`) per worker. Reads `fitz`, does
  native/vector extraction, classification, and orchestration.
- Pool 2 (`benchmark_jobs.py::run_benchmark`'s `compute_workers`, built
  via `pool.py::compute_pool` -- a `multiprocessing.Manager().Pool(...)`
  context manager reusable outside the benchmark, e.g. a notebook running
  one page through `run_pipeline(..., compute=...)`): a single shared pool
  every Pool-1 worker's page job dispatches its FAST tile detection and
  OCR crop recognition into -- proportionally, since it's one shared queue
  regardless of which page or which Pool-1 worker a job came from. Pool 2
  never imports `fitz`/`pymupdf`; its jobs (`OCR.fast_detect._detect_job`,
  `OCR.Paddle_OCR.ocr_backend._recognize_crops_job`) take only plain data
  (numpy arrays), never a `fitz`-backed object.

Why processes, never threads, for either pool: `PaddleRecBackend._ENGINE_CACHE`
and `FastDetector._MODEL_CACHE` are unlocked module-level singletons holding
engines that are not safe to call from multiple threads, and PyMuPDF is
not reentrant. Each worker process gets its own copies. Every Pool-1 worker
pins OMP / MKL / OpenBLAS to one thread so N workers do not oversubscribe the
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
    compute_pool,
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
    "compute_pool",
    "default_worker_count",
    "run_parallel",
    "warmup",
    "worker_init",
]
