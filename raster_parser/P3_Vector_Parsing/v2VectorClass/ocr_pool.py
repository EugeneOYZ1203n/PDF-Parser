"""`OcrPool` -- v2VectorClass's PaddleOCR calls, in this process or on a process pool.

With `RunSettings.workers > 1`, detection (one job per cluster render) and recognition (one job
per `OCR_BATCH_SIZE` batch, pass 1 and each retry rotation) run on one spawn
`ProcessPoolExecutor` for the page, opened the first time a call has more than one job. Each
worker builds its PaddleOCR engine once (`_worker_init`). Inputs are written as `.npz` under
`<dump_dir>/v2VectorClass/` and passed by path (the v1RasterToVec s3 / s4 pattern); the folder
is removed when the pool closes. Results come back in job order, and every batch is the same
list of crops as with `workers == 1`, so the output doesn't depend on `workers`.

With `workers <= 1`, or a call with a single job, the same backends run in this process.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from raster_parser.commons.progress import bar
from raster_parser.P3_Vector_Parsing.v2VectorClass.config import OCR_LANG, OCR_VERSION
from raster_parser.P3_Vector_Parsing.v2VectorClass.paddle_engine import (
    OcrBox,
    PaddleDetectBackend,
    PaddleRecBackend,
)

SPILL_FOLDER = "v2VectorClass"


def _worker_init(ocr_version: str, lang: str) -> None:
    """Limit BLAS threads and build this worker's PaddleOCR engine once."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    PaddleRecBackend.warmup(ocr_version, lang)


def _load(path: str) -> list[np.ndarray]:
    with np.load(path) as data:
        return [data[f"arr_{i}"] for i in range(len(data.files))]


def _detect_path_job(job: dict) -> list[np.ndarray]:
    """One cluster render (spilled) -> `PaddleDetectBackend.detect`."""
    (bgr,) = _load(job["path"])
    return PaddleDetectBackend(job["ocr_version"], job["lang"]).detect(bgr)


def _recognize_path_job(job: dict) -> list[OcrBox]:
    """One recognition batch (spilled) -> `recognize_crops`, or `recognize_crops_raw` when
    `job["raw"]`."""
    crops = _load(job["path"])
    backend = PaddleRecBackend(job["ocr_version"], job["lang"])
    return backend.recognize_crops_raw(crops) if job["raw"] else backend.recognize_crops(crops)


class OcrPool:
    """Detect / recognise through a process pool when `workers > 1` (see the module
    docstring). Use as a context manager; nothing is started until a call has more than one
    job."""

    def __init__(self, workers: int, dump_dir: "str | Path", *,
                 ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.workers = max(1, int(workers))
        self.dump_dir = Path(dump_dir)
        self.ocr_version = ocr_version
        self.lang = lang
        self._executor: ProcessPoolExecutor | None = None
        self._spill: Path | None = None
        self._n = 0

    def __enter__(self) -> "OcrPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._spill is not None:
            shutil.rmtree(self._spill, ignore_errors=True)
            self._spill = None

    def _parallel(self, n_jobs: int) -> bool:
        return self.workers > 1 and n_jobs > 1

    def _start(self) -> None:
        if self._executor is not None:
            return
        import multiprocessing as mp

        root = self.dump_dir / SPILL_FOLDER
        root.mkdir(parents=True, exist_ok=True)
        self._spill = Path(tempfile.mkdtemp(prefix="ocr_", dir=root))
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
            initargs=(self.ocr_version, self.lang),
        )

    def _spill_arrays(self, arrays: list[np.ndarray]) -> str:
        path = self._spill / f"j{self._n:06d}.npz"
        self._n += 1
        np.savez(path, *arrays)
        return str(path)

    def _run(self, fn, jobs: list[dict], desc: str) -> list:
        try:
            return list(bar(self._executor.map(fn, jobs), desc=desc, total=len(jobs)))
        finally:
            for job in jobs:
                Path(job["path"]).unlink(missing_ok=True)

    def detect(self, bgrs: list[np.ndarray]) -> list[list[np.ndarray]]:
        """One quad list per cluster render, in input order."""
        if not self._parallel(len(bgrs)):
            backend = PaddleDetectBackend(self.ocr_version, self.lang)
            return [backend.detect(bgr) for bgr in bar(bgrs, desc="detect clusters")]
        self._start()
        jobs = [{"path": self._spill_arrays([bgr]), "ocr_version": self.ocr_version, "lang": self.lang}
                for bgr in bgrs]
        return self._run(_detect_path_job, jobs, "detect clusters")

    def recognize(self, batches: list[list[np.ndarray]], *, raw: bool) -> list[list[OcrBox]]:
        """One `OcrBox` list per batch, in input order: `recognize_crops` (angle classifier,
        then recognise), or `recognize_crops_raw` when `raw`."""
        if not self._parallel(len(batches)):
            backend = PaddleRecBackend(self.ocr_version, self.lang)
            fn = backend.recognize_crops_raw if raw else backend.recognize_crops
            return [fn(batch) for batch in bar(batches, desc="recognise batches")]
        self._start()
        jobs = [{"path": self._spill_arrays(batch), "raw": raw,
                 "ocr_version": self.ocr_version, "lang": self.lang} for batch in batches]
        return self._run(_recognize_path_job, jobs, "recognise batches")
