"""`RunSettings` -- the run options every raster_vectorize / vector_classify backend receives, whichever
backend it is. A backend reads the fields it needs and ignores the rest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSettings:
    ocr_dpi: int = 300  # render DPI for OCR crops and raster passes
    use_angle_cls: bool = True  # PaddleOCR angle classifier
    workers: int = 1  # process-pool size for backends that parallelise
    dump_dir: Path | None = None  # root for intermediate files; None = each step's own default
    debug: bool = False  # also write debug images / dumps (else only what the result needs)
