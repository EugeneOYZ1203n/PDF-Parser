"""`RunSettings` -- the run options every Phase 2 / Phase 3 backend receives, whichever
backend it is. A backend reads the fields it needs and ignores the rest."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DUMP_DIR = Path("./tmp")


@dataclass(frozen=True)
class RunSettings:
    ocr_dpi: int = 300  # render DPI for OCR crops and raster passes
    use_angle_cls: bool = True  # PaddleOCR angle classifier
    workers: int = 1  # process-pool size for backends that parallelise
    dump_dir: Path = field(default=DEFAULT_DUMP_DIR)  # root for intermediate files
