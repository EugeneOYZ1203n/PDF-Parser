"""`RunSettings` -- the run options every raster_vectorize / vector_classify backend receives, whichever
backend it is. A backend reads the fields it needs and ignores the rest.

`on_debug_layer` / `on_debug_image` are debug hooks for tools (the pipeline report). They never
change a backend's output: with both left `None` (as `extract()` does) a backend renders and
keeps nothing extra. See `docs/ARCHITECTURE.md`, "Debug output".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

# (stage, label, "#rrggbb", one-page PDF bytes) -- one debug layer, called as soon as it is rendered.
DebugLayerSink = Callable[[str, str, str, bytes], None]
# (folder, name, make_image) -- one debug image; `make_image()` is only called if the tool keeps it.
DebugImageSink = Callable[[str, str, "Callable[[], Image.Image]"], None]


@dataclass(frozen=True)
class RunSettings:
    ocr_dpi: int = 300  # render DPI for OCR crops and raster passes
    use_angle_cls: bool = True  # PaddleOCR angle classifier
    workers: int = 1  # process-pool size for backends that parallelise
    dump_dir: Path | None = None  # root for intermediate files; None = each step's own default
    debug: bool = False  # also write debug images / dumps (else only what the result needs)
    on_debug_layer: DebugLayerSink | None = None  # debug only, see the module docstring
    on_debug_image: DebugImageSink | None = None  # debug only, see the module docstring
