"""Adapter: runs archive's legacy pipeline (`raster_parser.main_pipeline_extract.
extract`) completely unmodified and reshapes its `NativePDFElements` output into
rastervec's own `Text` shape so `metrics.evaluate_metrics` can score it on the
exact same metrics as the current pipeline -- see
`rastervec/notebooks/benchmark_vector_classification.ipynb`.

Archive is a plain sibling folder under the repo root (not an installed
package), so `_ensure_archive_importable` adds its path to `sys.path` lazily,
only when this module's functions are actually called -- nothing about
archive's own code is touched, copied, or reimplemented here, only its
*output shape* is translated. Using `archive/` at all is logged as a warning
(once per process), and any legacy-run failure is logged and re-raised, never
swallowed.

Archive's `raster_parser` needs **PaddleOCR 2.x** (now the repo default -- see
`OCR/Paddle_OCR/ocr_backend.py`) and **LibreOffice on PATH** (`import
raster_parser` launches a LibreOffice subprocess at import time). No
compatibility shim is installed. `run_archive_pipeline` is a manual smoke test
only; `to_cluster_ocr_results` is pure and unit-tested against a hand-built
archive-shaped object.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rastervec.helpers.geometry import compute_origin
from rastervec.logging_setup import get_logger
from rastervec.models import Text

_LOG = get_logger("eval.legacy_adapter")

if TYPE_CHECKING:
    from raster_parser.models import NativePDFElements as ArchiveNativePDFElements

_ARCHIVE_ROOT = Path(__file__).resolve().parents[3] / "archive"


class _ArchiveTextDTO(Protocol):
    x0: float
    y0: float
    x1: float
    y1: float
    word: str
    rotate: int


class _ArchiveVectorDTO(Protocol):
    x0: float
    y0: float
    x1: float
    y1: float
    fill: tuple[float, ...] | None
    color: tuple[float, ...] | None
    width: float
    dashes: str


class _ArchiveNativePDFElements(Protocol):
    words: list[_ArchiveTextDTO]
    vectors: list[_ArchiveVectorDTO]


def _ensure_archive_importable() -> None:
    """Adds the repo-root `archive/` folder to `sys.path` (once), so
    `import raster_parser...` resolves against archive's own tree -- archive
    has no `setup.py`/`pyproject.toml`, it's imported as a plain path root.

    No compatibility shim is installed: the repo now runs PaddleOCR 2.x, the
    API surface archive's OCR was written against. Nothing in `archive/` is
    modified. Logs a warning the first time `archive/` is put on the path."""
    root_str = str(_ARCHIVE_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        _LOG.warning(
            "Using archive/ (legacy pipeline): needs PaddleOCR 2.x (now the repo "
            "default) and LibreOffice on PATH; nothing in archive/ is modified."
        )


def run_archive_pipeline(
    pdf_path: str, page_index: int = 0, **extract_kwargs: Any,
) -> "ArchiveNativePDFElements":
    """Thin call-through to archive's own `main_pipeline_extract.extract` --
    nothing about archive's internals is touched or copied. `extract_kwargs`
    forwards straight to archive's own signature (`ocr_dpi`,
    `enable_raster_pass`, `workers`, `dump_dir`, `debug`, ...).

    Any failure (archive import, LibreOffice, PaddleOCR) is logged and
    re-raised -- never swallowed into an empty result."""
    _ensure_archive_importable()
    try:
        from raster_parser.main_pipeline_extract import extract

        return extract(pdf_path, page_index, **extract_kwargs)
    except Exception as exc:
        _LOG.warning("legacy pipeline failed: %s", exc)
        raise


def to_texts(
    elements: "_ArchiveNativePDFElements", page_index: int = 0,
) -> list[Text]:
    """Wraps each archive `TextDTO` word as a rastervec `Text` (source=
    "ocr") so `evaluate_metrics` scores it identically to the current
    pipeline's own OCR readings. `confidence` defaults to 1.0 since
    archive's `TextDTO` doesn't carry one; `direction` is derived from
    `rotate` (a quarter-turn int, archive's own precision)."""
    import math

    results: list[Text] = []
    for seq, word in enumerate(elements.words):
        bbox = (word.x0, word.y0, word.x1, word.y1)
        angle = math.radians(word.rotate)
        direction = (math.cos(angle), math.sin(angle))
        results.append(Text(
            text=word.word, bbox=bbox, direction=direction,
            origin=compute_origin(bbox, direction),
            font="", font_size=0.0, color=None, flags=0,
            ascender=None, descender=None, wmode=0,
            block_no=0, line_no=0, word_no=0,
            page_index=page_index, seqno=seq, confidence=1.0, source="ocr",
            orientation_source="ocr",
        ))
    return results


