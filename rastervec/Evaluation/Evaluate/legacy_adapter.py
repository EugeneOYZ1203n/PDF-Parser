"""Adapter: runs archive's legacy pipeline (`raster_parser.main_pipeline_extract.
extract`) completely unmodified and reshapes its `NativePDFElements` output into
rastervec's own `ClusterOcrResult`/`DrawingVector` shapes so `evaluate.
evaluate_pipeline` can score it on the exact same metrics as the current
pipeline -- see `rastervec/notebooks/benchmark_vector_classification.ipynb`.

Archive is a plain sibling folder under the repo root (not an installed
package), so `_ensure_archive_importable` adds its path to `sys.path` lazily,
only when this module's functions are actually called -- nothing about
archive's own code is touched, copied, or reimplemented here, only its
*output shape* is translated. `run_archive_pipeline` is a manual smoke test
only (real archive dependency chain: PaddleOCR, LibreOffice, autotrace --
same "not unit-testable" convention as `manual_label.py`/`benchmark.py`'s own
OCR-backed paths); `to_cluster_ocr_results`/`to_drawing_vectors` are pure and
unit-tested against a hand-built archive-shaped object.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rastervec.models import ClusterOcrResult, DrawingVector, TextVectorResult

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
    has no `setup.py`/`pyproject.toml`, it's imported as a plain path root."""
    root_str = str(_ARCHIVE_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def run_archive_pipeline(
    pdf_path: str, page_index: int = 0, **extract_kwargs: Any,
) -> "ArchiveNativePDFElements":
    """Thin call-through to archive's own `main_pipeline_extract.extract` --
    nothing about archive's internals is touched or copied. `extract_kwargs`
    forwards straight to archive's own signature (`ocr_dpi`,
    `enable_raster_pass`, `workers`, `dump_dir`, `debug`, ...)."""
    _ensure_archive_importable()
    from raster_parser.main_pipeline_extract import extract

    return extract(pdf_path, page_index, **extract_kwargs)


def to_cluster_ocr_results(
    elements: "_ArchiveNativePDFElements", page_index: int = 0,
) -> list[ClusterOcrResult]:
    """Wraps each archive `TextDTO` word as a rastervec `ClusterOcrResult` so
    `evaluate_pipeline` scores it identically to the current pipeline's own
    OCR readings. `cluster`/`paths` are left empty -- archive's `TextDTO`
    carries no back-reference to source vector geometry, and
    `evaluate_pipeline` only ever reads `ClusterOcrResult.resolved` for
    scoring, never `.cluster`. `confidence` defaults to 1.0 since archive's
    `TextDTO` doesn't carry one."""
    results: list[ClusterOcrResult] = []
    for word in elements.words:
        resolved = TextVectorResult(
            paths=[],
            text=word.word,
            confidence=1.0,
            bbox=(word.x0, word.y0, word.x1, word.y1),
            ocr_bbox=None,
            rotation_used=word.rotate,
            page_index=page_index,
        )
        results.append(ClusterOcrResult(cluster=[], resolved=resolved, ocr_seconds=0.0))
    return results


def to_drawing_vectors(
    elements: "_ArchiveNativePDFElements", page_index: int = 0,
) -> list[DrawingVector]:
    """Archive's own leftover (non-text) vectors, wrapped minimally as
    `DrawingVector`s -- `evaluate_pipeline` only ever reads
    `len(drawing_vectors)`, so full path-level fidelity isn't needed here."""
    results: list[DrawingVector] = []
    for vector in elements.vectors:
        results.append(
            DrawingVector(
                paths=[],
                bbox=(vector.x0, vector.y0, vector.x1, vector.y1),
                stroke_color=vector.color,
                fill_color=vector.fill,
                stroke_width=vector.width,
                dashed=bool(vector.dashes),
                page_index=page_index,
            )
        )
    return results
