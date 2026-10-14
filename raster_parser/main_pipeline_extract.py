"""`extract()` -- the library's entry point: one PDF page through the phase pipeline.

    Phase 1  native text, vectors and embedded images                   (always)
    Phase 2  raster pass, `p2` (default v1RasterToVec)                  (skipped when
                                                                         enable_raster_pass=False)
    Phase 3  vector parsing, `p3` (default v1VectorClass)               (always)
    Phase 4  combine everything                                          (always)

See `raster_parser.core` for the pipeline and the backend registry.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from raster_parser.commons.models import ExtractResult
from raster_parser.core.registry import DEFAULT_P2, DEFAULT_P3


def _resolve_dump_dir(
    dump_dir: str | Path | None,
) -> Path:
    """Directory for intermediates (LibreOffice, raster pass); default is ``./tmp`` (CWD-relative)."""
    from raster_parser.core.settings import DEFAULT_DUMP_DIR

    root = Path(dump_dir) if dump_dir is not None else DEFAULT_DUMP_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cleanup_dump_dir(dump_root: Path) -> None:
    """Remove the dump directory and all intermediates under it."""
    if dump_root.is_dir():
        shutil.rmtree(dump_root)


def extract(
    pdf_path: str | Path,
    page_index: int = 0,
    *,
    ocr_dpi: int = 300,
    use_angle_cls: bool = True,
    enable_raster_pass: bool = True,
    workers: int = 1,
    dump_dir: str | Path | None = None,
    debug: bool = False,
    p2: str = DEFAULT_P2,
    p3: str = DEFAULT_P3,
) -> ExtractResult:
    """Extract one page's text and vectors, as an `ExtractResult(page, texts, vectors)`.

    ``p2`` / ``p3`` pick the Phase 2 (raster pass) and Phase 3 (vector parsing) backends by
    name (``raster_parser.core.registry``); an unknown name raises ``ValueError``.
    ``enable_raster_pass=False`` skips Phase 2 whatever ``p2`` is.
    ``ocr_dpi``: render DPI for OCR crops and the raster pass. ``use_angle_cls``: PaddleOCR
    angle classifier. ``workers``: process-pool size for the raster pass.
    ``dump_dir``: directory for intermediate files (default ``./tmp``).
    ``debug``: when ``False`` (default) ``dump_dir`` is removed afterwards; ``True`` keeps it.
    """
    from raster_parser.core.pipeline import run_pipeline
    from raster_parser.core.registry import resolve_p2, resolve_p3
    from raster_parser.core.settings import RunSettings

    resolve_p2(p2)  # a bad name fails even when the raster pass is off
    resolve_p3(p3)
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    dump_root = _resolve_dump_dir(dump_dir)
    settings = RunSettings(ocr_dpi=ocr_dpi, use_angle_cls=use_angle_cls, workers=workers, dump_dir=dump_root)
    try:
        result = run_pipeline(
            pdf_path, page_index, p2=p2 if enable_raster_pass else "Stub", p3=p3, settings=settings,
        )
        return ExtractResult(page=result.page.meta, texts=result.texts, vectors=result.vectors)
    finally:
        if not debug:
            _cleanup_dump_dir(dump_root)
