"""`extract()` -- the library's entry point: one PDF page through the phase pipeline.

    native_read       native text, vectors and embedded images      (always)
    raster_vectorize  raster pass, `p2` (default v1_raster_to_vec)  (skipped when
                                                                    enable_raster_pass=False)
    vector_classify   vector parsing, `p3` (default v2_vector_class)  (always)
    output_assemble   combine everything                            (always)

See `raster_parser.core` for the pipeline and the backend registry.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from raster_parser.commons.logging_setup import get_logger
from raster_parser.models.result import ExtractResult
from raster_parser.core.registry import DEFAULT_VECTORIZE, DEFAULT_CLASSIFY

_LOG = get_logger("extract")


def _make_run_dir(dump_dir: str | Path | None) -> Path:
    """A new, empty directory for this run's intermediates: under ``dump_dir`` (created if needed),
    or under the system temp dir when ``dump_dir`` is None. Nothing already in ``dump_dir`` is touched."""
    if dump_dir is None:
        return Path(tempfile.mkdtemp(prefix="raster_parser_"))
    root = Path(dump_dir)
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="extract_", dir=root))


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
    p2: str = DEFAULT_VECTORIZE,
    p3: str = DEFAULT_CLASSIFY,
) -> ExtractResult:
    """Extract one page's text and vectors, as an `ExtractResult(page, texts, vectors)`.

    ``p2`` / ``p3`` pick the raster_vectorize (raster pass) and vector_classify (vector parsing) backends by
    name (``raster_parser.core.registry``); an unknown name raises ``ValueError``.
    ``enable_raster_pass=False`` skips raster_vectorize whatever ``p2`` is.
    ``ocr_dpi``: render DPI for OCR crops and the raster pass. ``use_angle_cls``: PaddleOCR
    angle classifier. ``workers``: process-pool size for the raster pass.
    ``dump_dir``: parent for this run's intermediate files (default: the system temp dir). Each call
    works in its own new sub-folder, so concurrent calls and files already in ``dump_dir`` are safe.
    ``debug``: when ``False`` (default) that sub-folder is removed afterwards; ``True`` keeps it.
    """
    from raster_parser.core.pipeline import run_pipeline
    from raster_parser.core.registry import resolve_vectorize, resolve_classify
    from raster_parser.models.settings import RunSettings

    resolve_vectorize(p2)  # a bad name fails even when the raster pass is off
    resolve_classify(p3)
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    run_dir = _make_run_dir(dump_dir)
    settings = RunSettings(
        ocr_dpi=ocr_dpi,
        use_angle_cls=use_angle_cls,
        workers=workers,
        dump_dir=run_dir,
        debug=debug,
    )
    try:
        return run_pipeline(
            pdf_path,
            page_index,
            p2=p2 if enable_raster_pass else "stub",
            p3=p3,
            settings=settings,
        )
    finally:
        if debug:
            _LOG.info("kept intermediates in %s", run_dir)
        else:
            shutil.rmtree(run_dir, ignore_errors=True)
