"""Autotrace a drawing PNG into ``list[Vector]``.

Flow:
  1. ``pyautotrace`` (``autotrace.Bitmap.trace``) -> SVG (outline by default; optional
     centerline; low noise-removal / light smoothing so small details are not despeckled away).
  2. PyMuPDF SVG -> PDF.
  3. ``page.get_drawings()`` -> ``Vector.from_drawing``, then **split** each drawing into
     one Vector per continuous stroke (autotrace emits one giant ``<path>`` with many ``M``
     subpaths; MuPDF keeps that as a single drawing).

Masks from colour-fill autotrace are already binary; this module does not binarize.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pymupdf as pymupdf
from autotrace import Bitmap, VectorFormat

from raster_parser.conversion.png_to_vectors import pdf_to_vectors, svg_to_pdf
from raster_parser.conversion.png_to_vectors import png_to_svg
from raster_parser.models import Vector

# Join tolerance (SVG/PDF units before dpi scale; ~pixels from autotrace).
_STROKE_JOIN_EPS = 0.05

# Defaults tuned to keep small / thin linework (autotrace's own defaults are aggressive:
# noise-removal=0.99 and filter-iterations=4 erase fine AEC detail).
_DEFAULT_ERROR_THRESHOLD = 0.5
_DEFAULT_NOISE_REMOVAL = 0.2
_DEFAULT_FILTER_ITERATIONS = 1















def autotrace_drawing(
    png_path: str | Path,
    *,
    centerline: bool = False,
    color_count: int = 2,
    error_threshold: float = _DEFAULT_ERROR_THRESHOLD,
    corner_always_threshold: float = 80.0,
    corner_surround: int = 4,
    noise_removal: float = _DEFAULT_NOISE_REMOVAL,
    filter_iterations: int = _DEFAULT_FILTER_ITERATIONS,
    despeckle_level: int = 0,
    preserve_width: bool = True,
    dpi: int | None = None,
    svg_out: str | Path | None = None,
    pdf_out: str | Path | None = None,
    scratch_dir: str | Path | None = None,
    split_strokes: bool = True,
) -> list[Vector]:
    """Trace a drawing PNG with autotrace and return ``list[Vector]``.

    PNG -> SVG -> PDF -> ``get_drawings`` -> Vector. No soft/adaptive binarize;
    pass an already binary (or otherwise ready) PNG.

    ``svg_out`` / ``pdf_out``: if given, those files are kept; otherwise temps are deleted.
    ``dpi``: if given, scale Vector coords from pixel-space into PDF points (``72/dpi``).
    ``scratch_dir``: parent for throwaway intermediates (temp svg/pdf when not kept).
    ``split_strokes``: one Vector per continuous stroke (linework default). Pass ``False``
    for filled shapes so subpath holes stay inside one Vector (see ``pdf_to_vectors``).

    Detail knobs (defaults prefer recall of thin linework over aggressive cleanup):
      ``noise_removal``     autotrace default is 0.99 (drops small marks); we use 0.2
      ``filter_iterations`` autotrace default is 4 (over-smooths); we use 1
      ``preserve_width``    keep stroke widths under ``--centerline``
    """
    src = Path(png_path)
    if not src.is_file():
        raise FileNotFoundError(f"PNG not found: {src}")

    if scratch_dir is not None:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix="autotrace_",
            dir=str(scratch_dir) if scratch_dir is not None else None,
        )
    )
    keep_svg = svg_out is not None
    keep_pdf = pdf_out is not None
    svg_path = Path(svg_out) if keep_svg else tmp_dir / (src.stem + ".svg")
    pdf_path = Path(pdf_out) if keep_pdf else tmp_dir / (src.stem + ".pdf")

    try:
        png_to_svg(
            src,
            svg_path,
            centerline=centerline,
            color_count=color_count,
            error_threshold=error_threshold,
            corner_always_threshold=corner_always_threshold,
            corner_surround=corner_surround,
            noise_removal=noise_removal,
            filter_iterations=filter_iterations,
            despeckle_level=despeckle_level,
            preserve_width=preserve_width,
        )
        svg_to_pdf(svg_path, pdf_path)
        return pdf_to_vectors(pdf_path, dpi=dpi, split_strokes=split_strokes)
    finally:
        if not keep_svg:
            svg_path.unlink(missing_ok=True)
        if not keep_pdf:
            pdf_path.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

