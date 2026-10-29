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
import pymupdf as fitz
from autotrace import Bitmap, VectorFormat

from raster_parser.conversion.png_to_vectors.png_to_svg import png_to_svg
from raster_parser.conversion.png_to_vectors.svg_to_pdf import svg_to_pdf
from raster_parser.helpers.geometry import PDF_POINTS_PER_INCH
from raster_parser.models import Vector

# Join tolerance (SVG/PDF units before dpi scale; ~pixels from autotrace).
_STROKE_JOIN_EPS = 0.05

# Defaults tuned to keep small / thin linework (autotrace's own defaults are aggressive:
# noise-removal=0.99 and filter-iterations=4 erase fine AEC detail).
# error_threshold: how far (px) a fitted curve may stray from the traced outline before
# autotrace splits it into more pieces.
_DEFAULT_ERROR_THRESHOLD = 0.5
_DEFAULT_NOISE_REMOVAL = 0.2
_DEFAULT_FILTER_ITERATIONS = 1
# A bend sharper than this many degrees is always a corner (autotrace's own default).
_DEFAULT_CORNER_ALWAYS_THRESHOLD_DEG = 80.0
# How many outline pixels on each side of a point autotrace looks at to decide whether it
# is a corner (autotrace's own default).
_DEFAULT_CORNER_SURROUND_PX = 4
# A drawing split into several strokes keeps its seqno on the first stroke; stroke i > 0
# gets seqno * stride + i, which stays unique and keeps the drawing's paint order.
_SUBSTROKE_SEQNO_STRIDE = 1_000_000


def _scale_point(p: fitz.Point, scale: float) -> fitz.Point:
    return fitz.Point(p.x * scale, p.y * scale)


def _scale_item(item: tuple, scale: float) -> tuple:
    """Scale one get_drawings() path segment into PDF-point space.

    Op codes: ``"l"`` line, ``"c"`` cubic Bezier curve, ``"re"`` rectangle, ``"qu"``
    quadrilateral (corners ul/ur/ll/lr = upper-left, upper-right, lower-left, lower-right).
    """
    op = item[0]
    if op == "l":
        return ("l", _scale_point(item[1], scale), _scale_point(item[2], scale))
    if op == "c":
        return (
            "c",
            _scale_point(item[1], scale),
            _scale_point(item[2], scale),
            _scale_point(item[3], scale),
            _scale_point(item[4], scale),
        )
    if op == "re":
        r = item[1]
        return ("re", fitz.Rect(r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale))
    if op == "qu":
        q = item[1]
        return (
            "qu",
            fitz.Quad(
                _scale_point(q.ul, scale),
                _scale_point(q.ur, scale),
                _scale_point(q.ll, scale),
                _scale_point(q.lr, scale),
            ),
        )
    return item


def _scale_vector(v: Vector, scale: float) -> Vector:
    """Map a Vector from SVG-PDF pixel-as-point space into PDF points (``72/dpi``).

    ``scissor`` is the clip rectangle PyMuPDF reports for the path (``extended=True``).
    """
    if scale == 1.0:
        return v
    x0, y0, x1, y1 = v.rect
    scissor = None
    if v.scissor is not None:
        sx0, sy0, sx1, sy1 = v.scissor
        scissor = (sx0 * scale, sy0 * scale, sx1 * scale, sy1 * scale)
    return Vector(
        items=[_scale_item(it, scale) for it in v.items],
        type=v.type,
        rect=(x0 * scale, y0 * scale, x1 * scale, y1 * scale),
        fill=v.fill,
        color=v.color,
        width=v.width * scale,
        even_odd=v.even_odd,
        close_path=v.close_path,
        dashes=v.dashes,
        fill_opacity=v.fill_opacity,
        stroke_opacity=v.stroke_opacity,
        seqno=v.seqno,
        layer=v.layer,
        line_cap=v.line_cap,
        line_join=v.line_join,
        level=v.level,
        scissor=scissor,
        blendmode=v.blendmode,
        isolated=v.isolated,
        knockout=v.knockout,
        opacity=v.opacity,
    )


def _item_points(item: tuple) -> list[fitz.Point]:
    """give me every coordinate point involved in this segment" (used to compute bounding boxes)."""
    op = item[0]
    if op == "l":
        return [item[1], item[2]]
    if op == "c":
        return [item[1], item[2], item[3], item[4]]
    if op == "re":
        r = item[1]
        return [fitz.Point(r.x0, r.y0), fitz.Point(r.x1, r.y1)]
    if op == "qu":
        q = item[1]
        return [q.ul, q.ur, q.ll, q.lr]
    return []


def _item_start(item: tuple) -> fitz.Point | None:
    # "where does this segment begin"
    op = item[0]
    if op in ("l", "c"):
        return item[1]
    if op == "qu":
        return item[1].ul
    if op == "re":
        r = item[1]
        return fitz.Point(r.x0, r.y0)
    return None


def _item_end(item: tuple) -> fitz.Point | None:
    # "where does this segment end"
    op = item[0]
    if op == "l":
        return item[2]
    if op == "c":
        return item[4]
    if op == "qu":
        return item[1].lr
    if op == "re":
        r = item[1]
        return fitz.Point(r.x1, r.y1)
    return None


def _items_rect(items: list) -> tuple[float, float, float, float]:
    # folds a whole list of segments down into one (min_x, min_y, max_x, max_y) bounding box
    xs: list[float] = []
    ys: list[float] = []
    for it in items:
        for p in _item_points(it):
            xs.append(float(p.x))
            ys.append(float(p.y))
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _split_items_into_strokes(items: list, *, join_eps: float = _STROKE_JOIN_EPS) -> list[list]:
    """Split a flat get_drawings() ``items`` list into continuous strokes.

    Autotrace writes one SVG ``<path d="M...M...M...">`` with a moveto per stroke.
    MuPDF folds that into a single drawing whose segments jump whenever a new ``M``
    starts. A new stroke begins when the next segment's start is not near the previous
    end (or the op is a standalone ``re``/``qu``).
    """
    if not items:
        return []
    strokes: list[list] = [[items[0]]]
    for it in items[1:]:
        op = it[0]
        # Rects / quads are closed primitives; always their own stroke.
        if op in ("re", "qu") or strokes[-1][-1][0] in ("re", "qu"):
            strokes.append([it])
            continue
        prev_end = _item_end(strokes[-1][-1])
        cur_start = _item_start(it)
        if prev_end is None or cur_start is None:
            strokes.append([it])
            continue
        dx = float(cur_start.x) - float(prev_end.x)
        dy = float(cur_start.y) - float(prev_end.y)
        if dx * dx + dy * dy > join_eps * join_eps:
            strokes.append([it])
        else:
            strokes[-1].append(it)
    return strokes


def _vectors_from_drawing(d: dict) -> list[Vector]:
    """One Vector per continuous stroke inside a MuPDF drawing dict."""
    base = Vector.from_drawing(d)
    parts = _split_items_into_strokes(base.items)
    if len(parts) <= 1:
        return [base]
    out: list[Vector] = []
    for i, items in enumerate(parts):
        out.append(
            Vector(
                items=items,
                type=base.type,
                rect=_items_rect(items),
                fill=base.fill,
                color=base.color,
                width=base.width,
                even_odd=base.even_odd,
                close_path=base.close_path,
                dashes=base.dashes,
                fill_opacity=base.fill_opacity,
                stroke_opacity=base.stroke_opacity,
                # Preserve drawing seqno; sub-index via local offset so they're distinct.
                seqno=base.seqno if i == 0 else base.seqno * _SUBSTROKE_SEQNO_STRIDE + i,
                layer=base.layer,
                line_cap=base.line_cap,
                line_join=base.line_join,
                level=base.level,
                scissor=base.scissor,
                blendmode=base.blendmode,
                isolated=base.isolated,
                knockout=base.knockout,
                opacity=base.opacity,
            )
        )
    return out


def pdf_to_vectors(
    pdf_path: Path,
    *,
    page_index: int = 0,
    dpi: int | None = None,
    split_strokes: bool = True,
) -> list[Vector]:
    """Read ``get_drawings(extended=True)`` from a PDF and build ``Vector`` objects (Option A).

    ``split_strokes=True``: each continuous stroke becomes its own Vector (autotrace's
    multi-``M`` path is split). Use ``False`` when the paths are *filled* shapes whose
    subpaths are winding/even-odd holes — splitting those and filling each subpath
    independently would turn outlined regions into solid blobs.
    If ``dpi`` is set, coordinates are scaled by ``72/dpi`` so they match a page that was
    rendered at that DPI (autotrace's SVG/PDF is typically 1 SVG unit = 1 pixel).
    """
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        raw = page.get_drawings(extended=True)
    finally:
        doc.close()
    vectors: list[Vector] = []
    for d in raw:
        if split_strokes:
            vectors.extend(_vectors_from_drawing(d))
        else:
            vectors.append(Vector.from_drawing(d))
    if dpi is not None and dpi > 0:
        scale = PDF_POINTS_PER_INCH / float(dpi)
        vectors = [_scale_vector(v, scale) for v in vectors]
    return vectors


def autotrace_drawing(
    png_path: str | Path,
    *,
    centerline: bool = False,
    color_count: int = 2,
    error_threshold: float = _DEFAULT_ERROR_THRESHOLD,
    corner_always_threshold: float = _DEFAULT_CORNER_ALWAYS_THRESHOLD_DEG,
    corner_surround: int = _DEFAULT_CORNER_SURROUND_PX,
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
