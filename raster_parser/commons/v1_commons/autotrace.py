"""Autotrace a PNG into shared `Vector`s: pyautotrace -> SVG -> PDF (PyMuPDF) -> `get_drawings()`.

From the original `conversion/png_to_vectors/` (png_to_svg, svg_to_pdf, pdf_to_vectors),
`autotrace/drawing_autotrace.py`, `transforms/scale_pdf_vector_items.py` and
`vectors/vector_items_processing.py`, rewritten on `Vector` (plain-tuple items).

A traced `Vector` follows one rule (`traced_vector`): colours rounded to 3 decimals, stroke
fields (`width`, `dashes`, `stroke_opacity`) only on stroked paths and `fill_opacity` only on
filled ones, `closePath` a bool.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
import pymupdf

from raster_parser.models.source import Source
from raster_parser.commons.helpers.geometry import (
    PDF_POINTS_PER_INCH,
    item_bbox,
    item_points,
    round_color,
    union_bbox,
)
from raster_parser.models.vector import Vector

# PyMuPDF path types: "s" stroked, "f" filled, "fs" filled and stroked.
_STROKE_TYPES = ("s", "fs")
_FILL_TYPES = ("f", "fs")

# Join tolerance (SVG/PDF units before the dpi scale; ~pixels from autotrace).
_STROKE_JOIN_EPS = 0.05

# Defaults tuned to keep small / thin linework (autotrace's own defaults are aggressive:
# noise-removal=0.99 and filter-iterations=4 erase fine AEC detail).
# error_threshold: how far (px) a fitted curve may stray from the traced outline before
# autotrace splits it into more pieces.
DEFAULT_ERROR_THRESHOLD = 0.5
DEFAULT_NOISE_REMOVAL = 0.2
DEFAULT_FILTER_ITERATIONS = 1
# A bend sharper than this many degrees is always a corner (autotrace's own default).
DEFAULT_CORNER_ALWAYS_THRESHOLD_DEG = 80.0
# How many outline pixels on each side of a point autotrace looks at to decide whether it
# is a corner (autotrace's own default).
DEFAULT_CORNER_SURROUND_PX = 4
# A drawing split into several strokes keeps its seqno on the first stroke; stroke i > 0
# gets seqno * stride + i, which stays unique and keeps the drawing's paint order.
_SUBSTROKE_SEQNO_STRIDE = 1_000_000


def traced_vector(v: Vector) -> Vector:
    """`v` with the traced-vector rule applied (see the module docstring); idempotent."""
    stroked = v.type in _STROKE_TYPES
    filled = v.type in _FILL_TYPES
    return replace(
        v,
        color=round_color(v.color),
        fill=round_color(v.fill),
        width=float(v.width or 0.0) if stroked else None,
        dashes=(v.dashes or None) if stroked else None,
        closePath=bool(v.closePath),
        stroke_opacity=float(1.0 if v.stroke_opacity is None else v.stroke_opacity)
        if stroked
        else None,
        fill_opacity=float(1.0 if v.fill_opacity is None else v.fill_opacity) if filled else None,
        seqno=int(v.seqno or 0),
        source=Source.FROM_IMAGE,
    )


def _item_start(item: tuple):
    pts = item_points(item)
    return pts[0] if pts else None


def _item_end(item: tuple):
    pts = item_points(item)
    if not pts:
        return None
    return pts[2] if item[0] == "qu" else pts[-1]  # a quad ends at its lower-right corner


def _items_rect(items: list) -> tuple[float, float, float, float]:
    boxes = [item_bbox(it) for it in items if item_points(it)]
    return union_bbox(boxes) if boxes else (0.0, 0.0, 0.0, 0.0)


def split_items_into_strokes(items: list, *, join_eps: float = _STROKE_JOIN_EPS) -> list[list]:
    """Split a flat `items` list into continuous strokes.

    Autotrace writes one SVG `<path d="M...M...M...">` with a moveto per stroke; MuPDF folds that
    into a single drawing whose segments jump whenever a new `M` starts. A new stroke begins when
    the next segment's start is not near the previous end (or the op is a standalone `re`/`qu`).
    """
    if not items:
        return []
    strokes: list[list] = [[items[0]]]
    for it in items[1:]:
        # Rects / quads are closed primitives; always their own stroke.
        if it[0] in ("re", "qu") or strokes[-1][-1][0] in ("re", "qu"):
            strokes.append([it])
            continue
        prev_end = _item_end(strokes[-1][-1])
        cur_start = _item_start(it)
        if prev_end is None or cur_start is None:
            strokes.append([it])
            continue
        dx = float(cur_start[0]) - float(prev_end[0])
        dy = float(cur_start[1]) - float(prev_end[1])
        if dx * dx + dy * dy > join_eps * join_eps:
            strokes.append([it])
        else:
            strokes[-1].append(it)
    return strokes


def _vectors_from_drawing(base: Vector) -> list[Vector]:
    """One `Vector` per continuous stroke of `base`; parts after the first get a derived seqno."""
    parts = split_items_into_strokes(base.items)
    if len(parts) <= 1:
        return [base]
    return [
        replace(
            base,
            items=items,
            rect=_items_rect(items),
            seqno=base.seqno if i == 0 else base.seqno * _SUBSTROKE_SEQNO_STRIDE + i,
        )
        for i, items in enumerate(parts)
    ]


def _scale_item(item: tuple, s: float) -> tuple:
    """Scale one item: ``"l"`` line, ``"c"`` cubic Bezier, ``"re"`` rectangle, ``"qu"`` quad."""
    op = item[0]
    if op == "l":
        return ("l", (item[1][0] * s, item[1][1] * s), (item[2][0] * s, item[2][1] * s))
    if op == "c":
        return ("c", *[(p[0] * s, p[1] * s) for p in item[1:5]], *item[5:])
    if op == "re":
        x0, y0, x1, y1 = item[1]
        return ("re", (x0 * s, y0 * s, x1 * s, y1 * s), *item[2:])
    if op == "qu":
        return ("qu", tuple((p[0] * s, p[1] * s) for p in item[1]))
    return item


def _scale_vector(v: Vector, s: float) -> Vector:
    """Map `v` from SVG-PDF pixel-as-point space into PDF points (`s = 72/dpi`)."""
    if s == 1.0:
        return v
    x0, y0, x1, y1 = v.rect
    return replace(
        v,
        items=[_scale_item(it, s) for it in v.items],
        rect=(x0 * s, y0 * s, x1 * s, y1 * s),
        scissor=tuple(c * s for c in v.scissor) if v.scissor is not None else None,
        width=v.width * s if v.width is not None else None,
    )


def pdf_to_vectors(
    pdf_path: Path,
    *,
    page_index: int = 0,
    dpi: int | None = None,
    split_strokes: bool = True,
) -> list[Vector]:
    """Traced `Vector`s from `get_drawings(extended=True)` of one PDF page.

    `split_strokes=True`: each continuous stroke becomes its own `Vector` (autotrace's
    multi-`M` path is split). Use `False` when the paths are *filled* shapes whose subpaths are
    winding/even-odd holes -- filling each subpath on its own would turn outlined regions into
    solid blobs. With `dpi`, coordinates are scaled by `72/dpi` to match a page rendered at that
    DPI (autotrace's SVG/PDF is 1 unit = 1 pixel).
    """
    doc = pymupdf.open(str(pdf_path))
    try:
        raw = doc[page_index].get_drawings(extended=True)
    finally:
        doc.close()
    vectors: list[Vector] = []
    for d in raw:
        base = traced_vector(Vector.from_pymupdf(d, page_index=0, seq=0))
        vectors.extend(_vectors_from_drawing(base) if split_strokes else [base])
    if dpi is not None and dpi > 0:
        scale = PDF_POINTS_PER_INCH / float(dpi)
        vectors = [_scale_vector(v, scale) for v in vectors]
    return vectors


def png_to_svg(
    inp: Path,
    out: Path,
    centerline: bool = False,
    color_count: int = 16,
    error_threshold: float = DEFAULT_ERROR_THRESHOLD,
    corner_always_threshold: float = DEFAULT_CORNER_ALWAYS_THRESHOLD_DEG,
    corner_surround: int = DEFAULT_CORNER_SURROUND_PX,
    noise_removal: float = DEFAULT_NOISE_REMOVAL,
    filter_iterations: int = DEFAULT_FILTER_ITERATIONS,
    despeckle_level: int = 0,
    preserve_width: bool = True,
) -> None:
    """Vectorize `inp` PNG to `out` SVG via pyautotrace (`Bitmap.trace`)."""
    from autotrace import Bitmap, VectorFormat

    inp = Path(inp)
    out = Path(out)
    if not inp.is_file():
        raise FileNotFoundError(f"image not found: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)

    # pyautotrace Bitmap expects HxWx3 RGB uint8.
    bgr = cv2.imread(str(inp), cv2.IMREAD_COLOR)
    if bgr is None:
        gray = cv2.imread(str(inp), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"could not read image: {inp}")
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        traced = Bitmap(rgb).trace(
            color_count=color_count,
            error_threshold=error_threshold,
            corner_always_threshold=corner_always_threshold,
            corner_surround=corner_surround,
            noise_removal=noise_removal,
            filter_iterations=filter_iterations,
            despeckle_level=despeckle_level,
            centerline=centerline,
            preserve_width=preserve_width if centerline else False,
        )
        traced.save(out, format=VectorFormat.SVG)
    except Exception as e:
        raise RuntimeError(f"pyautotrace failed on {inp}: {e}") from e


def svg_to_pdf(svg_path: Path, pdf_path: Path) -> None:
    """Convert an SVG file to PDF using PyMuPDF."""
    doc = pymupdf.open(str(svg_path))
    try:
        pdf_bytes = doc.convert_to_pdf()
    finally:
        doc.close()
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)


def autotrace_drawing(
    png_path: str | Path,
    *,
    centerline: bool = False,
    color_count: int = 2,
    error_threshold: float = DEFAULT_ERROR_THRESHOLD,
    corner_always_threshold: float = DEFAULT_CORNER_ALWAYS_THRESHOLD_DEG,
    corner_surround: int = DEFAULT_CORNER_SURROUND_PX,
    noise_removal: float = DEFAULT_NOISE_REMOVAL,
    filter_iterations: int = DEFAULT_FILTER_ITERATIONS,
    despeckle_level: int = 0,
    preserve_width: bool = True,
    dpi: int | None = None,
    svg_out: str | Path | None = None,
    pdf_out: str | Path | None = None,
    scratch_dir: str | Path | None = None,
    split_strokes: bool = True,
) -> list[Vector]:
    """Trace a drawing PNG with autotrace into traced `Vector`s.

    PNG -> SVG -> PDF -> `get_drawings` -> `Vector`. No binarisation; pass a ready PNG.

    `svg_out` / `pdf_out`: if given, those files are kept; otherwise temps are deleted.
    `dpi`: if given, coordinates are scaled from pixel-space into PDF points (`72/dpi`).
    `scratch_dir`: parent for throwaway intermediates. `split_strokes`: one `Vector` per
    continuous stroke (linework default); pass `False` for filled shapes so subpath holes stay
    inside one `Vector` (see `pdf_to_vectors`).

    Detail knobs (defaults prefer recall of thin linework over aggressive cleanup):
      `noise_removal`     autotrace default is 0.99 (drops small marks); we use 0.2
      `filter_iterations` autotrace default is 4 (over-smooths); we use 1
      `preserve_width`    keep stroke widths under `centerline`
    """
    src = Path(png_path)
    if not src.is_file():
        raise FileNotFoundError(f"PNG not found: {src}")

    if scratch_dir is not None:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix="autotrace_", dir=str(scratch_dir) if scratch_dir else None)
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
