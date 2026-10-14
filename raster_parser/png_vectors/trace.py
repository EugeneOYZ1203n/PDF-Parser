"""PNG -> SVG -> PDF -> vectors, for `extract_vectors_from_png`.

pyautotrace traces the PNG to SVG; PyMuPDF converts the SVG to PDF; each `get_drawings()` path
becomes a `TraceVector`, split into one per continuous stroke and scaled from SVG units to PDF
points. `draw_vectors` replays vectors onto a PDF (used for the alpha / transparency pass) and
`trace_vector_to_vector` gives the shared `Vector` this package returns.

Copied from the original raster_parser's `conversion/png_to_vectors/`, `vectors/`,
`transforms/`, `models/Vector_Class.py` (here `TraceVector`) and `rendering/pdf_render/`
(`draw_vectors`); the logic is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pymupdf as pymupdf
from autotrace import Bitmap, VectorFormat

from raster_parser.commons.helpers.fitz_geometry import plain_item
from raster_parser.commons.helpers.geometry import round_color
from raster_parser.commons.models import Vector


# --- from models/Vector_Class.py (the original raster_parser Vector, kept as this module's own
# trace record; adapter.py converts it to the shared Vector) ---

def _as_rect_tuple(r: Any) -> tuple[float, float, float, float]:
    """Normalize a MuPDF rect-like (or 4-tuple) to ``(x0, y0, x1, y1)``."""
    if r is None:
        return (0.0, 0.0, 0.0, 0.0)
    if hasattr(r, "x0") and hasattr(r, "y1"):
        return (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
    if isinstance(r, (list, tuple)) and len(r) >= 4:
        return (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
    return (0.0, 0.0, 0.0, 0.0)


def _as_line_cap(value: Any) -> Optional[Tuple[int, int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    if isinstance(value, (int, float)):
        i = int(value)
        return (i, i, i)
    return None


@dataclass
class TraceVector:
    """A single vector drawing (path) extracted from a page.

    Mirrors the dict shape returned by PyMuPDF's ``page.get_drawings()``,
    including fields present when ``extended=True`` (clips / groups):
    ``level``, ``scissor``, ``blendmode``, ``isolated``, ``knockout``, ``opacity``.
    Also maps stroke extras that appear in the base dict: ``lineCap``, ``lineJoin``.
    ``rect`` is a plain 4-tuple ``(x0, y0, x1, y1)``.
    """
    items: List[Any]
    type: str
    rect: tuple
    fill: Optional[tuple]
    color: Optional[tuple]
    width: float
    even_odd: bool
    close_path: bool
    dashes: str
    fill_opacity: float
    stroke_opacity: float
    seqno: int
    layer: Optional[str]
    # Stroke style (base get_drawings dict)
    line_cap: Optional[Tuple[int, int, int]] = None
    line_join: Optional[float] = None
    # extended=True extras (clips / transparency groups)
    level: int = 0
    scissor: Optional[tuple] = None
    blendmode: Optional[str] = None
    isolated: Optional[bool] = None
    knockout: Optional[bool] = None
    opacity: Optional[float] = None

    @classmethod
    def from_drawing(cls, d: Dict[str, Any]) -> "TraceVector":
        """Build a TraceVector from one entry of ``page.get_drawings([extended=True])``."""
        fill_op = d.get("fill_opacity")
        stroke_op = d.get("stroke_opacity")
        width = d.get("width")
        even_odd = d.get("even_odd")
        close_path = d.get("closePath")
        line_join = d.get("lineJoin")
        opacity = d.get("opacity")
        level = d.get("level")
        layer = d.get("layer")
        scissor = d.get("scissor")

        return cls(
            items=list(d.get("items") or []),
            type=d.get("type") or "",
            rect=_as_rect_tuple(d.get("rect")),
            fill=d.get("fill"),
            color=d.get("color"),
            width=float(width) if width is not None else 0.0,
            even_odd=bool(even_odd) if even_odd is not None else False,
            close_path=bool(close_path) if close_path is not None else False,
            dashes=(d.get("dashes") or "") if d.get("dashes") is not None else "",
            fill_opacity=float(fill_op) if fill_op is not None else 1.0,
            stroke_opacity=float(stroke_op) if stroke_op is not None else 1.0,
            seqno=int(d.get("seqno") or 0),
            layer=layer if layer else None,
            line_cap=_as_line_cap(d.get("lineCap")),
            line_join=float(line_join) if line_join is not None else None,
            level=int(level) if level is not None else 0,
            scissor=_as_rect_tuple(scissor) if scissor is not None else None,
            blendmode=d.get("blendmode"),
            isolated=d.get("isolated"),
            knockout=d.get("knockout"),
            opacity=float(opacity) if opacity is not None else None,
        )

    def coords_to_dict(self) -> Dict[str, Any]:
        return {
            "x0": self.rect[0],
            "y0": self.rect[1],
            "x1": self.rect[2],
            "y1": self.rect[3],
        }


# --- from rendering/pdf_render/reconstruct.py ---

def _finish_kwargs_reconstruct(path: dict) -> dict:
    kwargs: dict = {
        "width": path.get("width") or 0,
        "even_odd": path.get("even_odd", False),
        "closePath": path.get("closePath", False),
        "lineJoin": path.get("lineJoin", 0),
        "stroke_opacity": path.get("stroke_opacity", 1),
        "fill_opacity": path.get("fill_opacity", 1),
    }
    fill = path.get("fill")
    if fill is not None:
        kwargs["fill"] = fill
    color = path.get("color")
    if color is not None:
        kwargs["color"] = color
    dashes = path.get("dashes")
    if dashes:
        kwargs["dashes"] = dashes
    line_cap = path.get("lineCap")
    if line_cap is not None:
        kwargs["lineCap"] = line_cap[0] if isinstance(line_cap, (list, tuple)) else line_cap
    return kwargs


def _replay_items(shape: pymupdf.Shape, items: list) -> None:
    for item in items:
        op = item[0]
        if op == "l":
            shape.draw_line(item[1], item[2])
        elif op == "re":
            shape.draw_rect(item[1])
        elif op == "qu":
            shape.draw_quad(item[1])
        elif op == "c":
            shape.draw_bezier(item[1], item[2], item[3], item[4])
        else:
            raise ValueError(f"unhandled drawing op {op!r}: {item!r}")


def drawing_from_vector(v: TraceVector) -> dict:
    path = {
        "type": v.type,
        "items": v.items,
        "fill": v.fill,
        "color": v.color,
        "dashes": v.dashes,
        "even_odd": v.even_odd,
        "closePath": v.close_path,
        "width": v.width,
        "stroke_opacity": v.stroke_opacity,
        "fill_opacity": v.fill_opacity,
        "seqno": v.seqno,
        "layer": v.layer,
        "level": v.level,
    }
    if v.line_cap is not None:
        path["lineCap"] = v.line_cap
    if v.line_join is not None:
        path["lineJoin"] = v.line_join
    if v.scissor is not None:
        path["scissor"] = v.scissor
    if v.blendmode is not None:
        path["blendmode"] = v.blendmode
    if v.isolated is not None:
        path["isolated"] = v.isolated
    if v.knockout is not None:
        path["knockout"] = v.knockout
    if v.opacity is not None:
        path["opacity"] = v.opacity
    return path


# --- from rendering/pdf_render/dto_reconstruct_pipeline.py ---

def _page_size_from_vectors(
    vectors: list[TraceVector],
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    margin: float = 20.0,
) -> tuple[float, float]:
    if page_width is not None and page_height is not None:
        return page_width, page_height

    max_x = 0.0
    max_y = 0.0
    for vector in vectors:
        max_x = max(max_x, float(vector.rect[2]))
        max_y = max(max_y, float(vector.rect[3]))

    width = page_width if page_width is not None else max(max_x + margin, 400.0)
    height = page_height if page_height is not None else max(max_y + margin, 400.0)
    return width, height


def draw_vectors(
    vectors: list[TraceVector],
    out_pdf: Path | str,
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    page_rotation: int = 0,
) -> Path:
    """Replay all vectors onto a new single-page PDF."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    width, height = _page_size_from_vectors(
        vectors,
        page_width=page_width,
        page_height=page_height,
    )
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    if page_rotation:
        page.set_rotation(page_rotation)

    shape = page.new_shape()
    for vector in vectors:
        path = drawing_from_vector(vector)
        _replay_items(shape, path["items"])
        shape.finish(**_finish_kwargs_reconstruct(path))
    shape.commit()

    doc.save(out_pdf)
    doc.close()
    return out_pdf.resolve()


# --- from vectors/vector_items_processing.py ---

# Join tolerance (SVG/PDF units before dpi scale; ~pixels from autotrace).
_STROKE_JOIN_EPS = 0.05


def _item_points(item: tuple) -> list[pymupdf.Point]:
    """give me every coordinate point involved in this segment" (used to compute bounding boxes)."""
    op = item[0]
    if op == "l":
        return [item[1], item[2]]
    if op == "c":
        return [item[1], item[2], item[3], item[4]]
    if op == "re":
        r = item[1]
        return [pymupdf.Point(r.x0, r.y0), pymupdf.Point(r.x1, r.y1)]
    if op == "qu":
        q = item[1]
        return [q.ul, q.ur, q.ll, q.lr]
    return []


def _item_start(item: tuple) -> pymupdf.Point | None:
    #"where does this segment begin"
    op = item[0]
    if op in ("l", "c"):
        return item[1]
    if op == "qu":
        return item[1].ul
    if op == "re":
        r = item[1]
        return pymupdf.Point(r.x0, r.y0)
    return None


def _item_end(item: tuple) -> pymupdf.Point | None:
    #"where does this segment end"
    op = item[0]
    if op == "l":
        return item[2]
    if op == "c":
        return item[4]
    if op == "qu":
        return item[1].lr
    if op == "re":
        r = item[1]
        return pymupdf.Point(r.x1, r.y1)
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


def _vectors_from_drawing(d: dict) -> list[TraceVector]:
    """One TraceVector per continuous stroke inside a MuPDF drawing dict."""
    base = TraceVector.from_drawing(d)
    parts = _split_items_into_strokes(base.items)
    if len(parts) <= 1:
        return [base]
    out: list[TraceVector] = []
    for i, items in enumerate(parts):
        out.append(
            TraceVector(
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
                seqno=base.seqno if i == 0 else base.seqno * 1_000_000 + i,
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


# --- from transforms/scale_pdf_vector_items.py ---

def _scale_point(p: pymupdf.Point, scale: float) -> pymupdf.Point:
    return pymupdf.Point(p.x * scale, p.y * scale)


def _scale_item(item: tuple, scale: float) -> tuple:
    """Scale one get_drawings() path segment into PDF-point space."""
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
        return ("re", pymupdf.Rect(r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale))
    if op == "qu":
        q = item[1]
        return (
            "qu",
            pymupdf.Quad(
                _scale_point(q.ul, scale),
                _scale_point(q.ur, scale),
                _scale_point(q.ll, scale),
                _scale_point(q.lr, scale),
            ),
        )
    return item


def _scale_vector(v: TraceVector, scale: float) -> TraceVector:
    """Map a TraceVector from SVG-PDF pixel-as-point space into PDF points (``72/dpi``)."""
    if scale == 1.0:
        return v
    x0, y0, x1, y1 = v.rect
    scissor = None
    if v.scissor is not None:
        sx0, sy0, sx1, sy1 = v.scissor
        scissor = (sx0 * scale, sy0 * scale, sx1 * scale, sy1 * scale)
    return TraceVector(
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


# --- from conversion/png_to_vectors/pdf_to_vectors.py ---

def pdf_to_vectors(
    pdf_path: Path,
    *,
    page_index: int = 0,
    dpi: int | None = None,
    split_strokes: bool = True,
) -> list[TraceVector]:
    """Read ``get_drawings(extended=True)`` from a PDF and build ``TraceVector`` objects (Option A).

    ``split_strokes=True``: each continuous stroke becomes its own TraceVector (autotrace's
    multi-``M`` path is split). Use ``False`` when the paths are *filled* shapes whose
    subpaths are winding/even-odd holes — splitting those and filling each subpath
    independently would turn outlined regions into solid blobs.
    If ``dpi`` is set, coordinates are scaled by ``72/dpi`` so they match a page that was
    rendered at that DPI (autotrace's SVG/PDF is typically 1 SVG unit = 1 pixel).
    """
    doc = pymupdf.open(str(pdf_path))
    try:
        page = doc[page_index]
        raw = page.get_drawings(extended=True)
    finally:
        doc.close()
    vectors: list[TraceVector] = []
    for d in raw:
        if split_strokes:
            vectors.extend(_vectors_from_drawing(d))
        else:
            vectors.append(TraceVector.from_drawing(d))
    if dpi is not None and dpi > 0:
        scale = 72.0 / float(dpi)
        vectors = [_scale_vector(v, scale) for v in vectors]
    return vectors


# --- from conversion/png_to_vectors/png_to_svg.py ---

# Defaults tuned to keep small / thin linework (autotrace's own defaults are aggressive:
# noise-removal=0.99 and filter-iterations=4 erase fine AEC detail).
_DEFAULT_ERROR_THRESHOLD = 0.5


_DEFAULT_NOISE_REMOVAL = 0.2


_DEFAULT_FILTER_ITERATIONS = 1


def png_to_svg(
    inp: Path,
    out: Path,
    centerline: bool = False,
    color_count: int = 16,
    error_threshold: float = _DEFAULT_ERROR_THRESHOLD,
    corner_always_threshold: float = 80.0,
    corner_surround: int = 4,
    noise_removal: float = _DEFAULT_NOISE_REMOVAL,
    filter_iterations: int = _DEFAULT_FILTER_ITERATIONS,
    despeckle_level: int = 0,
    preserve_width: bool = True,
) -> None:
    """Vectorize ``inp`` PNG to ``out`` SVG via pyautotrace (``Bitmap.trace``)."""
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


# --- from conversion/png_to_vectors/svg_to_pdf.py ---

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


# --- TraceVector -> shared Vector (as the removed VectorDTO.from_vector + vector_from_dto did) ---

_STROKE_TYPES = ("s", "fs")
_FILL_TYPES = ("f", "fs")


def trace_vector_to_vector(tv: TraceVector, *, page_index: int = 0) -> Vector:
    """The shared `Vector` for one traced path: plain-tuple items, colours rounded to 3 decimals,
    stroke fields `None` on fill-only paths and fill opacity `None` on stroke-only ones."""
    stroked = tv.type in _STROKE_TYPES
    filled = tv.type in _FILL_TYPES
    return Vector(
        type=tv.type,
        items=[plain_item(item) for item in tv.items],
        color=round_color(tuple(float(c) for c in tv.color) if tv.color is not None else None),
        fill=round_color(tuple(float(c) for c in tv.fill) if tv.fill is not None else None),
        width=float(tv.width) if stroked else None,
        dashes=(tv.dashes or None) if stroked else None,
        closePath=bool(tv.close_path),
        lineCap=tv.line_cap if tv.line_cap is not None else 0,
        lineJoin=float(tv.line_join) if tv.line_join is not None else 0.0,
        even_odd=bool(tv.even_odd),
        stroke_opacity=float(tv.stroke_opacity) if stroked else None,
        fill_opacity=float(tv.fill_opacity) if filled else None,
        layer=tv.layer or None,
        rect=tuple(float(c) for c in tv.rect),
        scissor=tuple(float(c) for c in tv.scissor) if tv.scissor is not None else None,
        seqno=int(tv.seqno),
        blendmode=tv.blendmode,
        isolated=bool(tv.isolated),
        knockout=bool(tv.knockout),
        opacity=tv.opacity,
        page_index=page_index,
        level=int(tv.level),
    )
