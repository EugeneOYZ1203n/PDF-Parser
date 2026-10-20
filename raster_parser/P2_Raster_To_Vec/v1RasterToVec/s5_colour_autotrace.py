"""Step 5 -- colour-fill autotrace of the drawing layer.

1. Inventory non-background RGB; reserve black + white clusters for near-B/W ink.
2. Greedy-group the rest in HSV space; exclusive nearest-centroid masks.
3. Outline-autotrace each non-white mask (pyautotrace -> SVG -> PDF -> `get_drawings`, each
   drawing split into one vector per continuous stroke); drop solid page washes (high cover +
   high solidity/Q); keep thin page-frame paths and small white hole covers. Stamp remaining ink.
4. Write artefacts under ``{stem}_colours/``.

Copied from `autotrace/drawing_colour_fill_autotrace.py`, `autotrace/drawing_autotrace.py`,
`conversion/png_to_vectors/` (png_to_svg, svg_to_pdf, pdf_to_vectors),
`transforms/scale_pdf_vector_items.py`, `vectors/vector_items_processing.py`,
`helpers/geometry.py` (`_pt` / `_item_points` renamed `_geom_pt` / `_geom_item_points`),
`serialisation/deserialization.py` (`deserialize_vector`), `serialisation/serialization.py`,
`models/Vector_Class.py` (the original `Vector`, here `TraceVector`) and
`rendering/pdf_render/dto_reconstruct_pipeline.py` + `reconstruct.py` (`draw_vectors`).
Only the reached definitions; the logic is unchanged.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf as pymupdf
from autotrace import Bitmap, VectorFormat

from raster_parser.commons.helpers.fitz_geometry import plain_item
from raster_parser.commons.helpers.geometry import round_color
from raster_parser.commons.models import Vector
from raster_parser.commons.progress import bar
from pydantic import BaseModel, Field


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


# --- from serialisation/serialization.py ---

def _quad_to_json(value: Any) -> list[list[float]]:
    return [
        [float(value.ul.x), float(value.ul.y)],
        [float(value.ur.x), float(value.ur.y)],
        [float(value.ll.x), float(value.ll.y)],
        [float(value.lr.x), float(value.lr.y)],
    ]


def _to_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "ul") and hasattr(value, "ur") and hasattr(value, "ll") and hasattr(value, "lr"):
        return _quad_to_json(value)
    if hasattr(value, "x0") and hasattr(value, "y1"):
        return [float(value.x0), float(value.y0), float(value.x1), float(value.y1)]
    if hasattr(value, "x") and hasattr(value, "y"):
        return [float(value.x), float(value.y)]
    if isinstance(value, (list, tuple)):
        return [_to_json_value(v) for v in value]
    return str(value)


def drawing_item_to_dict(item: tuple | list) -> dict[str, Any]:
    """One ``get_drawings()`` path segment as a JSON-safe dict."""
    op = item[0]
    return {"op": op, "args": [_to_json_value(v) for v in item[1:]]}


def drawing_items_to_dicts(items: list[Any]) -> list[dict[str, Any]]:
    return [drawing_item_to_dict(item) for item in items]


# --- from models/VectorDTO_Class.py (fields + `VectorDTO.from_vector`) ---

class TraceVectorRecord(BaseModel):
    """JSON form of a `TraceVector` for this module's vector dumps (`*_vectors.json`), which
    `colour_fill_autotrace` writes and reads back. Same fields as the removed `VectorDTO`."""

    type: str
    x0: float
    y0: float
    x1: float
    y1: float
    items: List[dict[str, Any]] = Field(default_factory=list)
    fill: tuple[float, float, float] | None = None
    color: tuple[float, float, float] | None = None
    width: float = 0.0
    even_odd: bool = False
    close_path: bool = False
    dashes: str = ""
    fill_opacity: float = 1.0
    stroke_opacity: float = 1.0
    seqno: int = 0
    layer: str | None = None
    line_cap: tuple[int, int, int] | None = None
    line_join: float | None = None
    level: int = 0
    scissor: tuple[float, float, float, float] | None = None
    blendmode: str | None = None
    isolated: bool | None = None
    knockout: bool | None = None
    opacity: float | None = None



def trace_vector_to_record(vector: TraceVector) -> TraceVectorRecord:
    """`VectorDTO.from_vector` for a `TraceVector`."""
    fill = vector.fill
    color = vector.color
    return TraceVectorRecord(
        type=vector.type,
        x0=float(vector.rect[0]),
        y0=float(vector.rect[1]),
        x1=float(vector.rect[2]),
        y1=float(vector.rect[3]),
        items=drawing_items_to_dicts(vector.items),
        fill=tuple(float(c) for c in fill) if fill is not None else None,
        color=tuple(float(c) for c in color) if color is not None else None,
        width=vector.width,
        even_odd=vector.even_odd,
        close_path=vector.close_path,
        dashes=vector.dashes,
        fill_opacity=vector.fill_opacity,
        stroke_opacity=vector.stroke_opacity,
        seqno=vector.seqno,
        layer=vector.layer,
        line_cap=vector.line_cap,
        line_join=vector.line_join,
        level=vector.level,
        scissor=tuple(float(x) for x in vector.scissor)
        if vector.scissor is not None
        else None,
        blendmode=vector.blendmode,
        isolated=vector.isolated,
        knockout=vector.knockout,
        opacity=vector.opacity,
    )


# --- from helpers/geometry.py ---

_JOIN_EPS = 1e-3


def _geom_pt(p: Any) -> tuple[float, float]:
    if hasattr(p, "x"):
        return (float(p.x), float(p.y))
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return (float(p[0]), float(p[1]))
    raise TypeError(f"not a point: {p!r}")


def _geom_item_points(item: Any) -> list[tuple[float, float]]:
    """Sample points along one drawing op (chord approx for beziers)."""
    op = item[0]
    if op == "l":
        return [_geom_pt(item[1]), _geom_pt(item[2])]
    if op == "c":
        p0, p1, p2, p3 = _geom_pt(item[1]), _geom_pt(item[2]), _geom_pt(item[3]), _geom_pt(item[4])
        return [p0, p1, p2, p3]
    if op == "re":
        r = item[1]
        if hasattr(r, "x0"):
            x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
        else:
            x0, y0, x1, y1 = map(float, r[:4])
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    if op == "qu":
        q = item[1]
        pts = [_geom_pt(q[i]) for i in range(4)]
        pts.append(pts[0])
        return pts
    return []


def _split_loops(items: list) -> list[list[tuple[float, float]]]:
    """Split drawing items into continuous closed/open polylines."""
    if not items:
        return []
    loops: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []

    def flush() -> None:
        nonlocal cur
        if len(cur) >= 2:
            loops.append(cur)
        cur = []

    for it in items:
        op = it[0]
        if op in ("re", "qu"):
            flush()
            pts = _geom_item_points(it)
            if pts:
                loops.append(pts)
            continue
        pts = _geom_item_points(it)
        if not pts:
            flush()
            continue
        if not cur:
            cur = pts[:]
            continue
        prev_end = cur[-1]
        start = pts[0]
        dx = start[0] - prev_end[0]
        dy = start[1] - prev_end[1]
        if dx * dx + dy * dy > _JOIN_EPS * _JOIN_EPS:
            flush()
            cur = pts[:]
        else:
            cur.extend(pts[1:])
    flush()
    return loops


def _perimeter(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    p = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        p += math.hypot(dx, dy)
    dx = pts[0][0] - pts[-1][0]
    dy = pts[0][1] - pts[-1][1]
    if dx * dx + dy * dy > _JOIN_EPS * _JOIN_EPS:
        p += math.hypot(dx, dy)
    return p


def _area(pts: list[tuple[float, float]]) -> float:
    """Shoelace area (absolute); closes the ring if needed."""
    if len(pts) < 3:
        return 0.0
    ring = pts
    if abs(pts[0][0] - pts[-1][0]) > _JOIN_EPS or abs(pts[0][1] - pts[-1][1]) > _JOIN_EPS:
        ring = pts + [pts[0]]
    a = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        a += x0 * y1 - x1 * y0
    return abs(a) * 0.5


def isoperimetric_quotient(v: TraceVector) -> float:
    """``Q = 4*pi*A / P^2`` for ``v`` (area-weighted mean over continuous loops).

    Returns 0 if geometry is degenerate. Circle -> 1; thin stroke ribbon -> near 0.
    """
    loops = _split_loops(v.items or [])
    if not loops:
        x0, y0, x1, y1 = v.rect
        w = max(0.0, float(x1) - float(x0))
        h = max(0.0, float(y1) - float(y0))
        a = w * h
        p = 2.0 * (w + h)
        if p <= 0 or a <= 0:
            return 0.0
        return float(4.0 * math.pi * a / (p * p))

    total_a = 0.0
    weighted_q = 0.0
    for pts in loops:
        a = _area(pts)
        p = _perimeter(pts)
        if a <= 0 or p <= 0:
            continue
        q = 4.0 * math.pi * a / (p * p)
        weighted_q += a * q
        total_a += a
    if total_a <= 0:
        return 0.0
    return float(weighted_q / total_a)


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


# --- from serialisation/deserialization.py ---

_POINT_RE = re.compile(r"Point\(([^,]+),\s*([^)]+)\)")


def _quad_from_string(value: str) -> pymupdf.Quad:
    points = _POINT_RE.findall(value)
    if len(points) != 4:
        raise ValueError(f"unsupported quad string: {value!r}")
    corners = [pymupdf.Point(float(x), float(y)) for x, y in points]
    return pymupdf.Quad(corners[0], corners[1], corners[2], corners[3])


def _coords_to_pymupdf(value: list | tuple | str) -> pymupdf.Point | pymupdf.Rect | pymupdf.Quad:
    if isinstance(value, str):
        if value.startswith("Quad("):
            return _quad_from_string(value)
        raise ValueError(f"unsupported coordinate string: {value!r}")
    if len(value) == 2:
        return pymupdf.Point(float(value[0]), float(value[1]))
    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        return pymupdf.Rect(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    if len(value) == 4 and all(isinstance(v, (list, tuple)) for v in value):
        points = [pymupdf.Point(float(p[0]), float(p[1])) for p in value]
        return pymupdf.Quad(points[0], points[1], points[2], points[3])
    raise ValueError(f"unsupported coordinate list: {value!r}")


def drawing_item_from_dict(item: dict[str, Any]) -> tuple:
    """Rebuild one ``get_drawings()`` path segment from a serialized dict."""
    op = item["op"]
    args = item.get("args", [])
    if op == "re":
        return ("re", _coords_to_pymupdf(args[0]))
    if op == "l":
        return ("l", _coords_to_pymupdf(args[0]), _coords_to_pymupdf(args[1]))
    if op == "c":
        if len(args) != 4:
            raise ValueError(f"bezier expects 4 points, got {args!r}")
        return (
            "c",
            _coords_to_pymupdf(args[0]),
            _coords_to_pymupdf(args[1]),
            _coords_to_pymupdf(args[2]),
            _coords_to_pymupdf(args[3]),
        )
    if op == "qu":
        if len(args) == 1:
            return ("qu", _coords_to_pymupdf(args[0]))
        if len(args) == 4:
            return ("qu", _coords_to_pymupdf(args))
        raise ValueError(f"quad expects 1 or 4 args, got {args!r}")
    raise ValueError(f"unsupported drawing op {op!r}")


def drawing_items_from_dicts(items: list[dict[str, Any]]) -> list[tuple]:
    return [drawing_item_from_dict(item) for item in items]


def deserialize_vector(dto: TraceVectorRecord) -> TraceVector:
    """Convert one :class:`TraceVectorRecord` to pipeline :class:`TraceVector`."""
    return TraceVector(
        items=drawing_items_from_dicts(dto.items),
        type=dto.type,
        rect=(dto.x0, dto.y0, dto.x1, dto.y1),
        fill=dto.fill,
        color=dto.color,
        width=dto.width,
        even_odd=dto.even_odd,
        close_path=dto.close_path,
        dashes=dto.dashes,
        fill_opacity=dto.fill_opacity,
        stroke_opacity=dto.stroke_opacity,
        seqno=dto.seqno,
        layer=dto.layer,
        line_cap=dto.line_cap,
        line_join=dto.line_join,
        level=dto.level,
        scissor=dto.scissor,
        blendmode=dto.blendmode,
        isolated=dto.isolated,
        knockout=dto.knockout,
        opacity=dto.opacity,
    )


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


# --- from autotrace/drawing_autotrace.py ---

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
) -> list[TraceVector]:
    """Trace a drawing PNG with autotrace and return ``list[TraceVector]``.

    PNG -> SVG -> PDF -> ``get_drawings`` -> TraceVector. No soft/adaptive binarize;
    pass an already binary (or otherwise ready) PNG.

    ``svg_out`` / ``pdf_out``: if given, those files are kept; otherwise temps are deleted.
    ``dpi``: if given, scale TraceVector coords from pixel-space into PDF points (``72/dpi``).
    ``scratch_dir``: parent for throwaway intermediates (temp svg/pdf when not kept).
    ``split_strokes``: one TraceVector per continuous stroke (linework default). Pass ``False``
    for filled shapes so subpath holes stay inside one TraceVector (see ``pdf_to_vectors``).

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


# --- from autotrace/drawing_colour_fill_autotrace.py ---

_DEFAULT_BG_THRESH = 250


_DEFAULT_GROUP_DIST = 150.0


_DEFAULT_MIN_GROUP_PIXELS = 5000


_ACHROMA_S_MAX = 40


_BLACK_RGB = (0, 0, 0)


_WHITE_RGB = (255, 255, 255)


_BLACK_MEAN_MAX = 80


_WHITE_MEAN_MIN = 220  # below bg_thresh so page BG stays out of inventory


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB (..., 3) 0..255 → HSV distance embedding ``(S cos H, S sin H, V)``."""
    arr = np.asarray(rgb, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, 1, 3)
    elif arr.ndim == 2 and arr.shape[-1] == 3:
        arr = arr.reshape(1, -1, 3)
    u8 = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    ang = hsv[..., 0] * (2.0 * np.pi / 180.0)
    s_raw = hsv[..., 1]
    v = hsv[..., 2]
    s = np.where(s_raw < _ACHROMA_S_MAX, 0.0, s_raw * 1.5)
    out = np.stack([s * np.cos(ang), s * np.sin(ang), v], axis=-1)
    if single:
        return out.reshape(3)
    if np.asarray(rgb).ndim == 2:
        return out.reshape(-1, 3)
    return out


def _is_achromatic(rgb: tuple[int, int, int]) -> bool:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return (max(r, g, b) - min(r, g, b)) <= _ACHROMA_S_MAX


def _is_near_black(rgb: tuple[int, int, int]) -> bool:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return _is_achromatic((r, g, b)) and (r + g + b) / 3.0 <= _BLACK_MEAN_MAX


def _is_near_white(rgb: tuple[int, int, int]) -> bool:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return _is_achromatic((r, g, b)) and (r + g + b) / 3.0 >= _WHITE_MEAN_MIN


@dataclass
class ColourGroup:
    """One colour cluster (centroid RGB + pixel count; optional black/white kind)."""

    label: int
    rgb: tuple[int, int, int]
    n_pixels: int
    kind: str | None = None  # "black" | "white" | None

    @property
    def tag(self) -> str:
        """Stable folder/file tag: ``c{label:02d}_{rrggbb}``."""
        r, g, b = self.rgb
        return f"c{self.label:02d}_{r:02x}{g:02x}{b:02x}"


def _stamp_rgb(group: ColourGroup) -> tuple[int, int, int]:
    if group.kind == "black":
        return _BLACK_RGB
    if group.kind == "white":
        return _WHITE_RGB
    return group.rgb


@dataclass
class ColourFillTraceResult:
    """Artefacts from ``colour_fill_autotrace`` (groups, vectors, QA paths)."""

    groups: list[ColourGroup]
    group_vectors: dict[int, list[TraceVector]]
    combined_vectors: list[TraceVector]
    mask_paths: dict[int, Path]
    isolate_paths: dict[int, Path]
    autotrace_svg_paths: dict[int, Path]
    autotrace_pdf_paths: dict[int, Path]
    palette_path: Path | None
    summary_path: Path | None
    colours_dir: Path | None = None
    combined_pdf_path: Path | None = None
    vectors_txt_path: Path | None = None
    vectors_json_path: Path | None = None
    centroid_map_path: Path | None = None
    page_width: float | None = None
    page_height: float | None = None


def dump_colour_vectors_json(
    vectors: list[TraceVector],
    out_path: str | Path,
    *,
    page_width: float,
    page_height: float,
    source: str | Path | None = None,
) -> Path:
    """Serialize ``vectors`` as TraceVectorRecord JSON for stroke/fill post-processing."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(source) if source is not None else None,
        "page_width": float(page_width),
        "page_height": float(page_height),
        "vectors": [trace_vector_to_record(v).model_dump(mode="json") for v in vectors],
    }
    out_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return out_path.resolve()


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def unique_colours(
    bgr: np.ndarray,
    *,
    bg_thresh: int = _DEFAULT_BG_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """Unique non-background RGB (N×3) and counts, sorted by descending count."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    bg = (rgb[:, 0] >= bg_thresh) & (rgb[:, 1] >= bg_thresh) & (rgb[:, 2] >= bg_thresh)
    ink = rgb[~bg]
    if ink.size == 0:
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int64)
    packed = (
        ink[:, 0].astype(np.uint32) << 16
        | ink[:, 1].astype(np.uint32) << 8
        | ink[:, 2].astype(np.uint32)
    )
    uniq, counts = np.unique(packed, return_counts=True)
    order = np.argsort(-counts)
    uniq, counts = uniq[order], counts[order]
    rgb_u = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    return rgb_u, counts.astype(np.int64)


def group_colours(
    unique_rgb: np.ndarray,
    counts: np.ndarray,
    *,
    dist_thresh: float = _DEFAULT_GROUP_DIST,
    min_pixels: int = _DEFAULT_MIN_GROUP_PIXELS,
) -> list[ColourGroup]:
    """Reserve black/white; greedy-cluster everything else in HSV space."""
    if len(unique_rgb) == 0:
        return []

    black_i, white_i, rest = [], [], []
    for i, rgb in enumerate(unique_rgb):
        t = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        if _is_near_black(t):
            black_i.append(i)
        elif _is_near_white(t):
            white_i.append(i)
        else:
            rest.append(i)

    hsv_u = rgb_to_hsv(unique_rgb)
    free: list[tuple[np.ndarray, list[int], int]] = []  # cent_rgb, members, npix
    for i in rest:
        hsv_i = hsv_u[i]
        n = int(counts[i])
        best_j, best_d = -1, float("inf")
        for j, (cent, _m, _n) in enumerate(free):
            d = float(np.linalg.norm(hsv_i - rgb_to_hsv(cent)))
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= dist_thresh:
            _c, members, npix = free[best_j]
            members.append(i)
            mem = unique_rgb[members].astype(np.float64)
            w = counts[members].astype(np.float64)
            free[best_j] = ((mem * w[:, None]).sum(0) / w.sum(), members, npix + n)
        else:
            free.append((unique_rgb[i].astype(np.float64).copy(), [i], n))

    out: list[ColourGroup] = []
    for members, rgb, kind in (
        (black_i, _BLACK_RGB, "black"),
        (white_i, _WHITE_RGB, "white"),
    ):
        if members:
            out.append(ColourGroup(len(out), rgb, int(counts[members].sum()), kind))
    for cent, _members, npix in free:
        if npix < min_pixels:
            continue
        rgb = tuple(int(round(x)) for x in cent.tolist())
        out.append(ColourGroup(len(out), rgb, npix, None))
    return out


def _achroma_bw_masks(slab_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chroma = slab_rgb.max(2).astype(np.int16) - slab_rgb.min(2).astype(np.int16)
    lum = slab_rgb.astype(np.float32).mean(2)
    achro = chroma <= _ACHROMA_S_MAX
    return achro & (lum <= _BLACK_MEAN_MAX), achro & (lum >= _WHITE_MEAN_MIN)


def _reserved_indices(groups: list[ColourGroup]) -> tuple[int | None, int | None]:
    black_i = white_i = None
    for i, g in enumerate(groups):
        if g.kind == "black":
            black_i = i
        elif g.kind == "white":
            white_i = i
    return black_i, white_i


def _label_map(
    bgr: np.ndarray,
    groups: list[ColourGroup],
    *,
    dist_thresh: float,
    bg_thresh: int,
    chunk_rows: int = 512,
) -> np.ndarray:
    """HxW int32 labels (−1 = background / unassigned). Forces near-B/W to reserved.

    Nearest-centroid search is done **per group** (running min), not via a
    ``(H, W, N, 3)`` broadcast — that temporary alone can exceed 10GB on A3@300
    DPI when N is large.
    """
    h, w = bgr.shape[:2]
    labels = np.full((h, w), -1, dtype=np.int32)
    if not groups:
        return labels

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cents = rgb_to_hsv(np.array([g.rgb for g in groups], dtype=np.uint8)).astype(
        np.float32, copy=False
    )
    black_i, white_i = _reserved_indices(groups)
    n_groups = len(groups)

    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        slab = rgb[y0:y1]
        bg = (
            (slab[:, :, 0] >= bg_thresh)
            & (slab[:, :, 1] >= bg_thresh)
            & (slab[:, :, 2] >= bg_thresh)
        )
        near_b, near_w = _achroma_bw_masks(slab)
        hsv = rgb_to_hsv(slab).astype(np.float32, copy=False)

        best_d = np.full(hsv.shape[:2], np.inf, dtype=np.float32)
        arg = np.zeros(hsv.shape[:2], dtype=np.int32)
        for gi in range(n_groups):
            # (sh, sw) only — never materialize (sh, sw, N, 3).
            d = np.linalg.norm(hsv - cents[gi], axis=2).astype(np.float32, copy=False)
            closer = d < best_d
            best_d = np.where(closer, d, best_d)
            arg = np.where(closer, gi, arg)

        assign = (best_d <= dist_thresh) & ~bg
        if black_i is not None:
            hit = near_b & ~bg
            arg = np.where(hit, black_i, arg)
            assign |= hit
        if white_i is not None:
            hit = near_w & ~bg & ~near_b
            arg = np.where(hit, white_i, arg)
            assign |= hit
        slab_lab = np.full(assign.shape, -1, dtype=np.int32)
        slab_lab[assign] = arg[assign]
        labels[y0:y1] = slab_lab
        del hsv, best_d, arg, assign, slab_lab, near_b, near_w, bg
    return labels


def mask_to_bw_png(mask: np.ndarray, out: Path) -> Path:
    """Write a bool mask as black-ink / white-bg PNG."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.where(mask, 0, 255).astype(np.uint8))
    return out


def save_palette_preview(groups: list[ColourGroup], out: Path, *, swatch: int = 64) -> Path:
    """Write a horizontal RGB swatch strip for ``groups``."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not groups:
        img = np.full((swatch, swatch, 3), 255, dtype=np.uint8)
    else:
        img = np.full((swatch, swatch * len(groups), 3), 255, dtype=np.uint8)
        for i, g in enumerate(groups):
            r, gv, b = _stamp_rgb(g)
            img[:, i * swatch:(i + 1) * swatch] = (b, gv, r)
    cv2.imwrite(str(out), img)
    return out


def _is_white_fill(v: TraceVector, *, thresh: float = 0.98) -> bool:
    return v.fill is not None and all(c >= thresh for c in v.fill)


def _covers_most_of_page(
    v: TraceVector, page_w: float, page_h: float, *, cover: float = 0.95,
) -> bool:
    x0, y0, x1, y1 = v.rect
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) >= cover * page_w * page_h


# Full-page wash vs border frame (tightened to keep side grids).
_WASH_COVER = 0.95           # was 0.85 — tall side strips less often match


_WASH_MAX_SIMPLE_ITEMS = 8


_WASH_Q_MIN = 0.20           # compact solid


_WASH_MIN_SOLIDITY = 0.35    # path_area / bbox_area; below → thin frame, keep


_WASH_SOLID_SOLIDITY = 0.85  # very filled bbox → solid wash even if many items


def _path_fill_area(v: TraceVector) -> float:
    """Shoelace area of path loops (fallback: bbox area)."""
    loops = _split_loops(v.items or [])
    if not loops:
        x0, y0, x1, y1 = v.rect
        return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))
    return float(sum(_area(pts) for pts in loops))


def _is_page_wash(
    v: TraceVector,
    page_w: float,
    page_h: float,
    *,
    stamp_rgb: tuple[int, int, int],
    cover: float = _WASH_COVER,
) -> bool:
    """Drop solid page washes; keep thin page-frame / side-grid paths.

    Achromatic near-page paths are dropped only when they look *solid*:
    high fill/bbox solidity and high Q, optionally with few items.
    Low-solidity rings (border grids) are kept even if the bbox is page-sized.
    """
    if not _covers_most_of_page(v, page_w, page_h, cover=cover):
        return False
    if _is_white_fill(v):
        return True  # full-page white clear

    r, g, b = stamp_rgb
    chroma = max(r, g, b) - min(r, g, b)
    lum = (r + g + b) / 3.0
    if chroma > 40 or lum > 250:
        return False

    x0, y0, x1, y1 = v.rect
    bbox_a = max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))
    if bbox_a <= 0:
        return False
    solidity = _path_fill_area(v) / bbox_a
    if solidity < _WASH_MIN_SOLIDITY:
        return False  # thin frame / side grids

    q = isoperimetric_quotient(v)
    n_items = len(v.items or [])
    # Simple solid clear: few items AND compact Q.
    if n_items <= _WASH_MAX_SIMPLE_ITEMS and q >= _WASH_Q_MIN:
        return True
    # Complex outline but still a solid page plate.
    return solidity >= _WASH_SOLID_SOLIDITY and q >= _WASH_Q_MIN


def _stamp_group_vectors(
    vecs: list[TraceVector],
    fill_rgb: tuple[int, int, int],
    *,
    seq_base: int,
    layer: str,
    page_w: float,
    page_h: float,
) -> list[TraceVector]:
    """Stamp ink; drop page washes; keep smaller white fills as hole covers."""
    fill = (fill_rgb[0] / 255.0, fill_rgb[1] / 255.0, fill_rgb[2] / 255.0)
    out: list[TraceVector] = []
    for v in vecs:
        if _is_page_wash(v, page_w, page_h, stamp_rgb=fill_rgb):
            continue
        seqno = seq_base + len(out)
        if _is_white_fill(v):
            # Hole cover: stay white so hollow shapes don't become solid black.
            out.append(replace(
                v, fill=(1.0, 1.0, 1.0), color=None, width=0.0,
                fill_opacity=1.0, seqno=seqno, layer=layer,
            ))
        else:
            out.append(replace(
                v,
                type="f" if v.type in ("f", "fs") else v.type,
                fill=fill, color=None, width=0.0,
                fill_opacity=1.0, seqno=seqno, layer=layer,
            ))
    return out


def _colour_trace_worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _colour_trace_job(job: dict) -> dict:
    t0 = time.perf_counter()
    vecs = autotrace_drawing(
        job["mask_path"],
        centerline=False,
        color_count=2,
        noise_removal=0.0,
        filter_iterations=0,
        despeckle_level=0,
        dpi=job.get("dpi"),
        svg_out=job["svg_path"],
        pdf_out=job["pdf_path"],
        scratch_dir=job["scratch_dir"],
        split_strokes=False,
    )
    return {
        "label": int(job["label"]),
        "tag": str(job["tag"]),
        "t_trace": time.perf_counter() - t0,
        "n_raw": len(vecs),
        "vectors": [trace_vector_to_record(v).model_dump(mode="json") for v in vecs],
    }


def colour_fill_autotrace(
    drawing_png: str | Path,
    *,
    out_dir: str | Path | None = None,
    stem: str | None = None,
    dpi: int | None = 300,
    bg_thresh: int = _DEFAULT_BG_THRESH,
    group_dist: float = _DEFAULT_GROUP_DIST,
    min_group_pixels: int = _DEFAULT_MIN_GROUP_PIXELS,
    keep_intermediates: bool = True,
    workers: int = 1,
) -> ColourFillTraceResult:
    """Cluster colours, autotrace each non-white mask, combine stamped vectors."""
    src = Path(drawing_png)
    if not src.is_file():
        raise FileNotFoundError(f"PNG not found: {src}")

    t_all = time.perf_counter()
    parent = Path(out_dir) if out_dir is not None else src.parent
    parent.mkdir(parents=True, exist_ok=True)
    out_stem = stem if stem else src.stem
    colour_dir = parent / f"{out_stem}_colours"
    colour_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="colourfill_", dir=str(colour_dir)))
    n_workers = max(1, int(workers))

    t0 = time.perf_counter()
    bgr = _load_bgr(src)
    h, w = bgr.shape[:2]
    print(f"[colour] load {src.name} {w}x{h} ({time.perf_counter() - t0:.2f}s) -> {colour_dir}",
          flush=True)

    t0 = time.perf_counter()
    uniq, counts = unique_colours(bgr, bg_thresh=bg_thresh)
    groups = group_colours(uniq, counts, dist_thresh=group_dist, min_pixels=min_group_pixels)
    n_uniq = len(uniq)
    del uniq, counts
    print(
        f"[colour] {n_uniq} RGB -> {len(groups)} groups "
        f"(dist={group_dist}, min_px={min_group_pixels}) in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    for g in groups:
        print(f"  [{g.tag}] kind={g.kind} stamp={_stamp_rgb(g)} px={g.n_pixels}", flush=True)

    palette_path = save_palette_preview(groups, colour_dir / f"{out_stem}_colour_palette.png")
    t0 = time.perf_counter()
    lab = _label_map(bgr, groups, dist_thresh=group_dist, bg_thresh=bg_thresh)
    stamps = np.array(
        [(_stamp_rgb(g)[2], _stamp_rgb(g)[1], _stamp_rgb(g)[0]) for g in groups],
        dtype=np.uint8,
    ) if groups else np.zeros((0, 3), dtype=np.uint8)
    centroid = np.full_like(bgr, 255)
    if groups:
        hit = lab >= 0
        centroid[hit] = stamps[lab[hit]]
        del hit
    centroid_map_path = colour_dir / f"{out_stem}_colour_centroid_map.png"
    cv2.imwrite(str(centroid_map_path), centroid)
    del centroid, stamps
    print(f"[colour] masks+centroid map {time.perf_counter() - t0:.2f}s", flush=True)

    page_w = w * 72.0 / float(dpi) if dpi and dpi > 0 else float(w)
    page_h = h * 72.0 / float(dpi) if dpi and dpi > 0 else float(h)

    group_vectors: dict[int, list[TraceVector]] = {g.label: [] for g in groups}
    mask_paths: dict[int, Path] = {}
    isolate_paths: dict[int, Path] = {}
    autotrace_svg_paths: dict[int, Path] = {}
    autotrace_pdf_paths: dict[int, Path] = {}
    group_timings: list[tuple[str, float, float, int]] = []
    combined: list[TraceVector] = []

    jobs: list[dict] = []
    prep_meta: dict[int, dict] = {}
    for gi, group in enumerate(groups):
        i = gi + 1
        # One bool mask at a time — never hold N full-page masks.
        mask = lab == gi
        if group.kind == "white":
            print(f"[colour] ({i}/{len(groups)}) {group.tag}: white — skip autotrace", flush=True)
            del mask
            continue
        if not mask.any():
            print(f"[colour] ({i}/{len(groups)}) {group.tag}: empty — skip", flush=True)
            del mask
            continue

        tag = group.tag
        mask_path = colour_dir / f"{out_stem}_{tag}_mask.png"
        isolate_path = colour_dir / f"{out_stem}_{tag}_isolate.png"
        svg_path = colour_dir / f"{out_stem}_{tag}_autotrace.svg"
        pdf_path = colour_dir / f"{out_stem}_{tag}_autotrace.pdf"

        t0 = time.perf_counter()
        mask_to_bw_png(mask, mask_path)
        if keep_intermediates:
            iso = np.full_like(bgr, 255)
            iso[mask] = bgr[mask]
            cv2.imwrite(str(isolate_path), iso)
            del iso
        t_iso = time.perf_counter() - t0
        del mask

        jobs.append({
            "label": group.label, "tag": tag,
            "mask_path": str(mask_path), "svg_path": str(svg_path),
            "pdf_path": str(pdf_path), "scratch_dir": str(scratch / tag), "dpi": dpi,
        })
        prep_meta[group.label] = {
            "tag": tag, "stamp_rgb": _stamp_rgb(group),
            "mask_path": mask_path, "isolate_path": isolate_path,
            "svg_path": svg_path, "pdf_path": pdf_path,
            "t_iso": t_iso, "index": i,
        }

    # Page rasters no longer needed after mask PNGs are on disk.
    del lab, bgr

    trace_by_label: dict[int, dict] = {}
    t_trace_all = time.perf_counter()
    if jobs and (n_workers == 1 or len(jobs) == 1):
        for job in bar(jobs, desc="autotrace colours", unit="colour"):
            trace_by_label[int(job["label"])] = _colour_trace_job(job)
    elif jobs:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(n_workers, len(jobs)),
            mp_context=ctx,
            initializer=_colour_trace_worker_init,
        ) as pool:
            futs = {pool.submit(_colour_trace_job, j): int(j["label"]) for j in jobs}
            for fut in bar(as_completed(futs), desc="autotrace colours", total=len(futs), unit="colour"):
                label = futs[fut]
                trace_by_label[label] = fut.result()
                print(
                    f"[colour] autotrace {prep_meta[label]['tag']} "
                    f"paths={trace_by_label[label]['n_raw']} "
                    f"{trace_by_label[label]['t_trace']:.1f}s",
                    flush=True,
                )
    if jobs:
        print(f"[colour] autotraces done in {time.perf_counter() - t_trace_all:.2f}s "
              f"(workers={n_workers})", flush=True)

    seq_base = 0
    for group in groups:
        meta = prep_meta.get(group.label)
        if meta is None:
            continue
        trace = trace_by_label.pop(group.label)
        vecs = [
            deserialize_vector(TraceVectorRecord.model_validate(row))
            for row in trace["vectors"]
        ]
        n_raw = int(trace["n_raw"])
        t_trace = float(trace["t_trace"])
        del trace
        vecs.sort(key=lambda v: v.seqno)
        stamped = _stamp_group_vectors(
            vecs, meta["stamp_rgb"], seq_base=seq_base, layer=meta["tag"],
            page_w=page_w, page_h=page_h,
        )
        del vecs
        seq_base += len(stamped)
        group_vectors[group.label] = stamped
        combined.extend(stamped)
        autotrace_svg_paths[group.label] = meta["svg_path"]
        autotrace_pdf_paths[group.label] = meta["pdf_path"]
        if keep_intermediates:
            mask_paths[group.label] = meta["mask_path"]
            isolate_paths[group.label] = meta["isolate_path"]
        else:
            meta["mask_path"].unlink(missing_ok=True)
            meta["isolate_path"].unlink(missing_ok=True)

        group_timings.append((meta["tag"], meta["t_iso"], t_trace, len(stamped)))
        print(
            f"[colour] ({meta['index']}/{len(groups)}) {meta['tag']}: "
            f"autotrace={t_trace:.1f}s raw={n_raw} -> {len(stamped)}",
            flush=True,
        )

    combined_pdf_path = vectors_txt_path = vectors_json_path = None
    if combined:
        t0 = time.perf_counter()
        combined_pdf_path = colour_dir / f"{out_stem}_colour_combined.pdf"
        draw_vectors(combined, combined_pdf_path, page_width=page_w, page_height=page_h)
        vectors_txt_path = colour_dir / f"{out_stem}_colour_vectors.txt"
        vectors_txt_path.write_text(
            "\n".join(f"TraceVector {i}: {v}" for i, v in enumerate(combined)) + "\n",
            encoding="utf-8",
        )
        vectors_json_path = dump_colour_vectors_json(
            combined, colour_dir / f"{out_stem}_colour_vectors.json",
            page_width=page_w, page_height=page_h, source=src,
        )
        print(f"[colour] combined {len(combined)} vectors -> {combined_pdf_path.name} "
              f"({time.perf_counter() - t0:.2f}s)", flush=True)

    t_elapsed = time.perf_counter() - t_all
    print(f"[colour] DONE {src.name}: {len(groups)} groups, {len(combined)} vectors, "
          f"{t_elapsed:.2f}s", flush=True)

    summary_path = colour_dir / f"{out_stem}_colour_summary.txt"
    lines = [
        f"source: {src}",
        f"colours_dir: {colour_dir}",
        f"workers: {n_workers}",
        f"unique RGB (non-bg): {n_uniq}",
        f"colour groups: {len(groups)} (group_dist={group_dist}, min_group_pixels={min_group_pixels})",
        f"combined vectors: {len(combined)}",
        f"combined_pdf: {combined_pdf_path}",
        f"vectors_json: {vectors_json_path}",
        f"centroid_map: {centroid_map_path}",
        f"page_width: {page_w}",
        f"page_height: {page_h}",
        f"total_seconds: {t_elapsed:.2f}",
        "",
        "groups:",
    ]
    for g in groups:
        lines.append(
            f"  [{g.tag}] kind={g.kind} stamp={_stamp_rgb(g)} "
            f"pixels={g.n_pixels} vectors={len(group_vectors.get(g.label, []))}"
        )
    if group_timings:
        lines += ["", "timings (seconds):"]
        for tag, t_iso, t_trace, n_vec in group_timings:
            lines.append(f"  [{tag}] prep={t_iso:.2f} autotrace={t_trace:.2f} vectors={n_vec}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return ColourFillTraceResult(
        groups=groups,
        group_vectors=group_vectors,
        combined_vectors=combined,
        mask_paths=mask_paths,
        isolate_paths=isolate_paths,
        autotrace_svg_paths=autotrace_svg_paths,
        autotrace_pdf_paths=autotrace_pdf_paths,
        palette_path=palette_path,
        summary_path=summary_path,
        colours_dir=colour_dir,
        combined_pdf_path=combined_pdf_path,
        vectors_txt_path=vectors_txt_path,
        vectors_json_path=vectors_json_path,
        centroid_map_path=centroid_map_path,
        page_width=page_w,
        page_height=page_h,
    )


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
