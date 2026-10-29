from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any, Optional, Dict, Tuple


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
class Vector:
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
    def from_drawing(cls, d: Dict[str, Any]) -> "Vector":
        """Build a Vector from one entry of ``page.get_drawings([extended=True])``."""
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
