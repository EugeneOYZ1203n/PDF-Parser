"""Tk UI plumbing for `manual_vector_text_label.py` -- imported as `from _common import ...`
(running a script directly puts its own directory first on `sys.path`, so this resolves without
`scripts/label/` being a package).

Not itself runnable.
"""

from __future__ import annotations

import tkinter as tk

import pymupdf
from raster_parser.commons.helpers.geometry import item_points
from raster_parser.commons.helpers.layer_separation import (  # noqa: F401 -- re-exported for the label tools
    separate_by_color,
    separate_by_layer,
    separate_by_width,
)
from raster_parser.models.vector import Vector

MIN_ZOOM = 0.25
MAX_ZOOM = 6.0
ZOOM_STEP = 1.25
ROTATION_SNAP_DEG = 2.5

# Canvas-pixel movement below which a press/release is treated as a plain click.
DRAG_THRESHOLD_PX = 4

ZOOM_DEFAULT = 1.5
UNLABELLED_COLOR = "#3366ff"
LABELLED_COLOR = "#33aa33"
SELECTED_COLOR = "#ff8800"
ENTRY_COLOR = "#999999"


def get_display_matrix(pymupdf_page: "pymupdf.Page", zoom: float) -> "pymupdf.Matrix":
    """page-space (unrotated MediaBox) -> canvas-space, page rotation baked
    in. OVERLAY-ONLY: the page pixmap must be rendered zoom-only, since
    get_pixmap() bakes /Rotate itself (double-rotates otherwise)."""
    return pymupdf_page.rotation_matrix * pymupdf.Matrix(zoom, zoom)


class Tooltip:
    """Mouse-following tooltip, ported from the former debug_app.py
    (originally inspector/overlay_canvas.py)."""

    def __init__(self, parent: "tk.Widget"):
        self.parent = parent
        self.window: "tk.Toplevel | None" = None
        self.label: "tk.Label | None" = None

    def show(self, x: int, y: int, text: str) -> None:
        if self.window is None:
            self.window = tk.Toplevel(self.parent)
            self.window.overrideredirect(True)
            self.window.attributes("-topmost", True)
            self.label = tk.Label(
                self.window,
                text=text,
                justify="left",
                anchor="w",
                padx=8,
                pady=6,
                bg="#ffffe0",
                fg="#111111",
                relief="solid",
                borderwidth=1,
                font=("TkDefaultFont", 9),
            )
            self.label.pack()
        else:
            self.label.config(text=text)
        self.window.geometry(f"+{x + 15}+{y + 15}")
        self.window.deiconify()

    def hide(self) -> None:
        if self.window is not None:
            self.window.withdraw()


def _exact_width(v: Vector) -> float | None:
    return v.width


def separate_by_layer_color_width(vectors: list[Vector]) -> list[list[Vector]]:
    """`(layer, color, width)` buckets, flat -- the label tool's visibility
    filter (one checkbox per bucket). Widths are bucketed exactly, not rounded like
    `v2_vector_class` does (`commons.helpers.layer_separation`)."""
    buckets: list[list[Vector]] = []
    for layer_bucket in separate_by_layer(vectors).values():
        for color_bucket in separate_by_color(layer_bucket).values():
            buckets.extend(separate_by_width(color_bucket, key=_exact_width).values())
    return buckets


def draw_vector(
    canvas: "tk.Canvas", matrix: "pymupdf.Matrix", vector: Vector, color: str, width: int
) -> None:
    """Polyline of every item's own points through the display matrix
    (polygon outline for re/qu, line otherwise) -- one `Vector` can carry
    several items, all drawn."""
    for item in vector.items:
        coords: list[float] = []
        for x, y in item_points(item):
            p = pymupdf.Point(x, y) * matrix
            coords.extend([p.x, p.y])
        if len(coords) < 4:
            continue
        if item[0] in ("re", "qu"):
            canvas.create_polygon(*coords, outline=color, fill="", width=width, tags=("overlay",))
        else:
            canvas.create_line(*coords, fill=color, width=width, tags=("overlay",))
