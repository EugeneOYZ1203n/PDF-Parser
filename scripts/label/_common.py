"""Shared Tk UI plumbing for the `scripts/label/*.py` tools -- imported as
`from _common import ...` (relies on the sibling-import trick every script
here already uses: running a script directly puts its own directory first
on `sys.path`, so this resolves without needing `scripts/label/` to be a
package).

Not itself runnable.
"""
from __future__ import annotations

import tkinter as tk

import pymupdf as fitz

from rastervec.helpers.geometry import item_points
from rastervec.models import Vector

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


def get_display_matrix(fitz_page: "fitz.Page", zoom: float) -> "fitz.Matrix":
    """page-space (unrotated MediaBox) -> canvas-space, page rotation baked
    in. OVERLAY-ONLY: the page pixmap must be rendered zoom-only, since
    get_pixmap() bakes /Rotate itself (double-rotates otherwise)."""
    return fitz_page.rotation_matrix * fitz.Matrix(zoom, zoom)


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
                self.window, text=text, justify="left", anchor="w", padx=8, pady=6,
                bg="#ffffe0", fg="#111111", relief="solid", borderwidth=1,
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


def bezier_points(p0, p1, p2, p3, n: int = 24) -> list[tuple[float, float]]:
    """`n+1` sampled points along the cubic bezier `p0->p1->p2->p3` --
    shared by `raster_label.py`'s curve preview/render and
    `label_viewer.py`'s read-only overlay of the same `GeometryAnnotation`
    shape."""
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = (mt**3) * p0[0] + 3 * (mt**2) * t * p1[0] + 3 * mt * (t**2) * p2[0] + (t**3) * p3[0]
        y = (mt**3) * p0[1] + 3 * (mt**2) * t * p1[1] + 3 * mt * (t**2) * p2[1] + (t**3) * p3[1]
        pts.append((x, y))
    return pts


def draw_vector(canvas: "tk.Canvas", matrix: "fitz.Matrix", vector: Vector, color: str, width: int) -> None:
    """Polyline of every item's own points through the display matrix
    (polygon outline for re/qu, line otherwise) -- one `Vector` can carry
    several items, all drawn."""
    for item in vector.items:
        coords: list[float] = []
        for x, y in item_points(item):
            p = fitz.Point(x, y) * matrix
            coords.extend([p.x, p.y])
        if len(coords) < 4:
            continue
        if item[0] in ("re", "qu"):
            canvas.create_polygon(*coords, outline=color, fill="", width=width, tags=("overlay",))
        else:
            canvas.create_line(*coords, fill=color, width=width, tags=("overlay",))
