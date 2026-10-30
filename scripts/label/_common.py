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


def prefetch_order(page_index: int, page_count: int) -> list[int]:
    """Pages to process in the background: `page_index` first, then outwards (next, previous,
    next-but-one, ...), so the neighbours the labeller reaches first are ready first."""
    order = [page_index]
    for step in range(1, page_count):
        order += [p for p in (page_index + step, page_index - step) if 0 <= p < page_count]
    return order


def label_rows(entries) -> list[tuple[str, tuple]]:
    """One `(label_id, (page, text, rotation, n vectors))` row per label entry for the side list,
    by page then reading position (top to bottom, left to right). Pages are shown 1-based."""
    ordered = sorted(entries, key=lambda e: (e.page_index, round(e.bbox[1]), e.bbox[0]))
    return [
        (
            e.label_id,
            (e.page_index + 1, e.text, f"{e.expected_rotation}°", len(e.vector_signatures)),
        )
        for e in ordered
    ]


def focus_buckets(bucket_sigs: list[set[str]], label_sigs: set[str]) -> list[bool]:
    """Bucket visibility that shows only the buckets holding one of the label's vectors -- or every
    bucket when none does (a label whose vectors are no longer on the page)."""
    shown = [bool(sigs & label_sigs) for sigs in bucket_sigs]
    return shown if any(shown) else [True] * len(bucket_sigs)


def edit_label_vectors(label_sigs: list[str], hit_sigs: list[str]) -> list[str] | None:
    """A label's vector signatures after clicking / dragging over `hit_sigs` while it is
    selected: when every hit already belongs to the label they are removed, otherwise the hits
    not in it are added. `None` when the edit would leave the label with no vectors."""
    hits = list(dict.fromkeys(hit_sigs))
    current = set(label_sigs)
    if set(hits) <= current:
        kept = [s for s in label_sigs if s not in set(hits)]
        return kept or None
    return list(label_sigs) + [s for s in hits if s not in current]


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
