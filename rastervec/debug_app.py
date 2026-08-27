"""Debug Tkinter app for the rastervec demo pipeline.

Runs Pipeline (rastervec/pipeline.py) against the current page and lets
you step through each implemented stage's output with cycling arrows.
Only stages actually implemented (currently Reader, Native) appear in the
cycle -- a new stage joins once it has both a StageSpec in
Pipeline.STAGES and a view-renderer entry in _STAGE_RENDERERS below.
"""
from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import io

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageTk

if __name__ == "__main__" and __package__ is None:
    # Allow running this file directly (`python rastervec/debug_app.py`),
    # not just as a module (`python -m rastervec.debug_app`), by putting
    # the repo root -- the parent of this package -- on sys.path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import colorsys
import hashlib

import numpy as np

from rastervec.helpers import geometry
from rastervec.logging_setup import configure_logging
from rastervec.models import (
    ClusterOcrResult,
    DrawingVector,
    Page,
    TextVectorResult,
    TextWord,
    VectorPath,
)
from rastervec.OCR.Paddle_OCR.render_ocr import RenderOCR
from rastervec.pipeline import (
    FAST_PAGE_RENDER_DPI,
    SPATIAL_REGROUP_TOLERANCE_PX,
    ClusteringStageResult,
    FastPageResult,
    Pipeline,
    RotationCheck,
    StageOutput,
)
from rastervec.Reader.reader import Reader
from rastervec.renderer import Renderer
from rastervec.Vector_Classification.classification import SIGNATURE_ROUND_PX
from rastervec.Vector_Classification.items import item_filters as vc

REFERENCES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "references",
)

MIN_ZOOM = 0.25
MAX_ZOOM = 6.0
ZOOM_STEP = 1.25


class Tooltip:
    """Mouse-following tooltip, ported from inspector/overlay_canvas.py."""

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.window: tk.Toplevel | None = None
        self.label: tk.Label | None = None

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


@dataclass
class DebugAppState:
    page_index: int = 0
    zoom: float = 1.5
    stage_index: int = 0
    # Keyed by page_index -- the classification pipeline is fixed, so a
    # page's stage outputs never need more than one cache slot.
    stage_cache: dict[int, list[StageOutput]] = field(default_factory=dict)
    # stage key -> arbitrary dict of checkbox state, kept across redraws of
    # that stage (reset per page since it's cached with stage_cache anyway).
    filter_state: dict[str, dict] = field(default_factory=dict)
    # ocr_compare inspector: (page_index, cluster index) ->
    # {"image", "bbox_corners"} (see _ocr_cluster_preview). Deliberately
    # its own dict, not filter_state -- filter_state is keyed only by stage
    # key (shared across every page visit of that stage, fine for cheap UI
    # toggles), but this holds real recomputed render+OCR data that's only
    # valid for the exact page/cluster it was built from.
    ocr_detail_cache: dict[tuple, dict] = field(default_factory=dict)


_DEFAULT_PATH_COLOR = "#111827"
_DROPPED_COLOR = "#9ca3af"
_CENTROID_COLOR = "#dc2626"
_CENTROID_RADIUS = 3
_MAX_HOVER_HIGHLIGHT_PATHS = 400
_HOVER_TOLERANCE_PX = 5.0


_RENDERER = Renderer()  # stateless; the debug app's one shared instance
# RenderOCR's own PaddleOCR engine is cached at module scope inside
# helpers/render_ocr.py, so this instance is cheap -- but still shared
# rather than constructed per call, for symmetry with _RENDERER.
_RENDER_OCR = RenderOCR()


def _path_color_hex(path: VectorPath, default: str = _DEFAULT_PATH_COLOR) -> str:
    return _RENDERER.path_color_hex(path, default)


@dataclass
class RenderContext:
    canvas: tk.Canvas
    matrix: "fitz.Matrix"
    output: StageOutput
    tooltip: Tooltip
    side_panel: ttk.Frame
    filters: dict  # persistent per-stage checkbox state; mutate freely
    on_change: "Callable[[], None]"
    # Bumped once per DebugApp.redraw_overlay() call (owned by DebugApp,
    # same dict every render). A chunked/deferred draw callback scheduled
    # during one stage-visit checks this hasn't changed before touching
    # the canvas, so switching stage/page/zoom away mid-draw can't leave
    # ghost items drawn by a now-superseded render.
    epoch_box: dict = field(default_factory=lambda: {"value": 0})
    # The live Page for the page currently being viewed -- only
    # ocr_compare needs it today (to re-render a cluster on demand for its
    # inspector panel; see _ocr_cluster_preview), but it's cheap and
    # generally useful, so it's populated for every stage, not just that one.
    page: Page | None = None
    # DebugAppState.ocr_detail_cache, threaded through so
    # _render_ocr_compare_stage can key/read/write it without needing
    # a back-reference to DebugApp itself.
    ocr_detail_cache: dict[tuple, dict] | None = None
    # Every StageOutput for the current page, in Pipeline.STAGES order --
    # lets a stage renderer look up an *earlier* stage's own data (see
    # _stage_output_data), used by the reconstruction toggle's "Combined"
    # scope to draw more than just the current stage's own elements.
    all_outputs: list[StageOutput] | None = None


def _set_tag_visible(canvas: tk.Canvas, tag: str, visible: bool) -> None:
    """Show/hide every canvas item carrying `tag` in one Tk call. Toggling
    visibility this way (instead of deleting and recreating the items) is
    what keeps checkbox clicks snappy on stages with thousands of overlay
    items -- see the per-category tags used by the renderers below."""
    canvas.itemconfigure(tag, state="normal" if visible else "hidden")


def _add_category_checkbox(
    parent: ttk.Frame, canvas: tk.Canvas, text: str, var: tk.BooleanVar, tag: str, persist,
) -> None:
    """A checkbox whose toggle only flips the visibility of already-drawn
    canvas items tagged `tag` -- no re-render, so it stays fast no matter
    how many items that category has. `persist(value)` saves the new state
    for when the stage/page is revisited later."""

    def _command() -> None:
        _set_tag_visible(canvas, tag, var.get())
        persist(var.get())

    ttk.Checkbutton(parent, text=text, variable=var, command=_command).pack(
        anchor="w", padx=4, pady=1
    )


def _draw_vector_path(
    canvas: tk.Canvas, matrix: "fitz.Matrix", path: VectorPath, color: str | None = None,
    width: int = 2, tags=("overlay",), visible: bool = True,
) -> int | None:
    color = color or _path_color_hex(path)
    coords = []
    for x, y in path.points:
        p = fitz.Point(x, y) * matrix
        coords.extend([p.x, p.y])
    item_id = None
    if path.kind in ("re", "qu") and len(coords) >= 4:
        item_id = canvas.create_polygon(*coords, outline=color, fill="", width=width, tags=tags)
    elif len(coords) >= 4:
        item_id = canvas.create_line(*coords, fill=color, width=width, tags=tags)
    if item_id is not None and not visible:
        canvas.itemconfigure(item_id, state="hidden")
    return item_id


def _draw_bbox(
    canvas: tk.Canvas, matrix: "fitz.Matrix", bbox, color: str, width: int = 2, dash=None,
    tags=("overlay",), visible: bool = True,
) -> int:
    x0, y0, x1, y1 = bbox
    p0 = fitz.Point(x0, y0) * matrix
    p1 = fitz.Point(x1, y1) * matrix
    kwargs = {}
    if dash:
        kwargs["dash"] = dash
    item_id = canvas.create_rectangle(
        p0.x, p0.y, p1.x, p1.y, outline=color, width=width, tags=tags, **kwargs
    )
    if not visible:
        canvas.itemconfigure(item_id, state="hidden")
    return item_id


class _SpatialIndex:
    """Uniform grid over (bbox, payload) entries, for fast "what overlaps
    this rect" queries -- lets the debug app draw only what's currently
    visible on screen instead of every cluster/path up front, bounding the
    live canvas item count to the viewport rather than the total dataset
    size (a stage here can have tens of thousands of entries)."""

    _MAX_CELLS_PER_ENTRY = 64
    _MAX_QUERY_CELLS = 4000

    def __init__(
        self, entries: list[tuple[tuple[float, float, float, float], object]]
    ) -> None:
        self._entries = entries
        self._grid: dict[tuple[int, int], list[int]] = {}
        if not entries:
            self._cell_size = 1.0
            return

        sizes = sorted(max(b[2] - b[0], b[3] - b[1], 1e-6) for b, _ in entries)
        self._cell_size = max(1.0, min(sizes[len(sizes) // 2], 500.0))

        for index, (bbox, _payload) in enumerate(entries):
            for cell in self._entry_cells(bbox):
                self._grid.setdefault(cell, []).append(index)

    def _cell_for_point(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self._cell_size), int(y // self._cell_size))

    def _entry_cells(self, bbox: tuple[float, float, float, float]):
        x0, y0, x1, y1 = bbox
        cx0, cy0 = self._cell_for_point(x0, y0)
        cx1, cy1 = self._cell_for_point(x1, y1)
        if (cx1 - cx0 + 1) * (cy1 - cy0 + 1) > self._MAX_CELLS_PER_ENTRY:
            # One entry spans an enormous number of cells relative to the
            # grid resolution (a huge outlier bbox) -- index it by its
            # center only rather than flooding every cell it touches.
            yield self._cell_for_point((x0 + x1) / 2, (y0 + y1) / 2)
            return
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                yield (cx, cy)

    def query(self, rect: tuple[float, float, float, float]) -> list[object]:
        if not self._entries:
            return []
        rx0, ry0, rx1, ry1 = rect
        cx0, cy0 = self._cell_for_point(rx0, ry0)
        cx1, cy1 = self._cell_for_point(rx1, ry1)
        if (cx1 - cx0 + 1) * (cy1 - cy0 + 1) > self._MAX_QUERY_CELLS:
            # The query rect covers most/all of the indexed area (e.g.
            # zoomed all the way out) -- cheaper to just return everything
            # than to enumerate that many cells.
            return [payload for _bbox, payload in self._entries]

        seen: set[int] = set()
        results = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                for index in self._grid.get((cx, cy), ()):
                    if index in seen:
                        continue
                    seen.add(index)
                    bbox, payload = self._entries[index]
                    if (
                        bbox[0] <= rx1 and bbox[2] >= rx0
                        and bbox[1] <= ry1 and bbox[3] >= ry0
                    ):
                        results.append(payload)
        return results


def _visible_page_rect(
    ctx: RenderContext, margin: float = 0.2
) -> tuple[float, float, float, float]:
    """The currently visible canvas region, converted back to page-space
    through the inverse display matrix (so it lines up with VectorPath
    bboxes, which stay in unrotated MediaBox space per models.py), padded
    by `margin` so small scrolls don't immediately expose an empty edge
    before the next re-cull."""
    canvas = ctx.canvas
    x0 = canvas.canvasx(0)
    y0 = canvas.canvasy(0)
    x1 = canvas.canvasx(canvas.winfo_width())
    y1 = canvas.canvasy(canvas.winfo_height())
    dx = (x1 - x0) * margin
    dy = (y1 - y0) * margin
    x0, y0, x1, y1 = x0 - dx, y0 - dy, x1 + dx, y1 + dy

    inv = fitz.Matrix(ctx.matrix)
    inv.invert()
    corners = [fitz.Point(x, y) * inv for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    xs = [p.x for p in corners]
    ys = [p.y for p in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _bind_bucket_hover(
    ctx: RenderContext, *category_groups: tuple[list[list[VectorPath]], dict],
) -> None:
    """Hover is computed on demand (page-space hit-test against the group
    bboxes already in memory) instead of via one binding per drawn item --
    the groups are baked into a static image now, so there's nothing to
    bind to. Only the (small, capped) highlight for the currently-hovered
    group is ever drawn as real canvas items.

    Takes any number of `(groups, show_flag)` categories -- checked in the
    given order, skipping any category whose `show_flag["value"]` is
    False -- so callers with 2 categories (filter stages: this/previous)
    and callers with 5 (the clustering stage: 4 step categories +
    previous) share the same hit-testing logic.

    A group only counts as hovered when the cursor is over an actual member
    path's own bbox (padded by a small `_HOVER_TOLERANCE_PX`-screen-pixel
    tolerance, converted to page-space via the matrix's own scale so it
    stays a consistent ~5px regardless of zoom level -- thin lines are
    otherwise near-impossible to hit exactly), not just anywhere inside the
    group's aggregate bbox (which can have plenty of empty space between
    diagonally-placed members) -- the aggregate-bbox check runs first as a
    cheap reject so groups nowhere near the cursor are skipped without
    scanning their members."""
    inv = fitz.Matrix(ctx.matrix)
    inv.invert()
    hovered: dict = {"group": None}

    sx, sy = geometry.matrix_scale(ctx.matrix)
    tol_x = _HOVER_TOLERANCE_PX / sx if sx > 1e-6 else 0.0
    tol_y = _HOVER_TOLERANCE_PX / sy if sy > 1e-6 else 0.0

    def _group_hit(group: list[VectorPath], px: float, py: float) -> bool:
        x0, y0, x1, y1 = geometry.union_bbox([p.bbox for p in group])
        if not (x0 - tol_x <= px <= x1 + tol_x and y0 - tol_y <= py <= y1 + tol_y):
            return False
        return any(
            gx0 - tol_x <= px <= gx1 + tol_x and gy0 - tol_y <= py <= gy1 + tol_y
            for gx0, gy0, gx1, gy1 in (p.bbox for p in group)
        )

    def _find_group(px: float, py: float) -> list[VectorPath] | None:
        for groups, show in category_groups:
            if not show["value"]:
                continue
            for group in groups:
                if _group_hit(group, px, py):
                    return group
        return None

    def _show_tooltip(group: list[VectorPath]) -> None:
        x0, y0, x1, y1 = geometry.union_bbox([p.bbox for p in group])
        count_note = (
            f"{len(group)} path(s)"
            if len(group) <= _MAX_HOVER_HIGHLIGHT_PATHS
            else f"{len(group)} path(s) (showing first {_MAX_HOVER_HIGHLIGHT_PATHS})"
        )
        ctx.tooltip.show(
            ctx.canvas.winfo_pointerx(), ctx.canvas.winfo_pointery(),
            f"bbox: ({x0:.1f}, {y0:.1f}) - ({x1:.1f}, {y1:.1f})\n{count_note}",
        )

    def _on_motion(event) -> None:
        cx = ctx.canvas.canvasx(event.x)
        cy = ctx.canvas.canvasy(event.y)
        pdf_pt = fitz.Point(cx, cy) * inv
        group = _find_group(pdf_pt.x, pdf_pt.y)
        if group is hovered["group"]:
            if group is not None:
                _show_tooltip(group)
            return
        hovered["group"] = group
        ctx.canvas.delete("overlay_hover")
        if group is None:
            ctx.tooltip.hide()
            return
        for path in group[:_MAX_HOVER_HIGHLIGHT_PATHS]:
            _draw_vector_path(ctx.canvas, ctx.matrix, path, width=4, tags=("overlay_hover",))
        _show_tooltip(group)

    def _on_leave(_event) -> None:
        hovered["group"] = None
        ctx.canvas.delete("overlay_hover")
        ctx.tooltip.hide()

    ctx.canvas.bind("<Motion>", _on_motion)
    ctx.canvas.bind("<Leave>", _on_leave)


def _get_display_matrix(fitz_page: "fitz.Page", zoom: float) -> "fitz.Matrix":
    """page-space (unrotated) -> canvas-space, same rule as rastervec/Evaluation/inspector/inspector.py:
    TextWord/geometry is canonical unrotated MediaBox space (models.py),
    so this is the one transform both the pixmap and every overlay must
    go through to land in the same canvas space."""
    return fitz_page.rotation_matrix * fitz.Matrix(zoom, zoom)


def _stage_output_data(ctx: RenderContext, key: str) -> Any | None:
    """Looks up another stage's own StageOutput.data by key from
    ctx.all_outputs -- None if that stage hasn't run yet, errored, or
    ctx.all_outputs itself wasn't populated. Lets a stage renderer read an
    *earlier* stage's captured elements for the reconstruction toggle's
    "Combined" scope (see _add_reconstruction_toggle)."""
    for output in ctx.all_outputs or []:
        if output.key == key:
            return output.data if output.status == "ok" else None
    return None


def _add_reconstruction_toggle(ctx: RenderContext, build_own_fn, build_combined_fn) -> bool:
    """Adds a "Show Reconstructed Compare"/"Hide Reconstructed Compare"
    toggle button to the top of ctx.side_panel, for the (few) stages that
    support it -- native, ocr_compare, rotation_verify, drawing_vectors.
    Persisted per-stage in ctx.filters (survives redraws of that stage the
    same way every other checkbox here does), so switching stages away and
    back remembers whether it was showing.

    When active, a "This layer" / "Combined" radio pair also appears
    (state in ctx.filters["reconstruction_scope"], default "own"), picking
    which of `build_own_fn()` (just this stage's own captured elements) or
    `build_combined_fn()` (this stage's elements plus every earlier
    stage's, via _stage_output_data) is drawn -- both zero-arg callables
    the caller supplies so the actual Renderer.render_reconstructed_page
    call -- and whatever page_meta/element-list plumbing it needs -- stays
    local to each stage's own renderer.

    The built image is shown as a **drag-anywhere compare slider**: left
    of the divider is the reconstruction, right of it is the real original
    page pixmap (already the base canvas layer beneath, untouched -- the
    slider just crops how much of the reconstruction covers it). Divider
    position is state in ctx.filters["reconstruction_slider"] (a 0..1
    fraction, default 0.5). Dragging re-crops the *already-built* image
    (cheap PIL crop) and repositions the divider line directly via
    canvas.itemconfig/coords -- no re-render, no full stage redraw -- so
    the drag stays smooth even though building the reconstruction itself
    can be relatively expensive. Bindings are cleaned up centrally in
    DebugApp.redraw_overlay's unbind block on every stage switch.

    Returns whether the compare view is currently showing -- callers
    should skip drawing their normal overlay entirely when this is True,
    since the compare view replaces the overlay rather than layering
    under it."""
    state = ctx.filters.setdefault("show_reconstructed", {"value": False})

    def _toggle() -> None:
        state["value"] = not state["value"]
        ctx.on_change()

    label = "Hide Reconstructed Compare" if state["value"] else "Show Reconstructed Compare"
    ttk.Button(ctx.side_panel, text=label, command=_toggle).pack(
        anchor="w", fill="x", padx=4, pady=(4, 8)
    )

    if not state["value"]:
        return False

    scope = ctx.filters.setdefault("reconstruction_scope", {"value": "own"})
    scope_var = tk.StringVar(value=scope["value"])

    def _set_scope() -> None:
        scope["value"] = scope_var.get()
        ctx.on_change()

    row = ttk.Frame(ctx.side_panel)
    row.pack(anchor="w", padx=4, pady=(0, 4))
    for text, value in (("This layer", "own"), ("Combined", "combined")):
        ttk.Radiobutton(
            row, text=text, value=value, variable=scope_var, command=_set_scope,
        ).pack(side="left", padx=(0, 8))

    ttk.Label(
        ctx.side_panel, text="Drag: left = reconstructed, right = original",
    ).pack(anchor="w", padx=4, pady=(0, 8))

    image = build_combined_fn() if scope["value"] == "combined" else build_own_fn()
    slider = ctx.filters.setdefault("reconstruction_slider", {"value": 0.5})

    def _cropped_photo(frac: float) -> "ImageTk.PhotoImage":
        width = max(1, min(image.width, round(image.width * frac)))
        return ImageTk.PhotoImage(image.crop((0, 0, width, image.height)))

    photo = _cropped_photo(slider["value"])
    ctx.filters["_reconstructed_photo_ref"] = photo  # keep alive -- Tk won't retain it otherwise
    image_item = ctx.canvas.create_image(0, 0, anchor="nw", image=photo, tags=("overlay",))
    line_x = image.width * slider["value"]
    line_item = ctx.canvas.create_line(
        line_x, 0, line_x, image.height, fill="#2563eb", width=2, tags=("overlay",),
    )

    def _apply_slider(frac: float) -> None:
        frac = max(0.0, min(1.0, frac))
        slider["value"] = frac
        new_photo = _cropped_photo(frac)
        ctx.filters["_reconstructed_photo_ref"] = new_photo  # keep alive
        ctx.canvas.itemconfig(image_item, image=new_photo)
        new_x = image.width * frac
        ctx.canvas.coords(line_item, new_x, 0, new_x, image.height)

    def _frac_from_event(event) -> float:
        cx = ctx.canvas.canvasx(event.x)
        return (cx / image.width) if image.width else 0.0

    dragging = {"value": False}

    def _on_press(event) -> None:
        dragging["value"] = True
        _apply_slider(_frac_from_event(event))

    def _on_drag(event) -> None:
        if dragging["value"]:
            _apply_slider(_frac_from_event(event))

    def _on_release(_event) -> None:
        dragging["value"] = False

    ctx.canvas.bind("<ButtonPress-1>", _on_press)
    ctx.canvas.bind("<B1-Motion>", _on_drag)
    ctx.canvas.bind("<ButtonRelease-1>", _on_release)

    return True


def _render_reader_stage(ctx: RenderContext) -> None:
    # Base PDF only -- no overlay, confirms the pixmap itself is correct.
    pass


def _render_native_stage(ctx: RenderContext) -> None:
    words: list[TextWord] = ctx.output.data or []

    def _build() -> "Image.Image":
        return _RENDERER.render_reconstructed_page(
            ctx.page.meta, native_words=words, zoom=geometry.matrix_scale(ctx.matrix)[0],
        )

    # native is the first stage with this toggle -- nothing precedes it,
    # so "This layer" and "Combined" are the same render here.
    if ctx.page is not None and _add_reconstruction_toggle(ctx, _build, _build):
        return

    for word in words:
        coords = []
        for x, y in word.quad:
            p = fitz.Point(x, y) * ctx.matrix
            coords.extend([p.x, p.y])
        item_id = ctx.canvas.create_polygon(
            *coords, outline="#2563eb", fill="", width=1, tags=("overlay",)
        )
        ctx.canvas.tag_bind(
            item_id,
            "<Enter>",
            lambda _event, w=word: ctx.tooltip.show(
                ctx.canvas.winfo_pointerx(),
                ctx.canvas.winfo_pointery(),
                f"{w.text}\nfont: {w.font} {w.font_size:.1f}pt\nangle: {w.angle:.1f}deg",
            ),
        )
        ctx.canvas.tag_bind(item_id, "<Leave>", lambda _event: ctx.tooltip.hide())


def _render_vector_extract_stage(ctx: RenderContext) -> None:
    paths: list[VectorPath] = ctx.output.data or []
    kinds = sorted({p.kind for p in paths})
    active = ctx.filters.setdefault("kinds", {k: True for k in kinds})
    for k in kinds:
        active.setdefault(k, True)

    ttk.Label(ctx.side_panel, text=f"{len(paths)} path(s)").pack(anchor="w", padx=4, pady=(4, 6))
    for kind in kinds:
        count = sum(1 for p in paths if p.kind == kind)
        tag = f"kind_{kind}"
        var = tk.BooleanVar(value=active[kind])
        _add_category_checkbox(
            ctx.side_panel, ctx.canvas, f"{kind} ({count})", var, tag,
            persist=lambda v, k=kind: active.__setitem__(k, v),
        )

    for path in paths:
        _draw_vector_path(
            ctx.canvas, ctx.matrix, path,
            tags=("overlay", f"kind_{path.kind}"), visible=active.get(path.kind, True),
        )


def _render_layer_separation_stage(ctx: RenderContext) -> None:
    by_layer: dict[str, list[VectorPath]] = ctx.output.data or {}
    layers = sorted(by_layer.keys())
    active = ctx.filters.setdefault("layers", {name: True for name in layers})
    for name in layers:
        active.setdefault(name, True)

    ttk.Label(ctx.side_panel, text="Layers").pack(anchor="w", padx=4, pady=(4, 6))
    for name in layers:
        label = name or "(no layer)"
        tag = f"layer_{name}"
        var = tk.BooleanVar(value=active[name])
        _add_category_checkbox(
            ctx.side_panel, ctx.canvas, f"{label} ({len(by_layer[name])})", var, tag,
            persist=lambda v, n=name: active.__setitem__(n, v),
        )

    for name in layers:
        for path in by_layer[name]:
            _draw_vector_path(
                ctx.canvas, ctx.matrix, path,
                tags=("overlay", f"layer_{name}"), visible=active.get(name, True),
            )


def _render_color_separation_stage(ctx: RenderContext) -> None:
    by_layer_color: dict[str, dict[tuple, list[VectorPath]]] = ctx.output.data or {}
    active = ctx.filters.setdefault("entries", {})

    ttk.Label(ctx.side_panel, text="Layer / color").pack(anchor="w", padx=4, pady=(4, 6))

    entries = []  # (key, tag, layer, color, paths)
    for layer in sorted(by_layer_color.keys()):
        color_groups = by_layer_color[layer]
        for index, color in enumerate(sorted(color_groups.keys(), key=repr)):
            key = f"{layer}|{color}"
            tag = f"entry_{layer}_{index}"
            active.setdefault(key, True)
            entries.append((key, tag, layer, color, color_groups[color]))

    for key, tag, layer, color, paths in entries:
        label = f"{layer or '(no layer)'} / {color} ({len(paths)})"
        var = tk.BooleanVar(value=active[key])
        _add_category_checkbox(
            ctx.side_panel, ctx.canvas, label, var, tag,
            persist=lambda v, k=key: active.__setitem__(k, v),
        )

    for key, tag, layer, color, paths in entries:
        for path in paths:
            _draw_vector_path(
                ctx.canvas, ctx.matrix, path,
                tags=("overlay", tag), visible=active.get(key, True),
            )


_RECULL_CHUNK_SIZE = 400


def _cluster_size_stats_text(clusters: list[list[VectorPath]]) -> str:
    sizes = sorted(len(g) for g in clusters)
    if not sizes:
        return "0 cluster(s)"
    n = len(sizes)
    mean = sum(sizes) / n
    median = sizes[n // 2] if n % 2 else (sizes[n // 2 - 1] + sizes[n // 2]) / 2
    return (
        f"{n} cluster(s)\n"
        f"size min/median/mean/max:\n{sizes[0]} / {median:.1f} / {mean:.1f} / {sizes[-1]}"
    )


_CLUSTER_STEP_COLORS = [
    "#2563eb", "#7c3aed", "#ea580c", "#0d9488",
    "#6b7280", "#c026d3", "#65a30d", "#0284c7",
]

# Side (non-"kept") category colors, cycled per step so e.g. a step with
# two side categories gets two visually distinct, non-grey colors instead
# of everything defaulting to the same "dropped" grey.
_SIDE_CATEGORY_COLORS = [
    "#9ca3af", "#f59e0b", "#db2777", "#eab308", "#16a34a",
]


def _signature_color(signature: "vc.VectorSignature") -> str:
    """A stable color for a shape signature, derived from a hash of the
    signature itself -- the same shape always gets the same color across
    the whole page/run, with no need to pre-enumerate every signature
    (there can be hundreds)."""
    digest = hashlib.md5(repr(signature).encode()).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def _render_vector_type_colors(
    ctx: RenderContext, paths: list[VectorPath], signature_counts: dict,
) -> None:
    """Draws every path in `paths` colored by a hash of its shape
    signature (see vector_classification.vector_signature) -- same shape
    always gets the same color, so repeated symbols/hatching/ticks are
    visually obvious at a glance. Hover shows the signature's kind and its
    occurrence count in this bucket. Not viewport-culled/chunked like the
    per-step categories below -- this is an opt-in debug view, and the
    tag_bind-per-item hover pattern here mirrors _render_native_stage's,
    an existing precedent for drawing every item up front."""
    for path in paths:
        sig = vc.vector_signature(path, SIGNATURE_ROUND_PX)
        color = _signature_color(sig)
        item_id = _draw_vector_path(ctx.canvas, ctx.matrix, path, color=color, width=2)
        if item_id is None:
            continue
        count = signature_counts.get(sig, 0)
        ctx.canvas.tag_bind(
            item_id, "<Enter>",
            lambda _event, s=sig, c=count: ctx.tooltip.show(
                ctx.canvas.winfo_pointerx(), ctx.canvas.winfo_pointery(),
                f"kind: {s[0]}\noccurrences in bucket: {c}",
            ),
        )
        ctx.canvas.tag_bind(item_id, "<Leave>", lambda _event: ctx.tooltip.hide())


def _render_clustering_stage(ctx: RenderContext) -> None:
    """The fixed, non-configurable classification pipeline (see
    vector.py's module docstring) -- no runtime editing; tuning is done by
    editing vector.py's threshold constants. Each step can report any
    number of named categories (StepResult.categories) -- exactly one
    `role="kept"` (that step's surviving groups: bbox + centroid, in a
    distinct color per step) plus any number of `role="dropped"` side
    categories (e.g. "duplicate_runs", "repeated_pattern" -- what that
    step removed, classified as Drawing, its own color, no centroid), each
    independently toggleable, plus a live min/median/mean/max cluster-size
    summary for the kept side. Same viewport-culling + chunked-drawing
    approach as other high-volume stages, generalized to N categories per
    step instead of a fixed kept/dropped pair.

    A top-of-panel "Color by vector type" toggle replaces this whole view
    with every raw path (post step 1) colored by a hash of its shape
    signature, for visually spotting repeated shapes/textures."""
    results_by_group: dict[tuple, ClusteringStageResult] = ctx.output.data or {}
    any_result = next(iter(results_by_group.values()), None)
    step_labels = [step.label for step in any_result.steps] if any_result else []

    ttk.Label(ctx.side_panel, text=ctx.output.label).pack(anchor="w", padx=4, pady=(4, 6))

    show_types = ctx.filters.setdefault("show_vector_types", {"value": False})
    types_var = tk.BooleanVar(value=show_types["value"])

    def _toggle_types() -> None:
        show_types["value"] = types_var.get()
        ctx.on_change()

    ttk.Checkbutton(
        ctx.side_panel, text="Color by vector type", variable=types_var, command=_toggle_types,
    ).pack(anchor="w", padx=4, pady=(0, 8))

    if show_types["value"]:
        all_signature_paths = [
            p
            for result in results_by_group.values()
            for step in result.steps
            if step.signature_counts is not None
            for g in step.categories["kept"].groups
            for p in g
        ]
        merged_counts: dict = {}
        for result in results_by_group.values():
            for step in result.steps:
                if step.signature_counts:
                    for sig, count in step.signature_counts.items():
                        merged_counts[sig] = merged_counts.get(sig, 0) + count
        _render_vector_type_colors(ctx, all_signature_paths, merged_counts)
        return

    # --- per-step named categories ---
    # category_key -> list of that category's groups, merged across every
    # (layer, color) bucket, for step index i / category name.
    category_groups: dict[str, list[list[VectorPath]]] = {}
    category_role: dict[str, str] = {}
    category_step_index: dict[str, int] = {}
    category_names: list[str] = []  # display order: step 0's categories, step 1's, ...

    for i, label in enumerate(step_labels):
        names_this_step: list[str] = []
        for result in results_by_group.values():
            step = result.steps[i]
            for name in step.categories:
                if name not in names_this_step:
                    names_this_step.append(name)
        for name in names_this_step:
            key = f"{i}_{name}"
            category_names.append(key)
            category_step_index[key] = i
            groups: list[list[VectorPath]] = []
            role = "kept"
            for result in results_by_group.values():
                step = result.steps[i]
                cat = step.categories.get(name)
                if cat is None:
                    continue
                role = cat.role
                groups.extend(g for g in cat.groups if g)
            category_groups[key] = groups
            category_role[key] = role

    # Precomputed once (not per category key) -- each step's own dropped-
    # category keys, in display order, so a color index lookup below is
    # O(1) instead of rebuilding this same per-step filter+list().index()
    # on every non-kept category.
    dropped_keys_by_step: dict[int, list[str]] = {}
    for key in category_names:
        if category_role[key] != "kept":
            dropped_keys_by_step.setdefault(category_step_index[key], []).append(key)

    categories: dict[str, dict] = {}
    for key in category_names:
        i = category_step_index[key]
        role = category_role[key]
        is_kept = role == "kept"
        if is_kept:
            color = _CLUSTER_STEP_COLORS[i % len(_CLUSTER_STEP_COLORS)]
        else:
            side_index = dropped_keys_by_step[i].index(key)
            color = _SIDE_CATEGORY_COLORS[side_index % len(_SIDE_CATEGORY_COLORS)]
        groups = category_groups[key]
        categories[key] = {
            "show": ctx.filters.setdefault(f"show_{key}", {"value": is_kept and i == len(step_labels) - 1}),
            "entries": [(geometry.union_bbox([p.bbox for p in g]), g) for g in groups],
            "color": color,
            "width": 2 if is_kept else 1,
            "centroid": is_kept,
        }
    for state in categories.values():
        state["index"] = _SpatialIndex(state["entries"])
        state["drawn"] = {}  # id(payload) -> list[canvas item id]
        state["generation"] = {"value": 0}

    epoch = ctx.epoch_box["value"]

    def _draw_payload(category: str, payload: list[VectorPath]) -> list[int]:
        state = categories[category]
        bbox = geometry.union_bbox([p.bbox for p in payload])
        ids = [
            _draw_bbox(
                ctx.canvas, ctx.matrix, bbox, state["color"], width=state["width"],
                tags=("overlay", category),
            )
        ]
        if state["centroid"]:
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            centroid = fitz.Point(cx, cy) * ctx.matrix
            r = _CENTROID_RADIUS
            ids.append(
                ctx.canvas.create_oval(
                    centroid.x - r, centroid.y - r, centroid.x + r, centroid.y + r,
                    fill=_CENTROID_COLOR, outline="", tags=("overlay", category, "centroid"),
                )
            )
        return ids

    def _draw_chunk(category: str, items: list, generation: dict, gen: int, start: int) -> None:
        if ctx.epoch_box["value"] != epoch or generation["value"] != gen:
            return
        drawn = categories[category]["drawn"]
        end = min(start + _RECULL_CHUNK_SIZE, len(items))
        for payload in items[start:end]:
            drawn[id(payload)] = _draw_payload(category, payload)
        if end < len(items):
            ctx.canvas.after_idle(lambda: _draw_chunk(category, items, generation, gen, end))

    def _recull(category: str) -> None:
        state = categories[category]
        generation = state["generation"]
        generation["value"] += 1
        gen = generation["value"]
        drawn: dict = state["drawn"]

        if not state["show"]["value"]:
            for ids in drawn.values():
                for item_id in ids:
                    ctx.canvas.delete(item_id)
            drawn.clear()
            return

        rect = _visible_page_rect(ctx)
        wanted = state["index"].query(rect)
        wanted_ids = {id(payload) for payload in wanted}

        for payload_id in list(drawn.keys()):
            if payload_id not in wanted_ids:
                for item_id in drawn.pop(payload_id):
                    ctx.canvas.delete(item_id)

        to_draw = [payload for payload in wanted if id(payload) not in drawn]
        if to_draw:
            _draw_chunk(category, to_draw, generation, gen, 0)

    def _recull_all() -> None:
        for category in categories:
            _recull(category)

    ctx.filters["_recull_all"] = _recull_all

    for i, step_label in enumerate(step_labels):
        this_step_keys = [k for k in category_names if category_step_index[k] == i]
        ttk.Label(ctx.side_panel, text=f"{i + 1}: {step_label}", font=("TkDefaultFont", 9, "bold")).pack(
            anchor="w", padx=4, pady=(4, 0)
        )
        for key in this_step_keys:
            name = key.split("_", 1)[1]
            role = category_role[key]
            groups = category_groups[key]
            var = tk.BooleanVar(value=categories[key]["show"]["value"])
            label = f"{name} ({len(groups)})"
            _add_category_checkbox(
                ctx.side_panel, ctx.canvas, label, var, key,
                persist=lambda v, k=key: (categories[k]["show"].__setitem__("value", v), _recull(k)),
            )
            if role == "kept":
                ttk.Label(
                    ctx.side_panel, text=_cluster_size_stats_text(groups), justify="left",
                ).pack(anchor="w", padx=18, pady=(0, 2))
        ttk.Frame(ctx.side_panel, height=6).pack()

    _recull_all()
    _bind_bucket_hover(
        ctx,
        *[(category_groups[key], categories[key]["show"]) for key in category_names],
    )


def _render_drawing_vectors_stage(ctx: RenderContext) -> None:
    drawing_vectors: list[DrawingVector] = ctx.output.data or []

    def _build_own() -> "Image.Image":
        return _RENDERER.render_reconstructed_page(
            ctx.page.meta, drawing_vectors=drawing_vectors,
            zoom=geometry.matrix_scale(ctx.matrix)[0],
        )

    def _build_combined() -> "Image.Image":
        # rotation_verify runs after ocr_compare and reports its own
        # corrected-rotation copies via RotationCheck.resolved (see
        # pipeline.py's _run_rotation_verify) -- read those, not
        # ocr_compare's own (never-corrected) resolved readings, so this
        # combined view reflects the final orientation.
        native_words = _stage_output_data(ctx, "native") or []
        checks: list[RotationCheck] = _stage_output_data(ctx, "rotation_verify") or []
        ocr_results = [c.resolved for c in checks if _ocr_passed(c.resolved)]
        return _RENDERER.render_reconstructed_page(
            ctx.page.meta, native_words=native_words, ocr_results=ocr_results,
            drawing_vectors=drawing_vectors, zoom=geometry.matrix_scale(ctx.matrix)[0],
        )

    if ctx.page is not None and _add_reconstruction_toggle(ctx, _build_own, _build_combined):
        return

    filters = ctx.filters
    show_dashed = filters.setdefault("show_dashed", {"value": True})
    show_solid = filters.setdefault("show_solid", {"value": True})

    ttk.Label(ctx.side_panel, text=f"{len(drawing_vectors)} drawing vector(s)").pack(
        anchor="w", padx=4, pady=(4, 6)
    )

    dashed_var = tk.BooleanVar(value=show_dashed["value"])
    _add_category_checkbox(
        ctx.side_panel, ctx.canvas, "Dashed", dashed_var, "dv_dashed",
        persist=lambda v: show_dashed.__setitem__("value", v),
    )
    solid_var = tk.BooleanVar(value=show_solid["value"])
    _add_category_checkbox(
        ctx.side_panel, ctx.canvas, "Solid", solid_var, "dv_solid",
        persist=lambda v: show_solid.__setitem__("value", v),
    )

    for dv in drawing_vectors:
        color = dv.stroke_color or dv.fill_color
        hex_color = "#111827" if color is None else "#%02x%02x%02x" % tuple(
            round(c * 255) for c in color
        )
        tag = "dv_dashed" if dv.dashed else "dv_solid"
        visible = show_dashed["value"] if dv.dashed else show_solid["value"]
        _draw_bbox(
            ctx.canvas, ctx.matrix, dv.bbox, hex_color, width=1,
            tags=("overlay", tag), visible=visible,
        )


_OCR_PASSED_COLOR = "#059669"
_OCR_FAILED_COLOR = "#dc2626"
_OCR_SELECTED_WIDTH = 3
_OCR_PREVIEW_MAX_SIDE = 200
_OCR_BBOX_OVERLAY_COLOR = (220, 38, 38)
_OCR_BBOX_OVERLAY_COLOR_HEX = "#dc2626"


def _ocr_passed(result: TextVectorResult) -> bool:
    return bool(result.text.strip())


def _ocr_cluster_preview(result: TextVectorResult, page: Page) -> dict:
    """Renders one OCR'd text cluster upright (the same render `ocr_cluster`
    itself used) and re-OCRs it, purely to recover the detected-text bbox
    in that rendered image's own pixel space for the inspector's overlay --
    the production TextVectorResult keeps `text`/`confidence` (used
    directly below, not re-derived here) but not that bbox. No manual
    pre-rotation here: `RenderOCR.ocr()` (via `_boxes_from_page`'s
    `_undo_doc_rotation` correction) already maps PaddleOCR's own detected
    corners back into this upright render's own pixel space regardless of
    whatever internal doc-orientation rotation Paddle applied, so
    `result.rotation_used` is display-only here, not something this preview
    needs to pre-apply itself."""
    image = _RENDERER.render_vector_cluster(result.paths, dpi=300).convert("RGB")
    _text, _confidence, bbox_corners = _RENDER_OCR.ocr(image)
    return {"image": image, "bbox_corners": bbox_corners}


def _ocr_preview_photo(preview: dict) -> "ImageTk.PhotoImage":
    image = preview["image"].copy()
    scale = max(1, _OCR_PREVIEW_MAX_SIDE // max(image.width, image.height, 1))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    bbox_corners = preview["bbox_corners"]
    if bbox_corners:
        draw = ImageDraw.Draw(image)
        draw.polygon(
            [(x * scale, y * scale) for x, y in bbox_corners],
            outline=_OCR_BBOX_OVERLAY_COLOR, width=2,
        )
    return ImageTk.PhotoImage(image)


def _render_text_candidates_stage(ctx: RenderContext) -> None:
    """Every text-candidate cluster surviving the whole vector
    classification chain (across every (layer, color) bucket), before FAST
    has looked at any of them -- a plain bbox-per-cluster view, hover shows
    member-path count."""
    clusters: list[list[VectorPath]] = ctx.output.data or []

    ttk.Label(
        ctx.side_panel, text=f"{len(clusters)} text candidate cluster(s)",
    ).pack(anchor="w", padx=4, pady=(4, 6))

    show = ctx.filters.setdefault("show", {"value": True})
    var = tk.BooleanVar(value=show["value"])
    _add_category_checkbox(
        ctx.side_panel, ctx.canvas, "Show", var, "tc_all",
        persist=lambda v: show.__setitem__("value", v),
    )

    for cluster in clusters:
        if not cluster:
            continue
        bbox = geometry.union_bbox([p.bbox for p in cluster])
        item_id = _draw_bbox(
            ctx.canvas, ctx.matrix, bbox, _CLUSTER_STEP_COLORS[0], width=1,
            tags=("overlay", "tc_all"), visible=show["value"],
        )
        ctx.canvas.tag_bind(
            item_id, "<Enter>",
            lambda _event, c=cluster: ctx.tooltip.show(
                ctx.canvas.winfo_pointerx(), ctx.canvas.winfo_pointery(),
                f"{len(c)} path(s)",
            ),
        )
        ctx.canvas.tag_bind(item_id, "<Leave>", lambda _event: ctx.tooltip.hide())


_SIMILARITY_GROUP_COLORS = (
    "#059669", "#2563eb", "#dc2626", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#65a30d",
)


def _render_unique_clusters_stage(ctx: RenderContext) -> None:
    """Whole-page geometric similarity grouping of text-candidate clusters
    (Vector.group_similar_clusters) -- clusters judged the "same" shape
    (translation/rotation-tolerant) share a color and a FAST verdict (see
    fast_text_detect). Purely a visualization stage; hover shows which
    similarity group a cluster belongs to and its member count."""
    groups: list[list[list[VectorPath]]] = ctx.output.data or []

    ttk.Label(
        ctx.side_panel, text=f"{len(groups)} similarity group(s)",
    ).pack(anchor="w", padx=4, pady=(4, 6))

    show = ctx.filters.setdefault("show", {"value": True})
    var = tk.BooleanVar(value=show["value"])
    _add_category_checkbox(
        ctx.side_panel, ctx.canvas, "Show", var, "uc_all",
        persist=lambda v: show.__setitem__("value", v),
    )

    for gi, group in enumerate(groups):
        color = _SIMILARITY_GROUP_COLORS[gi % len(_SIMILARITY_GROUP_COLORS)]
        for cluster in group:
            if not cluster:
                continue
            bbox = geometry.union_bbox([p.bbox for p in cluster])
            item_id = _draw_bbox(
                ctx.canvas, ctx.matrix, bbox, color, width=1,
                tags=("overlay", "uc_all"), visible=show["value"],
            )
            ctx.canvas.tag_bind(
                item_id, "<Enter>",
                lambda _event, gi=gi, n=len(group): ctx.tooltip.show(
                    ctx.canvas.winfo_pointerx(), ctx.canvas.winfo_pointery(),
                    f"similarity group {gi + 1}\n{n} member cluster(s)",
                ),
            )
            ctx.canvas.tag_bind(item_id, "<Leave>", lambda _event: ctx.tooltip.hide())


def _fast_mask_overlay_image(base_image: "Image.Image", mask: "np.ndarray") -> "Image.Image":
    heat = (np.clip(mask, 0.0, 1.0) * 255).astype("uint8")
    zeros = Image.new("L", base_image.size, 0)
    heat_img = Image.merge("RGB", (Image.fromarray(heat), zeros, zeros))
    return Image.blend(base_image.convert("RGB"), heat_img, alpha=0.5)


def _render_fast_text_detect_stage(ctx: RenderContext) -> None:
    """FAST runs once per page, over a render of every extracted vector
    path on the page -- drawing content included, not just surviving
    text-candidate clusters (pipeline.py's FastPageResult) -- "Render"/
    "Detection heatmap" toggles control what's shown. Each candidate
    cluster's own bbox is drawn as a hover rect (green if it passed
    FAST_COMBINED_KEEP_THRESHOLD, red if dropped), resized/positioned at
    the current debug-app zoom (not a fixed DPI/72 factor) so it stays
    pixel-aligned with the page pixmap underneath and lives inside the
    same scrollregion."""
    result: FastPageResult = ctx.output.data

    ttk.Label(
        ctx.side_panel,
        text=f"{len(result.passed)} passed / {len(result.dropped)} dropped candidate cluster(s)",
    ).pack(anchor="w", padx=4, pady=(4, 6))

    def _fmt_seconds(value: float | None) -> str:
        return f"{value:.2f}s" if value is not None else "n/a"

    ttk.Label(
        ctx.side_panel,
        text=f"FAST detect time: {_fmt_seconds(result.detect_seconds)}",
    ).pack(anchor="w", padx=4, pady=(0, 6))

    show_render = ctx.filters.setdefault("show_render", {"value": True})
    show_detection = ctx.filters.setdefault("show_detection", {"value": True})

    def _toggle(state: dict, var: tk.BooleanVar) -> None:
        state["value"] = var.get()
        ctx.on_change()

    for label, state in (("Render", show_render), ("Detection heatmap", show_detection)):
        var = tk.BooleanVar(value=state["value"])
        ttk.Checkbutton(
            ctx.side_panel, text=label, variable=var,
            command=lambda v=var, s=state: _toggle(s, v),
        ).pack(anchor="w", padx=4, pady=1)

    image = result.page_image
    mask = result.page_mask

    if image is None:
        ttk.Label(ctx.side_panel, text="(no vector paths on this page)").pack(
            anchor="w", padx=4, pady=(8, 4)
        )
        return

    base = image.convert("RGB") if show_render["value"] else Image.new("RGB", image.size, (255, 255, 255))
    if show_detection["value"] and mask is not None:
        base = _fast_mask_overlay_image(base, mask)

    # Resize to match the page pixmap's own on-canvas size at the current
    # debug-app zoom, so it's pixel-aligned and lives inside the same
    # scrollregion (rather than a fixed FAST_PAGE_RENDER_DPI/72 factor
    # unrelated to the app's own zoom control).
    zoom = geometry.matrix_scale(ctx.matrix)[0]
    render_zoom = FAST_PAGE_RENDER_DPI / 72.0
    target_size = (
        max(1, round(base.width * zoom / render_zoom)),
        max(1, round(base.height * zoom / render_zoom)),
    )
    if target_size != base.size:
        base = base.resize(target_size, Image.NEAREST)

    # Drawn on the main canvas (left) rather than the side panel (right) --
    # this stage has no use for the original page picture underneath, so
    # the render occludes it the same way _add_reconstruction_toggle's
    # image does, tagged "overlay" so the next redraw cleans it up.
    photo = ImageTk.PhotoImage(base)
    ctx.filters["_photo_ref"] = photo  # keep a reference so Tk doesn't GC it
    ctx.canvas.create_image(0, 0, anchor="nw", image=photo, tags=("overlay",))

    for cluster, passed in (
        *((c, True) for c in result.passed),
        *((c, False) for c in result.dropped),
    ):
        if not cluster:
            continue
        bx0, by0, bx1, by1 = geometry.union_bbox([p.bbox for p in cluster])
        p0 = fitz.Point(bx0, by0) * ctx.matrix
        p1 = fitz.Point(bx1, by1) * ctx.matrix
        final_score = result.scores.get(id(cluster), 0.0)
        item_id = ctx.canvas.create_rectangle(
            p0.x, p0.y, p1.x, p1.y,
            outline=_OCR_PASSED_COLOR if passed else _OCR_FAILED_COLOR,
            width=1, tags=("overlay",),
        )
        ctx.canvas.tag_bind(
            item_id, "<Enter>",
            lambda _event, fs=final_score, p=passed: ctx.tooltip.show(
                ctx.canvas.winfo_pointerx(), ctx.canvas.winfo_pointery(),
                f"final (min over similarity group): {fs * 100:.1f}%\n"
                f"{'passed' if p else 'dropped'}",
            ),
        )
        ctx.canvas.tag_bind(item_id, "<Leave>", lambda _event: ctx.tooltip.hide())


def _render_spatial_regroup_stage(ctx: RenderContext) -> None:
    """spatial_regroup re-merges every FAST-passed cluster whose member
    paths actually overlap (SPATIAL_REGROUP_TOLERANCE_PX, ignoring the
    (layer, color) bucket and unique_clusters similarity group each
    cluster came from -- see pipeline.py's _run_spatial_regroup) before
    ocr_compare OCRs them. Draws each merged cluster's bbox; hover shows
    its member path count."""
    clusters: list[list[VectorPath]] = ctx.output.data or []
    fast_result: FastPageResult | None = _stage_output_data(ctx, "fast_text_detect")
    passed_count = len(fast_result.passed) if fast_result else len(clusters)

    ttk.Label(
        ctx.side_panel,
        text=f"{passed_count} FAST-passed cluster(s) -> {len(clusters)} regrouped "
        f"cluster(s) (tolerance={SPATIAL_REGROUP_TOLERANCE_PX:g}px)",
    ).pack(anchor="w", padx=4, pady=(4, 6))

    for cluster in clusters:
        if not cluster:
            continue
        bx0, by0, bx1, by1 = geometry.union_bbox([p.bbox for p in cluster])
        p0 = fitz.Point(bx0, by0) * ctx.matrix
        p1 = fitz.Point(bx1, by1) * ctx.matrix
        item_id = ctx.canvas.create_rectangle(
            p0.x, p0.y, p1.x, p1.y, outline="#8b5cf6", width=1, tags=("overlay",),
        )
        ctx.canvas.tag_bind(
            item_id, "<Enter>",
            lambda _event, n=len(cluster): ctx.tooltip.show(
                ctx.canvas.winfo_pointerx(), ctx.canvas.winfo_pointery(),
                f"{n} path(s) in merged cluster",
            ),
        )
        ctx.canvas.tag_bind(item_id, "<Leave>", lambda _event: ctx.tooltip.hide())




def _render_ocr_compare_stage(ctx: RenderContext) -> None:
    """Renders the ocr_compare stage: click a cluster's bbox to inspect
    it. Everything -- main canvas bbox/tooltip/pass-fail color,
    reconstruction toggle -- keys off `cluster_result.resolved` (see
    pipeline.py's _run_ocr_compare: one direct OCR call per
    spatial_regroup cluster, no fallback tiers)."""
    cluster_results: list[ClusterOcrResult] = ctx.output.data or []
    passed = [c for c in cluster_results if _ocr_passed(c.resolved)]
    failed = [c for c in cluster_results if not _ocr_passed(c.resolved)]

    ttk.Label(ctx.side_panel, text=f"{len(cluster_results)} OCR'd cluster(s)").pack(
        anchor="w", padx=4, pady=(4, 6)
    )

    def _fmt_seconds(value: float) -> str:
        return f"{value:.2f}s"

    all_seconds = [c.ocr_seconds for c in cluster_results]
    total = sum(all_seconds) if all_seconds else 0.0
    average = (total / len(all_seconds)) if all_seconds else 0.0
    ttk.Label(
        ctx.side_panel,
        text=f"OCR time: total {_fmt_seconds(total)}, avg {_fmt_seconds(average)} "
        f"({len(all_seconds)} call(s))",
    ).pack(anchor="w", padx=4, pady=(0, 6))

    def _build_own() -> "Image.Image":
        return _RENDERER.render_reconstructed_page(
            ctx.page.meta, ocr_results=[c.resolved for c in passed],
            zoom=geometry.matrix_scale(ctx.matrix)[0],
        )

    def _build_combined() -> "Image.Image":
        native_words = _stage_output_data(ctx, "native") or []
        return _RENDERER.render_reconstructed_page(
            ctx.page.meta, native_words=native_words,
            ocr_results=[c.resolved for c in passed],
            zoom=geometry.matrix_scale(ctx.matrix)[0],
        )

    if ctx.page is not None and _add_reconstruction_toggle(ctx, _build_own, _build_combined):
        return

    show_passed = ctx.filters.setdefault("show_passed", {"value": True})
    show_failed = ctx.filters.setdefault("show_failed", {"value": True})
    show_ocr_bbox = ctx.filters.setdefault("show_ocr_bbox", {"value": False})
    selected = ctx.filters.setdefault("selected_index", {"value": 0})

    def _visible(cluster_result: ClusterOcrResult) -> bool:
        return show_passed["value"] if _ocr_passed(cluster_result.resolved) else show_failed["value"]

    visible_indices = [i for i, c in enumerate(cluster_results) if _visible(c)]
    if selected["value"] not in visible_indices:
        selected["value"] = visible_indices[0] if visible_indices else None

    def _select(index: int) -> None:
        selected["value"] = index
        ctx.on_change()

    def _step_cluster(delta: int) -> None:
        if not visible_indices:
            return
        pos = visible_indices.index(selected["value"]) if selected["value"] in visible_indices else 0
        _select(visible_indices[(pos + delta) % len(visible_indices)])

    # Small counts per page (FAST-passed text clusters are rare compared to
    # raw path counts), so plain tag_bind hover -- like _render_native_stage
    # -- is plenty; no need for the filter stages' viewport-culling/
    # spatial-index machinery here.
    for i in visible_indices:
        result = cluster_results[i].resolved
        x0, y0 = fitz.Point(result.bbox[0], result.bbox[1]) * ctx.matrix
        x1, y1 = fitz.Point(result.bbox[2], result.bbox[3]) * ctx.matrix
        is_selected = i == selected["value"]
        color = _OCR_PASSED_COLOR if _ocr_passed(result) else _OCR_FAILED_COLOR
        item_id = ctx.canvas.create_rectangle(
            x0, y0, x1, y1, outline=color,
            width=_OCR_SELECTED_WIDTH if is_selected else 2, tags=("overlay",),
        )
        ctx.canvas.tag_bind(
            item_id,
            "<Enter>",
            lambda _event, r=result: ctx.tooltip.show(
                ctx.canvas.winfo_pointerx(),
                ctx.canvas.winfo_pointery(),
                f"resolved: {r.text!r}\nconfidence: {r.confidence:.2f}\n"
                f"rotation used: {r.rotation_used}deg\n{len(r.paths)} path(s)\n"
                "(click to inspect)",
            ),
        )
        ctx.canvas.tag_bind(item_id, "<Leave>", lambda _event: ctx.tooltip.hide())
        ctx.canvas.tag_bind(item_id, "<Button-1>", lambda _event, idx=i: _select(idx))

        if show_ocr_bbox["value"] and result.ocr_bbox is not None:
            ox0, oy0 = fitz.Point(result.ocr_bbox[0], result.ocr_bbox[1]) * ctx.matrix
            ox1, oy1 = fitz.Point(result.ocr_bbox[2], result.ocr_bbox[3]) * ctx.matrix
            ctx.canvas.create_rectangle(
                ox0, oy0, ox1, oy1, outline=_OCR_BBOX_OVERLAY_COLOR_HEX,
                width=1, dash=(3, 2), tags=("overlay",),
            )

    # --- inspector panel: rendered image (at rotation_used) + cluster nav
    # + resolved reading ---
    if selected["value"] is not None and ctx.page is not None and ctx.ocr_detail_cache is not None:
        index = selected["value"]
        cluster_result = cluster_results[index]
        result = cluster_result.resolved
        pos = visible_indices.index(index) + 1

        ttk.Separator(ctx.side_panel, orient="horizontal").pack(fill="x", padx=4, pady=6)

        nav = ttk.Frame(ctx.side_panel)
        nav.pack(fill="x", padx=4)
        ttk.Button(nav, text="< ←", width=4, command=lambda: _step_cluster(-1)).pack(side="left")
        ttk.Label(
            nav, text=f"Cluster {pos}/{len(visible_indices)}", anchor="center",
        ).pack(side="left", expand=True, fill="x")
        ttk.Button(nav, text="→ >", width=4, command=lambda: _step_cluster(1)).pack(side="right")

        ctx.canvas.focus_set()
        ctx.canvas.bind("<Left>", lambda _e: _step_cluster(-1))
        ctx.canvas.bind("<Right>", lambda _e: _step_cluster(1))

        cache_key = (ctx.page.meta.index, index)
        if cache_key not in ctx.ocr_detail_cache:
            ctx.ocr_detail_cache[cache_key] = _ocr_cluster_preview(result, ctx.page)
        preview = ctx.ocr_detail_cache[cache_key]

        # Drawn on the main canvas (left) rather than the side panel (right)
        # -- same as _render_fast_text_detect_stage's page render, tagged
        # "overlay" so the next redraw cleans it up.
        photo = _ocr_preview_photo(preview)
        ctx.filters["_photo_ref"] = photo  # keep a reference so Tk doesn't GC it
        ctx.canvas.create_image(0, 0, anchor="nw", image=photo, tags=("overlay",))

        lines = [
            "RESOLVED",
            f"  text: {result.text!r}",
            f"  confidence: {result.confidence:.3f}",
            f"  rotation used: {result.rotation_used}deg",
            f"  {len(result.paths)} path(s)",
            "  passed" if _ocr_passed(result) else "  failed (no text detected)",
            "",
            f"  {_fmt_seconds(cluster_result.ocr_seconds)}",
        ]
        info = tk.Text(
            ctx.side_panel, height=18, width=26, wrap="word", font=("TkDefaultFont", 8),
        )
        info.insert("1.0", "\n".join(lines))
        info.configure(state="disabled")
        info.pack(fill="both", expand=True, padx=4, pady=(4, 8))

    # --- bbox display checkbox ---
    ttk.Separator(ctx.side_panel, orient="horizontal").pack(fill="x", padx=4, pady=4)
    ocr_bbox_var = tk.BooleanVar(value=show_ocr_bbox["value"])

    def _on_ocr_bbox_toggle() -> None:
        show_ocr_bbox["value"] = ocr_bbox_var.get()
        ctx.on_change()

    ttk.Checkbutton(
        ctx.side_panel, text="Show Paddle OCR bbox", variable=ocr_bbox_var,
        command=_on_ocr_bbox_toggle,
    ).pack(anchor="w", padx=4, pady=1)

    # --- passed/failed filter, pinned to the bottom of the panel ---
    ttk.Separator(ctx.side_panel, orient="horizontal").pack(side="bottom", fill="x", padx=4, pady=4)
    failed_var = tk.BooleanVar(value=show_failed["value"])

    def _on_failed_toggle() -> None:
        show_failed["value"] = failed_var.get()
        ctx.on_change()

    ttk.Checkbutton(
        ctx.side_panel, text=f"Show failed ({len(failed)})", variable=failed_var,
        command=_on_failed_toggle,
    ).pack(side="bottom", anchor="w", padx=4, pady=1)

    passed_var = tk.BooleanVar(value=show_passed["value"])

    def _on_passed_toggle() -> None:
        show_passed["value"] = passed_var.get()
        ctx.on_change()

    ttk.Checkbutton(
        ctx.side_panel, text=f"Show passed ({len(passed)})", variable=passed_var,
        command=_on_passed_toggle,
    ).pack(side="bottom", anchor="w", padx=4, pady=1)


def _render_rotation_verify_stage(ctx: RenderContext) -> None:
    """New layer between ocr_compare and drawing_vectors (pipeline.py's
    _run_rotation_verify): for every OCR'd word, compares its text's own
    natural aspect ratio against its bbox as-is vs. rotated 90 deg -- a
    meaningfully closer fit after rotating produces `RotationCheck.
    resolved`, a *new* TextVectorResult with `rotation_used` flipped by
    +90 deg (ocr_compare's own reading is never mutated -- see
    pipeline.py's _run_rotation_verify). The reconstruction compare below
    reads that corrected `resolved`. Each checked word's bbox is drawn as
    a hover rect (orange + thicker outline if the fix was applied,
    grey/thin if not)."""
    checks: list[RotationCheck] = ctx.output.data or []
    applied = [c for c in checks if c.applied]

    def _build_own() -> "Image.Image":
        ocr_results = [c.resolved for c in checks if _ocr_passed(c.resolved)]
        return _RENDERER.render_reconstructed_page(
            ctx.page.meta, ocr_results=ocr_results, zoom=geometry.matrix_scale(ctx.matrix)[0],
        )

    def _build_combined() -> "Image.Image":
        native_words = _stage_output_data(ctx, "native") or []
        ocr_results = [c.resolved for c in checks if _ocr_passed(c.resolved)]
        return _RENDERER.render_reconstructed_page(
            ctx.page.meta, native_words=native_words, ocr_results=ocr_results,
            zoom=geometry.matrix_scale(ctx.matrix)[0],
        )

    if ctx.page is not None and _add_reconstruction_toggle(ctx, _build_own, _build_combined):
        return

    ttk.Label(
        ctx.side_panel, text=f"{len(applied)} rotation fix(es) applied / {len(checks)} checked",
    ).pack(anchor="w", padx=4, pady=(4, 6))

    for check in checks:
        x0, y0, x1, y1 = check.bbox
        p0 = fitz.Point(x0, y0) * ctx.matrix
        p1 = fitz.Point(x1, y1) * ctx.matrix
        item_id = ctx.canvas.create_rectangle(
            p0.x, p0.y, p1.x, p1.y,
            outline="#f97316" if check.applied else "#94a3b8",
            width=2 if check.applied else 1, tags=("overlay",),
        )
        ctx.canvas.tag_bind(
            item_id, "<Enter>",
            lambda _event, c=check: ctx.tooltip.show(
                ctx.canvas.winfo_pointerx(), ctx.canvas.winfo_pointery(),
                f"{c.text!r}\nerror as-is: {c.error_unrotated:.2f}  "
                f"error rotated: {c.error_rotated:.2f}\n"
                f"rotation: {c.before_rotation}deg -> {c.after_rotation}deg"
                + (" (fixed)" if c.applied else ""),
            ),
        )
        ctx.canvas.tag_bind(item_id, "<Leave>", lambda _event: ctx.tooltip.hide())


# Stage key -> view-render function. Add one entry here alongside each new
# StageSpec in rastervec/pipeline.py's Pipeline.STAGES.
_STAGE_RENDERERS = {
    "reader": _render_reader_stage,
    "native": _render_native_stage,
    "vector_extract": _render_vector_extract_stage,
    "layer_separation": _render_layer_separation_stage,
    "color_separation": _render_color_separation_stage,
    "clustering": _render_clustering_stage,
    "text_candidates": _render_text_candidates_stage,
    "unique_clusters": _render_unique_clusters_stage,
    "fast_text_detect": _render_fast_text_detect_stage,
    "spatial_regroup": _render_spatial_regroup_stage,
    "ocr_compare": _render_ocr_compare_stage,
    "rotation_verify": _render_rotation_verify_stage,
    "drawing_vectors": _render_drawing_vectors_stage,
}


class DebugApp:
    def __init__(
        self, root: tk.Tk, pdf_path: str, page_index: int = 0, final_stage: str | None = None,
    ) -> None:
        self.root = root
        self.pdf_path = os.path.abspath(pdf_path)
        # Fixed for the app's lifetime (a startup flag, not something the UI
        # changes), so it lives here rather than in DebugAppState -- it
        # doesn't need to be part of the stage_cache key. None runs every
        # stage, unchanged from before this existed. Threaded into every
        # Pipeline.run_page() call in _get_stage_outputs.
        self.final_stage = final_stage
        # Only the requested page is ever loaded (Reader.select()s it out of
        # the rest of the document) -- the debug app operates on one page at
        # a time, so there's no reason to pay to load the others.
        self.reader = Reader(self.pdf_path, page_index=page_index)
        self.pipeline = Pipeline()
        self.state = DebugAppState(page_index=0)

        self._photo = None
        self.tooltip: Tooltip | None = None
        # Bumped once per redraw_overlay() call; handed to RenderContext so
        # a stage's deferred/chunked draw callbacks can tell they've been
        # superseded by a later stage/page/zoom change and stop instead of
        # drawing ghost items onto the (shared) canvas.
        self._render_epoch_box: dict = {"value": 0}

        root.title(f"rastervec Debug — {os.path.basename(self.pdf_path)}")
        root.geometry("1400x900")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_menu()
        self._build_layout()
        self.render()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open PDF...", command=self.open_pdf_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _build_layout(self) -> None:
        page_bar = ttk.Frame(self.root)
        page_bar.pack(fill="x", side="top")

        self.page_label = ttk.Label(page_bar, text="Page - / -")
        self.page_label.pack(side="left", padx=6)

        ttk.Button(page_bar, text="Zoom -", command=lambda: self.change_zoom(-1)).pack(
            side="left", padx=(20, 2)
        )
        self.zoom_label = ttk.Label(page_bar, text="100%")
        self.zoom_label.pack(side="left", padx=6)
        ttk.Button(page_bar, text="Zoom +", command=lambda: self.change_zoom(1)).pack(
            side="left", padx=2
        )

        stage_bar = ttk.Frame(self.root)
        stage_bar.pack(fill="x", side="top")

        ttk.Button(
            stage_bar, text="< Prev Stage", command=lambda: self.change_stage(-1)
        ).pack(side="left", padx=2, pady=2)
        self.stage_label = ttk.Label(stage_bar, text="Stage - / -")
        self.stage_label.pack(side="left", padx=6)
        ttk.Button(
            stage_bar, text="Next Stage >", command=lambda: self.change_stage(1)
        ).pack(side="left", padx=2, pady=2)

        self.status_label = ttk.Label(stage_bar, text="", foreground="#b91c1c")
        self.status_label.pack(side="left", padx=20)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)

        canvas_frame = ttk.Frame(body)
        canvas_frame.pack(fill="both", expand=True, side="left")

        self.canvas = tk.Canvas(canvas_frame, bg="#808080")
        vbar = ttk.Scrollbar(
            canvas_frame, orient="vertical", command=self._make_scroll_command(self.canvas.yview)
        )
        hbar = ttk.Scrollbar(
            canvas_frame, orient="horizontal", command=self._make_scroll_command(self.canvas.xview)
        )
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.canvas.bind("<Configure>", lambda _event: self._on_viewport_changed())

        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # Scrollable side panel: the clustering stage alone can pack in up
        # to 8 dropdown+params blocks plus 16 kept/dropped checkboxes,
        # comfortably exceeding the panel's fixed height -- ttk.Frame has
        # no native scrolling, so the panel is really a Canvas holding one
        # inner ttk.Frame (self.side_panel, which every stage renderer
        # still packs directly into, unaware this wrapping exists).
        side_panel_container = ttk.Frame(body, width=260)
        side_panel_container.pack(fill="y", side="right")
        side_panel_container.pack_propagate(False)

        self.side_panel_canvas = tk.Canvas(side_panel_container, highlightthickness=0)
        side_panel_scrollbar = ttk.Scrollbar(
            side_panel_container, orient="vertical", command=self.side_panel_canvas.yview,
        )
        self.side_panel_canvas.configure(yscrollcommand=side_panel_scrollbar.set)
        self.side_panel_canvas.pack(side="left", fill="both", expand=True)
        side_panel_scrollbar.pack(side="right", fill="y")

        self.side_panel = ttk.Frame(self.side_panel_canvas)
        self._side_panel_window = self.side_panel_canvas.create_window(
            (0, 0), window=self.side_panel, anchor="nw"
        )

        def _on_side_panel_configure(_event=None) -> None:
            self.side_panel_canvas.configure(scrollregion=self.side_panel_canvas.bbox("all"))

        self.side_panel.bind("<Configure>", _on_side_panel_configure)

        def _on_side_panel_canvas_configure(event) -> None:
            # Keep the inner frame exactly as wide as the visible canvas so
            # its child widgets can wrap/fill horizontally; only the height
            # is ever meant to scroll.
            self.side_panel_canvas.itemconfigure(self._side_panel_window, width=event.width)

        self.side_panel_canvas.bind("<Configure>", _on_side_panel_canvas_configure)

        def _on_mousewheel(event) -> None:
            self.side_panel_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Only bind the wheel globally while the cursor is actually over the
        # side panel, so scrolling the main canvas elsewhere isn't hijacked.
        self.side_panel_canvas.bind(
            "<Enter>", lambda _e: self.side_panel_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        )
        self.side_panel_canvas.bind(
            "<Leave>", lambda _e: self.side_panel_canvas.unbind_all("<MouseWheel>")
        )

        self.tooltip = Tooltip(self.canvas)

    def _make_scroll_command(self, view_fn: "Callable") -> "Callable":
        """Wrap a canvas xview/yview so a scrollbar drag also triggers a
        viewport re-cull -- pure scrolling doesn't otherwise go through
        redraw_overlay()."""

        def _command(*args):
            view_fn(*args)
            self._on_viewport_changed()

        return _command

    def _on_viewport_changed(self) -> None:
        """Scrolled or resized: ask the current stage's renderer (if it
        registered one) to redraw just the delta for the new visible
        region, without rebuilding the whole stage (checkboxes, spatial
        indices, etc.)."""
        outputs = self.state.stage_cache.get(self.state.page_index)
        if not outputs:
            return
        stage_index = min(self.state.stage_index, len(outputs) - 1)
        output = outputs[stage_index]
        if output.status != "ok":
            return
        filters = self.state.filter_state.get(output.key)
        if not filters:
            return
        recull_all = filters.get("_recull_all")
        if recull_all is not None:
            recull_all()

    def open_pdf_dialog(self) -> None:
        initial_dir = REFERENCES_DIR if os.path.isdir(REFERENCES_DIR) else os.getcwd()
        path = filedialog.askopenfilename(
            title="Open PDF", initialdir=initial_dir, filetypes=[("PDF files", "*.pdf")]
        )
        if not path:
            return
        try:
            self.reader.close()
            self.reader = Reader(path, page_index=0)
            self.pdf_path = os.path.abspath(path)
            self.state = DebugAppState(page_index=0)
            self.root.title(f"rastervec Debug — {os.path.basename(path)}")
            self.render()
        except Exception as exc:
            messagebox.showerror("Unable to open PDF", str(exc))

    def change_zoom(self, direction: int) -> None:
        if direction > 0:
            self.state.zoom = min(MAX_ZOOM, self.state.zoom * ZOOM_STEP)
        elif direction < 0:
            self.state.zoom = max(MIN_ZOOM, self.state.zoom / ZOOM_STEP)
        self.render()

    def change_stage(self, delta: int) -> None:
        # Bounded by however many stages actually ran (self.final_stage may
        # truncate this well below len(self.pipeline.STAGES)), not the full
        # STAGES list -- _get_stage_outputs is cached, so this is free.
        stage_count = len(self._get_stage_outputs())
        new_index = self.state.stage_index + delta
        if not (0 <= new_index < stage_count):
            return
        self.state.stage_index = new_index
        self.redraw_overlay()

    def _get_stage_outputs(self) -> list[StageOutput]:
        page_index = self.state.page_index
        cached = self.state.stage_cache.get(page_index)
        if cached is not None:
            return cached
        outputs = self.pipeline.run_page(self.reader, page_index, final_stage=self.final_stage)
        self.state.stage_cache[page_index] = outputs
        return outputs

    def render(self) -> None:
        page = self.reader.get_page(self.state.page_index)
        matrix = fitz.Matrix(self.state.zoom, self.state.zoom)
        pixmap = page.fitz_page.get_pixmap(matrix=matrix)

        pil_image = Image.open(io.BytesIO(pixmap.pil_tobytes(format="PNG")))
        self._photo = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("page_image")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags=("page_image",))
        self.canvas.config(scrollregion=(0, 0, pixmap.width, pixmap.height))

        self.page_label.config(
            text=f"Page {page.meta.number} / {self.reader.original_page_count}"
        )
        self.zoom_label.config(text=f"{round(self.state.zoom * 100)}%")

        self.redraw_overlay()

    def redraw_overlay(self) -> None:
        # Bumping the epoch first means any chunked draw callback still in
        # flight from the previous stage/page/zoom (scheduled via
        # after_idle, so it may not have run yet even though we're about
        # to delete its items) will see a stale epoch and stop instead of
        # drawing more items after this cleanup runs.
        self._render_epoch_box["value"] += 1
        # Tag-based delete, not an id-list snapshot: a stage's chunked
        # draw can still be adding "overlay"-tagged items via after_idle
        # after redraw_overlay() has already returned once, so a
        # snapshot taken at the end of the previous call would miss them.
        self.canvas.delete("overlay")
        self.canvas.delete("overlay_hover")  # stray hover highlights from a switched-away stage
        self.canvas.unbind("<Motion>")  # only the vector-stage-buckets renderer rebinds these
        self.canvas.unbind("<Leave>")
        self.canvas.unbind("<Left>")  # only ocr_compare's inspector rebinds these
        self.canvas.unbind("<Right>")
        self.canvas.unbind("<ButtonPress-1>")  # only the reconstruction compare slider rebinds these
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        if self.tooltip is not None:
            self.tooltip.hide()
        for child in self.side_panel.winfo_children():
            child.destroy()
        self.side_panel_canvas.yview_moveto(0.0)  # reset scroll for the new stage's content

        outputs = self._get_stage_outputs()
        stage_count = len(outputs)
        stage_index = min(self.state.stage_index, stage_count - 1)
        self.state.stage_index = stage_index
        output = outputs[stage_index]

        self.stage_label.config(
            text=f"Stage {stage_index + 1}/{stage_count}: {output.label}"
        )

        if output.status == "error":
            self.status_label.config(text=f"stage failed: {output.error}")
            return
        self.status_label.config(text="")

        page = self.reader.get_page(self.state.page_index)
        matrix = _get_display_matrix(page.fitz_page, self.state.zoom)

        renderer = _STAGE_RENDERERS.get(output.key)
        if renderer is None:
            return

        filters = self.state.filter_state.setdefault(output.key, {})
        ctx = RenderContext(
            canvas=self.canvas,
            matrix=matrix,
            output=output,
            tooltip=self.tooltip,
            side_panel=self.side_panel,
            filters=filters,
            on_change=self.redraw_overlay,
            epoch_box=self._render_epoch_box,
            page=page,
            ocr_detail_cache=self.state.ocr_detail_cache,
            all_outputs=outputs,
        )
        renderer(ctx)

    def _on_close(self) -> None:
        try:
            self.reader.close()
        finally:
            self.root.destroy()


def _pick_initial_pdf() -> str | None:
    if not os.path.isdir(REFERENCES_DIR):
        return None
    pdfs = sorted(f for f in os.listdir(REFERENCES_DIR) if f.lower().endswith(".pdf"))
    return os.path.join(REFERENCES_DIR, pdfs[0]) if pdfs else None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug Tkinter app for the rastervec pipeline.")
    parser.add_argument("pdf", nargs="?", help="Path to the PDF to inspect.")
    parser.add_argument(
        "--page", type=int, default=0,
        help="0-based page index to load (default: 0). Only this page is "
        "loaded -- the app doesn't page through the rest of the document.",
    )
    parser.add_argument(
        "--final-stage",
        choices=Pipeline.stage_keys(),
        default=None,
        help="Stop the pipeline after this stage instead of running every stage -- e.g. "
        "--final-stage fast_text_detect skips ocr_compare (and the PaddleOCR "
        "engine it would otherwise build) entirely, so the stage-nav bar simply won't cycle "
        "past it. Default: run every stage.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging()

    pdf_path = args.pdf or _pick_initial_pdf()

    root = tk.Tk()
    if not pdf_path:
        pdf_path = filedialog.askopenfilename(title="Open PDF", filetypes=[("PDF files", "*.pdf")])
    if not pdf_path:
        root.destroy()
        return 0

    app = DebugApp(root, pdf_path, page_index=args.page, final_stage=args.final_stage)
    try:
        root.mainloop()
    finally:
        app.reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
