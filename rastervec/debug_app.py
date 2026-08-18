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
from typing import Callable

import io

import pymupdf as fitz
from PIL import Image, ImageTk

if __name__ == "__main__" and __package__ is None:
    # Allow running this file directly (`python rastervec/debug_app.py`),
    # not just as a module (`python -m rastervec.debug_app`), by putting
    # the repo root -- the parent of this package -- on sys.path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rastervec import geometry
from rastervec.logging_setup import configure_logging
from rastervec.models import DrawingVector, Page, TextWord, VectorPath
from rastervec.pipeline import ClusteringStageResult, Pipeline, StageOutput
from rastervec.reader import Reader
from rastervec.renderer import Renderer
from rastervec.vector import Vector

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
    # Keyed by (page_index, clustering_order) -- changing the clustering
    # stage's operation order is a full pipeline recompute (see
    # DebugApp._set_clustering_order), so it gets its own cache slot rather
    # than invalidating/replacing the page's other-order results (letting
    # the user flip back and forth between orders without recomputing).
    stage_cache: dict[tuple[int, tuple[str, ...]], list[StageOutput]] = field(
        default_factory=dict
    )
    # stage key -> arbitrary dict of checkbox state, kept across redraws of
    # that stage (reset per page since it's cached with stage_cache anyway).
    filter_state: dict[str, dict] = field(default_factory=dict)
    # The clustering stage's 4-operation order -- lives here (not in
    # filter_state) since changing it drives which cache slot above is used.
    clustering_order: list[str] = field(default_factory=lambda: list(Vector.CLUSTER_STEPS))


_DEFAULT_PATH_COLOR = "#111827"
_THIS_STAGE_COLOR = "#2563eb"
_PREVIOUS_COLOR = "#9ca3af"
_PENDING_GROUP_COLOR = "#059669"
_CENTROID_COLOR = "#dc2626"
_CENTROID_RADIUS = 3
_MAX_HOVER_HIGHLIGHT_PATHS = 400
_HOVER_TOLERANCE_PX = 5.0


_RENDERER = Renderer()  # stateless; the debug app's one shared instance


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
    # Only used by the clustering stage: the current 4-operation order, and
    # a callback to persist a new order + trigger the full pipeline
    # recompute it requires (see DebugApp._set_clustering_order).
    clustering_order: list[str] = field(default_factory=list)
    set_clustering_order: "Callable[[list[str]], None] | None" = None


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
    """page-space (unrotated) -> canvas-space, same rule as inspector/app.py:
    TextWord/geometry is canonical unrotated MediaBox space (models.py),
    so this is the one transform both the pixmap and every overlay must
    go through to land in the same canvas space."""
    return fitz_page.rotation_matrix * fitz.Matrix(zoom, zoom)


def _render_reader_stage(ctx: RenderContext) -> None:
    # Base PDF only -- no overlay, confirms the pixmap itself is correct.
    pass


def _render_native_stage(ctx: RenderContext) -> None:
    words: list[TextWord] = ctx.output.data or []

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


def _entry_bbox(item) -> tuple[float, float, float, float]:
    """item is either a group (list[VectorPath]) or a single VectorPath --
    filter stages mix both (path-level filters drop singleton paths,
    filter_aspect_ratio drops/keeps whole groups)."""
    return geometry.union_bbox([p.bbox for p in item]) if isinstance(item, list) else item.bbox


def _draw_filter_bucket_payload(ctx: RenderContext, category: str, payload) -> int | None:
    if not isinstance(payload, list):
        return _draw_vector_path(ctx.canvas, ctx.matrix, payload, tags=("overlay", category))
    if category == "bucket_this":
        color, width = _THIS_STAGE_COLOR, 2
    elif category == "bucket_previous":
        color, width = _PREVIOUS_COLOR, 1
    else:
        color, width = _PENDING_GROUP_COLOR, 1
    bbox = geometry.union_bbox([p.bbox for p in payload])
    return _draw_bbox(ctx.canvas, ctx.matrix, bbox, color, width=width, tags=("overlay", category))


def _render_filter_stage_buckets(ctx: RenderContext) -> None:
    """Shared renderer for the filter vector stages (filter_layout_panels,
    filter_large_bbox, filter_large_group_bbox, filter_aspect_ratio) -- each
    produces the same dict[(layer, color), VectorStageBuckets] shape (see
    pipeline.py). A filter stage never decides "text": what it drops is
    always drawing content (VectorStageBuckets's docstring), so the this/
    previous checkboxes read "classified as Drawing", not the generic
    "classified this stage".

    These stages can have tens of thousands of dropped items, which was
    laggy even with tag-based visibility toggling (thousands of live
    canvas items). Instead: build a spatial index per category once, and
    only ever draw the (small) subset whose bbox overlaps the current
    viewport, redrawing the delta on scroll/resize (_recull, wired to
    DebugApp._on_viewport_changed) and streaming even that in via small
    after_idle chunks so no single draw blocks the UI. Hover is computed
    on demand against the full in-memory bboxes regardless of what's
    currently drawn -- see _bind_bucket_hover."""
    buckets_by_group: dict = ctx.output.data or {}

    show_this = ctx.filters.setdefault("show_this_stage", {"value": True})
    show_previous = ctx.filters.setdefault("show_previous", {"value": False})
    show_pending = ctx.filters.setdefault("show_pending", {"value": True})

    all_this = [g for b in buckets_by_group.values() for g in b.this_stage if g]
    all_previous = [g for b in buckets_by_group.values() for g in b.previous if g]
    all_pending = [p for b in buckets_by_group.values() for p in b.pending]

    ttk.Label(ctx.side_panel, text=ctx.output.label).pack(anchor="w", padx=4, pady=(4, 6))

    categories = {
        "bucket_this": {
            "show": show_this, "entries": [(_entry_bbox(g), g) for g in all_this],
        },
        "bucket_previous": {
            "show": show_previous, "entries": [(_entry_bbox(g), g) for g in all_previous],
        },
        "bucket_pending": {
            "show": show_pending, "entries": [(_entry_bbox(p), p) for p in all_pending],
        },
    }
    for state in categories.values():
        state["index"] = _SpatialIndex(state["entries"])
        state["drawn"] = {}  # id(payload) -> canvas item id
        state["generation"] = {"value": 0}

    epoch = ctx.epoch_box["value"]

    def _draw_chunk(category: str, items: list, generation: dict, gen: int, start: int) -> None:
        if ctx.epoch_box["value"] != epoch or generation["value"] != gen:
            return  # superseded by a newer render/recull -- stop quietly
        drawn = categories[category]["drawn"]
        end = min(start + _RECULL_CHUNK_SIZE, len(items))
        for payload in items[start:end]:
            item_id = _draw_filter_bucket_payload(ctx, category, payload)
            if item_id is not None:
                drawn[id(payload)] = item_id
        if end < len(items):
            ctx.canvas.after_idle(lambda: _draw_chunk(category, items, generation, gen, end))

    def _recull(category: str) -> None:
        state = categories[category]
        generation = state["generation"]
        generation["value"] += 1
        gen = generation["value"]
        drawn: dict = state["drawn"]

        if not state["show"]["value"]:
            for item_id in drawn.values():
                ctx.canvas.delete(item_id)
            drawn.clear()
            return

        rect = _visible_page_rect(ctx)
        wanted = state["index"].query(rect)
        wanted_ids = {id(payload) for payload in wanted}

        for payload_id in list(drawn.keys()):
            if payload_id not in wanted_ids:
                ctx.canvas.delete(drawn.pop(payload_id))

        to_draw = [payload for payload in wanted if id(payload) not in drawn]
        if to_draw:
            _draw_chunk(category, to_draw, generation, gen, 0)

    def _recull_all() -> None:
        for category in categories:
            _recull(category)

    ctx.filters["_recull_all"] = _recull_all

    this_var = tk.BooleanVar(value=show_this["value"])
    _add_category_checkbox(
        ctx.side_panel, ctx.canvas, f"Classified as Drawing this round ({len(all_this)})",
        this_var, "bucket_this",
        persist=lambda v: (show_this.__setitem__("value", v), _recull("bucket_this")),
    )
    previous_var = tk.BooleanVar(value=show_previous["value"])
    _add_category_checkbox(
        ctx.side_panel, ctx.canvas, f"Previously classified as Drawing ({len(all_previous)})",
        previous_var, "bucket_previous",
        persist=lambda v: (show_previous.__setitem__("value", v), _recull("bucket_previous")),
    )
    pending_var = tk.BooleanVar(value=show_pending["value"])
    _add_category_checkbox(
        ctx.side_panel, ctx.canvas, f"Not yet classified ({len(all_pending)})", pending_var,
        "bucket_pending",
        persist=lambda v: (show_pending.__setitem__("value", v), _recull("bucket_pending")),
    )

    _recull_all()
    _bind_bucket_hover(ctx, (all_this, show_this), (all_previous, show_previous))


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


_CLUSTER_STEP_LABELS: dict[str, str] = {
    "cluster_spatial": "Spatial",
    "cluster_by_seq": "Sequence",
    "group_overlapping": "Overlap",
    "cluster_groups_by_dimension": "Dimension",
    "none": "None",
}
_CLUSTER_STEP_COLORS = ["#2563eb", "#7c3aed", "#ea580c", "#0d9488", "#6b7280"]
_ORDINAL_LABELS = ("1st", "2nd", "3rd", "4th")


def _render_clustering_stage(ctx: RenderContext) -> None:
    """The single configurable clustering/grouping stage: cluster_spatial,
    cluster_by_seq, group_overlapping, and cluster_groups_by_dimension all
    run here now, in whatever order the 4 dropdowns pick (default:
    Vector.CLUSTER_STEPS's order -- spatial, sequence, overlap, dimension).
    Changing a dropdown swaps it with whichever other dropdown currently
    holds that value (keeping the 4 dropdowns a valid permutation) and
    calls ctx.set_clustering_order, which persists the new order and
    triggers a full pipeline recompute for it (clustering's result depends
    on the whole chain up to this point, so there's no cheaper partial
    re-run).

    On the right: one checkbox per step showing that step's resulting
    clusters (bbox + centroid, in a distinct color per step so multiple
    can be compared at once) plus a live min/median/mean/max cluster-size
    summary (_cluster_size_stats_text, updated every time the order
    changes since each step's actual output changes), and a "previously
    classified as Drawing" checkbox for content the two filter stages
    before this one already dropped. Same viewport-culling + chunked-
    drawing approach as the filter stages, generalized from 2-3 categories
    to (4 steps + previous) here."""
    results_by_group: dict[tuple, ClusteringStageResult] = ctx.output.data or {}
    order = list(ctx.clustering_order) if ctx.clustering_order else list(Vector.CLUSTER_STEPS)

    ttk.Label(ctx.side_panel, text=ctx.output.label).pack(anchor="w", padx=4, pady=(4, 6))

    # --- order dropdowns ---
    label_to_key = {label: key for key, label in _CLUSTER_STEP_LABELS.items()}
    combo_vars: list[tk.StringVar] = []

    def _apply_order_change(changed_index: int) -> None:
        new_order = [label_to_key[var.get()] for var in combo_vars]
        chosen_key = new_order[changed_index]
        # "none" (skip this ordinal position) isn't part of the swap
        # permutation -- any number of dropdowns can be "none" at once, only
        # the 4 real steps need to stay unique.
        if chosen_key != "none":
            for j, key in enumerate(new_order):
                if j != changed_index and key == chosen_key:
                    # keep it a permutation: whatever was just displaced goes
                    # into the slot that used to hold the newly-chosen value.
                    new_order[j] = order[changed_index]
                    combo_vars[j].set(_CLUSTER_STEP_LABELS[order[changed_index]])
                    break
        if ctx.set_clustering_order is not None:
            ctx.set_clustering_order(new_order)

    for i, ordinal in enumerate(_ORDINAL_LABELS):
        row = ttk.Frame(ctx.side_panel)
        row.pack(anchor="w", fill="x", padx=4, pady=1)
        ttk.Label(row, text=f"{ordinal}:", width=5).pack(side="left")
        var = tk.StringVar(value=_CLUSTER_STEP_LABELS[order[i]])
        combo = ttk.Combobox(
            row, textvariable=var, state="readonly", width=11,
            values=list(_CLUSTER_STEP_LABELS.values()),
        )
        combo.pack(side="left")
        combo_vars.append(var)
        combo.bind("<<ComboboxSelected>>", lambda _event, i=i: _apply_order_change(i))

    ttk.Separator(ctx.side_panel, orient="horizontal").pack(fill="x", padx=4, pady=6)

    # --- per-step + previous categories ---
    all_previous = [g for result in results_by_group.values() for g in result.previous if g]
    step_groups: list[list[list[VectorPath]]] = [
        [
            g
            for result in results_by_group.values()
            for g in (result.steps[i] if i < len(result.steps) else [])
            if g
        ]
        for i in range(len(order))
    ]

    categories: dict[str, dict] = {
        "previous": {
            "show": ctx.filters.setdefault("show_previous", {"value": False}),
            "entries": [(geometry.union_bbox([p.bbox for p in g]), g) for g in all_previous],
            "color": _PREVIOUS_COLOR,
            "width": 1,
            "centroid": False,
        },
    }
    for i in range(len(order)):
        categories[f"step_{i}"] = {
            "show": ctx.filters.setdefault(
                f"show_step_{i}", {"value": i == len(order) - 1}
            ),
            "entries": [(geometry.union_bbox([p.bbox for p in g]), g) for g in step_groups[i]],
            "color": _CLUSTER_STEP_COLORS[i % len(_CLUSTER_STEP_COLORS)],
            "width": 2,
            "centroid": True,
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

    for i, ordinal in enumerate(_ORDINAL_LABELS[: len(order)]):
        key = f"step_{i}"
        state = categories[key]
        label = f"{ordinal}: {_CLUSTER_STEP_LABELS[order[i]]} ({len(step_groups[i])})"
        var = tk.BooleanVar(value=state["show"]["value"])
        _add_category_checkbox(
            ctx.side_panel, ctx.canvas, label, var, key,
            persist=lambda v, k=key: (categories[k]["show"].__setitem__("value", v), _recull(k)),
        )
        ttk.Label(
            ctx.side_panel, text=_cluster_size_stats_text(step_groups[i]), justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 6))

    previous_state = categories["previous"]
    previous_var = tk.BooleanVar(value=previous_state["show"]["value"])
    _add_category_checkbox(
        ctx.side_panel, ctx.canvas, f"Previously classified as Drawing ({len(all_previous)})",
        previous_var, "previous",
        persist=lambda v: (categories["previous"]["show"].__setitem__("value", v), _recull("previous")),
    )

    _recull_all()
    _bind_bucket_hover(
        ctx,
        *[(step_groups[i], categories[f"step_{i}"]["show"]) for i in range(len(order))],
        (all_previous, categories["previous"]["show"]),
    )


def _render_drawing_vectors_stage(ctx: RenderContext) -> None:
    drawing_vectors: list[DrawingVector] = ctx.output.data or []
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


# Stage key -> view-render function. Add one entry here alongside each new
# StageSpec in rastervec/pipeline.py's Pipeline.STAGES.
_STAGE_RENDERERS = {
    "reader": _render_reader_stage,
    "native": _render_native_stage,
    "vector_extract": _render_vector_extract_stage,
    "layer_separation": _render_layer_separation_stage,
    "color_separation": _render_color_separation_stage,
    "filter_layout_panels": _render_filter_stage_buckets,
    "filter_large_bbox": _render_filter_stage_buckets,
    "clustering": _render_clustering_stage,
    "filter_large_group_bbox": _render_filter_stage_buckets,
    "filter_aspect_ratio": _render_filter_stage_buckets,
    "drawing_vectors": _render_drawing_vectors_stage,
}


class DebugApp:
    def __init__(self, root: tk.Tk, pdf_path: str, page_index: int = 0) -> None:
        self.root = root
        self.pdf_path = os.path.abspath(pdf_path)
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

        side_panel_container = ttk.Frame(body, width=220)
        side_panel_container.pack(fill="y", side="right")
        side_panel_container.pack_propagate(False)
        self.side_panel = ttk.Frame(side_panel_container)
        self.side_panel.pack(fill="both", expand=True)

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
        cache_key = (self.state.page_index, tuple(self.state.clustering_order))
        outputs = self.state.stage_cache.get(cache_key)
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
        stage_count = len(self.pipeline.STAGES)
        new_index = self.state.stage_index + delta
        if not (0 <= new_index < stage_count):
            return
        self.state.stage_index = new_index
        self.redraw_overlay()

    def _get_stage_outputs(self) -> list[StageOutput]:
        page_index = self.state.page_index
        cache_key = (page_index, tuple(self.state.clustering_order))
        cached = self.state.stage_cache.get(cache_key)
        if cached is not None:
            return cached
        outputs = self.pipeline.run_page(
            self.reader, page_index, clustering_order=self.state.clustering_order,
        )
        self.state.stage_cache[cache_key] = outputs
        return outputs

    def _set_clustering_order(self, order: list[str]) -> None:
        """Changing the clustering stage's operation order changes what
        every group downstream of it looks like, so this is a full
        pipeline recompute (via _get_stage_outputs's new cache key) rather
        than a cheaper partial re-run."""
        self.state.clustering_order = order
        self.redraw_overlay()

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
        if self.tooltip is not None:
            self.tooltip.hide()
        for child in self.side_panel.winfo_children():
            child.destroy()

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
            clustering_order=list(self.state.clustering_order),
            set_clustering_order=self._set_clustering_order,
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

    app = DebugApp(root, pdf_path, page_index=args.page)
    try:
        root.mainloop()
    finally:
        app.reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
