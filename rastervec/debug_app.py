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
from PIL import Image, ImageDraw, ImageTk

if __name__ == "__main__" and __package__ is None:
    # Allow running this file directly (`python rastervec/debug_app.py`),
    # not just as a module (`python -m rastervec.debug_app`), by putting
    # the repo root -- the parent of this package -- on sys.path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rastervec import geometry
from rastervec.helpers.render_ocr import RenderOCR
from rastervec.logging_setup import configure_logging
from rastervec.models import DrawingVector, Page, TextVectorResult, TextWord, VectorPath
from rastervec.pipeline import ClusteringStageResult, Pipeline, StageOutput
from rastervec.reader import Reader
from rastervec.renderer import Renderer
from rastervec.vector_classification import (
    DEFAULT_PIPELINE,
    PAIRWISE_METRICS,
    SCALAR_METRICS,
    StepConfig,
    load_from_file,
    save_to_file,
)

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
    # Keyed by (page_index, steps_cache_key) -- changing the classification
    # step list (add/remove/reorder/edit any step) is a full pipeline
    # recompute (see DebugApp._set_classification_steps), so each distinct
    # step list gets its own cache slot rather than invalidating/replacing
    # the page's other-steps results (letting the user flip back and forth
    # without recomputing).
    stage_cache: dict[tuple, list[StageOutput]] = field(default_factory=dict)
    # stage key -> arbitrary dict of checkbox state, kept across redraws of
    # that stage (reset per page since it's cached with stage_cache anyway).
    filter_state: dict[str, dict] = field(default_factory=dict)
    # The clustering stage's step list -- lives here (not in filter_state)
    # since changing it drives which cache slot above is used.
    classification_steps: list[StepConfig] = field(
        default_factory=lambda: [StepConfig(**vars(s)) for s in DEFAULT_PIPELINE]
    )
    # Staged (not-yet-applied) edits for the clustering stage's step-list
    # editor -- decoupled from classification_steps so editing several rows
    # doesn't trigger a recompute after each one; only clicking "Apply
    # Changes" copies these over (see DebugApp._apply_pending_classification_
    # changes). Reset to mirror the committed value every time the
    # clustering stage is (re)entered.
    pending_classification_steps: list[StepConfig] = field(
        default_factory=lambda: [StepConfig(**vars(s)) for s in DEFAULT_PIPELINE]
    )
    # ocr_text_clusters inspector: (page_index, steps_cache_key, cluster
    # index) -> {"image", "bbox_corners"} (see _ocr_cluster_preview).
    # Deliberately its own dict, not filter_state -- filter_state is keyed
    # only by stage key (shared across every page/order visit of that
    # stage, fine for cheap UI toggles), but this holds real recomputed
    # render+OCR data that's only valid for the exact page/steps/cluster it
    # was built from.
    ocr_detail_cache: dict[tuple, dict] = field(default_factory=dict)
    # Last stage key redraw_overlay() actually rendered -- used to reset
    # pending_classification_steps only when the clustering stage is newly
    # (re)entered, not on every redraw (a row edit in the clustering stage
    # also calls redraw_overlay() to rebuild the editor's widgets; without
    # this guard that redraw would immediately stomp the just-made edit).
    last_stage_key: str | None = None


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
    # Only used by the clustering stage: the current committed step list,
    # and a callback to persist a new list + trigger the full pipeline
    # recompute it requires (see DebugApp._set_classification_steps).
    classification_steps: list[StepConfig] = field(default_factory=list)
    set_classification_steps: "Callable[[list[StepConfig]], None] | None" = None
    # Staged (uncommitted) step list the clustering stage's row editor
    # actually reads/writes -- see DebugAppState.pending_classification_
    # steps. Writing to this never triggers a redraw; only
    # apply_pending_classification_changes() (wired to the "Apply Changes"
    # button) copies it into classification_steps above and recomputes.
    pending_classification_steps: list[StepConfig] = field(default_factory=list)
    set_pending_classification_steps: "Callable[[list[StepConfig]], None] | None" = None
    apply_pending_classification_changes: "Callable[[], None] | None" = None
    # The live Page for the page currently being viewed -- only
    # ocr_text_clusters needs it today (to re-render a cluster on demand
    # for its inspector panel; see _ocr_cluster_preview), but it's cheap and
    # generally useful, so it's populated for every stage, not just that one.
    page: Page | None = None
    # DebugAppState.ocr_detail_cache, threaded through so
    # _render_ocr_text_clusters_stage can key/read/write it without needing
    # a back-reference to DebugApp itself.
    ocr_detail_cache: dict[tuple, dict] | None = None


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


def _add_reconstruction_toggle(ctx: RenderContext, build_image_fn) -> bool:
    """Adds a "Show Reconstructed PDF"/"Show Original PDF" toggle button to
    the top of ctx.side_panel, for the (few) stages that support it --
    native, drawing_vectors, ocr_text_clusters. Persisted per-stage in
    ctx.filters (survives redraws of that stage the same way every other
    checkbox here does), so switching stages away and back remembers
    whether reconstruction was showing.

    When active, this draws the reconstructed page (built lazily by
    calling `build_image_fn()`, a zero-arg callable the caller supplies so
    the actual Renderer.render_reconstructed_page call -- and whatever
    page_meta/element-list plumbing it needs -- stays local to each
    stage's own renderer) as one big canvas item, tagged "overlay" so the
    next redraw_overlay()'s canvas.delete("overlay") cleans it up
    automatically. It's drawn *on top of* the real page pixmap (which is
    never touched/hidden) rather than replacing it -- since the
    reconstruction covers the exact same rect at the exact same zoom, this
    fully occludes the original without risking the base "page_image"
    item's separate lifecycle (that item is only ever (re)created by
    DebugApp.render() on page/zoom changes, not on every stage switch).

    Returns whether reconstruction is currently showing -- callers should
    skip drawing their normal overlay entirely when this is True, since
    the toggle replaces the overlay rather than layering under it."""
    state = ctx.filters.setdefault("show_reconstructed", {"value": False})

    def _toggle() -> None:
        state["value"] = not state["value"]
        ctx.on_change()

    label = "Show Original PDF" if state["value"] else "Show Reconstructed PDF"
    ttk.Button(ctx.side_panel, text=label, command=_toggle).pack(
        anchor="w", fill="x", padx=4, pady=(4, 8)
    )

    if state["value"]:
        image = build_image_fn()
        photo = ImageTk.PhotoImage(image)
        ctx.filters["_reconstructed_photo_ref"] = photo  # keep alive -- Tk won't retain it otherwise
        ctx.canvas.create_image(0, 0, anchor="nw", image=photo, tags=("overlay",))

    return state["value"]


def _render_reader_stage(ctx: RenderContext) -> None:
    # Base PDF only -- no overlay, confirms the pixmap itself is correct.
    pass


def _render_native_stage(ctx: RenderContext) -> None:
    words: list[TextWord] = ctx.output.data or []

    if ctx.page is not None and _add_reconstruction_toggle(
        ctx, lambda: _RENDERER.render_reconstructed_page(
            ctx.page.meta, native_words=words, zoom=geometry.matrix_scale(ctx.matrix)[0],
        ),
    ):
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


def _steps_cache_key(steps: list[StepConfig]) -> tuple:
    """Hashable representation of a classification_steps list, for use as
    (part of) a DebugAppState.stage_cache key -- dataclasses/lists aren't
    hashable, and two calls with equal-but-freshly-built lists (as
    redraw_overlay's RenderContext construction does) must still hit the
    same cache slot."""
    def _hashable_params(params: dict) -> tuple:
        return tuple(sorted(params.items()))

    def _hashable_threshold(threshold) -> tuple | float | None:
        return tuple(threshold) if isinstance(threshold, list) else threshold

    return tuple(
        (
            s.kind, s.metric, s.condition, _hashable_threshold(s.threshold), s.scope,
            s.method, s.aggregate, s.aggregate_scope, _hashable_params(s.params), s.label,
        )
        for s in steps
    )


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

# Row-editor dropdown option lists -- mirrors step_config.py's own
# validation constants (kept private there, so duplicated here as plain
# tuples rather than imported).
_KINDS = ("filter", "cluster", "group")
_CONDITIONS = (">", ">=", "<", "<=", "==", "!=", "within", "outside")
_FILTER_SCOPES = ("item", "cluster", "items_in_cluster", "cluster_all_items")
_CLUSTER_METHODS = ("pairwise", "global")
_CLUSTER_SCOPES = ("within_group", "global_flatten")
_GROUP_SCOPES = ("path", "cluster")
_AGGREGATES = ("none", "mean", "median")
_AGGREGATE_SCOPES = ("global", "group")
_EXTRA_PARAM_CHOICES = {"bbox_source": ("path", "item")}


def _default_step_for_kind(kind: str) -> StepConfig:
    if kind == "filter":
        return StepConfig(kind="filter", metric="bbox_area_fraction", condition="<=", threshold=0.2, scope="item")
    if kind == "cluster":
        return StepConfig(kind="cluster", metric="spatial_gap", method="global", threshold=8.0)
    return StepConfig(kind="group", metric="spatial_gap", scope="cluster", threshold=8.0)


def _metric_options_for_step(step: StepConfig) -> dict:
    if step.kind == "cluster":
        return SCALAR_METRICS if step.method == "pairwise" else PAIRWISE_METRICS
    if step.kind == "group":
        return PAIRWISE_METRICS
    return SCALAR_METRICS


def _parse_threshold_text(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _render_clustering_stage(ctx: RenderContext) -> None:
    """The single configurable pipeline stage: a dynamic list of
    StepConfig entries (rastervec/vector_classification/step_config.py),
    each independently a filter/cluster/group step driven by a named
    metric from SCALAR_METRICS/PAIRWISE_METRICS (rastervec/
    vector_classification/metrics.py). Every StepConfig is self-contained
    -- no shared name-keyed param dict -- so two occurrences of the same
    metric at different positions hold fully independent thresholds/params
    (default: DEFAULT_PIPELINE, reproducing the original fixed 5-step
    pipeline).

    Editing is staged, not live: the row editor below reads/writes only
    ctx.pending_classification_steps (see DebugAppState's field of the
    same name) -- no recompute happens until the "Apply Changes" button
    calls ctx.apply_pending_classification_changes, which copies pending
    -> committed and triggers the full pipeline recompute (every step's
    result depends on the whole chain up to that point, so there's no
    cheaper partial re-run). Row edits do call ctx.on_change() to rebuild
    the editor's widgets (e.g. switching a row's kind changes which fields
    apply), but that's a cache hit against the *committed* steps (the
    stage-entry pending-reset in DebugApp.redraw_overlay only fires once
    per stage visit, not on every redraw -- see DebugAppState.
    last_stage_key), so it never triggers a real pipeline recompute.
    Because of this, two distinct step lists exist below: `pending_steps`
    drives the row editor (what the user is currently editing), while
    `committed_steps` drives the kept/dropped results section (what the
    last Apply actually ran) -- they can diverge until Apply is clicked.

    Save Config / Load Config persist/restore `pending_steps` as JSON
    (rastervec/vector_classification/step_config.py's to_json/from_json) --
    loading still requires clicking Apply to take effect, same as any
    other edit.

    Below the editor: two checkboxes per step -- "kept after step i" (that
    step's surviving groups: bbox + centroid, in a distinct color per
    step) and "dropped by step i" (only non-empty for a filter step --
    what it removed, classified as Drawing, grey, no centroid) -- plus a
    live min/median/mean/max cluster-size summary for the kept side. Same
    viewport-culling + chunked-drawing approach as other high-volume
    stages, generalized to 2 categories per step."""
    results_by_group: dict[tuple, ClusteringStageResult] = ctx.output.data or {}
    committed_steps = list(ctx.classification_steps) if ctx.classification_steps else [
        StepConfig(**vars(s)) for s in DEFAULT_PIPELINE
    ]
    pending_steps = list(ctx.pending_classification_steps) if ctx.pending_classification_steps else list(committed_steps)

    ttk.Label(ctx.side_panel, text=ctx.output.label).pack(anchor="w", padx=4, pady=(4, 6))

    def _apply_changes() -> None:
        if ctx.apply_pending_classification_changes is not None:
            ctx.apply_pending_classification_changes()

    button_row = ttk.Frame(ctx.side_panel)
    button_row.pack(anchor="w", fill="x", padx=4, pady=(0, 4))
    ttk.Button(button_row, text="Apply Changes", command=_apply_changes).pack(side="left")

    def _set_pending(new_steps: list[StepConfig]) -> None:
        if ctx.set_pending_classification_steps is not None:
            ctx.set_pending_classification_steps(new_steps)
        ctx.on_change()  # rebuild this stage's panel with the new rows -- no recompute (see docstring)

    def _save_config() -> None:
        path = filedialog.asksaveasfilename(
            title="Save classification config", defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        try:
            save_to_file(pending_steps, path)
        except Exception as exc:
            messagebox.showerror("Unable to save config", str(exc))

    def _load_config() -> None:
        path = filedialog.askopenfilename(
            title="Load classification config", filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        try:
            loaded = load_from_file(path)
        except Exception as exc:
            messagebox.showerror("Unable to load config", str(exc))
            return
        _set_pending(loaded)

    ttk.Button(button_row, text="Save Config", command=_save_config).pack(side="left", padx=(4, 0))
    ttk.Button(button_row, text="Load Config", command=_load_config).pack(side="left", padx=(4, 0))

    # --- dynamic step-list editor ---
    def _update_step(index: int, **changes) -> None:
        new_steps = [StepConfig(**vars(s)) for s in pending_steps]
        step = new_steps[index]
        for key, value in changes.items():
            setattr(step, key, value)
        _set_pending(new_steps)

    def _replace_step(index: int, new_step: StepConfig) -> None:
        new_steps = [StepConfig(**vars(s)) for s in pending_steps]
        new_steps[index] = new_step
        _set_pending(new_steps)

    def _move_step(index: int, delta: int) -> None:
        target = index + delta
        if not (0 <= target < len(pending_steps)):
            return
        new_steps = [StepConfig(**vars(s)) for s in pending_steps]
        new_steps[index], new_steps[target] = new_steps[target], new_steps[index]
        _set_pending(new_steps)

    def _delete_step(index: int) -> None:
        new_steps = [StepConfig(**vars(s)) for s in pending_steps]
        del new_steps[index]
        _set_pending(new_steps)

    def _add_step() -> None:
        _set_pending([StepConfig(**vars(s)) for s in pending_steps] + [_default_step_for_kind("filter")])

    editor = ttk.Frame(ctx.side_panel)
    editor.pack(anchor="w", fill="x")

    for i, step in enumerate(pending_steps):
        row_frame = ttk.LabelFrame(editor, text=f"Step {i + 1}")
        row_frame.pack(anchor="w", fill="x", padx=4, pady=3)

        top_row = ttk.Frame(row_frame)
        top_row.pack(anchor="w", fill="x", padx=4, pady=2)

        ttk.Label(top_row, text="Kind:", width=6).pack(side="left")
        kind_var = tk.StringVar(value=step.kind)
        kind_combo = ttk.Combobox(
            top_row, textvariable=kind_var, state="readonly", width=8, values=_KINDS,
        )
        kind_combo.pack(side="left")
        kind_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e, i=i, v=kind_var: _replace_step(i, _default_step_for_kind(v.get())),
        )

        ttk.Button(top_row, text="^", width=2, command=lambda i=i: _move_step(i, -1)).pack(side="left", padx=(6, 0))
        ttk.Button(top_row, text="v", width=2, command=lambda i=i: _move_step(i, 1)).pack(side="left")
        ttk.Button(top_row, text="X", width=2, command=lambda i=i: _delete_step(i)).pack(side="left")

        metric_row = ttk.Frame(row_frame)
        metric_row.pack(anchor="w", fill="x", padx=4, pady=2)
        ttk.Label(metric_row, text="Metric:", width=6).pack(side="left")
        metric_options = _metric_options_for_step(step)
        metric_labels = [spec.label for spec in metric_options.values()]
        label_to_metric = {spec.label: key for key, spec in metric_options.items()}
        current_metric_label = metric_options[step.metric].label if step.metric in metric_options else (
            metric_labels[0] if metric_labels else ""
        )
        metric_var = tk.StringVar(value=current_metric_label)
        metric_combo = ttk.Combobox(
            metric_row, textvariable=metric_var, state="readonly", width=26, values=metric_labels,
        )
        metric_combo.pack(side="left")
        metric_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e, i=i, v=metric_var, m=label_to_metric: _update_step(i, metric=m[v.get()]),
        )

        if step.kind == "cluster":
            method_row = ttk.Frame(row_frame)
            method_row.pack(anchor="w", fill="x", padx=4, pady=2)
            ttk.Label(method_row, text="Method:", width=6).pack(side="left")
            method_var = tk.StringVar(value=step.method or "global")
            method_combo = ttk.Combobox(
                method_row, textvariable=method_var, state="readonly", width=10, values=_CLUSTER_METHODS,
            )
            method_combo.pack(side="left")

            def _on_method_change(_e, i=i, v=method_var):
                new_method = v.get()
                default_metric = "spatial_gap" if new_method == "global" else next(iter(SCALAR_METRICS))
                _update_step(i, method=new_method, metric=default_metric)

            method_combo.bind("<<ComboboxSelected>>", _on_method_change)

            if step.method == "pairwise":
                scope_row = ttk.Frame(row_frame)
                scope_row.pack(anchor="w", fill="x", padx=4, pady=2)
                ttk.Label(scope_row, text="Scope:", width=6).pack(side="left")
                scope_var = tk.StringVar(value=step.scope if step.scope in _CLUSTER_SCOPES else _CLUSTER_SCOPES[0])
                scope_combo = ttk.Combobox(
                    scope_row, textvariable=scope_var, state="readonly", width=14, values=_CLUSTER_SCOPES,
                )
                scope_combo.pack(side="left")
                scope_combo.bind(
                    "<<ComboboxSelected>>",
                    lambda _e, i=i, v=scope_var: _update_step(i, scope=v.get()),
                )

        elif step.kind == "filter":
            scope_row = ttk.Frame(row_frame)
            scope_row.pack(anchor="w", fill="x", padx=4, pady=2)
            ttk.Label(scope_row, text="Scope:", width=6).pack(side="left")
            scope_var = tk.StringVar(value=step.scope if step.scope in _FILTER_SCOPES else _FILTER_SCOPES[0])
            scope_combo = ttk.Combobox(
                scope_row, textvariable=scope_var, state="readonly", width=16, values=_FILTER_SCOPES,
            )
            scope_combo.pack(side="left")
            scope_combo.bind(
                "<<ComboboxSelected>>",
                lambda _e, i=i, v=scope_var: _update_step(i, scope=v.get()),
            )

            cond_row = ttk.Frame(row_frame)
            cond_row.pack(anchor="w", fill="x", padx=4, pady=2)
            ttk.Label(cond_row, text="Cond:", width=6).pack(side="left")
            cond_var = tk.StringVar(value=step.condition or _CONDITIONS[0])
            cond_combo = ttk.Combobox(
                cond_row, textvariable=cond_var, state="readonly", width=8, values=_CONDITIONS,
            )
            cond_combo.pack(side="left")
            cond_combo.bind(
                "<<ComboboxSelected>>",
                lambda _e, i=i, v=cond_var: _update_step(
                    i, condition=v.get(),
                    threshold=[0.0, 1.0] if v.get() in ("within", "outside") else 1.0,
                ),
            )

            agg_row = ttk.Frame(row_frame)
            agg_row.pack(anchor="w", fill="x", padx=4, pady=2)
            ttk.Label(agg_row, text="Aggr:", width=6).pack(side="left")
            agg_var = tk.StringVar(value=step.aggregate or "none")
            agg_combo = ttk.Combobox(
                agg_row, textvariable=agg_var, state="readonly", width=8, values=_AGGREGATES,
            )
            agg_combo.pack(side="left")
            agg_combo.bind(
                "<<ComboboxSelected>>",
                lambda _e, i=i, v=agg_var: _update_step(
                    i, aggregate=None if v.get() == "none" else v.get(),
                ),
            )
            if step.aggregate:
                agg_scope_var = tk.StringVar(
                    value=step.aggregate_scope if step.aggregate_scope in _AGGREGATE_SCOPES else _AGGREGATE_SCOPES[0]
                )
                agg_scope_combo = ttk.Combobox(
                    agg_row, textvariable=agg_scope_var, state="readonly", width=8, values=_AGGREGATE_SCOPES,
                )
                agg_scope_combo.pack(side="left", padx=(4, 0))
                agg_scope_combo.bind(
                    "<<ComboboxSelected>>",
                    lambda _e, i=i, v=agg_scope_var: _update_step(i, aggregate_scope=v.get()),
                )

        else:  # group
            scope_row = ttk.Frame(row_frame)
            scope_row.pack(anchor="w", fill="x", padx=4, pady=2)
            ttk.Label(scope_row, text="Scope:", width=6).pack(side="left")
            scope_var = tk.StringVar(value=step.scope if step.scope in _GROUP_SCOPES else _GROUP_SCOPES[0])
            scope_combo = ttk.Combobox(
                scope_row, textvariable=scope_var, state="readonly", width=10, values=_GROUP_SCOPES,
            )
            scope_combo.pack(side="left")
            scope_combo.bind(
                "<<ComboboxSelected>>",
                lambda _e, i=i, v=scope_var: _update_step(i, scope=v.get()),
            )

        thr_row = ttk.Frame(row_frame)
        thr_row.pack(anchor="w", fill="x", padx=4, pady=2)
        ttk.Label(thr_row, text="Threshold:", width=9).pack(side="left")
        if step.kind == "filter" and step.condition in ("within", "outside"):
            lo, hi = (step.threshold if isinstance(step.threshold, list) else [0.0, 1.0])
            lo_var = tk.StringVar(value=str(lo))
            hi_var = tk.StringVar(value=str(hi))

            def _commit_range(i=i, lo_var=lo_var, hi_var=hi_var):
                lo_val = _parse_threshold_text(lo_var.get())
                hi_val = _parse_threshold_text(hi_var.get())
                if lo_val is None or hi_val is None:
                    return
                _update_step(i, threshold=[lo_val, hi_val])

            lo_entry = ttk.Entry(thr_row, textvariable=lo_var, width=6)
            lo_entry.pack(side="left")
            ttk.Label(thr_row, text="-").pack(side="left")
            hi_entry = ttk.Entry(thr_row, textvariable=hi_var, width=6)
            hi_entry.pack(side="left")
            lo_entry.bind("<Return>", lambda _e, c=_commit_range: c())
            lo_entry.bind("<FocusOut>", lambda _e, c=_commit_range: c())
            hi_entry.bind("<Return>", lambda _e, c=_commit_range: c())
            hi_entry.bind("<FocusOut>", lambda _e, c=_commit_range: c())
        else:
            thr_var = tk.StringVar(value=str(step.threshold if step.threshold is not None else 1.0))

            def _commit_threshold(i=i, v=thr_var):
                value = _parse_threshold_text(v.get())
                if value is None:
                    return
                _update_step(i, threshold=value)

            thr_entry = ttk.Entry(thr_row, textvariable=thr_var, width=8)
            thr_entry.pack(side="left")
            thr_entry.bind("<Return>", lambda _e, c=_commit_threshold: c())
            thr_entry.bind("<FocusOut>", lambda _e, c=_commit_threshold: c())

        metric_spec = metric_options.get(step.metric)
        if metric_spec is not None and metric_spec.extra_params:
            extra_row = ttk.Frame(row_frame)
            extra_row.pack(anchor="w", fill="x", padx=4, pady=2)
            for extra_param in metric_spec.extra_params:
                choices = _EXTRA_PARAM_CHOICES.get(extra_param, ())
                ttk.Label(extra_row, text=f"{extra_param}:", width=9).pack(side="left")
                current = step.params.get(extra_param, choices[0] if choices else "")
                extra_var = tk.StringVar(value=str(current))
                if choices:
                    extra_combo = ttk.Combobox(
                        extra_row, textvariable=extra_var, state="readonly", width=8, values=choices,
                    )
                    extra_combo.pack(side="left")

                    def _commit_extra(_e, i=i, p=extra_param, v=extra_var):
                        new_steps = [StepConfig(**vars(s)) for s in pending_steps]
                        new_steps[i].params = {**new_steps[i].params, p: v.get()}
                        _set_pending(new_steps)

                    extra_combo.bind("<<ComboboxSelected>>", _commit_extra)
                else:
                    extra_entry = ttk.Entry(extra_row, textvariable=extra_var, width=8)
                    extra_entry.pack(side="left")

                    def _commit_extra_text(_e, i=i, p=extra_param, v=extra_var):
                        new_steps = [StepConfig(**vars(s)) for s in pending_steps]
                        new_steps[i].params = {**new_steps[i].params, p: v.get()}
                        _set_pending(new_steps)

                    extra_entry.bind("<Return>", _commit_extra_text)
                    extra_entry.bind("<FocusOut>", _commit_extra_text)

        label_row = ttk.Frame(row_frame)
        label_row.pack(anchor="w", fill="x", padx=4, pady=2)
        ttk.Label(label_row, text="Label:", width=6).pack(side="left")
        label_var = tk.StringVar(value=step.label)

        def _commit_label(i=i, v=label_var):
            _update_step(i, label=v.get())

        label_entry = ttk.Entry(label_row, textvariable=label_var, width=20)
        label_entry.pack(side="left")
        label_entry.bind("<Return>", lambda _e, c=_commit_label: c())
        label_entry.bind("<FocusOut>", lambda _e, c=_commit_label: c())

    ttk.Button(editor, text="+ Add Step", command=_add_step).pack(anchor="w", fill="x", padx=4, pady=(2, 8))

    ttk.Separator(ctx.side_panel, orient="horizontal").pack(fill="x", padx=4, pady=6)

    # --- per-step kept/dropped categories (reflect the last *applied* run) ---
    steps = committed_steps
    kept_groups: list[list[list[VectorPath]]] = [
        [g for result in results_by_group.values() for g in (result.steps[i] if i < len(result.steps) else []) if g]
        for i in range(len(steps))
    ]
    dropped_groups: list[list[list[VectorPath]]] = [
        [g for result in results_by_group.values() for g in (result.dropped[i] if i < len(result.dropped) else []) if g]
        for i in range(len(steps))
    ]

    categories: dict[str, dict] = {}
    for i in range(len(steps)):
        categories[f"kept_{i}"] = {
            "show": ctx.filters.setdefault(f"show_kept_{i}", {"value": i == len(steps) - 1}),
            "entries": [(geometry.union_bbox([p.bbox for p in g]), g) for g in kept_groups[i]],
            "color": _CLUSTER_STEP_COLORS[i % len(_CLUSTER_STEP_COLORS)],
            "width": 2,
            "centroid": True,
        }
        categories[f"dropped_{i}"] = {
            "show": ctx.filters.setdefault(f"show_dropped_{i}", {"value": False}),
            "entries": [(geometry.union_bbox([p.bbox for p in g]), g) for g in dropped_groups[i]],
            "color": _DROPPED_COLOR,
            "width": 1,
            "centroid": False,
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

    for i, step in enumerate(steps):
        kept_key, dropped_key = f"kept_{i}", f"dropped_{i}"
        step_label = step.label or f"{step.kind}:{step.metric}"

        kept_var = tk.BooleanVar(value=categories[kept_key]["show"]["value"])
        _add_category_checkbox(
            ctx.side_panel, ctx.canvas,
            f"{i + 1}: {step_label} kept ({len(kept_groups[i])})", kept_var, kept_key,
            persist=lambda v, k=kept_key: (categories[k]["show"].__setitem__("value", v), _recull(k)),
        )
        ttk.Label(
            ctx.side_panel, text=_cluster_size_stats_text(kept_groups[i]), justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 2))

        dropped_var = tk.BooleanVar(value=categories[dropped_key]["show"]["value"])
        _add_category_checkbox(
            ctx.side_panel, ctx.canvas,
            f"{i + 1}: {step_label} dropped ({len(dropped_groups[i])})", dropped_var, dropped_key,
            persist=lambda v, k=dropped_key: (categories[k]["show"].__setitem__("value", v), _recull(k)),
        )
        ttk.Frame(ctx.side_panel, height=6).pack()

    _recull_all()
    _bind_bucket_hover(
        ctx,
        *[
            (kept_groups[i], categories[f"kept_{i}"]["show"])
            for i in range(len(steps))
        ],
        *[
            (dropped_groups[i], categories[f"dropped_{i}"]["show"])
            for i in range(len(steps))
        ],
    )


def _render_drawing_vectors_stage(ctx: RenderContext) -> None:
    drawing_vectors: list[DrawingVector] = ctx.output.data or []

    if ctx.page is not None and _add_reconstruction_toggle(
        ctx, lambda: _RENDERER.render_reconstructed_page(
            ctx.page.meta, drawing_vectors=drawing_vectors,
            zoom=geometry.matrix_scale(ctx.matrix)[0],
        ),
    ):
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


def _ocr_passed(result: TextVectorResult) -> bool:
    return bool(result.text.strip())


def _ocr_cluster_preview(result: TextVectorResult, page: Page) -> dict:
    """Renders one OCR'd text cluster at the single rotation it was
    actually read at (result.rotation_used) and re-OCRs just that one
    rotation, purely to recover the detected-text bbox in that rendered
    image's own pixel space for the inspector's overlay -- the production
    TextVectorResult keeps `text`/`confidence` (used directly below, not
    re-derived here) but not that bbox. One render + one OCR call per
    cluster -- callers only care which cluster this is, not the other 7
    rotations RenderOCR.ocr_cluster tried and discarded, so unlike an
    earlier version of this panel, this never replays all 8."""
    image = _RENDERER.render_vector_cluster(result.cluster_paths, page, dpi=300).convert("RGB")
    angle = result.rotation_used % 360
    if angle:
        image = image.rotate(
            -angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC,
        )
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


def _render_ocr_text_clusters_stage(ctx: RenderContext) -> None:
    results: list[TextVectorResult] = ctx.output.data or []
    passed = [r for r in results if _ocr_passed(r)]
    failed = [r for r in results if not _ocr_passed(r)]

    ttk.Label(ctx.side_panel, text=f"{len(results)} OCR'd text cluster(s)").pack(
        anchor="w", padx=4, pady=(4, 6)
    )

    if ctx.page is not None and _add_reconstruction_toggle(
        ctx, lambda: _RENDERER.render_reconstructed_page(
            ctx.page.meta, ocr_results=passed, zoom=geometry.matrix_scale(ctx.matrix)[0],
        ),
    ):
        return

    show_passed = ctx.filters.setdefault("show_passed", {"value": True})
    show_failed = ctx.filters.setdefault("show_failed", {"value": True})
    selected = ctx.filters.setdefault("selected_index", {"value": 0})

    def _visible(result: TextVectorResult) -> bool:
        return show_passed["value"] if _ocr_passed(result) else show_failed["value"]

    visible_indices = [i for i, r in enumerate(results) if _visible(r)]
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

    # Small counts per page (text-as-vector-paths clusters are rare
    # compared to raw path counts), so plain tag_bind hover -- like
    # _render_native_stage -- is plenty; no need for the filter stages'
    # viewport-culling/spatial-index machinery here.
    for i in visible_indices:
        result = results[i]
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
                f"{r.text!r}\nconfidence: {r.confidence:.2f}\n"
                f"rotation used: {r.rotation_used}deg\n{len(r.cluster_paths)} path(s)\n"
                "(click to inspect)",
            ),
        )
        ctx.canvas.tag_bind(item_id, "<Leave>", lambda _event: ctx.tooltip.hide())
        ctx.canvas.tag_bind(item_id, "<Button-1>", lambda _event, idx=i: _select(idx))

    # --- inspector panel: rendered image (at rotation_used) + cluster nav ---
    if selected["value"] is not None and ctx.page is not None and ctx.ocr_detail_cache is not None:
        index = selected["value"]
        result = results[index]
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

        cache_key = (ctx.page.meta.index, _steps_cache_key(ctx.classification_steps), index)
        if cache_key not in ctx.ocr_detail_cache:
            ctx.ocr_detail_cache[cache_key] = _ocr_cluster_preview(result, ctx.page)
        preview = ctx.ocr_detail_cache[cache_key]

        photo = _ocr_preview_photo(preview)
        ctx.filters["_photo_ref"] = photo  # keep a reference so Tk doesn't GC it
        tk.Label(ctx.side_panel, image=photo, bg="#808080").pack(padx=4, pady=(6, 4))

        lines = [
            f"text: {result.text!r}",
            f"confidence: {result.confidence:.3f}",
            f"rotation used: {result.rotation_used}deg",
            f"{len(result.cluster_paths)} path(s)",
            "passed" if _ocr_passed(result) else "failed (no text detected)",
        ]
        info = tk.Text(
            ctx.side_panel, height=7, width=26, wrap="word", font=("TkDefaultFont", 8),
        )
        info.insert("1.0", "\n".join(lines))
        info.configure(state="disabled")
        info.pack(fill="both", expand=True, padx=4, pady=(4, 8))

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


# Stage key -> view-render function. Add one entry here alongside each new
# StageSpec in rastervec/pipeline.py's Pipeline.STAGES.
_STAGE_RENDERERS = {
    "reader": _render_reader_stage,
    "native": _render_native_stage,
    "vector_extract": _render_vector_extract_stage,
    "layer_separation": _render_layer_separation_stage,
    "color_separation": _render_color_separation_stage,
    "clustering": _render_clustering_stage,
    "drawing_vectors": _render_drawing_vectors_stage,
    "ocr_text_clusters": _render_ocr_text_clusters_stage,
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
        cache_key = (self.state.page_index, _steps_cache_key(self.state.classification_steps))
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
        cache_key = (page_index, _steps_cache_key(self.state.classification_steps))
        cached = self.state.stage_cache.get(cache_key)
        if cached is not None:
            return cached
        outputs = self.pipeline.run_page(
            self.reader, page_index,
            classification_steps=self.state.classification_steps,
            final_stage=self.final_stage,
        )
        self.state.stage_cache[cache_key] = outputs
        return outputs

    def _set_classification_steps(self, steps: list[StepConfig]) -> None:
        """Changing the clustering stage's step list changes what every
        group downstream of it looks like, so this is a full pipeline
        recompute (via _get_stage_outputs's new cache key) rather than a
        cheaper partial re-run."""
        self.state.classification_steps = steps
        self.redraw_overlay()

    def _set_pending_classification_steps(self, steps: list[StepConfig]) -> None:
        """Stages a step-list edit without recomputing -- see
        _apply_pending_classification_changes."""
        self.state.pending_classification_steps = steps

    def _apply_pending_classification_changes(self) -> None:
        """"Apply Changes" button: commits the staged step-list edits and
        triggers the full pipeline recompute they require."""
        self.state.classification_steps = [
            StepConfig(**vars(s)) for s in self.state.pending_classification_steps
        ]
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
        self.canvas.unbind("<Left>")  # only ocr_text_clusters' inspector rebinds these
        self.canvas.unbind("<Right>")
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

        if output.key == "clustering" and self.state.last_stage_key != "clustering":
            # Only reset pending edits back to the committed steps when the
            # clustering stage is newly (re)entered -- not on every redraw,
            # since a row edit inside that stage also calls redraw_overlay()
            # (via on_change) to rebuild the editor's widgets, and that call
            # must not clobber the edit it's rendering.
            self.state.pending_classification_steps = [
                StepConfig(**vars(s)) for s in self.state.classification_steps
            ]
        self.state.last_stage_key = output.key

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
            classification_steps=[StepConfig(**vars(s)) for s in self.state.classification_steps],
            set_classification_steps=self._set_classification_steps,
            pending_classification_steps=[
                StepConfig(**vars(s)) for s in self.state.pending_classification_steps
            ],
            set_pending_classification_steps=self._set_pending_classification_steps,
            apply_pending_classification_changes=self._apply_pending_classification_changes,
            page=page,
            ocr_detail_cache=self.state.ocr_detail_cache,
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
        "--final-stage drawing_vectors skips ocr_text_clusters (and the PaddleOCR engine "
        "it would otherwise build) entirely, so the stage-nav bar simply won't cycle past "
        "it. Default: run every stage.",
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
