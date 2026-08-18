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

from rastervec.logging_setup import configure_logging
from rastervec.models import DrawingVector, Page, TextWord, VectorPath
from rastervec.pipeline import Pipeline, StageOutput
from rastervec.reader import Reader

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
    stage_cache: dict[int, list[StageOutput]] = field(default_factory=dict)
    # stage key -> arbitrary dict of checkbox state, kept across redraws of
    # that stage (reset per page since it's cached with stage_cache anyway).
    filter_state: dict[str, dict] = field(default_factory=dict)


_DEFAULT_PATH_COLOR = "#111827"
_THIS_STAGE_COLOR = "#2563eb"
_PREVIOUS_COLOR = "#9ca3af"
_MAX_HOVER_HIGHLIGHT_PATHS = 400


def _path_color_hex(path: VectorPath, default: str = _DEFAULT_PATH_COLOR) -> str:
    """The path's own stroke/fill color as a hex string -- overlays render
    the PDF's real color; any B/W-style simplification is purely an
    internal classification concern (e.g. Vector's heuristics), never
    something the debug view substitutes in place of the real color."""
    color = path.stroke_color if path.stroke_color is not None else path.fill_color
    if color is None:
        return default
    return "#%02x%02x%02x" % tuple(min(255, max(0, round(c * 255))) for c in color)


@dataclass
class RenderContext:
    canvas: tk.Canvas
    matrix: "fitz.Matrix"
    output: StageOutput
    tooltip: Tooltip
    side_panel: ttk.Frame
    filters: dict  # persistent per-stage checkbox state; mutate freely
    on_change: "Callable[[], None]"


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


def _group_bbox(group: list[VectorPath]) -> tuple[float, float, float, float]:
    x0 = min(p.bbox[0] for p in group)
    y0 = min(p.bbox[1] for p in group)
    x1 = max(p.bbox[2] for p in group)
    y1 = max(p.bbox[3] for p in group)
    return (x0, y0, x1, y1)


def _draw_cluster_group(
    ctx: RenderContext, group: list[VectorPath], color: str, *, width: int = 2, dash=None,
    tag: str = "overlay", visible: bool = True,
) -> None:
    """One cluster/singleton group's bbox, hoverable: hovering it shows the
    bbox coordinates and highlights (in their own real colors, thicker) the
    individual vector paths that make up the cluster."""
    if not group:
        return
    bbox = _group_bbox(group)
    rect_id = _draw_bbox(
        ctx.canvas, ctx.matrix, bbox, color, width=width, dash=dash,
        tags=("overlay", tag), visible=visible,
    )

    def _on_enter(_event, group=group, bbox=bbox):
        # Capped so hovering a mega-cluster (the clustering safety cap can
        # leave thousands of paths in one group -- see helpers/clustering.py)
        # doesn't stall the UI creating that many canvas items on the fly.
        highlighted = group[:_MAX_HOVER_HIGHLIGHT_PATHS]
        for path in highlighted:
            _draw_vector_path(ctx.canvas, ctx.matrix, path, width=4, tags=("overlay_hover",))
        x0, y0, x1, y1 = bbox
        count_note = (
            f"{len(group)} path(s)"
            if len(group) <= _MAX_HOVER_HIGHLIGHT_PATHS
            else f"{len(group)} path(s) (showing first {_MAX_HOVER_HIGHLIGHT_PATHS})"
        )
        ctx.tooltip.show(
            ctx.canvas.winfo_pointerx(),
            ctx.canvas.winfo_pointery(),
            f"bbox: ({x0:.1f}, {y0:.1f}) - ({x1:.1f}, {y1:.1f})\n{count_note}",
        )

    def _on_leave(_event):
        # One tag-delete instead of unpacking a (possibly huge) id list --
        # cheaper for Tk and needs no id bookkeeping on RenderContext.
        ctx.canvas.delete("overlay_hover")
        ctx.tooltip.hide()

    ctx.canvas.tag_bind(rect_id, "<Enter>", _on_enter)
    ctx.canvas.tag_bind(rect_id, "<Leave>", _on_leave)


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


def _render_vector_stage_buckets(ctx: RenderContext) -> None:
    """Shared renderer for the 2 filter + 3 cluster vector stages -- each
    produces the same dict[(layer, color), VectorStageBuckets] shape (see
    pipeline.py), so one view suffices. Three checkboxes only (not one per
    cluster/path): what this exact stage decided, what an earlier stage in
    this 5-stage pipeline already decided, and what's still pending. Hover
    a cluster/path bbox to see its extent and highlight its member paths."""
    buckets_by_group: dict = ctx.output.data or {}

    show_this = ctx.filters.setdefault("show_this_stage", {"value": True})
    show_previous = ctx.filters.setdefault("show_previous", {"value": False})
    show_pending = ctx.filters.setdefault("show_pending", {"value": True})

    total_this = sum(len(b.this_stage) for b in buckets_by_group.values())
    total_previous = sum(len(b.previous) for b in buckets_by_group.values())
    total_pending = sum(len(b.pending) for b in buckets_by_group.values())

    ttk.Label(ctx.side_panel, text=ctx.output.label).pack(anchor="w", padx=4, pady=(4, 6))

    this_var = tk.BooleanVar(value=show_this["value"])
    _add_category_checkbox(
        ctx.side_panel, ctx.canvas, f"Classified this stage ({total_this})", this_var,
        "bucket_this", persist=lambda v: show_this.__setitem__("value", v),
    )
    previous_var = tk.BooleanVar(value=show_previous["value"])
    _add_category_checkbox(
        ctx.side_panel, ctx.canvas, f"Previously classified ({total_previous})", previous_var,
        "bucket_previous", persist=lambda v: show_previous.__setitem__("value", v),
    )
    pending_var = tk.BooleanVar(value=show_pending["value"])
    _add_category_checkbox(
        ctx.side_panel, ctx.canvas, f"Not yet classified ({total_pending})", pending_var,
        "bucket_pending", persist=lambda v: show_pending.__setitem__("value", v),
    )

    for group_buckets in buckets_by_group.values():
        for group in group_buckets.this_stage:
            _draw_cluster_group(
                ctx, group, _THIS_STAGE_COLOR, width=2,
                tag="bucket_this", visible=show_this["value"],
            )
        for group in group_buckets.previous:
            _draw_cluster_group(
                ctx, group, _PREVIOUS_COLOR, width=1, dash=(3, 2),
                tag="bucket_previous", visible=show_previous["value"],
            )
        for path in group_buckets.pending:
            _draw_vector_path(
                ctx.canvas, ctx.matrix, path,
                tags=("overlay", "bucket_pending"), visible=show_pending["value"],
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
    "filter_layout_panels": _render_vector_stage_buckets,
    "filter_background_fill": _render_vector_stage_buckets,
    "cluster_spatial": _render_vector_stage_buckets,
    "cluster_by_dimension": _render_vector_stage_buckets,
    "cluster_by_seq": _render_vector_stage_buckets,
    "drawing_vectors": _render_drawing_vectors_stage,
}


class DebugApp:
    def __init__(self, root: tk.Tk, pdf_path: str) -> None:
        self.root = root
        self.pdf_path = os.path.abspath(pdf_path)
        self.reader = Reader(self.pdf_path)
        self.pipeline = Pipeline()
        self.state = DebugAppState()

        self._photo = None
        self._overlay_ids: list[int] = []
        self.tooltip: Tooltip | None = None

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

        ttk.Button(page_bar, text="< Prev Page", command=lambda: self.change_page(-1)).pack(
            side="left", padx=2, pady=2
        )
        self.page_label = ttk.Label(page_bar, text="Page - / -")
        self.page_label.pack(side="left", padx=6)
        ttk.Button(page_bar, text="Next Page >", command=lambda: self.change_page(1)).pack(
            side="left", padx=2, pady=2
        )

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
        vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

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

    def open_pdf_dialog(self) -> None:
        initial_dir = REFERENCES_DIR if os.path.isdir(REFERENCES_DIR) else os.getcwd()
        path = filedialog.askopenfilename(
            title="Open PDF", initialdir=initial_dir, filetypes=[("PDF files", "*.pdf")]
        )
        if not path:
            return
        try:
            self.reader.close()
            self.reader = Reader(path)
            self.pdf_path = os.path.abspath(path)
            self.state = DebugAppState()
            self.root.title(f"rastervec Debug — {os.path.basename(path)}")
            self.render()
        except Exception as exc:
            messagebox.showerror("Unable to open PDF", str(exc))

    def change_page(self, delta: int) -> None:
        new_index = self.state.page_index + delta
        if not (0 <= new_index < self.reader.page_count()):
            return
        self.state.page_index = new_index
        self.render()

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
        cached = self.state.stage_cache.get(page_index)
        if cached is not None:
            return cached
        outputs = self.pipeline.run_page(self.reader, page_index)
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
            text=f"Page {self.state.page_index + 1} / {self.reader.page_count()}"
        )
        self.zoom_label.config(text=f"{round(self.state.zoom * 100)}%")

        self.redraw_overlay()

    def redraw_overlay(self) -> None:
        if self._overlay_ids:
            self.canvas.delete(*self._overlay_ids)
            self._overlay_ids = []
        self.canvas.delete("overlay_hover")  # stray hover highlights from a switched-away stage
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
        )
        renderer(ctx)
        self._overlay_ids = list(self.canvas.find_withtag("overlay"))

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

    app = DebugApp(root, pdf_path)
    try:
        root.mainloop()
    finally:
        app.reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
