"""Raster labelling: a Tk UI for drawing ground-truth text bounding boxes
plus straight-line/bezier-curve annotations directly over a rasterized page
-- no live `Vector` extraction/clustering at all (a raster tool has no
vectors to select). Same skeleton as `vector_label.py` (Reader-backed page
render, zoom/pan, save on close/page-change, inline label bar), simpler.

Works on any PDF page the same way every other tool does
(`page.get_pixmap()`), including a pre-rasterized PDF from
`scripts/rasterize_pdf.py` (which itself just emits a normal
one-image-per-page PDF, so it opens identically to any other source PDF).

Three tools (toolbar radio buttons):

- **Text**: click-drag-release a bbox (same rubber-band mechanics as
  `vector_label.py`'s selection drag); the inline label bar (text +
  rotation slider + direction arrow, ported verbatim) then creates a
  `LabelEntry(source="raster", vector_signatures=[])` on Apply.
- **Line**: click, drag (live preview), release -- two points ->
  `GeometryAnnotation(kind="l", points=[p1, p2])`.
- **Curve**: click four times in sequence (start, 2 control points, end) --
  live straight-guide preview, then a sampled cubic-bezier preview once all
  4 points exist. The 4th click finalizes ->
  `GeometryAnnotation(kind="c", points=[p1, p2, p3, p4])`. `Escape` cancels
  an in-progress line/curve.

No cluster/path "mode" toggle, no Group/Ungroup, no undo-stack -- a flat
per-page list of placed annotations, right-click to remove a mis-click.

Not unit-testable (a real Tk event loop). Smoke-test manually:

    .venv/Scripts/python.exe scripts/label/raster_label.py path/to.pdf --page 0

1. Text tool: drag a box over some text, type + rotate + Apply -- a solid
   box with text appears; hovering shows it.
2. Line tool: click-drag-release along a line -- a solid segment appears.
3. Curve tool: click 4 points -- a bezier curve appears; Escape mid-way
   cancels the in-progress curve.
4. Right-click an annotation to delete it. Save (or close) writes the
   label JSON (`entries` + `geometry_entries`).
"""
from __future__ import annotations

import argparse
import math
import sys
import uuid
from pathlib import Path
from tkinter import ttk
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tkinter as tk

import pymupdf as fitz

from rastervec.Evaluation.Labelling.label_schema import (
    GeometryAnnotation,
    LabelEntry,
    LabelSet,
    load_labels,
    save_labels,
)
from rastervec.helpers.geometry import bbox_contains
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.paths import output_dir
from rastervec.Reader.reader import Reader

_LOG = get_logger("raster_label")

MIN_ZOOM = 0.25
MAX_ZOOM = 6.0
ZOOM_STEP = 1.25
ROTATION_SNAP_DEG = 2.5
_DRAG_THRESHOLD_PX = 4

_ZOOM = 1.5
_ENTRY_COLOR = "#33aa33"
_STALE_COLOR = "#999999"
_SELECTED_COLOR = "#ff8800"
_LINE_COLOR = "#3366ff"
_CURVE_COLOR = "#cc33cc"


def _get_display_matrix(fitz_page: "fitz.Page", zoom: float) -> "fitz.Matrix":
    return fitz_page.rotation_matrix * fitz.Matrix(zoom, zoom)


class Tooltip:
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


def _bezier_points(p0, p1, p2, p3, n: int = 24) -> list[tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = (mt**3) * p0[0] + 3 * (mt**2) * t * p1[0] + 3 * mt * (t**2) * p2[0] + (t**3) * p3[0]
        y = (mt**3) * p0[1] + 3 * (mt**2) * t * p1[1] + 3 * mt * (t**2) * p2[1] + (t**3) * p3[1]
        pts.append((x, y))
    return pts


class RasterLabelApp:
    def __init__(self, pdf_path: str, page_index: int, out_path: str) -> None:
        self.pdf_path = pdf_path
        self.out_path = out_path
        self.reader = Reader(pdf_path)
        self.labels = (
            load_labels(out_path) if Path(out_path).exists()
            else LabelSet(pdf_path=pdf_path)
        )

        self._tool: Literal["text", "line", "curve"] = "text"
        self._active_label_id: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_moved = False
        self._drag_rect_id: int | None = None
        self._curve_points: list[tuple[float, float]] = []
        self._curve_preview_ids: list[int] = []
        self.zoom = _ZOOM

        self.root = tk.Tk()
        self.tooltip = Tooltip(self.root)
        self._build_layout()
        self._bind_events()
        self._load_page(page_index)

    # ---- per-page state -------------------------------------------------

    def _load_page(self, page_index: int) -> None:
        self.page_index = max(0, min(self.reader.page_count() - 1, page_index))
        self.page = self.reader.get_page(self.page_index)
        self.matrix = _get_display_matrix(self.page.fitz_page, self.zoom)
        self._curve_points.clear()
        self._active_label_id = None
        self.root.title(
            f"Raster Label -- {Path(self.pdf_path).name} "
            f"page {self.page_index + 1}/{self.reader.page_count()}"
        )
        self._sync_page_entry()
        self._clear_label_panel()
        self._render()

    def _page_entries(self) -> list[LabelEntry]:
        return [e for e in self.labels.entries if e.page_index == self.page_index]

    def _page_geometry(self) -> list[GeometryAnnotation]:
        return [g for g in self.labels.geometry_entries if g.page_index == self.page_index]

    # ---- layout -----------------------------------------------------------

    def _build_layout(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Zoom -", command=lambda: self._change_zoom(-1)).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(bar, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Zoom +", command=lambda: self._change_zoom(1)).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._tool_var = tk.StringVar(value=self._tool)
        for value, label in (("text", "Text"), ("line", "Line"), ("curve", "Curve")):
            ttk.Radiobutton(
                bar, text=label, value=value, variable=self._tool_var,
                command=self._on_tool_change,
            ).pack(side=tk.LEFT)

        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(bar, text="<", width=3, command=lambda: self._change_page(-1)).pack(side=tk.LEFT, padx=1)
        self._page_var = tk.StringVar(value="1")
        page_entry = ttk.Entry(bar, textvariable=self._page_var, width=4, justify="center")
        page_entry.pack(side=tk.LEFT)
        page_entry.bind("<Return>", self._on_page_entry_return)
        self._page_count_label = ttk.Label(bar, text="/ 1")
        self._page_count_label.pack(side=tk.LEFT, padx=(1, 1))
        ttk.Button(bar, text=">", width=3, command=lambda: self._change_page(1)).pack(side=tk.LEFT, padx=1)

        self._status = ttk.Label(bar, text="")
        self._status.pack(side=tk.RIGHT, padx=8)

        label_bar = ttk.Frame(self.root)
        label_bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(label_bar, text="Text").pack(side=tk.LEFT, padx=(4, 2))
        self._text_var = tk.StringVar()
        self._text_entry = ttk.Entry(label_bar, textvariable=self._text_var, width=40)
        self._text_entry.pack(side=tk.LEFT)
        self._text_entry.bind("<Return>", lambda _e: self._apply_label())

        ttk.Label(label_bar, text="Rot").pack(side=tk.LEFT, padx=(8, 2))
        self._rot_var = tk.DoubleVar(value=0.0)
        self._rot_scale = ttk.Scale(
            label_bar, from_=-180.0, to=180.0, orient="horizontal", length=160,
            variable=self._rot_var, command=self._on_rotation_change,
        )
        self._rot_scale.pack(side=tk.LEFT, padx=(0, 4))
        self._rot_readout = ttk.Label(label_bar, text="0.0°", width=7)
        self._rot_readout.pack(side=tk.LEFT)

        ttk.Button(label_bar, text="Apply", command=self._apply_label).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(label_bar, text="Delete", command=self._delete_label).pack(side=tk.LEFT, padx=2)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="#808080")
        vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        ttk.Button(self.root, text="Save", command=self.save).pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_events(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.canvas.bind("<Button-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _e: self.tooltip.hide())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_wheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        self.root.bind("<Escape>", lambda _e: self._cancel_in_progress())
        self.root.bind("<Next>", lambda _e: self._change_page(1))
        self.root.bind("<Prior>", lambda _e: self._change_page(-1))

    # ---- coordinate helpers -------------------------------------------

    def _page_point(self, event: "tk.Event") -> "fitz.Point":
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        return fitz.Point(cx, cy) * ~self.matrix

    # ---- rendering ----------------------------------------------------

    def _render(self) -> None:
        self.canvas.delete("all")
        pix = self.page.fitz_page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
        self._photo = tk.PhotoImage(data=pix.tobytes("ppm"))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.config(scrollregion=(0, 0, pix.width, pix.height))

        for entry in self._page_entries():
            rect = fitz.Rect(entry.cluster_bbox) * self.matrix
            selected = entry.label_id == self._active_label_id
            color = _SELECTED_COLOR if selected else _ENTRY_COLOR
            width = 3 if selected else 2
            self.canvas.create_rectangle(
                rect.x0, rect.y0, rect.x1, rect.y1, outline=color, width=width,
                tags=("entry", entry.label_id),
            )

        for geo in self._page_geometry():
            self._draw_geometry(geo, tag=f"geo:{id(geo)}")

        self.zoom_label.config(text=f"{round(self.zoom * 100)}%")
        self._status.config(
            text=f"{self._tool} tool  |  {len(self._page_entries())} text label(s)  "
                 f"|  {len(self._page_geometry())} line/curve annotation(s)"
        )
        self._refresh_label_hint()
        self._draw_rotation_arrow()

    def _draw_geometry(self, geo: GeometryAnnotation, *, tag: str, preview: bool = False) -> None:
        color = _LINE_COLOR if geo.kind == "l" else _CURVE_COLOR
        if geo.kind == "l" and len(geo.points) == 2:
            pts = geo.points
        elif geo.kind == "c" and len(geo.points) == 4:
            pts = _bezier_points(*geo.points)
        else:
            pts = geo.points
        coords: list[float] = []
        for x, y in pts:
            p = fitz.Point(x, y) * self.matrix
            coords.extend([p.x, p.y])
        if len(coords) < 4:
            return
        dash = (3, 2) if preview else None
        self.canvas.create_line(
            *coords, fill=color, width=2, dash=dash, tags=("overlay", "geometry", tag),
        )

    # ---- zoom / nav (unchanged pattern) --------------------------------

    def _change_zoom(self, direction: int) -> None:
        if direction > 0:
            self.zoom = min(MAX_ZOOM, self.zoom * ZOOM_STEP)
        else:
            self.zoom = max(MIN_ZOOM, self.zoom / ZOOM_STEP)
        self.matrix = _get_display_matrix(self.page.fitz_page, self.zoom)
        self._render()

    def _on_wheel(self, event: "tk.Event") -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_shift_wheel(self, event: "tk.Event") -> None:
        self.canvas.xview_scroll(int(-event.delta / 120), "units")

    def _on_ctrl_wheel(self, event: "tk.Event") -> None:
        self._change_zoom(1 if event.delta > 0 else -1)

    def _sync_page_entry(self) -> None:
        self._page_var.set(str(self.page_index + 1))
        self._page_count_label.config(text=f"/ {self.reader.page_count()}")

    def _change_page(self, delta: int) -> None:
        self._goto_page(self.page_index + delta)

    def _on_page_entry_return(self, _event: "tk.Event") -> None:
        try:
            self._goto_page(int(self._page_var.get()) - 1)
        except ValueError:
            self._sync_page_entry()

    def _goto_page(self, new_index: int) -> None:
        new_index = max(0, min(self.reader.page_count() - 1, new_index))
        if new_index == self.page_index:
            self._sync_page_entry()
            return
        self.save()
        self.tooltip.hide()
        self._load_page(new_index)

    # ---- tool switching -------------------------------------------------

    def _on_tool_change(self) -> None:
        self._tool = self._tool_var.get()
        self._cancel_in_progress()
        self._clear_label_panel()
        self._render()

    def _cancel_in_progress(self) -> None:
        self._curve_points.clear()
        for cid in self._curve_preview_ids:
            self.canvas.delete(cid)
        self._curve_preview_ids.clear()
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None
        self._drag_start = None

    # ---- mouse: text/line drag, curve click-sequence -------------------

    def _on_left_press(self, event: "tk.Event") -> None:
        if self._tool == "curve":
            self._on_curve_click(event)
            return
        self._drag_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self._drag_moved = False

    def _on_left_drag(self, event: "tk.Event") -> None:
        if self._tool == "curve" or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if not self._drag_moved and (
            abs(x1 - x0) < _DRAG_THRESHOLD_PX and abs(y1 - y0) < _DRAG_THRESHOLD_PX
        ):
            return
        self._drag_moved = True
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
        if self._tool == "text":
            self._drag_rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=_SELECTED_COLOR, width=1, dash=(3, 2), tags=("selrect",),
            )
        else:
            self._drag_rect_id = self.canvas.create_line(
                x0, y0, x1, y1, fill=_LINE_COLOR, width=2, dash=(3, 2), tags=("selrect",),
            )

    def _on_left_release(self, event: "tk.Event") -> None:
        if self._tool == "curve":
            return
        start = self._drag_start
        self._drag_start = None
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None
        if start is None or not self._drag_moved:
            if self._tool == "text" and start is not None:
                self._click_select_text(fitz.Point(*start) * ~self.matrix)
            return
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        inv = ~self.matrix
        p0, p1 = fitz.Point(*start) * inv, fitz.Point(cx, cy) * inv
        if self._tool == "text":
            rect = (min(p0.x, p1.x), min(p0.y, p1.y), max(p0.x, p1.x), max(p0.y, p1.y))
            self._start_new_text_label(rect)
        else:  # line
            geo = GeometryAnnotation(
                page_index=self.page_index, kind="l", points=[(p0.x, p0.y), (p1.x, p1.y)],
            )
            self.labels.geometry_entries.append(geo)
            self._render()

    def _on_curve_click(self, event: "tk.Event") -> None:
        pt = self._page_point(event)
        self._curve_points.append((pt.x, pt.y))
        for cid in self._curve_preview_ids:
            self.canvas.delete(cid)
        self._curve_preview_ids.clear()
        if len(self._curve_points) < 4:
            coords: list[float] = []
            for x, y in self._curve_points:
                p = fitz.Point(x, y) * self.matrix
                coords.extend([p.x, p.y])
            if len(coords) >= 4:
                cid = self.canvas.create_line(
                    *coords, fill=_CURVE_COLOR, width=1, dash=(2, 2), tags=("overlay",),
                )
                self._curve_preview_ids.append(cid)
            return
        geo = GeometryAnnotation(page_index=self.page_index, kind="c", points=list(self._curve_points))
        self._curve_points.clear()
        self.labels.geometry_entries.append(geo)
        self._render()

    # ---- text label lifecycle ------------------------------------------

    def _click_select_text(self, pt: "fitz.Point") -> None:
        hit = next(
            (e for e in self._page_entries() if bbox_contains(e.cluster_bbox, pt.x, pt.y)), None,
        )
        if hit is None:
            self._active_label_id = None
            self._clear_label_panel()
        else:
            self._active_label_id = hit.label_id
            self._active_bbox = hit.cluster_bbox
            self._prefill_label_bar(hit)
        self._render()

    def _start_new_text_label(self, rect: tuple[float, float, float, float]) -> None:
        self._active_label_id = None
        self._active_bbox = rect
        self._text_var.set("")
        self._rot_var.set(0.0)
        self._rot_readout.config(text="0.0°")
        self._refresh_label_hint()
        self._render()
        self._draw_pending_rect(rect)
        self._text_entry.focus_set()
        self._text_entry.selection_range(0, tk.END)

    def _draw_pending_rect(self, rect: tuple[float, float, float, float]) -> None:
        r = fitz.Rect(rect) * self.matrix
        self.canvas.create_rectangle(
            r.x0, r.y0, r.x1, r.y1, outline=_SELECTED_COLOR, width=2, tags=("overlay", "pending"),
        )

    def _clear_label_panel(self) -> None:
        self._active_label_id = None
        self._active_bbox = None
        self._text_var.set("")
        self._rot_var.set(0.0)
        self._rot_readout.config(text="0.0°")
        self._refresh_label_hint()

    def _prefill_label_bar(self, entry: LabelEntry) -> None:
        self._text_var.set(entry.text)
        self._rot_var.set(float(entry.expected_rotation))
        self._rot_readout.config(text=f"{entry.expected_rotation:.1f}°")
        self._refresh_label_hint()

    def _refresh_label_hint(self) -> None:
        pass  # status bar already summarizes counts; label bar is self-explanatory

    def _on_rotation_change(self, value: str) -> None:
        snapped = round(float(value) / ROTATION_SNAP_DEG) * ROTATION_SNAP_DEG
        self._rot_var.set(snapped)
        self._rot_readout.config(text=f"{snapped:.1f}°")
        self._draw_rotation_arrow()

    def _draw_rotation_arrow(self) -> None:
        self.canvas.delete("rotation_arrow")
        bbox = getattr(self, "_active_bbox", None)
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        length = max(x1 - x0, y1 - y0) / 2 or 1.0
        angle_rad = math.radians(self._rot_var.get())
        tip_x = cx + length * math.cos(angle_rad)
        tip_y = cy + length * math.sin(angle_rad)
        c1 = fitz.Point(cx, cy) * self.matrix
        c2 = fitz.Point(tip_x, tip_y) * self.matrix
        self.canvas.create_line(
            c1.x, c1.y, c2.x, c2.y, fill=_SELECTED_COLOR, width=2,
            arrow=tk.LAST, tags=("overlay", "rotation_arrow"),
        )

    def _apply_label(self) -> None:
        bbox = getattr(self, "_active_bbox", None)
        if bbox is None:
            return
        text = self._text_var.get().strip()
        if not text:
            return
        rotation = int(round(self._rot_var.get()))
        if self._active_label_id is not None:
            idx = self._entry_by_label_id(self._active_label_id)
            if idx is not None:
                entry = self.labels.entries[idx]
                entry.text = text
                entry.expected_rotation = rotation
        else:
            n = len(self._page_entries())
            entry = LabelEntry(
                page_index=self.page_index, cluster_bbox=bbox,
                cluster_signature=f"raster:{self.page_index}:{n}",
                label_id=uuid.uuid4().hex,
                text=text, source="raster", expected_rotation=rotation,
                vector_signatures=[],
            )
            self.labels.entries.append(entry)
        self._clear_label_panel()
        self._render()

    def _entry_by_label_id(self, label_id: str) -> int | None:
        for i, entry in enumerate(self.labels.entries):
            if entry.label_id == label_id:
                return i
        return None

    def _delete_label(self) -> None:
        if self._active_label_id is None:
            return
        idx = self._entry_by_label_id(self._active_label_id)
        if idx is not None:
            del self.labels.entries[idx]
        self._clear_label_panel()
        self._render()

    # ---- right-click: delete a text label or a line/curve --------------

    def _on_right_click(self, event: "tk.Event") -> None:
        pt = self._page_point(event)
        for i, entry in enumerate(self._page_entries()):
            if bbox_contains(entry.cluster_bbox, pt.x, pt.y):
                real_idx = self.labels.entries.index(entry)
                del self.labels.entries[real_idx]
                if entry.label_id == self._active_label_id:
                    self._clear_label_panel()
                self._render()
                return
        for geo in self._page_geometry():
            if self._point_near_geometry(pt, geo):
                self.labels.geometry_entries.remove(geo)
                self._render()
                return

    def _point_near_geometry(self, pt: "fitz.Point", geo: GeometryAnnotation, tol: float = 6.0) -> bool:
        pts = geo.points if geo.kind == "l" else _bezier_points(*geo.points)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if _point_segment_distance(pt.x, pt.y, x0, y0, x1, y1) <= tol:
                return True
        return False

    # ---- hover -------------------------------------------------------

    def _on_motion(self, event: "tk.Event") -> None:
        pt = self._page_point(event)
        for entry in self._page_entries():
            if bbox_contains(entry.cluster_bbox, pt.x, pt.y):
                self.tooltip.show(
                    event.x_root, event.y_root, f'"{entry.text}"  rot={entry.expected_rotation}',
                )
                return
        self.tooltip.hide()

    # ---- persistence ------------------------------------------------

    def save(self) -> None:
        save_labels(self.labels, self.out_path)
        _LOG.info(
            "saved %d label(s), %d geometry annotation(s) to %s",
            len(self.labels.entries), len(self.labels.geometry_entries), self.out_path,
        )

    def _on_close(self) -> None:
        self.save()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
        self.reader.close()


def _point_segment_distance(px, py, x0, y0, x1, y1) -> float:
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw ground-truth text boxes + line/curve annotations over a rasterized page.")
    parser.add_argument("pdf", help="Path to the input PDF.")
    parser.add_argument("--page", type=int, default=0, help="0-based page index.")
    parser.add_argument(
        "--out", default=None,
        help="Path to save/load the label JSON file "
        "(default: outputs/labels/<pdf stem>.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)
    out = args.out or str(output_dir("labels") / f"{Path(args.pdf).stem}.json")
    app = RasterLabelApp(args.pdf, args.page, out)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
