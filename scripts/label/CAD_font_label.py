"""CAD-font character + baseline labelling -- step 1 of
`docs/cad_font_vector_recognition.md`. A small Tk UI with two independent
modes:

- **Baseline**: drag out a new baseline (a line with a direction, saved as
  `Baseline(origin, direction)`), or click an existing one to select it as
  active -- drawn extended across the page's own mediabox, active baseline
  highlighted. Selecting a baseline shows a side-panel checklist of every
  `cad_font` character label on the page, checked iff currently assigned to
  the active baseline (`LabelEntry.baseline_id`); toggling a checkbox
  assigns/unassigns it. Right-click deletes the nearest baseline and clears
  `baseline_id` on any label that referenced it.
- **Label**: identical to `scripts/label/vector_label.py`'s own flow (flat
  vector pool, click/ctrl-click/drag-select, inline text+rotation label bar,
  click-a-labelled-vector edits in place) -- fully independent of baseline
  state; it never reads or writes `baseline_id`. New entries get
  `source="cad_font"`.

Per this feature's "commons-only" import constraint, `extract_vectors` comes
from `rastervec.P1_Reading_Native.vector_extract` (not
`rastervec.pipelines._steps`, which `vector_label.py` uses), and there is no
layer/color/width bucket-filter side panel (its backing
`separate_by_layer_color_width` lives in the off-limits
`rastervec.pipelines.current`) -- `self.vectors` is always the full,
unfiltered, always-visible/clickable pool.

Not unit-testable (a real Tk event loop). Smoke-test manually:

    .venv/Scripts/python.exe scripts/label/CAD_font_label.py path/to.pdf --page 0

1. Window opens in Baseline mode. Drag across a run of CAD text -- a solid
   line appears, extended to the page edges, with an arrowhead; it becomes
   active, and the side panel shows a checklist of cad_font labels on the
   page (empty at first).
2. Switch to Label mode -- select vectors for a character, type text,
   Apply; the vectors turn green. Click one again to edit in place, same as
   `vector_label.py`.
3. Switch back to Baseline mode, re-select the baseline, check a few
   characters in the side panel -- their `baseline_id` is set. Drag a
   second baseline -- the checklist state is independent per baseline.
4. Right-click a baseline -- it's deleted, and any character it had been
   assigned to shows unchecked again under every remaining baseline.
5. Page nav / close -- Save (`label_schema.save_labels`) persists both
   `entries` and `baselines`.
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

from _common import (
    DRAG_THRESHOLD_PX,
    ENTRY_COLOR,
    LABELLED_COLOR,
    MAX_ZOOM,
    MIN_ZOOM,
    ROTATION_SNAP_DEG,
    SELECTED_COLOR,
    Tooltip,
    UNLABELLED_COLOR,
    ZOOM_DEFAULT,
    ZOOM_STEP,
    draw_vector,
    get_display_matrix,
)
from rastervec.Evaluation.Labelling.label_schema import (
    Baseline,
    LabelEntry,
    LabelSet,
    cluster_signature,
    load_labels,
    path_signature,
    save_labels,
    vector_signatures_for,
)
from rastervec.commons.helpers.geometry import (
    bbox_area,
    bbox_contains,
    bboxes_intersect,
    union_bbox,
)
from rastervec.commons.logging_setup import configure_logging, get_logger
from rastervec.commons.models import Vector
from rastervec.commons.paths import output_dir
from rastervec.P1_Reading_Native.reader import Reader
from rastervec.P1_Reading_Native.vector_extract import extract_vectors

_LOG = get_logger("cad_font_label")

_BASELINE_COLOR = "#00b3b3"
_BASELINE_HIT_TOL_PX = 6.0


class CadFontLabelApp:
    def __init__(self, pdf_path: str, page_index: int, out_path: str) -> None:
        self.pdf_path = pdf_path
        self.out_path = out_path
        self.reader = Reader(pdf_path)
        self.labels = (
            load_labels(out_path) if Path(out_path).exists()
            else LabelSet(pdf_path=pdf_path)
        )

        self._mode: Literal["baseline", "label"] = "baseline"

        # ---- Label-mode state (verbatim VectorLabelApp fields) ----
        self.selected: set[int] = set()
        self._active_label_id: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_moved = False
        self._drag_rect_id: int | None = None

        # ---- Baseline-mode state ----
        self._active_baseline_id: str | None = None
        self._baseline_drag_start: tuple[float, float] | None = None
        self._baseline_drag_moved = False
        self._baseline_preview_id: int | None = None
        self._baseline_checkbox_vars: dict[str, "tk.BooleanVar"] = {}

        self.zoom = ZOOM_DEFAULT

        # tk.Tk() must exist before any tk.StringVar()/tk.BooleanVar() is
        # created -- those need a default root window.
        self.root = tk.Tk()
        self.tooltip = Tooltip(self.root)
        self._mode_var = tk.StringVar(value=self._mode)

        self._build_layout()
        self._bind_events()
        self._load_page(page_index)

    # ---- per-page state -------------------------------------------------

    def _load_page(self, page_index: int) -> None:
        self.page_index = max(0, min(self.reader.page_count() - 1, page_index))
        self.page = self.reader.get_page(self.page_index)
        self.vectors: list[Vector] = extract_vectors(self.page)
        self._vectors_by_id: dict[int, Vector] = {id(v): v for v in self.vectors}

        self.selected.clear()
        self._active_label_id = None
        self._active_baseline_id = None
        self.matrix = get_display_matrix(self.page.fitz_page, self.zoom)

        self.root.title(
            f"CAD Font Label -- {Path(self.pdf_path).name} "
            f"page {self.page_index + 1}/{self.reader.page_count()}"
        )
        self._sync_page_entry()
        self._clear_label_panel()
        self._render()

    def _cad_font_page_entries(self) -> list[LabelEntry]:
        return [
            e for e in self.labels.entries
            if e.page_index == self.page_index and e.source == "cad_font"
        ]

    def _page_baselines(self) -> list[Baseline]:
        return [b for b in self.labels.baselines if b.page_index == self.page_index]

    def _labelled_sig_map(self) -> dict[str, LabelEntry]:
        out: dict[str, LabelEntry] = {}
        for entry in self._cad_font_page_entries():
            for sig in entry.vector_signatures:
                out[sig] = entry
        return out

    # ---- layout ---------------------------------------------------------

    def _build_layout(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Zoom -", command=lambda: self._change_zoom(-1)).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(bar, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Zoom +", command=lambda: self._change_zoom(1)).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        for value, label in (("baseline", "Baseline"), ("label", "Label")):
            ttk.Radiobutton(
                bar, text=label, value=value, variable=self._mode_var,
                command=self._on_mode_change,
            ).pack(side=tk.LEFT)

        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(bar, text="Clear selection", command=self._on_clear_click).pack(side=tk.LEFT, padx=2)

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

        # ---- inline label bar (Label mode only; disabled otherwise) ----
        label_bar = ttk.Frame(self.root)
        label_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(label_bar, text="Label:").pack(side=tk.LEFT, padx=(4, 2))
        self._label_target_lbl = ttk.Label(label_bar, text="(select vectors)", width=28)
        self._label_target_lbl.pack(side=tk.LEFT)

        ttk.Label(label_bar, text="Text").pack(side=tk.LEFT, padx=(8, 2))
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

        self._apply_btn = ttk.Button(label_bar, text="Apply", command=self._apply_label)
        self._apply_btn.pack(side=tk.LEFT, padx=(8, 2))
        self._delete_btn = ttk.Button(label_bar, text="Delete", command=self._delete_label)
        self._delete_btn.pack(side=tk.LEFT, padx=2)
        self._label_bar_widgets = [self._text_entry, self._rot_scale, self._apply_btn, self._delete_btn]

        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(body)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="#808080")
        vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self._side_panel = ttk.Frame(body, width=220)
        self._side_panel.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(self.root, text="Save", command=self.save).pack(side=tk.BOTTOM, fill=tk.X)

        self._set_label_bar_enabled(False)

    def _set_label_bar_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in self._label_bar_widgets:
            widget.configure(state=state)

    def _rebuild_side_panel(self) -> None:
        for child in self._side_panel.winfo_children():
            child.destroy()
        if self._mode != "baseline":
            return
        if self._active_baseline_id is None:
            ttk.Label(
                self._side_panel,
                text="Drag to create a baseline,\nor click one to select it.",
                justify="left",
            ).pack(anchor="w", padx=4, pady=4)
            return
        ttk.Label(
            self._side_panel, text=f"Characters on baseline {self._active_baseline_id[:8]}",
        ).pack(anchor="w", padx=4, pady=(4, 2))
        entries = self._cad_font_page_entries()
        if not entries:
            ttk.Label(self._side_panel, text="(no cad_font labels on this page yet)").pack(
                anchor="w", padx=4,
            )
            return
        self._baseline_checkbox_vars = {}
        for entry in entries:
            var = tk.BooleanVar(value=(entry.baseline_id == self._active_baseline_id))
            self._baseline_checkbox_vars[entry.label_id] = var
            label = f'"{entry.text}"  ({entry.label_id[:8]})'
            ttk.Checkbutton(
                self._side_panel, text=label, variable=var,
                command=lambda lid=entry.label_id: self._toggle_baseline_assignment(lid),
            ).pack(anchor="w", padx=4)

    def _toggle_baseline_assignment(self, label_id: str) -> None:
        idx = self._entry_by_label_id(label_id)
        if idx is None:
            return
        entry = self.labels.entries[idx]
        checked = self._baseline_checkbox_vars[label_id].get()
        entry.baseline_id = self._active_baseline_id if checked else None
        self._render()

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
        self.root.bind("<Escape>", lambda _e: self._on_escape())
        self.root.bind("<Next>", lambda _e: self._change_page(1))
        self.root.bind("<Prior>", lambda _e: self._change_page(-1))

    # ---- mode switching --------------------------------------------------

    def _on_mode_change(self) -> None:
        self._mode = self._mode_var.get()
        self._cancel_in_progress()
        self._set_label_bar_enabled(self._mode == "label")
        self._render()

    def _cancel_in_progress(self) -> None:
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None
        self._drag_start = None
        self._drag_moved = False
        if self._baseline_preview_id is not None:
            self.canvas.delete(self._baseline_preview_id)
            self._baseline_preview_id = None
        self._baseline_drag_start = None
        self._baseline_drag_moved = False

    def _on_escape(self) -> None:
        if self._mode == "label":
            self._clear_selection()
        else:
            self._cancel_in_progress()
            self._active_baseline_id = None
            self._render()

    def _on_clear_click(self) -> None:
        if self._mode == "label":
            self._clear_selection()
        else:
            self._active_baseline_id = None
            self._render()

    # ---- coordinate helpers -------------------------------------------

    def _page_point(self, event: "tk.Event") -> "fitz.Point":
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        return fitz.Point(cx, cy) * ~self.matrix

    def _clip_line_to_page_bbox(
        self, origin: tuple[float, float], direction: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Clips the infinite line `origin + t * direction` to the page's
        own mediabox (Liang-Barsky style per-axis `t` narrowing, no `[0, 1]`
        clamp since the line is infinite) -- a baseline "stretches the
        whole page", so it's drawn across its entire real page-space
        extent rather than the live scroll viewport (simpler, always
        correct, same visible result)."""
        ox, oy = origin
        dx, dy = direction
        x0, y0, x1, y1 = self.page.meta.mediabox
        t_min, t_max = -1e9, 1e9
        for d, o, lo, hi in ((dx, ox, x0, x1), (dy, oy, y0, y1)):
            if abs(d) < 1e-9:
                if not (lo <= o <= hi):
                    return None
                continue
            t_lo, t_hi = (lo - o) / d, (hi - o) / d
            if t_lo > t_hi:
                t_lo, t_hi = t_hi, t_lo
            t_min, t_max = max(t_min, t_lo), min(t_max, t_hi)
        if t_min > t_max:
            return None
        return (ox + dx * t_min, oy + dy * t_min), (ox + dx * t_max, oy + dy * t_max)

    def _nearest_baseline(self, pt: "fitz.Point") -> "Baseline | None":
        """Perpendicular distance from `pt` to each baseline's own infinite
        line (a baseline conceptually spans the whole page, so clicking
        anywhere near its extension selects it), within a small
        zoom-adjusted pixel tolerance."""
        tol_page = _BASELINE_HIT_TOL_PX / self.zoom
        best: "Baseline | None" = None
        best_dist = tol_page
        for b in self._page_baselines():
            dx, dy = b.direction
            px, py = pt.x - b.origin[0], pt.y - b.origin[1]
            dist = abs(px * dy - py * dx)
            if dist < best_dist:
                best_dist, best = dist, b
        return best

    # ---- rendering ----------------------------------------------------

    def _render(self) -> None:
        self.canvas.delete("all")
        # Zoom-only: get_pixmap() bakes /Rotate itself, so self.matrix
        # (rotation already folded in) stays overlay-only.
        pix = self.page.fitz_page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
        self._photo = tk.PhotoImage(data=pix.tobytes("ppm"))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.config(scrollregion=(0, 0, pix.width, pix.height))

        if self._mode == "label":
            self._render_label_mode()
        else:
            self._render_baseline_mode()

        self.zoom_label.config(text=f"{round(self.zoom * 100)}%")
        self._rebuild_side_panel()

    def _render_label_mode(self) -> None:
        labelled = self._labelled_sig_map()
        shown = 0
        for v in self.vectors:
            shown += 1
            vid = id(v)
            if vid in self.selected:
                color, width = SELECTED_COLOR, 3
            elif path_signature(v) in labelled:
                color, width = LABELLED_COLOR, 2
            else:
                color, width = UNLABELLED_COLOR, 1
            draw_vector(self.canvas, self.matrix, v, color, width)

        live_sigs = {path_signature(v) for v in self.vectors}
        for entry in self._cad_font_page_entries():
            if entry.vector_signatures and any(s in live_sigs for s in entry.vector_signatures):
                continue
            rect = fitz.Rect(entry.cluster_bbox) * self.matrix
            self.canvas.create_rectangle(
                rect.x0, rect.y0, rect.x1, rect.y1, outline=ENTRY_COLOR, width=1, dash=(4, 3),
                tags=("entry", entry.label_id),
            )

        self._status.config(
            text=f"Label mode  |  {shown} vectors  |  {len(self.selected)} selected  |  "
                 f"{len(self._cad_font_page_entries())} labels (page)  |  "
                 f"{sum(1 for e in self.labels.entries if e.source == 'cad_font')} total"
        )
        self._refresh_label_hint()
        self._draw_rotation_arrow()

    def _render_baseline_mode(self) -> None:
        for b in self._page_baselines():
            clipped = self._clip_line_to_page_bbox(b.origin, b.direction)
            if clipped is None:
                continue
            p0, p1 = clipped
            c0 = fitz.Point(*p0) * self.matrix
            c1 = fitz.Point(*p1) * self.matrix
            active = b.baseline_id == self._active_baseline_id
            color = SELECTED_COLOR if active else _BASELINE_COLOR
            width = 3 if active else 2
            self.canvas.create_line(
                c0.x, c0.y, c1.x, c1.y, fill=color, width=width, arrow=tk.LAST,
                tags=("baseline", b.baseline_id),
            )
        self._status.config(
            text=f"Baseline mode  |  {len(self._page_baselines())} baseline(s) (page)  |  "
                 f"active: {self._active_baseline_id[:8] if self._active_baseline_id else '(none)'}"
        )

    # ---- zoom -----------------------------------------------------------

    def _change_zoom(self, direction: int) -> None:
        if direction > 0:
            self.zoom = min(MAX_ZOOM, self.zoom * ZOOM_STEP)
        else:
            self.zoom = max(MIN_ZOOM, self.zoom / ZOOM_STEP)
        self.matrix = get_display_matrix(self.page.fitz_page, self.zoom)
        self._render()

    def _on_wheel(self, event: "tk.Event") -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_shift_wheel(self, event: "tk.Event") -> None:
        self.canvas.xview_scroll(int(-event.delta / 120), "units")

    def _on_ctrl_wheel(self, event: "tk.Event") -> None:
        self._change_zoom(1 if event.delta > 0 else -1)

    # ---- page navigation -------------------------------------------------

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

    # ---- mouse dispatch (mode-aware) -------------------------------------

    def _on_left_press(self, event: "tk.Event") -> None:
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if self._mode == "label":
            self._drag_start = (cx, cy)
            self._drag_moved = False
        else:
            self._baseline_drag_start = (cx, cy)
            self._baseline_drag_moved = False

    def _on_left_drag(self, event: "tk.Event") -> None:
        if self._mode == "label":
            self._label_left_drag(event)
        else:
            self._baseline_left_drag(event)

    def _on_left_release(self, event: "tk.Event") -> None:
        if self._mode == "label":
            self._label_left_release(event)
        else:
            self._baseline_left_release(event)

    def _on_right_click(self, event: "tk.Event") -> None:
        pt = self._page_point(event)
        if self._mode == "label":
            self._label_right_click(pt)
        else:
            self._baseline_right_click(pt)

    # ---- baseline mode: drag-create / click-select / delete --------------

    def _baseline_left_drag(self, event: "tk.Event") -> None:
        if self._baseline_drag_start is None:
            return
        x0, y0 = self._baseline_drag_start
        x1, y1 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if not self._baseline_drag_moved and (
            abs(x1 - x0) < DRAG_THRESHOLD_PX and abs(y1 - y0) < DRAG_THRESHOLD_PX
        ):
            return
        self._baseline_drag_moved = True
        if self._baseline_preview_id is not None:
            self.canvas.delete(self._baseline_preview_id)
        self._baseline_preview_id = self.canvas.create_line(
            x0, y0, x1, y1, fill=_BASELINE_COLOR, width=2, dash=(3, 2), arrow=tk.LAST,
            tags=("baseline_preview",),
        )

    def _baseline_left_release(self, event: "tk.Event") -> None:
        start = self._baseline_drag_start
        self._baseline_drag_start = None
        if self._baseline_preview_id is not None:
            self.canvas.delete(self._baseline_preview_id)
            self._baseline_preview_id = None
        if start is None:
            return
        if not self._baseline_drag_moved:
            pt = fitz.Point(*start) * ~self.matrix
            hit = self._nearest_baseline(pt)
            self._active_baseline_id = hit.baseline_id if hit else None
            self._render()
            return

        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        p0 = fitz.Point(*start) * ~self.matrix
        p1 = fitz.Point(cx, cy) * ~self.matrix
        dx, dy = p1.x - p0.x, p1.y - p0.y
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        direction = (dx / length, dy / length)
        baseline = Baseline(
            page_index=self.page_index, baseline_id=uuid.uuid4().hex,
            origin=(p0.x, p0.y), direction=direction,
        )
        self.labels.baselines.append(baseline)
        self._active_baseline_id = baseline.baseline_id
        self._render()

    def _baseline_right_click(self, pt: "fitz.Point") -> None:
        hit = self._nearest_baseline(pt)
        if hit is None:
            return
        self.labels.baselines.remove(hit)
        for entry in self.labels.entries:
            if entry.baseline_id == hit.baseline_id:
                entry.baseline_id = None
        if hit.baseline_id == self._active_baseline_id:
            self._active_baseline_id = None
        self._render()

    # ---- label mode: verbatim VectorLabelApp selection/edit flow ---------

    def _focus_text_entry(self) -> None:
        self._text_entry.focus_set()
        self._text_entry.selection_range(0, tk.END)

    def _clear_selection(self) -> None:
        self.selected.clear()
        self._active_label_id = None
        self._clear_label_panel()
        self._render()

    def _label_left_drag(self, event: "tk.Event") -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if not self._drag_moved and (
            abs(x1 - x0) < DRAG_THRESHOLD_PX and abs(y1 - y0) < DRAG_THRESHOLD_PX
        ):
            return
        self._drag_moved = True
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
        self._drag_rect_id = self.canvas.create_rectangle(
            x0, y0, x1, y1, outline=SELECTED_COLOR, width=1, dash=(3, 2), tags=("selrect",),
        )

    def _label_left_release(self, event: "tk.Event") -> None:
        start = self._drag_start
        self._drag_start = None
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None
        if start is None:
            return
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if not self._drag_moved:
            self._click_select(fitz.Point(cx, cy) * ~self.matrix)
            return
        inv = ~self.matrix
        p0, p1 = fitz.Point(*start) * inv, fitz.Point(cx, cy) * inv
        rect = (min(p0.x, p1.x), min(p0.y, p1.y), max(p0.x, p1.x), max(p0.y, p1.y))
        self._area_select(rect)

    def _click_select(self, pt: "fitz.Point") -> None:
        labelled = self._labelled_sig_map()
        hit = None
        for v in self.vectors:
            if bbox_contains(v.bbox, pt.x, pt.y):
                if hit is None or bbox_area(v.bbox) < bbox_area(hit.bbox):
                    hit = v
        if hit is None:
            return
        sig = path_signature(hit)
        entry = labelled.get(sig)
        if entry is not None:
            target_sigs = set(entry.vector_signatures)
            self.selected = {
                id(v) for v in self.vectors if path_signature(v) in target_sigs
            }
            self._active_label_id = entry.label_id
            self._prefill_label_bar(entry)
        else:
            self.selected.symmetric_difference_update({id(hit)})
            self._active_label_id = None
        self._render()
        if self.selected:
            self._focus_text_entry()

    def _items_in_rect(self, rect) -> set[int]:
        labelled = self._labelled_sig_map()
        hits: set[int] = set()
        for v in self.vectors:
            if path_signature(v) in labelled:
                continue
            if bboxes_intersect(v.bbox, rect):
                hits.add(id(v))
        return hits

    def _area_select(self, rect) -> None:
        hits = self._items_in_rect(rect)
        if not hits:
            return
        if hits <= self.selected:
            self.selected -= hits
        else:
            self.selected |= hits
        self._active_label_id = None
        self._render()
        if self.selected:
            self._focus_text_entry()

    def _label_right_click(self, pt: "fitz.Point") -> None:
        labelled = self._labelled_sig_map()
        hit = None
        for v in self.vectors:
            if bbox_contains(v.bbox, pt.x, pt.y):
                if hit is None or bbox_area(v.bbox) < bbox_area(hit.bbox):
                    hit = v
        if hit is None:
            return
        entry = labelled.get(path_signature(hit))
        if entry is None:
            return
        idx = self._entry_by_label_id(entry.label_id)
        if idx is None:
            return
        del self.labels.entries[idx]
        if entry.label_id == self._active_label_id:
            self._clear_label_panel()
        self._render()

    # ---- inline label bar ------------------------------------------------

    def _clear_label_panel(self) -> None:
        self._active_label_id = None
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
        if self._active_label_id is not None:
            self._label_target_lbl.config(text=f"editing 1 label ({len(self.selected)} vectors)")
        elif self.selected:
            self._label_target_lbl.config(text=f"{len(self.selected)} vector(s) selected -- Apply to label")
        else:
            self._label_target_lbl.config(text="(select vectors)")

    def _on_rotation_change(self, value: str) -> None:
        snapped = round(float(value) / ROTATION_SNAP_DEG) * ROTATION_SNAP_DEG
        self._rot_var.set(snapped)
        self._rot_readout.config(text=f"{snapped:.1f}°")
        self._draw_rotation_arrow()

    def _selection_bbox(self):
        if self._active_label_id is not None or self.selected:
            boxes = [self._vectors_by_id[i].bbox for i in self.selected if i in self._vectors_by_id]
            if boxes:
                return union_bbox(boxes)
        return None

    def _draw_rotation_arrow(self) -> None:
        self.canvas.delete("rotation_arrow")
        if self._mode != "label" or not self.selected:
            return
        bbox = self._selection_bbox()
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
            c1.x, c1.y, c2.x, c2.y, fill=SELECTED_COLOR, width=2,
            arrow=tk.LAST, tags=("overlay", "rotation_arrow"),
        )

    def _apply_label(self) -> None:
        if not self.selected:
            return
        text = self._text_var.get().strip()
        if not text:
            return
        rotation = int(round(self._rot_var.get()))
        vectors = [self._vectors_by_id[i] for i in self.selected if i in self._vectors_by_id]
        vsigs = vector_signatures_for(vectors)
        bbox = union_bbox([v.bbox for v in vectors])

        if self._active_label_id is not None:
            idx = self._entry_by_label_id(self._active_label_id)
            if idx is not None:
                entry = self.labels.entries[idx]
                entry.text = text
                entry.expected_rotation = rotation
                entry.vector_signatures = vsigs
                entry.cluster_bbox = bbox
                entry.cluster_signature = cluster_signature(vectors)
                # baseline_id is deliberately untouched -- Label mode never
                # reads or writes it.
        else:
            entry = LabelEntry(
                page_index=self.page_index, cluster_bbox=bbox,
                cluster_signature=cluster_signature(vectors),
                label_id=uuid.uuid4().hex,
                text=text, source="cad_font", expected_rotation=rotation,
                vector_signatures=vsigs,
            )
            self.labels.entries.append(entry)

        self.selected.clear()
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
        if idx is None:
            return
        del self.labels.entries[idx]
        self.selected.clear()
        self._clear_label_panel()
        self._render()

    # ---- hover -------------------------------------------------------

    def _on_motion(self, event: "tk.Event") -> None:
        pt = self._page_point(event)
        if self._mode == "label":
            self._label_hover(pt, event)
        else:
            self._baseline_hover(pt, event)

    def _label_hover(self, pt: "fitz.Point", event: "tk.Event") -> None:
        labelled = self._labelled_sig_map()
        hit = None
        for v in self.vectors:
            if bbox_contains(v.bbox, pt.x, pt.y):
                if hit is None or bbox_area(v.bbox) < bbox_area(hit.bbox):
                    hit = v
        if hit is not None:
            entry = labelled.get(path_signature(hit))
            text = (
                f'"{entry.text}"  rot={entry.expected_rotation}  baseline={entry.baseline_id or "(none)"}'
                if entry else f"{hit.type}  seqno={hit.seqno}  -- unlabelled"
            )
            self.tooltip.show(event.x_root, event.y_root, text)
            return
        self.tooltip.hide()

    def _baseline_hover(self, pt: "fitz.Point", event: "tk.Event") -> None:
        hit = self._nearest_baseline(pt)
        if hit is not None:
            n_assigned = sum(
                1 for e in self._cad_font_page_entries() if e.baseline_id == hit.baseline_id
            )
            self.tooltip.show(event.x_root, event.y_root, f"baseline {hit.baseline_id[:8]}  ({n_assigned} label(s))")
            return
        self.tooltip.hide()

    # ---- persistence ------------------------------------------------

    def save(self) -> None:
        save_labels(self.labels, self.out_path)
        _LOG.info(
            "saved %d label(s), %d baseline(s) to %s",
            len(self.labels.entries), len(self.labels.baselines), self.out_path,
        )

    def _on_close(self) -> None:
        self.save()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
        self.reader.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label CAD-vector characters + baselines.")
    parser.add_argument("pdf", help="Path to the input PDF.")
    parser.add_argument("--page", type=int, default=0, help="0-based page index.")
    parser.add_argument(
        "--out", default=None,
        help="Path to save/load the label JSON file "
        "(default: outputs/labels/<pdf stem>_cad_font.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)
    out = args.out or str(output_dir("labels") / f"{Path(args.pdf).stem}_cad_font.json")
    app = CadFontLabelApp(args.pdf, args.page, out)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
