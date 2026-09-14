"""Vector labelling: a small Tk UI for typing ground-truth text + rotation
onto raw extracted `Vector`s, saved via `label_schema.save_labels`. Replaces
`manual_label.py` (`ManualLabelApp` -> `VectorLabelApp`), fixing its broken
import (it called the now-archived `pipelines.sub_pipelines.
vector_classification.classify_vectors`) and dropping the old cluster/path
dual-mode + Group/Ungroup editor entirely.

There is only ever **one flat pool of raw vectors** for the page
(`extract_vectors(page)`) -- every vector is always individually clickable.
`separate_by_layer_color_width` (`rastervec.pipelines.current`) buckets
those vectors into one visibility checkbox per `(layer, color, width)` in
the side panel -- a filter, not a selection/clustering unit; its spatial
merge (`cluster_buckets`) is not used at all here.

Selecting vectors (plain/ctrl click or a rubber-band drag, over *unlabelled*
vectors only -- reuses the old path-mode selection mechanics verbatim) +
Apply creates a new label. Clicking a single **already-labelled** vector
instead selects that label's whole vector set, pre-fills the label bar, and
enters "editing" it (`self._active_label_id`); further select/deselect +
Apply overwrites that same entry in place rather than creating a duplicate.
A rubber-band drag never enters this edit mode by itself. Delete removes
the entry currently being edited (no-op if none is active).

Labelled vectors render green; unlabelled stay grey/blue. "Hide labelled"
hides every vector already in some label. The rotation control is a
continuous `ttk.Scale` (2.5 degree snap) with a live direction-arrow overlay,
replacing the old 4-value dropdown. The text entry auto-focuses after *any*
selection (click or drag), not just right-click.

`get_display_matrix` (page-space -> canvas-space) and `Tooltip` are
unchanged from `manual_label.py`, ported originally from the former
`debug_app.py`.

Not unit-testable (a real Tk event loop). Smoke-test manually:

    .venv/Scripts/python.exe scripts/label/vector_label.py path/to.pdf --page 0

1. A window opens showing the page with every raw vector drawn (grey/blue).
   Bucket checkboxes in the side panel toggle groups of vectors on/off;
   "Hide labelled" hides already-labelled ones. Zoom -/+, scrollbars, mouse
   wheel, Ctrl+wheel all work; overlays stay aligned on a rotated page.
2. Click or drag-select a few unlabelled vectors (they turn orange); the
   text field auto-focuses. Type text, drag the rotation slider (arrow
   overlay follows), Apply -- the vectors turn green, selection clears,
   fields reset.
3. Click one of those green vectors -- the whole label's vector set
   re-selects (orange), the label bar pre-fills. Add/remove a vector via
   drag, Apply again -- same label entry updates in place (still one entry).
   Delete -- removes it (vectors turn grey again).
4. Click `>` -- next page loads with its own vector pool; click `<` back --
   labels are still there.
5. "Save" (or close the window) writes the label JSON via
   `label_schema.save_labels`.
"""
from __future__ import annotations

import argparse
import math
import sys
import uuid
from pathlib import Path
from tkinter import ttk

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
from rastervec.pipelines._steps import extract_vectors
from rastervec.pipelines.current import separate_by_layer_color_width
from rastervec.Reader.reader import Reader
from rastervec.commons.renderer._shapes import path_color_hex

_LOG = get_logger("vector_label")


class VectorLabelApp:
    def __init__(self, pdf_path: str, page_index: int, out_path: str) -> None:
        self.pdf_path = pdf_path
        self.out_path = out_path

        self.reader = Reader(pdf_path)

        # One LabelSet covers every page of the PDF -- each LabelEntry
        # carries its own page_index, so a page switch never rewrites it.
        self.labels = (
            load_labels(out_path) if Path(out_path).exists()
            else LabelSet(pdf_path=pdf_path)
        )

        self.selected: set[int] = set()
        self._active_label_id: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_moved = False
        self._drag_rect_id: int | None = None
        self.zoom = ZOOM_DEFAULT

        self.root = tk.Tk()
        self.tooltip = Tooltip(self.root)

        self._build_layout()
        self._bind_events()
        self._load_page(page_index)

    # ---- per-page state -------------------------------------------------

    def _load_page(self, page_index: int) -> None:
        """(Re)run extraction for `page_index` and rebuild every per-page
        working structure. `self.reader`, `self.labels` and the Tk widgets
        outlive this."""
        self.page_index = max(0, min(self.reader.page_count() - 1, page_index))
        self.page = self.reader.get_page(self.page_index)
        self.vectors: list[Vector] = extract_vectors(self.page)
        self._vectors_by_id: dict[int, Vector] = {id(v): v for v in self.vectors}
        self.buckets: list[list[Vector]] = separate_by_layer_color_width(self.vectors)

        self.selected.clear()
        self._active_label_id = None
        self.matrix = get_display_matrix(self.page.fitz_page, self.zoom)

        self._build_bucket_panel()

        self.root.title(
            f"Vector Label -- {Path(self.pdf_path).name} "
            f"page {self.page_index + 1}/{self.reader.page_count()}"
        )
        self._sync_page_entry()
        self._clear_label_panel()
        self._render()

    def _page_entries(self) -> list[LabelEntry]:
        """Label entries for the page currently shown."""
        return [e for e in self.labels.entries if e.page_index == self.page_index]

    def _labelled_sig_map(self) -> dict[str, LabelEntry]:
        """`path_signature` -> owning entry, for every labelled vector on
        this page."""
        out: dict[str, LabelEntry] = {}
        for entry in self._page_entries():
            for sig in entry.vector_signatures:
                out[sig] = entry
        return out

    def _visible_vectors(self):
        """Vectors currently drawn/selectable: bucket-visible, and (if
        "Hide labelled" is set) not already in any label."""
        labelled = self._labelled_sig_map() if self._hide_labelled.get() else {}
        for bi, bucket in enumerate(self.buckets):
            if not self.bucket_visible[bi].get():
                continue
            for v in bucket:
                if labelled and path_signature(v) in labelled:
                    continue
                yield v

    # ---- layout ---------------------------------------------------------

    def _build_layout(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Zoom -", command=lambda: self._change_zoom(-1)).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(bar, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Zoom +", command=lambda: self._change_zoom(1)).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(bar, text="Clear selection", command=self._clear_selection).pack(side=tk.LEFT, padx=2)

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

        # ---- inline label bar ----
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

        ttk.Button(label_bar, text="Apply", command=self._apply_label).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(label_bar, text="Delete", command=self._delete_label).pack(side=tk.LEFT, padx=2)

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

        side = ttk.Frame(body, width=220)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(side, text="Buckets (layer / color / width)").pack(anchor="w", padx=4, pady=(4, 0))
        self._hide_labelled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            side, text="Hide labelled", variable=self._hide_labelled, command=self._render,
        ).pack(anchor="w", padx=4, pady=4)
        self._bucket_panel = ttk.Frame(side)
        self._bucket_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4)

        ttk.Button(self.root, text="Save", command=self.save).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_bucket_panel(self) -> None:
        for child in self._bucket_panel.winfo_children():
            child.destroy()
        self.bucket_visible: dict[int, "tk.BooleanVar"] = {}
        for i, bucket in enumerate(self.buckets):
            var = tk.BooleanVar(value=True)
            self.bucket_visible[i] = var
            sample = bucket[0]
            color = path_color_hex(sample)
            width = sample.width
            layer = sample.layer or "(no layer)"
            label = f"{layer}  {color}  w={width}  ({len(bucket)})"
            ttk.Checkbutton(
                self._bucket_panel, text=label, variable=var, command=self._render,
            ).pack(anchor="w")

    def _bind_events(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.canvas.bind("<Button-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _e: self.tooltip.hide())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_wheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        self.root.bind("<Escape>", lambda _e: self._clear_selection())
        self.root.bind("<Next>", lambda _e: self._change_page(1))
        self.root.bind("<Prior>", lambda _e: self._change_page(-1))

    # ---- coordinate helpers -------------------------------------------

    def _page_point(self, event: "tk.Event") -> "fitz.Point":
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        return fitz.Point(cx, cy) * ~self.matrix

    # ---- rendering ----------------------------------------------------

    def _render(self) -> None:
        self.canvas.delete("all")
        # Zoom-only: get_pixmap() bakes the page's /Rotate itself, so passing
        # self.matrix (rotation already folded in) would double-rotate the
        # bitmap relative to the overlays. self.matrix stays overlay-only.
        pix = self.page.fitz_page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
        self._photo = tk.PhotoImage(data=pix.tobytes("ppm"))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.config(scrollregion=(0, 0, pix.width, pix.height))

        labelled = self._labelled_sig_map()
        shown = 0
        for v in self._visible_vectors():
            shown += 1
            vid = id(v)
            if vid in self.selected:
                color, width = SELECTED_COLOR, 3
            elif path_signature(v) in labelled:
                color, width = LABELLED_COLOR, 2
            else:
                color, width = UNLABELLED_COLOR, 1
            draw_vector(self.canvas, self.matrix, v, color, width)

        # Entries with no live vector representation here at all (every
        # source="native" entry, plus a source="vector" entry left stale by
        # an edit elsewhere) -- dashed grey, so this window still doubles as
        # a viewer for them.
        live_sigs = {path_signature(v) for v in self.vectors}
        for entry in self._page_entries():
            if entry.vector_signatures and any(s in live_sigs for s in entry.vector_signatures):
                continue
            rect = fitz.Rect(entry.cluster_bbox) * self.matrix
            self.canvas.create_rectangle(
                rect.x0, rect.y0, rect.x1, rect.y1, outline=ENTRY_COLOR, width=1, dash=(4, 3),
                tags=("entry", entry.label_id),
            )

        self.zoom_label.config(text=f"{round(self.zoom * 100)}%")
        self._status.config(
            text=f"{shown}/{len(self.vectors)} vectors shown  |  "
                 f"{len(self.selected)} selected  |  {len(self._page_entries())} labels (page)  "
                 f"|  {len(self.labels.entries)} total"
        )
        self._refresh_label_hint()
        self._draw_rotation_arrow()

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
        self.save()  # persist the current page's labels before switching
        self.tooltip.hide()
        self._load_page(new_index)

    # ---- selection --------------------------------------------------

    def _clear_selection(self) -> None:
        self.selected.clear()
        self._active_label_id = None
        self._clear_label_panel()
        self._render()

    def _on_left_press(self, event: "tk.Event") -> None:
        self._drag_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self._drag_moved = False

    def _on_left_drag(self, event: "tk.Event") -> None:
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
            x0, y0, x1, y1, outline=SELECTED_COLOR, width=1, dash=(3, 2),
            tags=("selrect",),
        )

    def _on_left_release(self, event: "tk.Event") -> None:
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

    def _focus_text_entry(self) -> None:
        self._text_entry.focus_set()
        self._text_entry.selection_range(0, tk.END)

    def _click_select(self, pt: "fitz.Point") -> None:
        labelled = self._labelled_sig_map()
        hit = None
        for v in self._visible_vectors():
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
        """Ids of unlabelled, currently-visible vectors whose bbox
        intersects `rect` -- a drag never grabs an already-labelled vector
        (that's the single-click "edit label" path's job)."""
        labelled = self._labelled_sig_map()
        hits: set[int] = set()
        for v in self._visible_vectors():
            if path_signature(v) in labelled:
                continue
            if bboxes_intersect(v.bbox, rect):
                hits.add(id(v))
        return hits

    def _area_select(self, rect) -> None:
        hits = self._items_in_rect(rect)
        if not hits:
            return
        # All-selected area -> deselect it; otherwise add it to the selection.
        if hits <= self.selected:
            self.selected -= hits
        else:
            self.selected |= hits
        self._active_label_id = None
        self._render()
        if self.selected:
            self._focus_text_entry()

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
        if not self.selected:
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
            return  # nothing to write; use Delete to remove an existing label
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
        else:
            entry = LabelEntry(
                page_index=self.page_index, cluster_bbox=bbox,
                cluster_signature=cluster_signature(vectors),
                label_id=uuid.uuid4().hex,
                text=text, source="vector", expected_rotation=rotation,
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
        labelled = self._labelled_sig_map()
        hit = None
        for v in self._visible_vectors():
            if bbox_contains(v.bbox, pt.x, pt.y):
                if hit is None or bbox_area(v.bbox) < bbox_area(hit.bbox):
                    hit = v
        if hit is not None:
            entry = labelled.get(path_signature(hit))
            text = (
                f'"{entry.text}"  rot={entry.expected_rotation}' if entry
                else f"{hit.type}  seqno={hit.seqno}  -- unlabelled"
            )
            self.tooltip.show(event.x_root, event.y_root, text)
            return
        live_sigs = {path_signature(v) for v in self.vectors}
        for entry in self._page_entries():
            if entry.vector_signatures and any(s in live_sigs for s in entry.vector_signatures):
                continue
            if bbox_contains(entry.cluster_bbox, pt.x, pt.y):
                self.tooltip.show(event.x_root, event.y_root, f'"{entry.text}"  ({entry.source})')
                return
        self.tooltip.hide()

    # ---- persistence ------------------------------------------------

    def save(self) -> None:
        save_labels(self.labels, self.out_path)
        _LOG.info("saved %d label(s) to %s", len(self.labels.entries), self.out_path)

    def _on_close(self) -> None:
        self.save()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
        self.reader.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manually label raw vectors with ground-truth text.")
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
    app = VectorLabelApp(args.pdf, args.page, out)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
