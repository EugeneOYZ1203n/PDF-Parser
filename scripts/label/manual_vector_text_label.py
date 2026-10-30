"""Manual vector-text labelling: a small Tk UI for typing ground-truth text + rotation onto the
raw extracted `Vector`s that draw text in the original PDF, saved via `label_schema.save_labels`.
This is the only labelling tool: native-text and raster ground truth are derived by the
benchmark, so a label file holds hand-labelled vector text only.

**Pages are processed in the background.** Opening the tool shows the page at once and starts one
worker process (`seed_labels.prepare_page`) that extracts every page's vectors and, unless
`--no-seed`, reads its text with `v2_vector_class` for draft labels -- the opened page first, then
outwards (`_common.prefetch_order`). `<` / `>` (and PageUp / PageDown, the page box) only go to a
page that is ready; the bar shows how many are. A page with no labels yet is **seeded** with its
drafts the first time it is shown: each text the backend read becomes an entry tied to the vectors
it was read from. Correct, delete or extend them like any other label.

There is only ever **one flat pool of raw vectors** for the page -- every vector is always
individually clickable. `separate_by_layer_color_width` (`_common.py`) buckets those vectors into
one visibility checkbox per `(layer, color, width)` in the side panel -- a filter, not a
selection/clustering unit.

**The label list** (side panel) shows every label of the PDF: page, text, rotation and vector
count. Selecting a row (or clicking one of a label's vectors) selects that label: its vectors are
highlighted, the label bar is pre-filled, and only the buckets the label's vectors are in stay
visible (`_common.focus_buckets`; "Show all buckets" or deselecting restores them). A row on
another page goes to that page once it is ready. While a label is selected, clicking or dragging
over vectors edits its vector set at once (`_common.edit_label_vectors`): unlabelled vectors are
added, the label's own vectors removed (never the last one); a click on another label's vector
selects that label instead. Apply (or Enter) writes the text and rotation, Delete removes the
label. Ctrl / shift-click rows to select several labels: their vectors are highlighted and
Delete (the button, or the Delete key in the list) removes them all after a confirmation.

With no label selected, selecting vectors (plain/ctrl click or a rubber-band drag, over
*unlabelled* vectors only) + Apply creates a new label. Labelled vectors render green; unlabelled
stay blue. "Hide labelled" hides every vector already in some other label. The rotation control
is a continuous `ttk.Scale` (2.5 degree snap) with a live direction-arrow overlay. The text entry
auto-focuses after any selection.

`get_display_matrix` (page-space -> canvas-space) and `Tooltip` live in `_common.py`.

Not unit-testable (a real Tk event loop; the pure helpers are, in `_common.py`). Smoke-test
manually:

    python scripts/label/manual_vector_text_label.py path/to.pdf --page 0

1. The page shows at once ("processing ..." until its vectors arrive); the bar counts the pages
   ready. `<` / `>` appear as the neighbouring pages finish.
2. The label list fills with the page's drafts. Select a row: its vectors turn orange, the label
   bar pre-fills and only that label's buckets stay shown. Click an unlabelled vector: it joins
   the label (the vector count in the list goes up); click one of the label's vectors: it leaves.
   Edit the text, Apply.
3. Escape: the selection clears and every bucket is shown again. Drag over unlabelled vectors,
   type, Apply -- a new label appears in the list.
4. Select a row of another (ready) page: the tool goes there and selects it.
5. "Save" (or close the window) writes the label JSON via `label_schema.save_labels`; labels are
   also saved on every page change.
"""

from __future__ import annotations

import argparse
import math
import sys
import uuid
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tkinter as tk

import pymupdf
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
    edit_label_vectors,
    focus_buckets,
    get_display_matrix,
    label_rows,
    prefetch_order,
)
from raster_parser.evaluation.labelling.label_schema import (
    LabelEntry,
    LabelSet,
    load_labels,
    path_signature,
    save_labels,
    vector_signatures_for,
)
from raster_parser.commons.helpers.geometry import (
    bbox_area,
    bbox_contains,
    bboxes_intersect,
    union_bbox,
)
from raster_parser.commons.logging_setup import configure_logging, get_logger
from raster_parser.commons.workers import pin_blas_threads, spawn_pool
from raster_parser.models.vector import Vector
from raster_parser.commons.paths import output_dir
from _common import separate_by_layer_color_width
from raster_parser.evaluation.labelling.seed_labels import DEFAULT_SEED_BACKEND, prepare_page
from raster_parser.phase_native_read.reader import Reader
from raster_parser.commons.renderer._shapes import path_color_hex

_LOG = get_logger("manual_vector_text_label")

# How often (ms) the UI checks the background worker for finished pages.
_POLL_MS = 250


class VectorLabelApp:
    def __init__(self, pdf_path: str, page_index: int, out_path: str, *, seed: bool = True) -> None:
        self.pdf_path = pdf_path
        self.out_path = out_path
        self.seed = seed
        self._seeded_pages: set[int] = set()

        self.reader = Reader(pdf_path)

        # One LabelSet covers every page of the PDF -- each LabelEntry
        # carries its own page_index, so a page switch never rewrites it.
        self.labels = (
            load_labels(out_path) if Path(out_path).exists() else LabelSet(pdf_path=pdf_path)
        )

        # Background page processing: page -> (vectors, draft labels) once its worker job is done.
        self._ready: dict[int, tuple[list[Vector], list[LabelEntry]]] = {}
        self._jobs: dict[int, Future] = {}
        self._pool = spawn_pool(1, initializer=partial(pin_blas_threads, paddle=True))

        self.selected: set[int] = set()
        self._active_label_id: str | None = None
        self._multi_ids: set[str] = set()  # labels selected together in the list (2 or more)
        self._saved_visibility: list[bool] | None = None  # bucket visibility before a label focus
        self._drag_start: tuple[float, float] | None = None
        self._drag_moved = False
        self._drag_rect_id: int | None = None
        self.zoom = ZOOM_DEFAULT

        self.root = tk.Tk()
        self.tooltip = Tooltip(self.root)

        self._build_layout()
        self._bind_events()
        page_index = max(0, min(self.reader.page_count() - 1, page_index))
        labelled_pages = {e.page_index for e in self.labels.entries}
        for p in prefetch_order(page_index, self.reader.page_count()):
            # A page that already has labels is never seeded: skip reading its text.
            seed_p = seed and p not in labelled_pages
            self._jobs[p] = self._pool.submit(prepare_page, pdf_path, p, seed=seed_p)
        self._load_page(page_index)
        self._refresh_nav()
        self.root.after(_POLL_MS, self._poll_jobs)

    # ---- background processing ------------------------------------------

    def _poll_jobs(self) -> None:
        """Collect finished pages; fill the shown page when its own job is the one that finished."""
        for p, job in list(self._jobs.items()):
            if not job.done():
                continue
            del self._jobs[p]
            try:
                self._ready[p] = job.result()
            except Exception:  # a page that fails to process stays empty, the rest go on
                _LOG.exception("page %d: processing failed", p)
                self._ready[p] = ([], [])
            if p == self.page_index:
                self._populate_page()
        self._refresh_nav()
        if self._jobs:
            self.root.after(_POLL_MS, self._poll_jobs)

    # ---- per-page state -------------------------------------------------

    def _load_page(self, page_index: int) -> None:
        """Show `page_index` (its vectors, if processed, else "processing ...") and rebuild every
        per-page working structure. `self.reader`, `self.labels` and the Tk widgets outlive this."""
        self.page_index = page_index
        self.page = self.reader.get_page(self.page_index)
        self.matrix = get_display_matrix(self.page.pymupdf_page, self.zoom)
        self.root.title(
            f"Vector text labels -- {Path(self.pdf_path).name} "
            f"page {self.page_index + 1}/{self.reader.page_count()}"
        )
        self._sync_page_entry()
        self._populate_page()

    def _populate_page(self) -> None:
        vectors, drafts = self._ready.get(self.page_index, ([], []))
        self.vectors: list[Vector] = list(vectors)
        self._vectors_by_id: dict[int, Vector] = {id(v): v for v in self.vectors}
        self.buckets: list[list[Vector]] = separate_by_layer_color_width(self.vectors)
        self._bucket_sigs = [{path_signature(v) for v in b} for b in self.buckets]
        if self.page_index in self._ready:
            self._seed_page_if_unlabelled(drafts)

        self.selected.clear()
        self._active_label_id = None
        self._multi_ids = set()
        self._saved_visibility = None
        self._build_bucket_panel()
        self._clear_label_panel()
        self._refresh_label_list()
        self._render()

    def _seed_page_if_unlabelled(self, drafts: list[LabelEntry]) -> None:
        """Add the page's draft labels when it has none yet (once per page)."""
        if not self.seed or self.page_index in self._seeded_pages or self._page_entries():
            return
        self._seeded_pages.add(self.page_index)
        self.labels.entries.extend(drafts)
        _LOG.info(
            "page %d: %d draft label(s) from %s -- review them before saving",
            self.page_index,
            len(drafts),
            DEFAULT_SEED_BACKEND,
        )

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

    def _active_entry(self) -> LabelEntry | None:
        idx = self._entry_by_label_id(self._active_label_id) if self._active_label_id else None
        return self.labels.entries[idx] if idx is not None else None

    def _visible_vectors(self):
        """Vectors currently drawn/selectable: bucket-visible, and (if
        "Hide labelled" is set) not already in a label other than the selected one."""
        labelled = self._labelled_sig_map() if self._hide_labelled.get() else {}
        for bi, bucket in enumerate(self.buckets):
            if not self.bucket_visible[bi].get():
                continue
            for v in bucket:
                owner = labelled.get(path_signature(v)) if labelled else None
                if owner is not None and owner.label_id != self._active_label_id:
                    continue
                yield v

    # ---- layout ---------------------------------------------------------

    def _build_layout(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Zoom -", command=lambda: self._change_zoom(-1)).pack(
            side=tk.LEFT, padx=2
        )
        self.zoom_label = ttk.Label(bar, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Zoom +", command=lambda: self._change_zoom(1)).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(bar, text="Clear selection", command=self._clear_selection).pack(
            side=tk.LEFT, padx=2
        )

        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # `<` / `>` sit in fixed-width slots so showing / hiding them never shifts the page box.
        prev_slot = ttk.Frame(bar, width=34, height=26)
        prev_slot.pack(side=tk.LEFT, padx=1)
        prev_slot.pack_propagate(False)
        self._prev_btn = ttk.Button(prev_slot, text="<", command=lambda: self._change_page(-1))
        self._page_var = tk.StringVar(value="1")
        page_entry = ttk.Entry(bar, textvariable=self._page_var, width=4, justify="center")
        page_entry.pack(side=tk.LEFT)
        page_entry.bind("<Return>", self._on_page_entry_return)
        self._page_count_label = ttk.Label(bar, text="/ 1")
        self._page_count_label.pack(side=tk.LEFT, padx=(1, 1))
        next_slot = ttk.Frame(bar, width=34, height=26)
        next_slot.pack(side=tk.LEFT, padx=1)
        next_slot.pack_propagate(False)
        self._next_btn = ttk.Button(next_slot, text=">", command=lambda: self._change_page(1))
        self._ready_label = ttk.Label(bar, text="")
        self._ready_label.pack(side=tk.LEFT, padx=8)

        self._status = ttk.Label(bar, text="")
        self._status.pack(side=tk.RIGHT, padx=8)

        # ---- inline label bar ----
        label_bar = ttk.Frame(self.root)
        label_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(label_bar, text="Label:").pack(side=tk.LEFT, padx=(4, 2))
        self._label_target_lbl = ttk.Label(label_bar, text="(select vectors)", width=34)
        self._label_target_lbl.pack(side=tk.LEFT)

        ttk.Label(label_bar, text="Text").pack(side=tk.LEFT, padx=(8, 2))
        self._text_var = tk.StringVar()
        self._text_entry = ttk.Entry(label_bar, textvariable=self._text_var, width=40)
        self._text_entry.pack(side=tk.LEFT)
        self._text_entry.bind("<Return>", lambda _e: self._apply_label())

        ttk.Label(label_bar, text="Rot").pack(side=tk.LEFT, padx=(8, 2))
        self._rot_var = tk.DoubleVar(value=0.0)
        self._rot_scale = ttk.Scale(
            label_bar,
            from_=-180.0,
            to=180.0,
            orient="horizontal",
            length=160,
            variable=self._rot_var,
            command=self._on_rotation_change,
        )
        self._rot_scale.pack(side=tk.LEFT, padx=(0, 4))
        self._rot_readout = ttk.Label(label_bar, text="0.0°", width=7)
        self._rot_readout.pack(side=tk.LEFT)

        ttk.Button(label_bar, text="Apply", command=self._apply_label).pack(
            side=tk.LEFT, padx=(8, 2)
        )
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

        side = ttk.Frame(body, width=320)
        side.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(side, text="Labels").pack(anchor="w", padx=4, pady=(4, 0))
        list_frame = ttk.Frame(side)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4)
        self._label_list = ttk.Treeview(
            list_frame,
            columns=("page", "text", "rot", "n"),
            show="headings",
            selectmode="extended",
            height=16,
        )
        for col, title, width, anchor in (
            ("page", "Page", 40, "center"),
            ("text", "Text", 150, "w"),
            ("rot", "Rot", 50, "e"),
            ("n", "Vectors", 55, "e"),
        ):
            self._label_list.heading(col, text=title)
            self._label_list.column(col, width=width, anchor=anchor, stretch=col == "text")
        list_bar = ttk.Scrollbar(list_frame, orient="vertical", command=self._label_list.yview)
        self._label_list.configure(yscrollcommand=list_bar.set)
        self._label_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_bar.pack(side=tk.RIGHT, fill=tk.Y)
        self._label_list.bind("<<TreeviewSelect>>", self._on_list_select)
        self._label_list.bind("<Delete>", lambda _e: self._delete_label())

        ttk.Separator(side, orient="horizontal").pack(fill=tk.X, pady=6)
        bucket_head = ttk.Frame(side)
        bucket_head.pack(fill=tk.X, padx=4)
        ttk.Label(bucket_head, text="Buckets (layer / color / width)").pack(side=tk.LEFT)
        ttk.Button(bucket_head, text="Show all", command=self._show_all_buckets).pack(side=tk.RIGHT)
        self._hide_labelled = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            side,
            text="Hide labelled",
            variable=self._hide_labelled,
            command=self._render,
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
                self._bucket_panel,
                text=label,
                variable=var,
                command=self._render,
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

    def _page_point(self, event: "tk.Event") -> "pymupdf.Point":
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        return pymupdf.Point(cx, cy) * ~self.matrix

    # ---- rendering ----------------------------------------------------

    def _render(self) -> None:
        self.canvas.delete("all")
        # Zoom-only: get_pixmap() bakes the page's /Rotate itself, so passing
        # self.matrix (rotation already folded in) would double-rotate the
        # bitmap relative to the overlays. self.matrix stays overlay-only.
        pix = self.page.pymupdf_page.get_pixmap(matrix=pymupdf.Matrix(self.zoom, self.zoom))
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

        # Entries with no live vector representation here (left stale by an edit elsewhere) --
        # dashed grey, so they can still be found and fixed.
        live_sigs = {path_signature(v) for v in self.vectors}
        for entry in self._page_entries():
            if entry.vector_signatures and any(s in live_sigs for s in entry.vector_signatures):
                continue
            rect = pymupdf.Rect(entry.bbox) * self.matrix
            self.canvas.create_rectangle(
                rect.x0,
                rect.y0,
                rect.x1,
                rect.y1,
                outline=ENTRY_COLOR,
                width=1,
                dash=(4, 3),
                tags=("entry", entry.label_id),
            )

        self.zoom_label.config(text=f"{round(self.zoom * 100)}%")
        if self.page_index not in self._ready:
            self._status.config(text="processing this page ...")
        else:
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
        self.matrix = get_display_matrix(self.page.pymupdf_page, self.zoom)
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

    def _refresh_nav(self) -> None:
        """Show `<` / `>` only when that page is processed, and count the pages ready."""
        for btn, page in (
            (self._prev_btn, self.page_index - 1),
            (self._next_btn, self.page_index + 1),
        ):
            if page in self._ready:
                btn.pack(fill=tk.BOTH, expand=True)
            else:
                btn.pack_forget()
        total = self.reader.page_count()
        self._ready_label.config(
            text="" if len(self._ready) == total else f"{len(self._ready)}/{total} pages ready"
        )

    def _change_page(self, delta: int) -> None:
        self._goto_page(self.page_index + delta)

    def _on_page_entry_return(self, _event: "tk.Event") -> None:
        try:
            self._goto_page(int(self._page_var.get()) - 1)
        except ValueError:
            self._sync_page_entry()

    def _goto_page(self, new_index: int) -> bool:
        """Go to `new_index` if it is processed; False (and a status note) otherwise."""
        new_index = max(0, min(self.reader.page_count() - 1, new_index))
        if new_index == self.page_index:
            self._sync_page_entry()
            return True
        if new_index not in self._ready:
            self._sync_page_entry()
            self._status.config(text=f"page {new_index + 1} is still being processed")
            return False
        self.save()  # persist the current page's labels before switching
        self.tooltip.hide()
        self._load_page(new_index)
        self._refresh_nav()
        return True

    # ---- label list ---------------------------------------------------

    def _refresh_label_list(self) -> None:
        """Rebuild the side list from `self.labels`, keeping the selected label highlighted."""
        self._label_list.delete(*self._label_list.get_children())
        for label_id, values in label_rows(self.labels.entries):
            self._label_list.insert("", tk.END, iid=label_id, values=values)
        if self._multi_ids:
            self._label_list.selection_set(
                [i for i in self._multi_ids if self._label_list.exists(i)]
            )
        elif self._active_label_id and self._label_list.exists(self._active_label_id):
            self._label_list.selection_set(self._active_label_id)
            self._label_list.see(self._active_label_id)
        else:
            self._label_list.selection_set(())

    def _on_list_select(self, _event: "tk.Event") -> None:
        chosen = self._label_list.selection()
        if len(chosen) > 1:
            if set(chosen) != self._multi_ids:  # else the list echoing its own refresh
                self._select_many(chosen)
            return
        if self._multi_ids and chosen:
            self._drop_multi()  # back from several labels to one
        if not chosen or chosen[0] == self._active_label_id:
            return  # nothing chosen, or the list echoing a selection made on the canvas
        idx = self._entry_by_label_id(chosen[0])
        if idx is None:
            return
        entry = self.labels.entries[idx]
        if entry.page_index != self.page_index and not self._goto_page(entry.page_index):
            self._refresh_label_list()  # that page isn't ready: keep the current selection
            return
        self._activate_label(entry, scroll=True)

    # ---- selection --------------------------------------------------

    def _select_many(self, label_ids) -> None:
        """Several labels chosen in the list: highlight their vectors on this page, ready for
        Delete. Nothing is edited while more than one label is selected."""
        self._active_label_id = None
        self._restore_buckets()
        self._multi_ids = set(label_ids)
        sigs = {
            s
            for e in self.labels.entries
            if e.label_id in self._multi_ids
            for s in e.vector_signatures
        }
        self.selected = {id(v) for v in self.vectors if path_signature(v) in sigs}
        self._text_var.set("")
        self._refresh_label_list()
        self._render()

    def _drop_multi(self) -> None:
        if self._multi_ids:
            self._multi_ids = set()
            self.selected.clear()

    def _activate_label(self, entry: LabelEntry, *, scroll: bool = False) -> None:
        """Select `entry`: its vectors, the label bar, and only the buckets it is in."""
        target_sigs = set(entry.vector_signatures)
        self.selected = {id(v) for v in self.vectors if path_signature(v) in target_sigs}
        self._active_label_id = entry.label_id
        self._prefill_label_bar(entry)
        if self._saved_visibility is None:
            self._saved_visibility = [var.get() for var in self.bucket_visible.values()]
        for var, shown in zip(
            self.bucket_visible.values(), focus_buckets(self._bucket_sigs, target_sigs)
        ):
            var.set(shown)
        if scroll:
            self._scroll_to(entry.bbox)
        self._refresh_label_list()
        self._render()
        self._focus_text_entry()

    def _show_all_buckets(self) -> None:
        for var in self.bucket_visible.values():
            var.set(True)
        self._saved_visibility = None
        self._render()

    def _restore_buckets(self) -> None:
        if self._saved_visibility is not None:
            for var, shown in zip(self.bucket_visible.values(), self._saved_visibility):
                var.set(shown)
            self._saved_visibility = None

    def _scroll_to(self, bbox) -> None:
        """Scroll the canvas so `bbox` (page space) is in view."""
        rect = pymupdf.Rect(bbox) * self.matrix
        x0, y0, x1, y1 = (float(c) for c in self.canvas.cget("scrollregion").split())
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        view_w = self.canvas.winfo_width() or 1
        view_h = self.canvas.winfo_height() or 1
        self.canvas.xview_moveto(max(0.0, ((rect.x0 + rect.x1) / 2 - view_w / 2) / w))
        self.canvas.yview_moveto(max(0.0, ((rect.y0 + rect.y1) / 2 - view_h / 2) / h))

    def _clear_selection(self) -> None:
        self.selected.clear()
        self._active_label_id = None
        self._multi_ids = set()
        self._restore_buckets()
        self._clear_label_panel()
        self._refresh_label_list()
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
            x0,
            y0,
            x1,
            y1,
            outline=SELECTED_COLOR,
            width=1,
            dash=(3, 2),
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
            self._click_select(pymupdf.Point(cx, cy) * ~self.matrix)
            return
        inv = ~self.matrix
        p0, p1 = pymupdf.Point(*start) * inv, pymupdf.Point(cx, cy) * inv
        rect = (min(p0.x, p1.x), min(p0.y, p1.y), max(p0.x, p1.x), max(p0.y, p1.y))
        self._area_select(rect)

    def _focus_text_entry(self) -> None:
        self._text_entry.focus_set()
        self._text_entry.selection_range(0, tk.END)

    def _click_select(self, pt: "pymupdf.Point") -> None:
        self._drop_multi()
        labelled = self._labelled_sig_map()
        hit = None
        for v in self._visible_vectors():
            if bbox_contains(v.bbox, pt.x, pt.y):
                if hit is None or bbox_area(v.bbox) < bbox_area(hit.bbox):
                    hit = v
        if hit is None:
            return
        entry = labelled.get(path_signature(hit))
        active = self._active_entry()
        if active is not None and (entry is None or entry.label_id == active.label_id):
            self._edit_active_label([hit])  # add an unlabelled vector / remove one of its own
            return
        if entry is not None:
            self._activate_label(entry)
            return
        self.selected.symmetric_difference_update({id(hit)})
        self._render()
        if self.selected:
            self._focus_text_entry()

    def _items_in_rect(self, rect, *, include_label: str | None = None) -> list[Vector]:
        """Currently-visible vectors whose bbox intersects `rect`: unlabelled ones, plus those of
        label `include_label` -- a drag never grabs another label's vectors."""
        labelled = self._labelled_sig_map()
        hits: list[Vector] = []
        for v in self._visible_vectors():
            owner = labelled.get(path_signature(v))
            if owner is not None and owner.label_id != include_label:
                continue
            if bboxes_intersect(v.bbox, rect):
                hits.append(v)
        return hits

    def _area_select(self, rect) -> None:
        self._drop_multi()
        active = self._active_entry()
        hits = self._items_in_rect(rect, include_label=active.label_id if active else None)
        if not hits:
            return
        if active is not None:
            self._edit_active_label(hits)
            return
        ids = {id(v) for v in hits}
        # All-selected area -> deselect it; otherwise add it to the selection.
        if ids <= self.selected:
            self.selected -= ids
        else:
            self.selected |= ids
        self._render()
        if self.selected:
            self._focus_text_entry()

    def _edit_active_label(self, hits: list[Vector]) -> None:
        """Add `hits` to the selected label, or remove them when they all belong to it."""
        entry = self._active_entry()
        new_sigs = edit_label_vectors(entry.vector_signatures, [path_signature(v) for v in hits])
        if new_sigs is None:
            self._status.config(text="a label keeps at least one vector -- use Delete to remove it")
            return
        members = [v for v in self.vectors if path_signature(v) in set(new_sigs)]
        entry.vector_signatures = new_sigs
        entry.bbox = union_bbox([v.bbox for v in members]) if members else entry.bbox
        self.selected = {id(v) for v in members}
        self._refresh_label_list()
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
        if self._multi_ids:
            self._label_target_lbl.config(
                text=f"{len(self._multi_ids)} labels selected -- Delete removes them"
            )
        elif self._active_label_id is not None:
            self._label_target_lbl.config(
                text=f"editing label ({len(self.selected)} vectors) -- click to add / remove"
            )
        elif self.selected:
            self._label_target_lbl.config(
                text=f"{len(self.selected)} vector(s) selected -- Apply to label"
            )
        else:
            self._label_target_lbl.config(text="(select vectors or a label)")

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
        c1 = pymupdf.Point(cx, cy) * self.matrix
        c2 = pymupdf.Point(tip_x, tip_y) * self.matrix
        self.canvas.create_line(
            c1.x,
            c1.y,
            c2.x,
            c2.y,
            fill=SELECTED_COLOR,
            width=2,
            arrow=tk.LAST,
            tags=("overlay", "rotation_arrow"),
        )

    def _apply_label(self) -> None:
        text = self._text_var.get().strip()
        if not text or self._multi_ids:
            return  # nothing to write (or several labels selected); Delete removes labels
        rotation = int(round(self._rot_var.get()))
        active = self._active_entry()
        if active is not None and not self.selected:
            # A label whose vectors are no longer on the page: text and rotation only.
            active.text, active.expected_rotation = text, rotation
            self._clear_selection()
            return
        if not self.selected:
            return
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
                entry.bbox = bbox
        else:
            entry = LabelEntry(
                page_index=self.page_index,
                bbox=bbox,
                label_id=uuid.uuid4().hex,
                text=text,
                source="vector",
                expected_rotation=rotation,
                vector_signatures=vsigs,
            )
            self.labels.entries.append(entry)

        self._clear_selection()

    def _entry_by_label_id(self, label_id: str) -> int | None:
        for i, entry in enumerate(self.labels.entries):
            if entry.label_id == label_id:
                return i
        return None

    def _delete_label(self) -> None:
        if self._multi_ids:
            n = len(self._multi_ids)
            if not self._confirm(f"Delete {n} labels?"):
                return
            self.labels.entries = [
                e for e in self.labels.entries if e.label_id not in self._multi_ids
            ]
            self._clear_selection()
            self._status.config(text=f"deleted {n} labels")
            return
        if self._active_label_id is None:
            return
        idx = self._entry_by_label_id(self._active_label_id)
        if idx is None:
            return
        del self.labels.entries[idx]
        self._clear_selection()

    def _confirm(self, question: str) -> bool:
        return messagebox.askyesno("Delete labels", question, parent=self.root)

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
                f'"{entry.text}"  rot={entry.expected_rotation}'
                if entry
                else f"{hit.type}  seqno={hit.seqno}  -- unlabelled"
            )
            self.tooltip.show(event.x_root, event.y_root, text)
            return
        live_sigs = {path_signature(v) for v in self.vectors}
        for entry in self._page_entries():
            if entry.vector_signatures and any(s in live_sigs for s in entry.vector_signatures):
                continue
            if bbox_contains(entry.bbox, pt.x, pt.y):
                self.tooltip.show(event.x_root, event.y_root, f'"{entry.text}"  ({entry.source})')
                return
        self.tooltip.hide()

    # ---- persistence ------------------------------------------------

    def save(self) -> None:
        save_labels(self.labels, self.out_path)
        _LOG.info("saved %d label(s) to %s", len(self.labels.entries), self.out_path)

    def _on_close(self) -> None:
        self.save()
        self._stop_workers()
        self.root.destroy()

    def _stop_workers(self) -> None:
        """Drop queued pages and end the worker now: a page being read can take a while (OCR),
        and nobody waits for it once the window is closed."""
        # ProcessPoolExecutor has no public terminate before Python 3.14; shutdown() drops its
        # process table, so take it first.
        processes = list((getattr(self._pool, "_processes", None) or {}).values())
        self._pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            process.terminate()

    def run(self) -> None:
        self.root.mainloop()
        self.reader.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label the vector text of a PDF with its ground-truth text."
    )
    parser.add_argument("pdf", help="Path to the input PDF.")
    parser.add_argument("--page", type=int, default=0, help="0-based page index.")
    parser.add_argument(
        "--out",
        default=None,
        help="Path to save/load the label JSON file (default: outputs/labels/<pdf stem>.json).",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Don't pre-fill unlabelled pages with v2_vector_class's text.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)
    out = args.out or str(output_dir("labels") / f"{Path(args.pdf).stem}.json")
    app = VectorLabelApp(args.pdf, args.page, out, seed=not args.no_seed)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
