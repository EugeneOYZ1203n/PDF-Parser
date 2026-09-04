"""Manual labelling: a small Tk UI for editing vector-text clusters on a
rendered page and typing their ground-truth text, saved via
`label_schema.save_labels`. `_get_display_matrix` (the page-space ->
canvas-space transform rule, see `rastervec/models.py`'s coordinate-space
docstring) and the `Tooltip` class were ported from the former
`debug_app.py` when it was removed. `pipeline.run_page_context` gets the
same text-candidate clusters the pipeline's "Text Candidates" stage
produces, rather than re-implementing extraction/clustering/rendering
here.

Beyond plain labelling it's a light cluster *editor*: the pipeline's
clustering is not always right (a word split across two clusters, or two
words merged into one), so before assigning text you can regroup. Two
edit modes:

- **cluster mode** -- left-click toggles a whole cluster's selection;
  `Group` merges every selected cluster into one; `Ungroup` splits a
  selected cluster back into its pre-spatial "groups" (or, for an
  already-edited cluster, into one-path-per-cluster).
- **path mode** -- left-click toggles an individual `VectorPath`; `Group`
  builds a brand-new cluster from exactly the selected paths, pulling
  each out of whatever cluster currently owns it.

Instead of a single click you can also left-click-drag a rubber-band box:
every cluster/path (mode-dependent) whose bbox intersects the box is
added to the selection, or -- if all of them were already selected --
removed from it, so the same drag both selects and deselects an area.

`Ctrl+Z` undoes the last group/ungroup. The inline label bar below the
toolbar -- a Text entry + a Rotation dropdown that stay visible (an
earlier version used chained `simpledialog` popups that could vanish
behind the topmost hover tooltip) -- writes to whichever clusters are
currently targeted: **every selected cluster** (left-click / rubber-band
in cluster mode -- one `Apply` labels them all with the same text +
rotation), or a **single** cluster you right-click (which drops any
selection and pre-fills the fields from its existing label). `Apply` (or
Enter in the Text entry) writes the `LabelEntry`(s) then clears the
selection, empties the Text field and resets Rotation to 0; `Delete`
removes the targeted label(s) the same way. Hovering a cluster shows its
assigned text.

The `<` / `>` buttons (or `PageUp` / `PageDown`) move between pages of the
same PDF without relaunching; the current labels are saved on every page
change, and one `LabelSet` covers every page (each `LabelEntry` carries
its own `page_index`), so overlays, hover and the label bar only ever act
on the page currently shown.

`LabelEntry`s loaded from `--out` whose signature matches no current
cluster on the current page (every `source="auto"` entry, plus manual
entries left stale by an edit) are drawn as dashed grey boxes, so this
same window doubles as an auto-label viewer -- see `view_auto_labels.py`.

Not unit-testable (a real Tk event loop). Smoke-test manually:

    .venv/Scripts/python.exe -m rastervec.Evaluation.Labelling.manual_label \
        path/to.pdf --page 0 --out labels.json

1. A window opens showing the page with every surviving text-candidate
   cluster's bbox in blue. `Zoom -`/`Zoom +`, the scrollbars, mouse
   wheel (Shift = horizontal), and Ctrl+wheel (zoom) all work; overlays
   stay aligned with the page on a rotated page.
2. In cluster mode click two adjacent clusters (they turn orange) then
   `Group` -- one merged bbox. Select it and `Ungroup` -- it splits back.
   Drag a box across several clusters -- all turn orange; drag the same
   box again -- they all clear.
3. Switch to path mode, click or drag-box a few paths, `Group` -- a new
   cluster of exactly those paths; the clusters they came from lose them
   (empty ones vanish). `Ctrl+Z` reverts.
4. Right-click a cluster -- the label bar shows it as the target; type
   text, pick a rotation, `Apply` -- the bbox turns green, the selection
   clears and the fields reset (Text empty, Rotation 0); hovering it shows
   `"<text>"  rot=<n>`. The Text/Rotation fields stay on screen the whole
   time. Select several clusters instead (no right-click) and `Apply` --
   every one of them gets that same label.
5. Click `>` -- the next page loads (title shows `page 2/N`); its own
   clusters appear and the label bar resets. Label something, click `<`
   back -- the page-1 labels are still there.
6. Click "Save" (or close the window) to write `labels.json` via
   `label_schema.save_labels`. Re-running against the same PDF loads
   existing labels at `--out` first, so a session can be resumed
   (edited-cluster signatures won't re-match, so those show as dashed
   grey boxes on reload).
"""
from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pymupdf as fitz

from rastervec.Evaluation.Labelling.label_schema import (
    LabelEntry,
    LabelSet,
    cluster_signature,
    load_labels,
    save_labels,
)
from rastervec.helpers.geometry import (
    bbox_area,
    bbox_contains,
    bboxes_intersect,
    union_bbox,
)
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.models import VectorPath
from rastervec.pipeline import run_page_context
from rastervec.Reader.reader import Reader

_LOG = get_logger("manual_label")

# Ported from the former debug_app.py when it was removed.
MIN_ZOOM = 0.25
MAX_ZOOM = 6.0
ZOOM_STEP = 1.25


def _get_display_matrix(fitz_page: "fitz.Page", zoom: float) -> "fitz.Matrix":
    """page-space (unrotated MediaBox) -> canvas-space, page rotation baked in."""
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


_ZOOM = 1.5
_UNLABELLED_COLOR = "#3366ff"
_LABELLED_COLOR = "#33aa33"
_SELECTED_COLOR = "#ff8800"
_ENTRY_COLOR = "#999999"
_PATH_COLOR = "#7a7a7a"


def _draw_path(canvas: tk.Canvas, matrix: "fitz.Matrix", path: VectorPath, color: str, width: int):
    """Port of debug_app._draw_vector_path: polyline of path.points through
    the display matrix (polygon outline for re/qu, line otherwise)."""
    coords: list[float] = []
    for x, y in path.points:
        p = fitz.Point(x, y) * matrix
        coords.extend([p.x, p.y])
    if len(coords) < 4:
        return
    if path.kind in ("re", "qu"):
        canvas.create_polygon(*coords, outline=color, fill="", width=width, tags=("overlay",))
    else:
        canvas.create_line(*coords, fill=color, width=width, tags=("overlay",))


# Canvas-pixel movement below which a press/release is treated as a plain click.
_DRAG_THRESHOLD_PX = 4


class ManualLabelApp:
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

        self.mode = "cluster"
        self.selected: set[int] = set()
        self._drag_start: tuple[float, float] | None = None
        self._drag_moved = False
        self._drag_rect_id: int | None = None
        self._undo_stack: list[list[list[VectorPath]]] = []
        # (cluster_signature, cluster_bbox) the label bar currently edits.
        self._label_target: tuple[str, tuple[float, float, float, float]] | None = None
        self.zoom = _ZOOM

        self.root = tk.Tk()
        self.tooltip = Tooltip(self.root)

        self._build_layout()
        self._bind_events()
        self._load_page(page_index)

    # ---- per-page state -------------------------------------------------

    def _load_page(self, page_index: int) -> None:
        """(Re)run the pipeline through Text Candidates for `page_index` and
        rebuild every per-page working structure. `self.reader`, `self.labels`
        and the Tk widgets outlive this."""
        self.page_index = max(0, min(self.reader.page_count() - 1, page_index))
        self.ctx = run_page_context(
            self.reader, self.page_index, final_stage="text_candidates"
        )

        # Mutable working list -- all rendering/hit-testing/labelling uses this.
        # The pipeline's own cluster lists are referenced directly (never
        # mutated in place -- every edit builds new lists) so their id() still
        # matches ctx.cluster_groups keys for Ungroup.
        self.working_clusters: list[list[VectorPath]] = [
            cluster for cluster in (self.ctx.text_clusters or []) if cluster
        ]
        # id(original cluster) -> its pre-spatial "groups", for Ungroup.
        self._lineage: dict[int, list[list[VectorPath]]] = dict(
            self.ctx.cluster_groups or {}
        )

        self.selected.clear()
        self._undo_stack.clear()
        self.matrix = _get_display_matrix(self.ctx.page.fitz_page, self.zoom)

        self.root.title(
            f"Manual Label -- {Path(self.pdf_path).name} "
            f"page {self.page_index + 1}/{self.reader.page_count()}"
        )
        self._sync_page_entry()
        self._clear_label_panel()
        self._render()

    def _page_entries(self) -> list[LabelEntry]:
        """Label entries for the page currently shown."""
        return [e for e in self.labels.entries if e.page_index == self.page_index]

    # ---- layout ---------------------------------------------------------

    def _build_layout(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Zoom -", command=lambda: self._change_zoom(-1)).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(bar, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Zoom +", command=lambda: self._change_zoom(1)).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._mode_var = tk.StringVar(value=self.mode)
        ttk.Radiobutton(
            bar, text="Cluster", value="cluster", variable=self._mode_var,
            command=self._on_mode_change,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            bar, text="Path", value="path", variable=self._mode_var,
            command=self._on_mode_change,
        ).pack(side=tk.LEFT)

        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(bar, text="Group", command=self._group).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Ungroup", command=self._ungroup).pack(side=tk.LEFT, padx=2)
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

        # ---- inline label bar (replaces the old simpledialog popups) ----
        label_bar = ttk.Frame(self.root)
        label_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(label_bar, text="Label:").pack(side=tk.LEFT, padx=(4, 2))
        self._label_target_lbl = ttk.Label(label_bar, text="(right-click a cluster)", width=28)
        self._label_target_lbl.pack(side=tk.LEFT)

        ttk.Label(label_bar, text="Text").pack(side=tk.LEFT, padx=(8, 2))
        self._text_var = tk.StringVar()
        self._text_entry = ttk.Entry(label_bar, textvariable=self._text_var, width=44)
        self._text_entry.pack(side=tk.LEFT)
        self._text_entry.bind("<Return>", lambda _e: self._apply_label())

        ttk.Label(label_bar, text="Rot").pack(side=tk.LEFT, padx=(8, 2))
        self._rot_var = tk.StringVar(value="0")
        ttk.Combobox(
            label_bar, textvariable=self._rot_var, width=5, state="readonly",
            values=("0", "90", "180", "270"),
        ).pack(side=tk.LEFT)

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
        self.root.bind("<Control-z>", lambda _e: self._undo())
        self.root.bind("<Escape>", lambda _e: self._clear_selection())
        self.root.bind("<Next>", lambda _e: self._change_page(1))
        self.root.bind("<Prior>", lambda _e: self._change_page(-1))

    # ---- coordinate helpers -------------------------------------------

    def _page_point(self, event: "tk.Event") -> "fitz.Point":
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        return fitz.Point(cx, cy) * ~self.matrix

    def _cluster_bbox(self, cluster: list[VectorPath]):
        return union_bbox([p.bbox for p in cluster])

    def _iter_paths(self):
        for ci, cluster in enumerate(self.working_clusters):
            for path in cluster:
                yield ci, path

    # ---- rendering ----------------------------------------------------

    def _render(self) -> None:
        self.canvas.delete("all")
        pix = self.ctx.page.fitz_page.get_pixmap(matrix=self.matrix)
        self._photo = tk.PhotoImage(data=pix.tobytes("ppm"))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.config(scrollregion=(0, 0, pix.width, pix.height))

        page_entries = self._page_entries()
        labelled = {e.cluster_signature for e in page_entries}
        live_sigs = set()

        if self.mode == "cluster":
            for idx, cluster in enumerate(self.working_clusters):
                if not cluster:
                    continue
                sig = cluster_signature(cluster)
                live_sigs.add(sig)
                rect = fitz.Rect(self._cluster_bbox(cluster)) * self.matrix
                if idx in self.selected:
                    color, width = _SELECTED_COLOR, 3
                elif sig in labelled:
                    color, width = _LABELLED_COLOR, 2
                else:
                    color, width = _UNLABELLED_COLOR, 2
                self.canvas.create_rectangle(
                    rect.x0, rect.y0, rect.x1, rect.y1, outline=color, width=width,
                    tags=("cluster", f"c{idx}"),
                )
        else:
            for _ci, path in self._iter_paths():
                selected = id(path) in self.selected
                _draw_path(
                    self.canvas, self.matrix, path,
                    _SELECTED_COLOR if selected else _PATH_COLOR,
                    3 if selected else 1,
                )
            live_sigs = {cluster_signature(c) for c in self.working_clusters if c}

        # Entries with no matching live cluster (auto labels, stale manual edits).
        for entry in page_entries:
            if entry.cluster_signature in live_sigs:
                continue
            rect = fitz.Rect(entry.cluster_bbox) * self.matrix
            self.canvas.create_rectangle(
                rect.x0, rect.y0, rect.x1, rect.y1, outline=_ENTRY_COLOR, width=1, dash=(4, 3),
                tags=("entry", entry.cluster_signature),
            )

        self.zoom_label.config(text=f"{round(self.zoom * 100)}%")
        self._status.config(
            text=f"{self.mode} mode  |  {len(self.working_clusters)} clusters  "
                 f"|  {len(self.selected)} selected  |  {len(page_entries)} labels "
                 f"(page)  |  {len(self.labels.entries)} total"
        )
        self._refresh_label_hint()

    # ---- mode / zoom -------------------------------------------------

    def _on_mode_change(self) -> None:
        self.mode = self._mode_var.get()
        self.selected.clear()
        self._render()

    def _change_zoom(self, direction: int) -> None:
        if direction > 0:
            self.zoom = min(MAX_ZOOM, self.zoom * ZOOM_STEP)
        else:
            self.zoom = max(MIN_ZOOM, self.zoom / ZOOM_STEP)
        self.matrix = _get_display_matrix(self.ctx.page.fitz_page, self.zoom)
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

    # ---- selection / editing --------------------------------------------

    def _push_undo(self) -> None:
        # Outer list only -- inner cluster lists are never mutated in place, so
        # sharing the references keeps id()-based lineage valid after an undo.
        self._undo_stack.append(list(self.working_clusters))

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        self.working_clusters = self._undo_stack.pop()
        self.selected.clear()
        self._render()

    def _clear_selection(self) -> None:
        self.selected.clear()
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
            abs(x1 - x0) < _DRAG_THRESHOLD_PX and abs(y1 - y0) < _DRAG_THRESHOLD_PX
        ):
            return
        self._drag_moved = True
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
        self._drag_rect_id = self.canvas.create_rectangle(
            x0, y0, x1, y1, outline=_SELECTED_COLOR, width=1, dash=(3, 2),
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

    def _click_select(self, pt: "fitz.Point") -> None:
        if self.mode == "cluster":
            for idx, cluster in enumerate(self.working_clusters):
                if cluster and bbox_contains(self._cluster_bbox(cluster), pt.x, pt.y):
                    self.selected.symmetric_difference_update({idx})
                    break
        else:
            hit = None
            for _ci, path in self._iter_paths():
                if bbox_contains(path.bbox, pt.x, pt.y):
                    if hit is None or bbox_area(path.bbox) < bbox_area(hit.bbox):
                        hit = path
            if hit is not None:
                self.selected.symmetric_difference_update({id(hit)})
        self._render()

    def _items_in_rect(self, rect) -> set[int]:
        """Selection keys (cluster index / id(path)) whose bbox intersects `rect`."""
        hits: set[int] = set()
        if self.mode == "cluster":
            for idx, cluster in enumerate(self.working_clusters):
                if cluster and bboxes_intersect(self._cluster_bbox(cluster), rect):
                    hits.add(idx)
        else:
            for _ci, path in self._iter_paths():
                if bboxes_intersect(path.bbox, rect):
                    hits.add(id(path))
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
        self._render()

    def _group(self) -> None:
        if len(self.selected) < 2:
            return
        self._push_undo()
        if self.mode == "cluster":
            chosen = [self.working_clusters[i] for i in sorted(self.selected)]
            merged: list[VectorPath] = [p for cluster in chosen for p in cluster]
            carried = next((c for c in chosen if cluster_signature(c) in
                            {e.cluster_signature for e in self._page_entries()}), None)
            self.working_clusters = [
                c for i, c in enumerate(self.working_clusters) if i not in self.selected
            ]
            self.working_clusters.append(merged)
            if carried is not None:
                self._retarget_entry(cluster_signature(carried), merged)
        else:
            chosen_ids = set(self.selected)
            picked: list[VectorPath] = []
            new_clusters: list[list[VectorPath]] = []
            for cluster in self.working_clusters:
                kept = []
                for path in cluster:
                    (picked if id(path) in chosen_ids else kept).append(path)
                if kept:
                    new_clusters.append(kept)
            if len(picked) < 2:
                self._undo_stack.pop()
                return
            new_clusters.append(picked)
            self.working_clusters = new_clusters
        self.selected.clear()
        self._render()

    def _ungroup(self) -> None:
        if self.mode != "cluster" or not self.selected:
            return
        self._push_undo()
        result: list[list[VectorPath]] = []
        for idx, cluster in enumerate(self.working_clusters):
            if idx not in self.selected:
                result.append(cluster)
                continue
            groups = self._lineage.get(id(cluster))
            if groups:
                result.extend([list(g) for g in groups if g])
            else:
                result.extend([[p] for p in cluster])
        self.working_clusters = result
        self.selected.clear()
        self._render()

    # ---- labelling ----------------------------------------------------

    def _retarget_entry(self, old_sig: str, cluster: list[VectorPath]) -> None:
        for entry in self._page_entries():
            if entry.cluster_signature == old_sig:
                entry.cluster_signature = cluster_signature(cluster)
                entry.cluster_bbox = self._cluster_bbox(cluster)
                break

    # ---- inline label bar ------------------------------------------------

    def _clear_label_panel(self) -> None:
        self._label_target = None
        self._text_var.set("")
        self._rot_var.set("0")
        self._refresh_label_hint()

    def _refresh_label_hint(self) -> None:
        """Keep the label-bar's target text in sync: an explicit right-click
        target wins, else a cluster-mode selection (Apply labels all of it),
        else the idle prompt."""
        if self._label_target is not None:
            sig, bbox = self._label_target
            self._label_target_lbl.config(text=f"1 cluster @ ({bbox[0]:.0f}, {bbox[1]:.0f})")
        elif self.mode == "cluster" and self.selected:
            self._label_target_lbl.config(
                text=f"{len(self.selected)} selected cluster(s) — Apply labels all"
            )
        else:
            self._label_target_lbl.config(text="(right-click or select clusters)")

    def _on_right_click(self, event: "tk.Event") -> None:
        self.tooltip.hide()
        if self.mode != "cluster":
            return
        pt = self._page_point(event)
        hit = next(
            (c for c in self.working_clusters
             if c and bbox_contains(self._cluster_bbox(c), pt.x, pt.y)),
            None,
        )
        if hit is None:
            return
        sig = cluster_signature(hit)
        bbox = self._cluster_bbox(hit)
        # Right-click targets one specific cluster -- drop any multi-selection
        # so Apply is unambiguous.
        self.selected.clear()
        self._label_target = (sig, bbox)
        existing = next((e for e in self._page_entries() if e.cluster_signature == sig), None)
        self._text_var.set(existing.text if existing else "")
        self._rot_var.set(str(existing.expected_rotation if existing else 0))
        self._render()
        self._text_entry.focus_set()
        self._text_entry.selection_range(0, tk.END)

    def _matching_entry_index(self, sig: str) -> int | None:
        for i, entry in enumerate(self.labels.entries):
            if entry.page_index == self.page_index and entry.cluster_signature == sig:
                return i
        return None

    def _label_targets(self) -> list[tuple[str, tuple[float, float, float, float]]]:
        """(signature, bbox) for every cluster Apply/Delete should act on: a
        whole cluster-mode selection if there is one, else the single
        right-click target."""
        if self.mode == "cluster" and self.selected:
            return [
                (cluster_signature(c), self._cluster_bbox(c))
                for i, c in enumerate(self.working_clusters)
                if i in self.selected and c
            ]
        if self._label_target is not None:
            return [self._label_target]
        return []

    def _apply_label(self) -> None:
        targets = self._label_targets()
        if not targets:
            return
        text = self._text_var.get().strip()
        if not text:
            return  # nothing to write; use Delete to remove an existing label
        try:
            rotation = int(self._rot_var.get())
        except ValueError:
            rotation = 0
        for sig, bbox in targets:
            entry = LabelEntry(
                page_index=self.page_index, cluster_bbox=bbox, cluster_signature=sig,
                text=text, source="manual", expected_rotation=rotation,
            )
            existing = self._matching_entry_index(sig)
            if existing is not None:
                self.labels.entries[existing] = entry
            else:
                self.labels.entries.append(entry)
        self.selected.clear()
        self._clear_label_panel()  # clears text + resets rotation to 0
        self._render()

    def _delete_label(self) -> None:
        targets = self._label_targets()
        if not targets:
            return
        removed = False
        for sig, _bbox in targets:
            existing = self._matching_entry_index(sig)
            if existing is not None:
                del self.labels.entries[existing]
                removed = True
        if not removed:
            return
        self.selected.clear()
        self._clear_label_panel()
        self._render()

    # ---- hover -------------------------------------------------------

    def _on_motion(self, event: "tk.Event") -> None:
        pt = self._page_point(event)
        page_entries = self._page_entries()
        entries_by_sig = {e.cluster_signature: e for e in page_entries}

        if self.mode == "cluster":
            for cluster in self.working_clusters:
                if not cluster or not bbox_contains(self._cluster_bbox(cluster), pt.x, pt.y):
                    continue
                entry = entries_by_sig.get(cluster_signature(cluster))
                text = (
                    f'"{entry.text}"  rot={entry.expected_rotation}'
                    if entry else f"{len(cluster)} path(s) - unlabelled"
                )
                self.tooltip.show(event.x_root, event.y_root, text)
                return
        else:
            hit = None
            for _ci, path in self._iter_paths():
                if bbox_contains(path.bbox, pt.x, pt.y):
                    if hit is None or bbox_area(path.bbox) < bbox_area(hit.bbox):
                        hit = path
            if hit is not None:
                self.tooltip.show(event.x_root, event.y_root, f"{hit.kind}  seq={hit.seq}")
                return

        for entry in page_entries:
            if entry.cluster_signature in entries_by_sig and bbox_contains(
                entry.cluster_bbox, pt.x, pt.y
            ):
                self.tooltip.show(
                    event.x_root, event.y_root, f'"{entry.text}"  ({entry.source})'
                )
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
    parser = argparse.ArgumentParser(description="Manually label vector-text clusters.")
    parser.add_argument("pdf", help="Path to the input PDF.")
    parser.add_argument("--page", type=int, default=0, help="0-based page index.")
    parser.add_argument("--out", required=True, help="Path to save/load the label JSON file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)
    app = ManualLabelApp(args.pdf, args.page, args.out)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
