"""Read-only viewer over one PDF's `master_label.py` label folder --
`native_labels.json` / `vector_labels.json` / `raster_labels.json` all
overlaid on the page at once, independently toggleable, so a whole labelled
PDF can be sanity-checked before trusting it for regression testing. No
editing, no save button.

    .venv/Scripts/python.exe scripts/label/label_viewer.py PATH/TO.pdf
    .venv/Scripts/python.exe scripts/label/label_viewer.py outputs/labels/<stem>_label

Either the source PDF path or the label folder itself resolves to the same
folder (`outputs/labels/<stem>_label/`, per `master_label.py`'s naming
convention); the viewer always renders `original.pdf` inside that folder.

Four independent checkboxes:

- **Native** -- `native_labels.json` entries as bbox rectangles.
- **Vector** -- `vector_labels.json` entries as bbox rectangles, plus their
  actual backing `Vector`s (resolved by `path_signature` against a fresh
  `extract_vectors(page)`), so the real labelled geometry is visible.
- **Raster geometry** -- `raster_labels.json.geometry_entries`, dashed for
  `source="auto"` (machine-derived from the original vectors), solid for
  `source="manual"` (hand-traced over an embedded image).
- **Raster text** -- `raster_labels.json` text entries, dashed for a
  `"vecsync:"`-prefixed `label_id` (re-synced from vector labels), solid
  for a genuine hand-drawn raster entry.

Page nav (`<`/`>`, PageUp/PageDown) + zoom + hover tooltips only.

Not unit-testable (a real Tk event loop). Smoke-test manually against a
folder `master_label.py` has already produced: confirm each checkbox
toggles its own overlay independently and hovering each kind shows the
right text/attrs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tkinter import ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tkinter as tk

import pymupdf as fitz

from _common import (
    ENTRY_COLOR,
    MAX_ZOOM,
    MIN_ZOOM,
    ZOOM_DEFAULT,
    ZOOM_STEP,
    Tooltip,
    bezier_points,
    draw_vector,
    get_display_matrix,
)

from rastervec.Evaluation.Labelling.label_schema import (
    LabelEntry,
    LabelSet,
    load_labels,
    path_signature,
)
from rastervec.commons.helpers.geometry import bbox_contains
from rastervec.commons.logging_setup import configure_logging, get_logger
from rastervec.commons.paths import output_dir
from rastervec.pipelines._steps import extract_vectors
from rastervec.Reader.reader import Reader

_LOG = get_logger("label_viewer")

NATIVE_COLOR = "#e67e22"
VECTOR_BOX_COLOR = "#33aa33"
VECTOR_LINE_COLOR = "#3366ff"
RASTER_TEXT_COLOR = "#cc33cc"
RASTER_GEOMETRY_COLOR = "#00bcd4"


def resolve_label_folder(path: str) -> Path:
    """`path` may be the source PDF or the `<stem>_label` folder itself --
    either resolves to the same folder, per `master_label.py`'s naming
    convention (`outputs/labels/<stem>_label/`)."""
    p = Path(path)
    if p.is_dir():
        return p
    stem = p.stem
    if stem.endswith("_label"):
        stem = stem[: -len("_label")]
    return output_dir("labels") / f"{stem}_label"


def _load_or_empty(path: Path, pdf_path: str) -> LabelSet:
    return load_labels(str(path)) if path.exists() else LabelSet(pdf_path=pdf_path)


class LabelViewerApp:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        original_path = folder / "original.pdf"
        if not original_path.exists():
            raise FileNotFoundError(f"{original_path} not found -- run master_label.py on this PDF first")
        self.original_path = original_path

        self.native_labels = _load_or_empty(folder / "native_labels.json", str(original_path))
        self.vector_labels = _load_or_empty(folder / "vector_labels.json", str(original_path))
        self.raster_labels = _load_or_empty(folder / "raster_labels.json", str(original_path))

        self.reader = Reader(str(original_path))
        self.zoom = ZOOM_DEFAULT

        self.root = tk.Tk()
        self.tooltip = Tooltip(self.root)
        self._build_layout()
        self._bind_events()
        self._load_page(0)

    # ---- per-page state -------------------------------------------------

    def _load_page(self, page_index: int) -> None:
        self.page_index = max(0, min(self.reader.page_count() - 1, page_index))
        self.page = self.reader.get_page(self.page_index)
        self._page_vectors = extract_vectors(self.page) if self.show_vector.get() else []
        self.matrix = get_display_matrix(self.page.fitz_page, self.zoom)
        self.root.title(
            f"Label Viewer -- {self.folder.name} page {self.page_index + 1}/{self.reader.page_count()}"
        )
        self._sync_page_entry()
        self._render()

    def _entries_on_page(self, labels: LabelSet) -> list[LabelEntry]:
        return [e for e in labels.entries if e.page_index == self.page_index]

    # ---- layout -----------------------------------------------------------

    def _build_layout(self) -> None:
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(bar, text="Zoom -", command=lambda: self._change_zoom(-1)).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(bar, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Zoom +", command=lambda: self._change_zoom(1)).pack(side=tk.LEFT, padx=2)

        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.show_native = tk.BooleanVar(value=True)
        self.show_vector = tk.BooleanVar(value=True)
        self.show_raster_geometry = tk.BooleanVar(value=True)
        self.show_raster_text = tk.BooleanVar(value=True)
        for text, var in (
            ("Native", self.show_native), ("Vector", self.show_vector),
            ("Raster geometry", self.show_raster_geometry), ("Raster text", self.show_raster_text),
        ):
            ttk.Checkbutton(bar, text=text, variable=var, command=self._on_toggle).pack(side=tk.LEFT, padx=4)

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

    def _bind_events(self) -> None:
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _e: self.tooltip.hide())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_wheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
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

        if self.show_native.get():
            for entry in self._entries_on_page(self.native_labels):
                self._draw_box(entry.cluster_bbox, NATIVE_COLOR, dash=None)

        if self.show_vector.get():
            live_sigs = {path_signature(v): v for v in self._page_vectors}
            for entry in self._entries_on_page(self.vector_labels):
                self._draw_box(entry.cluster_bbox, VECTOR_BOX_COLOR, dash=None)
                for sig in entry.vector_signatures:
                    v = live_sigs.get(sig)
                    if v is not None:
                        draw_vector(self.canvas, self.matrix, v, VECTOR_LINE_COLOR, 1)

        if self.show_raster_geometry.get():
            for geo in self.raster_labels.geometry_entries:
                if geo.page_index != self.page_index:
                    continue
                self._draw_geometry(geo)

        if self.show_raster_text.get():
            for entry in self._entries_on_page(self.raster_labels):
                dash = (4, 3) if entry.label_id.startswith("vecsync:") else None
                self._draw_box(entry.cluster_bbox, RASTER_TEXT_COLOR, dash=dash)

        self.zoom_label.config(text=f"{round(self.zoom * 100)}%")
        self._status.config(
            text=f"native {len(self._entries_on_page(self.native_labels))}  |  "
                 f"vector {len(self._entries_on_page(self.vector_labels))}  |  "
                 f"raster text {len(self._entries_on_page(self.raster_labels))}  |  "
                 f"raster geometry {sum(1 for g in self.raster_labels.geometry_entries if g.page_index == self.page_index)}"
        )

    def _draw_box(self, bbox, color: str, *, dash) -> None:
        rect = fitz.Rect(bbox) * self.matrix
        self.canvas.create_rectangle(
            rect.x0, rect.y0, rect.x1, rect.y1, outline=color, width=2, dash=dash,
            tags=("overlay",),
        )

    def _draw_geometry(self, geo) -> None:
        color = RASTER_GEOMETRY_COLOR
        pts = bezier_points(*geo.points) if geo.kind == "c" and len(geo.points) == 4 else geo.points
        coords: list[float] = []
        for x, y in pts:
            p = fitz.Point(x, y) * self.matrix
            coords.extend([p.x, p.y])
        if len(coords) < 4:
            return
        dash = (2, 2) if geo.source == "auto" else None
        self.canvas.create_line(*coords, fill=color, width=1, dash=dash, tags=("overlay",))

    # ---- zoom / nav -----------------------------------------------------

    def _on_toggle(self) -> None:
        if self.show_vector.get() and not self._page_vectors:
            self._page_vectors = extract_vectors(self.page)
        self._render()

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
        self.tooltip.hide()
        self._load_page(new_index)

    # ---- hover -------------------------------------------------------

    def _on_motion(self, event: "tk.Event") -> None:
        pt = self._page_point(event)
        if self.show_native.get():
            for e in self._entries_on_page(self.native_labels):
                if bbox_contains(e.cluster_bbox, pt.x, pt.y):
                    self.tooltip.show(event.x_root, event.y_root, f'native: "{e.text}"  rot={e.expected_rotation}')
                    return
        if self.show_vector.get():
            for e in self._entries_on_page(self.vector_labels):
                if bbox_contains(e.cluster_bbox, pt.x, pt.y):
                    self.tooltip.show(event.x_root, event.y_root, f'vector: "{e.text}"  rot={e.expected_rotation}')
                    return
        if self.show_raster_text.get():
            for e in self._entries_on_page(self.raster_labels):
                if bbox_contains(e.cluster_bbox, pt.x, pt.y):
                    kind = "re-synced" if e.label_id.startswith("vecsync:") else "manual"
                    self.tooltip.show(event.x_root, event.y_root, f'raster ({kind}): "{e.text}"')
                    return
        self.tooltip.hide()

    def run(self) -> None:
        self.root.mainloop()
        self.reader.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only viewer over a master_label.py label folder.")
    parser.add_argument("path", help="Source PDF path or its <stem>_label folder.")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)
    folder = resolve_label_folder(args.path)
    app = LabelViewerApp(folder)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
