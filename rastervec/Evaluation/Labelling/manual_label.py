"""Manual labelling: a small Tk UI for clicking a vector-text cluster on a
rendered page and typing its ground-truth text, saved via
`label_schema.save_labels`. Reuses `debug_app._get_display_matrix` (the
same page-space -> canvas-space transform rule the debug app and inspector
tool both use, see `rastervec/models.py`'s coordinate-space docstring) and
`pipeline.run_page_context` to get the same text-candidate clusters the
debug app's "Text Candidates" stage would show, rather than
re-implementing extraction/clustering/rendering here.

Not unit-testable (a real Tk event loop). Smoke-test manually:

    .venv/Scripts/python.exe -m rastervec.Evaluation.Labelling.manual_label \
        path/to.pdf --page 0 --out labels.json

1. A window opens showing the page with every surviving text-candidate
   cluster's bbox drawn in blue.
2. Click inside a cluster's bbox -- it turns green and a text-entry dialog
   pops up; type the ground-truth text and press OK (Cancel skips it).
3. Click "Save" (or close the window) to write `labels.json` via
   `label_schema.save_labels`. Re-running against the same PDF/page loads
   any existing labels at `--out` first, so a labelling session can be
   resumed.
"""
from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import simpledialog

import pymupdf as fitz

from rastervec.debug_app import _get_display_matrix
from rastervec.Evaluation.Labelling.label_schema import (
    LabelEntry,
    LabelSet,
    cluster_signature,
    load_labels,
    save_labels,
)
from rastervec.helpers.geometry import union_bbox
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.pipeline import run_page_context
from rastervec.Reader.reader import Reader

_LOG = get_logger("manual_label")

_ZOOM = 1.5
_UNLABELLED_COLOR = "#3366ff"
_LABELLED_COLOR = "#33aa33"


class ManualLabelApp:
    def __init__(self, pdf_path: str, page_index: int, out_path: str) -> None:
        self.pdf_path = pdf_path
        self.page_index = page_index
        self.out_path = out_path

        self.reader = Reader(pdf_path)
        ctx = run_page_context(self.reader, page_index, final_stage="text_candidates")
        self.ctx = ctx
        self.clusters = ctx.text_clusters or []
        self._cluster_bboxes = {
            id(cluster): union_bbox([p.bbox for p in cluster])
            for cluster in self.clusters if cluster
        }

        self.labels = (
            load_labels(out_path) if Path(out_path).exists()
            else LabelSet(pdf_path=pdf_path)
        )
        self._labelled_signatures = {e.cluster_signature for e in self.labels.entries}

        self.matrix = _get_display_matrix(ctx.page.fitz_page, _ZOOM)

        self.root = tk.Tk()
        self.root.title(f"Manual Label -- {Path(pdf_path).name} page {page_index}")
        self.canvas = tk.Canvas(self.root, bg="#808080")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        save_btn = tk.Button(self.root, text="Save", command=self.save)
        save_btn.pack(side=tk.BOTTOM, fill=tk.X)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.canvas.bind("<Button-1>", self._on_click)

        self._render()

    def _render(self) -> None:
        self.canvas.delete("all")
        pix = self.ctx.page.fitz_page.get_pixmap(matrix=self.matrix)
        self._photo = tk.PhotoImage(data=pix.tobytes("ppm"))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.config(scrollregion=(0, 0, pix.width, pix.height))

        for cluster in self.clusters:
            if not cluster:
                continue
            bbox = self._cluster_bboxes[id(cluster)]
            rect = fitz.Rect(bbox) * self.matrix
            sig = cluster_signature(cluster)
            color = _LABELLED_COLOR if sig in self._labelled_signatures else _UNLABELLED_COLOR
            self.canvas.create_rectangle(
                rect.x0, rect.y0, rect.x1, rect.y1, outline=color, width=2,
                tags=("cluster", sig),
            )

    def _on_click(self, event: "tk.Event") -> None:
        cx, cy = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        inv = ~self.matrix
        page_pt = fitz.Point(cx, cy) * inv

        hit = None
        for cluster in self.clusters:
            if not cluster:
                continue
            bbox = self._cluster_bboxes[id(cluster)]
            if bbox[0] <= page_pt.x <= bbox[2] and bbox[1] <= page_pt.y <= bbox[3]:
                hit = cluster
                break
        if hit is None:
            return

        bbox = self._cluster_bboxes[id(hit)]
        sig = cluster_signature(hit)
        text = simpledialog.askstring("Label cluster", "Ground-truth text:", parent=self.root)
        if text is None:
            return

        self.labels.entries = [e for e in self.labels.entries if e.cluster_signature != sig]
        self.labels.entries.append(
            LabelEntry(
                page_index=self.page_index, cluster_bbox=bbox,
                cluster_signature=sig, text=text, source="manual",
            )
        )
        self._labelled_signatures.add(sig)
        self._render()

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
