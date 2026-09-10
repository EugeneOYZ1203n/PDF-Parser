"""Tkinter viewer for one pipeline-report per-PDF folder.

Shows the source page on the left; the right sidebar has one checkbox per
generated stage PDF (with its colour legend). A checked stage's page is
rendered and alpha-composited over the source page (near-white pixels made
transparent). Zoom / pan / page navigation.

Replaces the viewing half of
`rastervec/notebooks/pipeline_stage_visualization.ipynb`.

    .venv/Scripts/python.exe scripts/pipeline_report_viewer.py <run>/<pdf-stem>
"""
from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_MIN_ZOOM, _MAX_ZOOM, _ZOOM_STEP = 0.25, 6.0, 1.25


def _page_image(doc: "fitz.Document", page_pos: int, zoom: float) -> "Image.Image":
    page = doc[min(page_pos, doc.page_count - 1)]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _as_overlay(img: "Image.Image") -> "Image.Image":
    """Make near-white pixels transparent so only drawn ink composites."""
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).copy()
    white = (arr[:, :, 0] > 245) & (arr[:, :, 1] > 245) & (arr[:, :, 2] > 245)
    arr[:, :, 3] = np.where(white, 0, 255)
    return Image.fromarray(arr)


class ViewerApp:
    def __init__(self, doc_dir: Path) -> None:
        manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
        self.pages: list[int] = manifest["pages"]
        self.legend: dict = manifest.get("color_legend", {})
        self.source = fitz.open(manifest["source_pdf"])
        self.stage_docs: dict[str, fitz.Document] = {}
        for name in manifest["stages"]:
            fp = doc_dir / name
            if fp.exists():
                self.stage_docs[name] = fitz.open(str(fp))

        self.page_pos = 0
        self.zoom = 1.5
        self._photo = None
        self._vars: dict[str, tk.BooleanVar] = {}

        self.root = tk.Tk()
        self.root.title(f"pipeline report -- {doc_dir.name}")
        self.root.geometry("1400x900")
        self._build()
        self.redraw()

    # ---- layout ---------------------------------------------------------
    def _build(self) -> None:
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned)
        paned.add(left, weight=4)
        nav = ttk.Frame(left)
        nav.pack(fill="x")
        ttk.Button(nav, text="< Prev", command=lambda: self.flip(-1)).pack(side="left")
        ttk.Button(nav, text="Next >", command=lambda: self.flip(1)).pack(side="left")
        ttk.Button(nav, text="Zoom -", command=lambda: self.rezoom(-1)).pack(side="left", padx=(12, 0))
        ttk.Button(nav, text="Zoom +", command=lambda: self.rezoom(1)).pack(side="left")
        self.nav_label = ttk.Label(nav, text="")
        self.nav_label.pack(side="left", padx=12)

        cframe = ttk.Frame(left)
        cframe.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(cframe, background="#606060")
        vs = ttk.Scrollbar(cframe, orient="vertical", command=self.canvas.yview)
        hs = ttk.Scrollbar(cframe, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        cframe.rowconfigure(0, weight=1)
        cframe.columnconfigure(0, weight=1)

        right = ttk.Frame(paned, width=320)
        paned.add(right, weight=1)
        ttk.Label(right, text="Layers", font=("", 11, "bold")).pack(anchor="w", pady=(6, 2))
        for name in self.stage_docs:
            var = tk.BooleanVar(value=False)
            self._vars[name] = var
            ttk.Checkbutton(right, text=name, variable=var, command=self.redraw).pack(anchor="w")
            for label, hexc in self.legend.get(name, []):
                row = ttk.Frame(right)
                row.pack(anchor="w", padx=20)
                sw = tk.Canvas(row, width=12, height=12, highlightthickness=0)
                sw.create_rectangle(0, 0, 12, 12, fill=hexc, outline=hexc)
                sw.pack(side="left")
                ttk.Label(row, text=label).pack(side="left", padx=4)

    # ---- interaction ---------------------------------------------------
    def flip(self, delta: int) -> None:
        self.page_pos = max(0, min(len(self.pages) - 1, self.page_pos + delta))
        self.redraw()

    def rezoom(self, direction: int) -> None:
        self.zoom = max(_MIN_ZOOM, min(
            _MAX_ZOOM, self.zoom * (_ZOOM_STEP if direction > 0 else 1 / _ZOOM_STEP)
        ))
        self.redraw()

    def redraw(self) -> None:
        base = _page_image(self.source, self.pages[self.page_pos], self.zoom).convert("RGBA")
        for name, var in self._vars.items():
            if not var.get():
                continue
            layer = _page_image(self.stage_docs[name], self.page_pos, self.zoom)
            if layer.size != base.size:
                layer = layer.resize(base.size)
            base = Image.alpha_composite(base, _as_overlay(layer))
        self._photo = ImageTk.PhotoImage(base)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, base.width, base.height))
        self.nav_label.configure(
            text=f"page {self.pages[self.page_pos]}  "
                 f"({self.page_pos + 1}/{len(self.pages)})   zoom {self.zoom:.0%}"
        )

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("doc_dir", type=Path, help="A per-PDF folder from generate_pipeline_report.py.")
    args = p.parse_args(argv)
    ViewerApp(args.doc_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
