"""Tkinter viewer for one pipeline-report per-PDF folder.

Shows the source page on the left; the right sidebar has one checkbox per
generated layer PDF, grouped by stage. A checked layer is rasterized with a
real alpha channel (`get_pixmap(alpha=True)` -- a stage PDF page has no
background, so it is genuinely transparent where nothing was drawn and
correctly anti-aliased at every edge) and alpha-composited over the base.

Each visual element is its own single-purpose PDF, so toggling a layer just
loads / doesn't load that file -- no colour-keying, no anti-alias fringe.

- **source page** checkbox toggles the underlying page (off = layers
  composite over white).
- each stage has an "(all)" checkbox that toggles its whole group.

Zoom / pan / page navigation.

    .venv/Scripts/python.exe scripts/pipeline_report_viewer.py <run>/<pdf-stem>
"""
from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pymupdf as fitz
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_MIN_ZOOM, _MAX_ZOOM, _ZOOM_STEP = 0.25, 6.0, 1.25


def _page_image(
    doc: "fitz.Document", page_pos: int, zoom: float, *, alpha: bool = False
) -> "Image.Image":
    page = doc[min(page_pos, doc.page_count - 1)]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=alpha)
    return Image.frombytes("RGBA" if alpha else "RGB", (pix.width, pix.height), pix.samples)


class ViewerApp:
    def __init__(self, doc_dir: Path) -> None:
        manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
        self.pages: list[int] = manifest["pages"]
        self.source = fitz.open(manifest["source_pdf"])

        # manifest "layers": [{stage, layer, file, color}] in draw order.
        self.layers: list[dict] = [
            entry for entry in manifest.get("layers", [])
            if (doc_dir / entry["file"]).exists()
        ]
        self.layer_docs: dict[str, fitz.Document] = {
            entry["file"]: fitz.open(str(doc_dir / entry["file"])) for entry in self.layers
        }

        self.page_pos = 0
        self.zoom = 1.5
        self._photo = None
        self._vars: dict[str, tk.BooleanVar] = {}
        self._source_var: "tk.BooleanVar | None" = None

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

        # scrollable sidebar
        outer = ttk.Frame(paned, width=340)
        paned.add(outer, weight=1)
        side_canvas = tk.Canvas(outer, highlightthickness=0, width=320)
        sbar = ttk.Scrollbar(outer, orient="vertical", command=side_canvas.yview)
        right = ttk.Frame(side_canvas)
        right.bind(
            "<Configure>",
            lambda _e: side_canvas.configure(scrollregion=side_canvas.bbox("all")),
        )
        side_canvas.create_window((0, 0), window=right, anchor="nw")
        side_canvas.configure(yscrollcommand=sbar.set)
        side_canvas.pack(side="left", fill="both", expand=True)
        sbar.pack(side="right", fill="y")

        ttk.Label(right, text="Layers", font=("", 11, "bold")).pack(anchor="w", pady=(6, 2))
        self._source_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            right, text="source page", variable=self._source_var, command=self.redraw
        ).pack(anchor="w")
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=4)

        current_stage = None
        for entry in self.layers:
            file, stage, label, hexc = (
                entry["file"], entry["stage"], entry["layer"], entry.get("color", "#888888")
            )
            if stage != current_stage:
                current_stage = stage
                hdr = ttk.Frame(right)
                hdr.pack(anchor="w", fill="x", pady=(8, 0))
                ttk.Label(hdr, text=stage, font=("", 9, "bold")).pack(side="left")
                ttk.Button(
                    hdr, text="all", width=4,
                    command=lambda s=stage: self._toggle_stage(s, True),
                ).pack(side="left", padx=(6, 0))
                ttk.Button(
                    hdr, text="none", width=5,
                    command=lambda s=stage: self._toggle_stage(s, False),
                ).pack(side="left")

            var = tk.BooleanVar(value=False)
            self._vars[file] = var
            row = ttk.Frame(right)
            row.pack(anchor="w", padx=14)
            sw = tk.Canvas(row, width=12, height=12, highlightthickness=0)
            sw.create_rectangle(0, 0, 12, 12, fill=hexc, outline=hexc)
            sw.pack(side="left")
            ttk.Checkbutton(
                row, text=label, variable=var, command=self.redraw
            ).pack(side="left", padx=4)

    # ---- interaction ---------------------------------------------------
    def _toggle_stage(self, stage: str, value: bool) -> None:
        for entry in self.layers:
            if entry["stage"] == stage:
                self._vars[entry["file"]].set(value)
        self.redraw()

    def flip(self, delta: int) -> None:
        self.page_pos = max(0, min(len(self.pages) - 1, self.page_pos + delta))
        self.redraw()

    def rezoom(self, direction: int) -> None:
        self.zoom = max(_MIN_ZOOM, min(
            _MAX_ZOOM, self.zoom * (_ZOOM_STEP if direction > 0 else 1 / _ZOOM_STEP)
        ))
        self.redraw()

    def redraw(self) -> None:
        src = _page_image(self.source, self.pages[self.page_pos], self.zoom).convert("RGBA")
        if self._source_var is not None and self._source_var.get():
            base = src
        else:
            base = Image.new("RGBA", src.size, (255, 255, 255, 255))
        for entry in self.layers:
            if not self._vars[entry["file"]].get():
                continue
            layer = _page_image(
                self.layer_docs[entry["file"]], self.page_pos, self.zoom, alpha=True
            )
            if layer.size != base.size:
                layer = layer.resize(base.size)
            base = Image.alpha_composite(base, layer)
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
    # PowerShell mangles a quoted path with a trailing backslash: the `\"` is
    # read as an escaped quote, leaving a literal `"` on the end of the arg.
    doc_dir = Path(str(args.doc_dir).rstrip('"').rstrip("\\/") or ".")
    ViewerApp(doc_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
