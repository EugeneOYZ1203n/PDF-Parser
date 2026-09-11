"""Tkinter viewer for 1 or 2 pipeline-report per-PDF folders.

Left pane = the source page. Right side = one **panel per folder**, side by
side; each panel has one checkbox per generated layer PDF, grouped by stage
(including the `benchmark` group: `auto_bbox`, `auto_text`, `manual_bbox`,
`manual_text`). A checked layer is rasterized with a real alpha channel
(`get_pixmap(alpha=True)`) and alpha-composited over the base, in panel
order -- so you can show e.g. the OCR output of pipeline A together with the
clusters of pipeline B.

- **source page** checkbox toggles the underlying page (off = layers
  composite over white).
- each stage has "(all)" / "(none)" buttons.

Zoom / pan / page navigation.

    .venv/Scripts/python.exe scripts/pipeline_report_viewer.py <run>/<pdf-stem> [<run2>/<pdf-stem>]
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


def resolve_doc_dir(doc_dir: Path) -> tuple[str, Path]:
    """`(label, manifest_path)` for one report doc-dir. Trailing `"` / `\\`
    / `/` (PowerShell quoting) are already stripped by the CLI."""
    manifest = doc_dir / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"{doc_dir} has no manifest.json")
    return doc_dir.name, manifest


def _page_image(doc: "fitz.Document", page_pos: int, zoom: float, *, alpha: bool = False) -> "Image.Image":
    page = doc[min(page_pos, doc.page_count - 1)]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=alpha)
    return Image.frombytes("RGBA" if alpha else "RGB", (pix.width, pix.height), pix.samples)


class _Panel:
    def __init__(self, idx: int, doc_dir: Path) -> None:
        self.idx = idx
        label, manifest_path = resolve_doc_dir(doc_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.label = f"{label}  [{manifest.get('variant', manifest.get('engine', '?'))}]"
        self.pages: list[int] = manifest["pages"]
        self.source_pdf: str = manifest["source_pdf"]
        self.layers: list[dict] = [
            e for e in manifest.get("layers", []) if (doc_dir / e["file"]).exists()
        ]
        self.layer_docs: dict[str, fitz.Document] = {
            e["file"]: fitz.open(str(doc_dir / e["file"])) for e in self.layers
        }


class ViewerApp:
    def __init__(self, doc_dirs: list[Path]) -> None:
        self.panels = [_Panel(i, d) for i, d in enumerate(doc_dirs)]
        self.source = fitz.open(self.panels[0].source_pdf)
        self.pages = self.panels[0].pages
        for p in self.panels[1:]:
            if p.pages != self.pages:
                print(f"warning: {p.label} pages {p.pages} != {self.pages}; using the first")

        self.page_pos = 0
        self.zoom = 1.5
        self._photo = None
        self._vars: dict[tuple[int, str], tk.BooleanVar] = {}
        self._source_var: "tk.BooleanVar | None" = None

        self.root = tk.Tk()
        self.root.title("pipeline report -- " + " | ".join(p.label for p in self.panels))
        self.root.geometry("1600x900")
        self._build()
        self.redraw()

    # ---- layout --------------------------------------------------------
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
        self._source_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(nav, text="source page", variable=self._source_var,
                        command=self.redraw).pack(side="left", padx=12)
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

        side = ttk.Frame(paned)
        paned.add(side, weight=2)
        for panel in self.panels:
            self._build_panel(side, panel)

    def _build_panel(self, parent: ttk.Frame, panel: _Panel) -> None:
        outer = ttk.Frame(parent, width=330)
        outer.pack(side="left", fill="both", expand=True, padx=2)
        side_canvas = tk.Canvas(outer, highlightthickness=0, width=310)
        sbar = ttk.Scrollbar(outer, orient="vertical", command=side_canvas.yview)
        right = ttk.Frame(side_canvas)
        right.bind("<Configure>",
                   lambda _e, c=side_canvas: c.configure(scrollregion=c.bbox("all")))
        side_canvas.create_window((0, 0), window=right, anchor="nw")
        side_canvas.configure(yscrollcommand=sbar.set)
        side_canvas.pack(side="left", fill="both", expand=True)
        sbar.pack(side="right", fill="y")

        ttk.Label(right, text=panel.label, font=("", 10, "bold"),
                  wraplength=300).pack(anchor="w", pady=(6, 2))
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=2)

        current_stage = None
        for entry in panel.layers:
            file, stage, label = entry["file"], entry["stage"], entry["layer"]
            hexc = entry.get("color", "#888888")
            if stage != current_stage:
                current_stage = stage
                hdr = ttk.Frame(right)
                hdr.pack(anchor="w", fill="x", pady=(8, 0))
                ttk.Label(hdr, text=stage, font=("", 9, "bold")).pack(side="left")
                ttk.Button(hdr, text="all", width=4,
                           command=lambda s=stage, p=panel: self._toggle_stage(p, s, True)
                           ).pack(side="left", padx=(6, 0))
                ttk.Button(hdr, text="none", width=5,
                           command=lambda s=stage, p=panel: self._toggle_stage(p, s, False)
                           ).pack(side="left")
            var = tk.BooleanVar(value=False)
            self._vars[(panel.idx, file)] = var
            row = ttk.Frame(right)
            row.pack(anchor="w", padx=12)
            sw = tk.Canvas(row, width=12, height=12, highlightthickness=0)
            sw.create_rectangle(0, 0, 12, 12, fill=hexc, outline=hexc)
            sw.pack(side="left")
            ttk.Checkbutton(row, text=label, variable=var,
                            command=self.redraw).pack(side="left", padx=4)

    # ---- interaction --------------------------------------------------
    def _toggle_stage(self, panel: _Panel, stage: str, value: bool) -> None:
        for entry in panel.layers:
            if entry["stage"] == stage:
                self._vars[(panel.idx, entry["file"])].set(value)
        self.redraw()

    def flip(self, delta: int) -> None:
        self.page_pos = max(0, min(len(self.pages) - 1, self.page_pos + delta))
        self.redraw()

    def rezoom(self, direction: int) -> None:
        self.zoom = max(_MIN_ZOOM, min(
            _MAX_ZOOM, self.zoom * (_ZOOM_STEP if direction > 0 else 1 / _ZOOM_STEP)))
        self.redraw()

    def redraw(self) -> None:
        src = _page_image(self.source, self.pages[self.page_pos], self.zoom).convert("RGBA")
        base = src if (self._source_var and self._source_var.get()) \
            else Image.new("RGBA", src.size, (255, 255, 255, 255))
        for panel in self.panels:
            for entry in panel.layers:
                if not self._vars[(panel.idx, entry["file"])].get():
                    continue
                layer = _page_image(panel.layer_docs[entry["file"]], self.page_pos,
                                    self.zoom, alpha=True)
                if layer.size != base.size:
                    layer = layer.resize(base.size)
                base = Image.alpha_composite(base, layer)
        self._photo = ImageTk.PhotoImage(base)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, base.width, base.height))
        self.nav_label.configure(
            text=f"page {self.pages[self.page_pos]}  "
                 f"({self.page_pos + 1}/{len(self.pages)})   zoom {self.zoom:.0%}")

    def run(self) -> None:
        self.root.mainloop()


def _scrub(arg: str) -> Path:
    return Path(str(arg).rstrip('"').rstrip("\\/") or ".")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("doc_dir", nargs="+", type=Path,
                   help="1 or 2 per-PDF folders from generate_pipeline_report.py.")
    args = p.parse_args(argv)
    if len(args.doc_dir) > 2:
        p.error("at most 2 doc-dirs")
    ViewerApp([_scrub(d) for d in args.doc_dir]).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
