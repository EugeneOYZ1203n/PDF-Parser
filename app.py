"""Entry point for the PDF layer inspector."""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk

import pymupdf as fitz

import pdf_model
from control_panel import ControlPanel
from layers import build_layers, filter_items
from overlay_canvas import PageView

REFERENCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references")
MIN_ZOOM = 0.25
MAX_ZOOM = 6.0
ZOOM_STEP = 1.25


class AppState:
    def __init__(self):
        self.page_index = 0
        self.zoom = 1.5
        # per-page caches keyed by page index
        self.items_cache: dict[int, dict[str, list]] = {}
        self.color_cache: dict[int, tuple[dict, dict]] = {}


class InspectorApp:
    def __init__(self, root: tk.Tk, pdf_path: str):
        self.root = root
        self.doc = pdf_model.PdfDocument(pdf_path)
        self.state = AppState()
        self.layers = build_layers(pdf_model)
        self.layers_by_key = {layer.key: layer for layer in self.layers}

        root.title(f"PDF Layer Inspector — {os.path.basename(pdf_path)}")
        root.geometry("1400x900")

        self._build_menu()

        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        self.page_view = PageView(paned, on_page_change=self.change_page, on_zoom_change=self.change_zoom)
        paned.add(self.page_view, weight=3)

        self.control_panel = ControlPanel(paned, self.layers, on_change=self.redraw)
        paned.add(self.control_panel, weight=1)

        self.render_page()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open PDF...", command=self.open_pdf_dialog)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def open_pdf_dialog(self) -> None:
        initial_dir = REFERENCES_DIR if os.path.isdir(REFERENCES_DIR) else os.getcwd()
        path = filedialog.askopenfilename(
            title="Open PDF",
            initialdir=initial_dir,
            filetypes=[("PDF files", "*.pdf")],
        )
        if not path:
            return
        self.doc.close()
        self.doc = pdf_model.PdfDocument(path)
        self.state = AppState()
        self.root.title(f"PDF Layer Inspector — {os.path.basename(path)}")
        self.render_page()

    def change_page(self, delta: int) -> None:
        new_index = self.state.page_index + delta
        if 0 <= new_index < self.doc.page_count:
            self.state.page_index = new_index
            self.render_page()

    def change_zoom(self, direction: int) -> None:
        if direction > 0:
            self.state.zoom = min(MAX_ZOOM, self.state.zoom * ZOOM_STEP)
        else:
            self.state.zoom = max(MIN_ZOOM, self.state.zoom / ZOOM_STEP)
        self.render_page()

    def _get_page_items(self) -> dict[str, list]:
        idx = self.state.page_index
        if idx not in self.state.items_cache:
            page = self.doc.get_page(idx)
            cache = {}
            for layer in self.layers:
                cache[layer.key] = layer.extractor(page)
            self.state.items_cache[idx] = cache
            self.state.color_cache[idx] = pdf_model.collect_drawing_colors(page)
        return self.state.items_cache[idx]

    def render_page(self) -> None:
        pixmap = self.doc.render_pixmap(self.state.page_index, self.state.zoom)
        self.page_view.set_page_image(pixmap)
        self.page_view.set_nav_state(self.state.page_index, self.doc.page_count, self.state.zoom)

        self._get_page_items()
        strokes, fills = self.state.color_cache[self.state.page_index]
        self.control_panel.refresh_dynamic_options("drawings", "stroke_color", strokes)
        self.control_panel.refresh_dynamic_options("drawings", "fill_color", fills)

        self.redraw()

    def redraw(self) -> None:
        items_cache = self._get_page_items()
        active_layers = self.control_panel.get_active_layers()
        matrix = fitz.Matrix(self.state.zoom, self.state.zoom)

        items_by_layer = {}
        for layer_key in active_layers:
            layer = self.layers_by_key[layer_key]
            items = items_cache.get(layer_key, [])
            if layer.subfilters:
                active_filters = self.control_panel.get_active_filters(layer_key)
                items = filter_items(items, active_filters)
            items_by_layer[layer_key] = (layer.color, items)

        self.page_view.draw_items(items_by_layer, matrix)


def pick_initial_pdf() -> str | None:
    if os.path.isdir(REFERENCES_DIR):
        pdfs = sorted(f for f in os.listdir(REFERENCES_DIR) if f.lower().endswith(".pdf"))
        if pdfs:
            return os.path.join(REFERENCES_DIR, pdfs[0])
    return None


def main() -> None:
    root = tk.Tk()
    pdf_path = pick_initial_pdf()
    if not pdf_path:
        pdf_path = filedialog.askopenfilename(title="Open PDF", filetypes=[("PDF files", "*.pdf")])
        if not pdf_path:
            root.destroy()
            return
    InspectorApp(root, pdf_path)
    root.mainloop()


if __name__ == "__main__":
    main()
