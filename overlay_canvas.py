"""Left pane: page navigation/zoom bar + scrollable canvas that shows the
rendered page pixmap with layer overlays drawn on top."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import ImageTk

from layers import OverlayItem

import pymupdf as fitz


class PageView(ttk.Frame):
    def __init__(self, master, on_page_change=None, on_zoom_change=None, **kwargs):
        super().__init__(master, **kwargs)
        self.on_page_change = on_page_change
        self.on_zoom_change = on_zoom_change

        self._photo = None  # keep a reference so Tk doesn't garbage-collect it
        self._overlay_ids: list[int] = []

        self._build_nav_bar()

        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#808080")
        vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

    def _build_nav_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="top")

        self.prev_btn = ttk.Button(bar, text="< Prev", command=lambda: self.on_page_change and self.on_page_change(-1))
        self.prev_btn.pack(side="left", padx=2, pady=2)

        self.page_label = ttk.Label(bar, text="Page - / -")
        self.page_label.pack(side="left", padx=6)

        self.next_btn = ttk.Button(bar, text="Next >", command=lambda: self.on_page_change and self.on_page_change(1))
        self.next_btn.pack(side="left", padx=2, pady=2)

        ttk.Button(bar, text="Zoom -", command=lambda: self.on_zoom_change and self.on_zoom_change(-1)).pack(side="left", padx=(20, 2))
        self.zoom_label = ttk.Label(bar, text="100%")
        self.zoom_label.pack(side="left", padx=6)
        ttk.Button(bar, text="Zoom +", command=lambda: self.on_zoom_change and self.on_zoom_change(1)).pack(side="left", padx=2)

    def set_nav_state(self, page_index: int, page_count: int, zoom: float) -> None:
        self.page_label.config(text=f"Page {page_index + 1} / {page_count}")
        self.zoom_label.config(text=f"{round(zoom * 100)}%")

    def set_page_image(self, pixmap: "fitz.Pixmap") -> None:
        pil_image = pixmap.pil_image()
        self._photo = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("page_image")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo, tags="page_image")
        self.canvas.config(scrollregion=(0, 0, pixmap.width, pixmap.height))

    def clear_overlays(self) -> None:
        if self._overlay_ids:
            self.canvas.delete(*self._overlay_ids)
            self._overlay_ids = []

    def draw_items(self, items_by_layer: dict[str, tuple[str, list[OverlayItem]]], matrix: "fitz.Matrix") -> None:
        self.clear_overlays()
        for _layer_key, (color, items) in items_by_layer.items():
            for item in items:
                self._draw_item(item, color, matrix)

    def _draw_item(self, item: OverlayItem, color: str, matrix: "fitz.Matrix") -> None:
        if item.shape == "line":
            p1, p2 = item.points[0] * matrix, item.points[1] * matrix
            oid = self.canvas.create_line(p1.x, p1.y, p2.x, p2.y, fill=color, width=1)
        elif item.shape == "polygon":
            coords = []
            for p in item.points:
                p2 = p * matrix
                coords.extend([p2.x, p2.y])
            oid = self.canvas.create_polygon(*coords, outline=color, fill="", width=1)
        else:
            rect = item.bbox * matrix
            oid = self.canvas.create_rectangle(rect.x0, rect.y0, rect.x1, rect.y1, outline=color, width=1)
        self._overlay_ids.append(oid)
