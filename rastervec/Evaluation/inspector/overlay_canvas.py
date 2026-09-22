"""Left pane: page navigation/zoom bar + scrollable canvas.

Displays the rendered PDF page with layer overlays.

Supports:
    - Axis-aligned rectangles
    - Rotated quads
    - Polygons
    - Lines
    - Hover metadata
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from PIL import ImageTk

import pymupdf as fitz

from rastervec.Evaluation.inspector.layers import OverlayItem
from rastervec.Evaluation.inspector.overlay_metadata_format import format_metadata
from rastervec.Evaluation.inspector.overlay_tooltip import Tooltip  # noqa: F401 -- re-exported for callers

ItemColor = str | Callable[[OverlayItem], str]

SELECTION_COLOR = "#facc15"

# Canvas-pixel movement below which a press/release is treated as a plain
# click (clearing the selection) rather than a drag.
SELECT_DRAG_THRESHOLD_PX = 4


class PageView(ttk.Frame):

    def __init__(
        self,
        master,
        on_page_change=None,
        on_zoom_change=None,
        on_selection_change=None,
        **kwargs,
    ):

        super().__init__(
            master,
            **kwargs,
        )

        self.on_page_change = on_page_change
        self.on_zoom_change = on_zoom_change
        self.on_selection_change = on_selection_change

        # Kept alive so Tk doesn't garbage-collect the displayed image.
        self._photo = None

        self._overlay_ids: list[int] = []

        self._overlay_items: dict[
            int,
            OverlayItem,
        ] = {}

        self._overlay_layers: dict[
            int,
            str,
        ] = {}

        self._hovered_overlay_id: int | None = None

        self._tooltip = Tooltip(
            self
        )

        self._page_matrix = fitz.Matrix(
            1,
            1,
        )

        # Rubber-band region selection (page-space; canvas-space while
        # actively dragging).
        self._select_start: tuple[float, float] | None = None
        self._select_rect_id: int | None = None
        self._selection_overlay_id: int | None = None
        self._selection_rect: fitz.Rect | None = None

        self._build_nav_bar()


        canvas_frame = ttk.Frame(
            self
        )

        canvas_frame.pack(
            fill="both",
            expand=True,
        )

        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#808080",
        )

        vbar = ttk.Scrollbar(
            canvas_frame,
            orient="vertical",
            command=self.canvas.yview,
        )

        hbar = ttk.Scrollbar(
            canvas_frame,
            orient="horizontal",
            command=self.canvas.xview,
        )

        self.canvas.configure(
            yscrollcommand=vbar.set,
            xscrollcommand=hbar.set,
        )

        self.canvas.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        vbar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        hbar.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        canvas_frame.rowconfigure(
            0,
            weight=1,
        )

        canvas_frame.columnconfigure(
            0,
            weight=1,
        )


        self.canvas.bind(
            "<Motion>",
            self._on_mouse_move,
        )

        self.canvas.bind(
            "<Leave>",
            self._on_mouse_leave,
        )

        self.canvas.bind(
            "<Button-1>",
            self._on_select_press,
        )

        self.canvas.bind(
            "<B1-Motion>",
            self._on_select_drag,
        )

        self.canvas.bind(
            "<ButtonRelease-1>",
            self._on_select_release,
        )


    def _build_nav_bar(self) -> None:

        bar = ttk.Frame(
            self
        )

        bar.pack(
            fill="x",
            side="top",
        )

        self.prev_btn = ttk.Button(
            bar,
            text="< Prev",
            command=lambda: (
                self.on_page_change(-1)
                if self.on_page_change
                else None
            ),
        )

        self.prev_btn.pack(
            side="left",
            padx=2,
            pady=2,
        )

        self.page_label = ttk.Label(
            bar,
            text="Page - / -",
        )

        self.page_label.pack(
            side="left",
            padx=6,
        )

        self.next_btn = ttk.Button(
            bar,
            text="Next >",
            command=lambda: (
                self.on_page_change(1)
                if self.on_page_change
                else None
            ),
        )

        self.next_btn.pack(
            side="left",
            padx=2,
            pady=2,
        )

        ttk.Button(
            bar,
            text="Zoom -",
            command=lambda: (
                self.on_zoom_change(-1)
                if self.on_zoom_change
                else None
            ),
        ).pack(
            side="left",
            padx=(20, 2),
        )

        self.zoom_label = ttk.Label(
            bar,
            text="100%",
        )

        self.zoom_label.pack(
            side="left",
            padx=6,
        )

        ttk.Button(
            bar,
            text="Zoom +",
            command=lambda: (
                self.on_zoom_change(1)
                if self.on_zoom_change
                else None
            ),
        ).pack(
            side="left",
            padx=2,
        )

    def set_nav_state(
        self,
        page_index: int,
        page_count: int,
        zoom: float,
    ) -> None:

        self.page_label.config(
            text=f"Page {page_index + 1} / {page_count}"
        )

        self.zoom_label.config(
            text=f"{round(zoom * 100)}%"
        )


    def set_page_image(
        self,
        pixmap: "fitz.Pixmap",
    ) -> None:
        """Set the rendered PDF page."""

        pil_image = pixmap.pil_image()

        self._photo = ImageTk.PhotoImage(
            pil_image
        )

        self.canvas.delete(
            "page_image"
        )

        self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self._photo,
            tags=("page_image",),
        )

        self.canvas.config(
            scrollregion=(
                0,
                0,
                pixmap.width,
                pixmap.height,
            )
        )

        self._hide_tooltip()


    def clear_overlays(self) -> None:

        if self._overlay_ids:

            self.canvas.delete(
                *self._overlay_ids
            )

        self._overlay_ids.clear()

        self._overlay_items.clear()

        self._overlay_layers.clear()

        self._hovered_overlay_id = None

        self._hide_tooltip()

    def draw_items(
        self,
        items_by_layer: dict[
            str,
            tuple[ItemColor, list[OverlayItem]],
        ],
        matrix: "fitz.Matrix",
    ) -> None:
        """Draw all active layer items.

        `matrix` must be the same page->canvas transformation used to
        render the page.

        This matrix should include:

            page rotation
            zoom
        """

        self.clear_overlays()

        self._page_matrix = matrix

        for layer_key, (
            color,
            items,
        ) in items_by_layer.items():

            for item in items:

                self._draw_item(
                    layer_key,
                    item,
                    color,
                    matrix,
                )

        self._redraw_selection_overlay(
            matrix
        )


    def _register_overlay(
        self,
        canvas_id: int,
        layer_key: str,
        item: OverlayItem,
    ) -> None:

        self._overlay_ids.append(
            canvas_id
        )

        self._overlay_items[
            canvas_id
        ] = item

        self._overlay_layers[
            canvas_id
        ] = layer_key

    def _draw_item(
        self,
        layer_key: str,
        item: OverlayItem,
        color: ItemColor,
        matrix: "fitz.Matrix",
    ) -> None:

        if callable(color):
            color = color(item)

        oid: int | None = None


        if item.shape == "quad":

            if item.quad is None:
                return

            quad = item.quad * matrix

            coords = [
                quad.ul.x,
                quad.ul.y,

                quad.ur.x,
                quad.ur.y,

                quad.lr.x,
                quad.lr.y,

                quad.ll.x,
                quad.ll.y,
            ]

            oid = self.canvas.create_polygon(
                *coords,
                outline=color,
                fill="",
                width=2,
                tags=("overlay",),
            )


        elif item.shape == "line":

            if (
                not item.points
                or len(item.points) < 2
            ):
                return

            p1 = (
                item.points[0]
                * matrix
            )

            p2 = (
                item.points[1]
                * matrix
            )

            oid = self.canvas.create_line(
                p1.x,
                p1.y,
                p2.x,
                p2.y,
                fill=color,
                width=1,
                tags=("overlay",),
            )


        elif item.shape == "polygon":

            if not item.points:
                return

            coords = []

            for point in item.points:

                transformed = (
                    point * matrix
                )

                coords.extend(
                    [
                        transformed.x,
                        transformed.y,
                    ]
                )

            oid = self.canvas.create_polygon(
                *coords,
                outline=color,
                fill="",
                width=1,
                tags=("overlay",),
            )


        else:

            rect = (
                item.bbox
                * matrix
            )

            oid = self.canvas.create_rectangle(
                rect.x0,
                rect.y0,
                rect.x1,
                rect.y1,
                outline=color,
                width=1,
                tags=("overlay",),
            )

        if oid is not None:

            self._register_overlay(
                oid,
                layer_key,
                item,
            )


    def _on_mouse_move(
        self,
        event: tk.Event,
    ) -> None:

        canvas_x = self.canvas.canvasx(
            event.x
        )

        canvas_y = self.canvas.canvasy(
            event.y
        )

        overlapping = self.canvas.find_overlapping(
            canvas_x,
            canvas_y,
            canvas_x,
            canvas_y,
        )

        overlay_id = self._find_overlay(
            overlapping
        )

        if overlay_id is None:

            self._hovered_overlay_id = None

            self._hide_tooltip()

            return

        self._hovered_overlay_id = (
            overlay_id
        )

        item = self._overlay_items.get(
            overlay_id
        )

        if item is None:

            self._hide_tooltip()

            return

        layer_key = (
            self._overlay_layers.get(
                overlay_id,
                "",
            )
        )

        text = format_metadata(
            item,
            layer_key,
        )

        screen_x = (
            self.canvas.winfo_rootx()
            + event.x
        )

        screen_y = (
            self.canvas.winfo_rooty()
            + event.y
        )

        self._tooltip.show(
            screen_x,
            screen_y,
            text,
        )

    def _find_overlay(
        self,
        canvas_ids: tuple[int, ...],
    ) -> int | None:

        for canvas_id in reversed(
            canvas_ids
        ):

            if canvas_id in self._overlay_items:

                return canvas_id

        return None

    def _on_mouse_leave(
        self,
        _event: tk.Event,
    ) -> None:

        self._hovered_overlay_id = None

        self._hide_tooltip()

    def _hide_tooltip(self) -> None:

        self._tooltip.hide()


    # ---- region selection ----------------------------------------------

    def get_selection(self) -> fitz.Rect | None:
        """Return the current selection rectangle in page-space, if any."""

        return self._selection_rect

    def clear_selection(self) -> None:
        """Clear the current selection, if any, and notify the callback."""

        if self._selection_overlay_id is not None:
            self.canvas.delete(
                self._selection_overlay_id
            )
            self._selection_overlay_id = None

        if self._selection_rect is None:
            return

        self._selection_rect = None

        if self.on_selection_change:
            self.on_selection_change(None)

    def _redraw_selection_overlay(
        self,
        matrix: "fitz.Matrix",
    ) -> None:
        """Redraw the persistent selection rectangle for the given matrix.

        Called from `draw_items` so the selection stays correctly placed
        across zoom/layer redraws, and from `_on_select_release` for
        immediate feedback right after a drag.
        """

        if self._selection_overlay_id is not None:
            self.canvas.delete(
                self._selection_overlay_id
            )
            self._selection_overlay_id = None

        if self._selection_rect is None:
            return

        rect = self._selection_rect * matrix

        self._selection_overlay_id = self.canvas.create_rectangle(
            rect.x0,
            rect.y0,
            rect.x1,
            rect.y1,
            outline=SELECTION_COLOR,
            width=2,
            dash=(6, 3),
            tags=("selection",),
        )

    def _on_select_press(
        self,
        event: tk.Event,
    ) -> None:

        self._select_start = (
            self.canvas.canvasx(event.x),
            self.canvas.canvasy(event.y),
        )

    def _on_select_drag(
        self,
        event: tk.Event,
    ) -> None:

        if self._select_start is None:
            return

        x0, y0 = self._select_start

        x1 = self.canvas.canvasx(event.x)
        y1 = self.canvas.canvasy(event.y)

        if self._select_rect_id is not None:
            self.canvas.delete(
                self._select_rect_id
            )

        self._select_rect_id = self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            outline=SELECTION_COLOR,
            width=1,
            dash=(3, 2),
            tags=("selrect",),
        )

    def _on_select_release(
        self,
        event: tk.Event,
    ) -> None:

        start = self._select_start
        self._select_start = None

        if self._select_rect_id is not None:
            self.canvas.delete(
                self._select_rect_id
            )
            self._select_rect_id = None

        if start is None:
            return

        x0, y0 = start

        x1 = self.canvas.canvasx(event.x)
        y1 = self.canvas.canvasy(event.y)

        if (
            abs(x1 - x0) < SELECT_DRAG_THRESHOLD_PX
            and abs(y1 - y0) < SELECT_DRAG_THRESHOLD_PX
        ):
            self.clear_selection()
            return

        inverse = ~self._page_matrix

        p0 = fitz.Point(x0, y0) * inverse
        p1 = fitz.Point(x1, y1) * inverse

        self._selection_rect = fitz.Rect(
            min(p0.x, p1.x),
            min(p0.y, p1.y),
            max(p0.x, p1.x),
            max(p0.y, p1.y),
        )

        self._redraw_selection_overlay(
            self._page_matrix
        )

        if self.on_selection_change:
            self.on_selection_change(
                self._selection_rect
            )

