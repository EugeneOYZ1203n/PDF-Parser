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

from PIL import ImageTk

import pymupdf as fitz

from rastervec.Evaluation.inspector.layers import OverlayItem


class Tooltip:
    """Simple mouse-following tooltip for the canvas."""

    def __init__(self, parent: tk.Widget):
        self.parent = parent

        self.window: tk.Toplevel | None = None
        self.label: tk.Label | None = None

    def show(
        self,
        x: int,
        y: int,
        text: str,
    ) -> None:

        if self.window is None:
            self.window = tk.Toplevel(
                self.parent
            )

            self.window.overrideredirect(
                True
            )

            self.window.attributes(
                "-topmost",
                True,
            )

            self.label = tk.Label(
                self.window,
                text=text,
                justify="left",
                anchor="w",
                padx=8,
                pady=6,
                bg="#ffffe0",
                fg="#111111",
                relief="solid",
                borderwidth=1,
                font=("TkDefaultFont", 9),
            )

            self.label.pack()

        else:
            assert self.label is not None

            self.label.config(
                text=text
            )

        self.window.geometry(
            f"+{x + 15}+{y + 15}"
        )

        self.window.deiconify()

    def hide(self) -> None:
        if self.window is not None:
            self.window.withdraw()


class PageView(ttk.Frame):

    def __init__(
        self,
        master,
        on_page_change=None,
        on_zoom_change=None,
        **kwargs,
    ):

        super().__init__(
            master,
            **kwargs,
        )

        self.on_page_change = on_page_change
        self.on_zoom_change = on_zoom_change

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
            tuple[str, list[OverlayItem]],
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
        color: str,
        matrix: "fitz.Matrix",
    ) -> None:

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

        text = self._format_metadata(
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


    def _format_metadata(
        self,
        item: OverlayItem,
        layer_key: str,
    ) -> str:

        metadata = item.get_metadata()

        lines = []

        title = layer_key.upper()

        if item.kind:
            title += f" — {item.kind}"

        lines.append(title)
        lines.append("")

        preferred_order = [


            "text",
            "image_index",
            "xref",
            "type",
            "operation",


            "x",
            "y",
            "width",
            "height",
            "drawing_width",
            "drawing_height",
            "display_width",
            "display_height",

            "area",
            "display_area",
            "drawing_area",

            "aspect_ratio",

            "bbox",
            "drawing_bbox",


            "angle",
            "rotation",
            "direction",

            "transform",
            "scale",


            "origin",
            "font",
            "font_size",
            "flags",
            "color",
            "ascender",
            "descender",


            "width_px",
            "height_px",
            "pixel_count",

            "colorspace",
            "image_colorspace",

            "bpc",
            "image_bpc",

            "xres",
            "yres",
            "effective_dpi",

            "has_mask",
            "smask",

            "extension",
            "compressed_size",
            "compressed_size_kb",

            "digest",


            "start",
            "end",
            "length",

            "path_type",
            "stroke_color",
            "fill_color",

            "stroke_width",
            "stroke_opacity",
            "fill_opacity",

            "line_cap",
            "line_join",

            "dashes",
            "close_path",
            "even_odd",

            "layer",

            "operation_count",
            "item_index",
            "seqno",

            "corners",
            "control_points",
            "control_polygon_length",


            "author",
            "contents",
            "subject",
            "opacity",
            "creationDate",
            "modDate",
        ]

        displayed = set()

        for key in preferred_order:

            if key not in metadata:
                continue

            value = metadata[key]

            if value is None:
                continue

            lines.append(
                f"{self._pretty_key(key)}: "
                f"{self._format_value(value)}"
            )

            displayed.add(key)

        for key, value in metadata.items():

            if key in displayed:
                continue

            if value is None:
                continue

            lines.append(
                f"{self._pretty_key(key)}: "
                f"{self._format_value(value)}"
            )

        return "\n".join(lines)

    @staticmethod
    def _pretty_key(
        key: str,
    ) -> str:

        return key.replace(
            "_",
            " ",
        ).title()

    @staticmethod
    def _format_value(value):

        if value is None:
            return ""

        if isinstance(value, float):
            return f"{value:.3f}"

        if isinstance(value, bool):
            return "Yes" if value else "No"

        if isinstance(value, tuple):

            return "(" + ", ".join(
                PageView._format_value(v)
                for v in value
            ) + ")"

        if isinstance(value, list):

            if not value:
                return "[]"

            return "[" + ", ".join(
                PageView._format_value(v)
                for v in value
            ) + "]"

        if isinstance(value, int):

            return str(value)

        return str(value)