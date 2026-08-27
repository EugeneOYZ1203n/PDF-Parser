"""Entry point for the PDF layer inspector.

Architecture
------------

PDF extraction:
    pdf_model.py
        |
        | objects in PDF page coordinates
        v
    OverlayItem
        |
        | page rotation + zoom
        v
    PageView
        |
        v
    Tk canvas

Rotation handling
-----------------

All extracted geometry remains in PDF page coordinates.

For display, the following transformation is applied:

    PDF page coordinates
        -> page.rotation_matrix
        -> zoom
        -> canvas coordinates

This is important for rotated text because OverlayItem.quad contains the
actual four corners of the text geometry. The quad is transformed during
rendering rather than being converted into an axis-aligned rectangle.
"""

from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pymupdf as fitz

# This file lives at rastervec/Evaluation/inspector/inspector.py -- three
# dirname() calls from its own directory reaches the repo root (inspector
# -> Evaluation -> rastervec -> repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))

if __name__ == "__main__" and __package__ is None:
    # Allow running this file directly (`python rastervec/Evaluation/
    # inspector/inspector.py`), not just as a module (`python -m
    # rastervec.Evaluation.inspector.inspector`), by putting the repo root
    # on sys.path.
    sys.path.insert(0, _REPO_ROOT)

from rastervec.Evaluation.inspector import pdf_model
from rastervec.Evaluation.inspector.control_panel import ControlPanel
from rastervec.Evaluation.inspector.layers import build_layers, filter_items
from rastervec.Evaluation.inspector.overlay_canvas import PageView


REFERENCES_DIR = os.path.join(_REPO_ROOT, "references")

MIN_ZOOM = 0.25
MAX_ZOOM = 6.0
ZOOM_STEP = 1.25


class AppState:
    """State associated with the currently displayed document."""

    def __init__(self) -> None:
        self.page_index: int = 0
        self.zoom: float = 1.5

        # page_index -> {layer_key: [OverlayItem, ...]}
        self.items_cache: dict[int, dict[str, list]] = {}

        # page_index -> (stroke_options, fill_options)
        self.color_cache: dict[
            int,
            tuple[dict, dict],
        ] = {}

        self.rotation_cache: dict[int, int] = {}


class InspectorApp:
    """Main PDF layer inspector."""

    def __init__(
        self,
        root: tk.Tk,
        pdf_path: str,
    ) -> None:
        self.root = root
        self.pdf_path = os.path.abspath(pdf_path)

        self.doc = pdf_model.PdfDocument(
            self.pdf_path
        )

        self.state = AppState()

        self.layers = build_layers(
            pdf_model
        )

        self.layers_by_key = {
            layer.key: layer
            for layer in self.layers
        }

        self.page_rotation: int = 0
        self.page_rect: fitz.Rect = fitz.Rect()

        self._configure_window()
        self._build_menu()
        self._build_layout()

        self.render_page()


    def _configure_window(self) -> None:
        """Configure the main application window."""

        filename = os.path.basename(
            self.pdf_path
        )

        self.root.title(
            f"PDF Layer Inspector — {filename}"
        )

        self.root.geometry(
            "1400x900"
        )

        self.root.minsize(
            900,
            600,
        )


    def _build_menu(self) -> None:
        """Build the application menu."""

        menubar = tk.Menu(
            self.root
        )

        file_menu = tk.Menu(
            menubar,
            tearoff=0,
        )

        file_menu.add_command(
            label="Open PDF...",
            command=self.open_pdf_dialog,
        )

        file_menu.add_separator()

        file_menu.add_command(
            label="Close",
            command=self._on_close,
        )

        menubar.add_cascade(
            label="File",
            menu=file_menu,
        )

        self.root.config(
            menu=menubar
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self._on_close,
        )

    def _build_layout(self) -> None:
        """Build the main two-pane layout."""

        paned = ttk.PanedWindow(
            self.root,
            orient="horizontal",
        )

        paned.pack(
            fill="both",
            expand=True,
        )


        self.page_view = PageView(
            paned,
            on_page_change=self.change_page,
            on_zoom_change=self.change_zoom,
        )

        paned.add(
            self.page_view,
            weight=3,
        )


        self.control_panel = ControlPanel(
            paned,
            self.layers,
            on_change=self.redraw,
        )

        paned.add(
            self.control_panel,
            weight=1,
        )


    def _get_page(self) -> fitz.Page:
        """Return the currently displayed page."""

        return self.doc.get_page(
            self.state.page_index
        )

    def _get_page_rotation(
        self,
        page: fitz.Page | None = None,
    ) -> int:
        """Return the page's PDF rotation in degrees."""

        if page is None:
            page = self._get_page()

        return int(
            page.rotation
        ) % 360


    def _get_display_matrix(
        self,
        page: fitz.Page | None = None,
    ) -> fitz.Matrix:
        """Return the page-space -> canvas-space transform.

        Extracted objects are always stored in PDF page coordinates.

        The transformation is:

            page coordinates
                |
                | page.rotation_matrix
                v
            rotated display coordinates
                |
                | zoom
                v
            canvas coordinates

        Importantly, this transform is applied to the actual geometry.

        Therefore a text OverlayItem with a rotated Quad will remain
        rotated rather than becoming an axis-aligned rectangle.
        """

        if page is None:
            page = self._get_page()

        rotation = page.rotation_matrix

        zoom = fitz.Matrix(
            self.state.zoom,
            self.state.zoom,
        )

        return rotation * zoom


    def _update_page_metadata(
        self,
        page: fitz.Page,
    ) -> None:
        """Update cached information about the current page."""

        self.page_rotation = self._get_page_rotation(
            page
        )

        self.page_rect = fitz.Rect(
            page.rect
        )

        self.state.rotation_cache[
            self.state.page_index
        ] = self.page_rotation


    def open_pdf_dialog(self) -> None:
        """Open a PDF using the file picker."""

        initial_dir = (
            REFERENCES_DIR
            if os.path.isdir(REFERENCES_DIR)
            else os.getcwd()
        )

        path = filedialog.askopenfilename(
            title="Open PDF",
            initialdir=initial_dir,
            filetypes=[
                ("PDF files", "*.pdf"),
            ],
        )

        if not path:
            return

        try:
            self._load_pdf(path)
        except Exception as exc:
            messagebox.showerror(
                "Unable to open PDF",
                str(exc),
            )

    def _load_pdf(
        self,
        path: str,
    ) -> None:
        """Replace the currently opened PDF."""

        path = os.path.abspath(
            os.path.expanduser(path)
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"PDF not found:\n{path}"
            )

        if not path.lower().endswith(
            ".pdf"
        ):
            raise ValueError(
                f"Expected a PDF file:\n{path}"
            )

        # Open the new document before touching any state, so a failed
        # open leaves the existing PDF intact.
        new_doc = pdf_model.PdfDocument(
            path
        )

        old_doc = self.doc

        self.doc = new_doc
        self.pdf_path = path
        self.state = AppState()

        old_doc.close()

        self.root.title(
            "PDF Layer Inspector — "
            + os.path.basename(path)
        )

        self.render_page()


    def change_page(
        self,
        delta: int,
    ) -> None:
        """Move between pages."""

        new_index = (
            self.state.page_index
            + delta
        )

        if not (
            0
            <= new_index
            < self.doc.page_count
        ):
            return

        self.state.page_index = new_index

        self.render_page()

    def change_zoom(
        self,
        direction: int,
    ) -> None:
        """Change page zoom."""

        if direction > 0:
            new_zoom = (
                self.state.zoom
                * ZOOM_STEP
            )

            self.state.zoom = min(
                MAX_ZOOM,
                new_zoom,
            )

        elif direction < 0:
            new_zoom = (
                self.state.zoom
                / ZOOM_STEP
            )

            self.state.zoom = max(
                MIN_ZOOM,
                new_zoom,
            )

        self.render_page()


    def _get_page_items(
        self,
    ) -> dict[str, list]:
        """Extract all configured layers for the current page.

        Extraction happens once per page and is cached.

        Extractors should return OverlayItems whose geometry is in
        unscaled PDF page coordinates.
        """

        page_index = (
            self.state.page_index
        )

        cached = self.state.items_cache.get(
            page_index
        )

        if cached is not None:
            return cached

        page = self._get_page()

        self._update_page_metadata(
            page
        )

        items: dict[str, list] = {}

        for layer in self.layers:
            items[layer.key] = (
                layer.extractor(page)
            )

        self.state.items_cache[
            page_index
        ] = items

        self.state.color_cache[
            page_index
        ] = (
            pdf_model.collect_drawing_colors(
                page
            )
        )

        return items


    def render_page(self) -> None:
        """Render the PDF page and all active overlays."""

        page = self._get_page()

        self._update_page_metadata(
            page
        )

        pixmap = self.doc.render_pixmap(
            self.state.page_index,
            self.state.zoom,
        )

        self.page_view.set_page_image(
            pixmap
        )

        self.page_view.set_nav_state(
            self.state.page_index,
            self.doc.page_count,
            self.state.zoom,
        )

        self._get_page_items()

        strokes, fills = (
            self.state.color_cache[
                self.state.page_index
            ]
        )


        self.control_panel.refresh_dynamic_options(
            "drawings",
            "stroke_color",
            strokes,
        )

        self.control_panel.refresh_dynamic_options(
            "drawings",
            "fill_color",
            fills,
        )


        self.redraw()


    def redraw(self) -> None:
        """Redraw all currently enabled layers."""

        items_cache = (
            self._get_page_items()
        )

        active_layers = (
            self.control_panel.get_active_layers()
        )

        transform = (
            self._get_display_matrix()
        )

        items_by_layer: dict[
            str,
            tuple[str, list],
        ] = {}

        for layer_key in active_layers:

            layer = self.layers_by_key[
                layer_key
            ]

            items = items_cache.get(
                layer_key,
                [],
            )

            if layer.subfilters:
                active_filters = (
                    self.control_panel.get_active_filters(
                        layer_key
                    )
                )

                items = filter_items(
                    items,
                    active_filters,
                )

            items_by_layer[
                layer_key
            ] = (
                layer.color,
                items,
            )

        self.page_view.draw_items(
            items_by_layer,
            transform,
        )


    def _on_close(self) -> None:
        """Close the application."""

        try:
            self.doc.close()
        finally:
            self.root.destroy()

    def close(self) -> None:
        """Close the PDF document."""

        self.doc.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect the layers and objects "
            "contained in a PDF."
        )
    )

    parser.add_argument(
        "pdf",
        nargs="?",
        help="Path to the PDF to inspect.",
    )

    return parser.parse_args()


def pick_initial_pdf() -> str | None:
    """Pick the first PDF in the references directory."""

    if not os.path.isdir(
        REFERENCES_DIR
    ):
        return None

    pdfs = sorted(
        filename
        for filename in os.listdir(
            REFERENCES_DIR
        )
        if filename.lower().endswith(
            ".pdf"
        )
    )

    if not pdfs:
        return None

    return os.path.join(
        REFERENCES_DIR,
        pdfs[0],
    )


def pick_pdf_with_dialog() -> str | None:
    """Open a PDF selection dialog."""

    root = tk.Tk()
    root.withdraw()

    try:
        return filedialog.askopenfilename(
            title="Open PDF",
            filetypes=[
                ("PDF files", "*.pdf"),
            ],
        )
    finally:
        root.destroy()


def resolve_pdf_path(
    requested_path: str | None,
) -> str | None:
    """Resolve the PDF path.

    Priority:

        1. CLI argument
        2. First PDF in references/
        3. File picker
    """

    if requested_path:

        path = os.path.abspath(
            os.path.expanduser(
                requested_path
            )
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"PDF not found:\n{path}"
            )

        if not path.lower().endswith(
            ".pdf"
        ):
            raise ValueError(
                f"Expected a PDF file:\n{path}"
            )

        return path

    path = pick_initial_pdf()

    if path:
        return path

    return pick_pdf_with_dialog()


def main() -> None:
    args = parse_args()

    try:
        pdf_path = resolve_pdf_path(
            args.pdf
        )

    except Exception as exc:
        print(
            f"Error: {exc}"
        )

        raise SystemExit(1)

    if not pdf_path:
        return

    root = tk.Tk()

    app: InspectorApp | None = None

    try:
        app = InspectorApp(
            root,
            pdf_path,
        )

        root.mainloop()

    finally:

        if app is not None:
            app.close()

        try:
            root.destroy()

        except tk.TclError:
            pass


if __name__ == "__main__":
    main()