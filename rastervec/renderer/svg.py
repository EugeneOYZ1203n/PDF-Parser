"""SVG rendering.

Thin wrapper over PyMuPDF's own `get_svg_image()` so the SVG concern has a
home in the renderer package. For a page with native text, PyMuPDF emits
each glyph as a filled SVG `<path>` (never an SVG `<text>` element or an
embedded raster) -- the "text-as-filled-vector-paths" shape this project's
Vector_Classification pipeline is built to reconstruct. See
`Evaluation/Conversion/conversion.py`, which round-trips exactly this
output back into a PDF for known-answer testing.
"""
from __future__ import annotations

from rastervec.models import Page


def render_page_svg(page: Page) -> str:
    """The page rendered to an SVG string via PyMuPDF's `get_svg_image()`."""
    return page.fitz_page.get_svg_image()
