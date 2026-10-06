"""Phase 4's final-output PDF: the one place a pipeline run's final
`(texts, vectors)` becomes a reconstructed page.

Every caller that wants "what did the pipeline produce, as a page"
(`core/api.py::extract_svg`, the report's `reconstructed` layer, the
benchmark's per-page reconstructions) goes through here instead of
composing its own -- debug layers aside, nothing else builds output PDFs.
Vectors keep their real paint (`vector_spec(recolor=None)`), text its real
colour, each `Text` rotated to its own angle and scaled to fit its own box
(`commons/renderer/draw.py::text_spec` -- font size from the box height,
never sheared). Font family isn't preserved (always base14 Helvetica).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rastervec.commons.renderer.draw import render_specs_pdf, text_spec, vector_spec
from rastervec.commons.renderer.pdf import rasterize_pdf

if TYPE_CHECKING:
    from PIL import Image

    from rastervec.commons.models import PageMeta, Text, Vector


def output_specs(texts: "list[Text]", vectors: "list[Vector]") -> list:
    """Draw specs for a final output page: vectors (real paint) under text."""
    return [vector_spec(v) for v in vectors or []] + [text_spec(t) for t in texts or []]


def render_output_pdf(page_meta: "PageMeta", texts: "list[Text]", vectors: "list[Vector]") -> bytes:
    """One-page PDF of the final output, sized/rotated to `page_meta`. Text
    stays real (selectable/searchable) PDF text."""
    return render_specs_pdf(page_meta, output_specs(texts, vectors))


def render_output_page(
    page_meta: "PageMeta", texts: "list[Text]", vectors: "list[Vector]", *, zoom: float = 1.0,
) -> "Image.Image":
    """`render_output_pdf` rasterized at `zoom` -- pixel-comparable with the
    source page's own `get_pixmap` at the same zoom."""
    return rasterize_pdf(render_output_pdf(page_meta, texts, vectors), zoom=zoom)
