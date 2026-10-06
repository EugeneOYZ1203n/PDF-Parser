"""Stable public export surface for other codebases embedding this
pipeline -- not the dev-facing report/benchmark tooling under Evaluation/.
Wraps `core.pipeline.run_pipeline` with a minimal, deliberately small
signature."""
from __future__ import annotations

from rastervec.commons.models import Text, Vector
from rastervec.core.pipeline import run_pipeline
from rastervec.core.registry import DEFAULT_P2, DEFAULT_P3


def extract(
    pdf_path: str, page_index: int = 0, *, p2: str = DEFAULT_P2, p3: str = DEFAULT_P3,
) -> tuple[list[Text], list[Vector]]:
    """Run the pipeline on one page, returning `(texts, vectors)`."""
    result = run_pipeline(pdf_path, page_index, p2=p2, p3=p3)
    return result.texts, result.vectors


def extract_svg(
    pdf_path: str, page_index: int = 0, *, p2: str = DEFAULT_P2, p3: str = DEFAULT_P3,
) -> bytes:
    """Run the pipeline on one page and re-render its final text+drawing
    content as one page of SVG, via Phase 4's own output renderer
    (`P4_Output_Organization.render_output_pdf` -- the one final-output
    page builder), so callers get one code path for "what did the pipeline
    produce", whether they want structured objects (`extract`) or a
    renderable page (`extract_svg`)."""
    import pymupdf as fitz

    from rastervec.P4_Output_Organization import render_output_pdf

    result = run_pipeline(pdf_path, page_index, p2=p2, p3=p3)
    pdf_bytes = render_output_pdf(result.page.meta, result.texts, result.vectors)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc[0].get_svg_image().encode("utf-8")
    finally:
        doc.close()
