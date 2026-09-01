from __future__ import annotations

from rastervec.models import Page, PageMeta
from rastervec.renderer import render_page_svg


def test_render_page_svg_returns_svg_string(synthetic_pdf_factory):
    doc = synthetic_pdf_factory([
        {"texts": [{"point": (20, 40), "text": "hello"}]},
    ])
    try:
        fitz_page = doc[0]
        page = Page(
            doc_path="<mem>",
            meta=PageMeta(
                index=0, number=1, mediabox=tuple(fitz_page.mediabox),
                rotation=fitz_page.rotation, width=fitz_page.rect.width,
                height=fitz_page.rect.height,
            ),
            fitz_page=fitz_page,
        )
        svg = render_page_svg(page)
    finally:
        doc.close()

    assert isinstance(svg, str)
    assert "<svg" in svg
