from __future__ import annotations

from pathlib import Path

import pymupdf as pymupdf

def svg_to_pdf(svg_path: Path, pdf_path: Path) -> None:
    """Convert an SVG file to PDF using PyMuPDF."""
    doc = pymupdf.open(str(svg_path))
    try:
        pdf_bytes = doc.convert_to_pdf()
    finally:
        doc.close()
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)