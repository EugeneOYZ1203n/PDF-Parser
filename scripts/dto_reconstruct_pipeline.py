"""CLI: convert serialized TextDTO / VectorDTO lists back to native objects.

Usage:
    uv run scripts/dto_reconstruct_pipeline.py drawing.pdf [--page N]
    uv run scripts/dto_reconstruct_pipeline.py INPUT.json

Writes ``--out-json`` (DTO), ``--out-words``, ``--out-vectors``, and ``--out-pdf``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf as fitz

from raster_parser.rendering.pdf_render.dto_reconstruct_pipeline import (
    draw_vectors,
    output_as_dto_json,
    reconstruct_from_output,
)
from raster_parser.models import NativePDFElements


def main() -> None:
    """CLI: reconstruct PDF drawings from TextDTO/VectorDTO JSON or a PDF."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="NativePDFElements JSON or PDF to extract first")
    ap.add_argument("--page", type=int, default=0, help="PDF page index (PDF input only)")
    ap.add_argument("--out-words", type=Path, default=Path("out_words.txt"))
    ap.add_argument("--out-vectors", type=Path, default=Path("out_vectors.txt"))
    ap.add_argument("--out-pdf", type=Path, default=Path("vectors.pdf"))
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("output.json"),
        help="Write NativePDFElements DTO JSON (default: output.json)",
    )
    ap.add_argument("--ocr-dpi", type=int, default=300, metavar="N")
    args = ap.parse_args()

    if not args.input.is_file():
        ap.error(f"input not found: {args.input}")

    page_width: float | None = None
    page_height: float | None = None
    page_rotation = 0

    if args.input.suffix.lower() == ".json":
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        output = NativePDFElements.model_validate(payload)
    else:
        from raster_parser.main_pipeline_extract import extract

        with fitz.open(args.input) as doc:
            src_page = doc[args.page]
            page_width = float(src_page.mediabox.width)
            page_height = float(src_page.mediabox.height)
            page_rotation = int(src_page.rotation or 0)

        output = extract(
            args.input,
            page_index=args.page,
            ocr_dpi=args.ocr_dpi,
        )

    out_json = output_as_dto_json(output, args.out_json)

    result = reconstruct_from_output(
        output,
        out_txt_words=args.out_words,
        out_txt_vectors=args.out_vectors,
    )

    out_pdf = draw_vectors(
        result.vectors,
        args.out_pdf,
        page_width=page_width,
        page_height=page_height,
        page_rotation=page_rotation,
    )

    print(f"Wrote {out_json}")
    print(f"Wrote {result.out_txt_words}")
    print(f"Wrote {result.out_txt_vectors}")
    print(f"Wrote {out_pdf}")
    print(f"Words: {len(result.words)}, vectors: {len(result.vectors)}")


if __name__ == "__main__":
    main()
