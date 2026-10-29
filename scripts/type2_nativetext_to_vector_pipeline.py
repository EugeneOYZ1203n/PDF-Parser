"""Dump vectorized native text as lines in one output file.

Usage:
    uv run scripts/type2_nativetext_to_vector_pipeline.py PDF_PATH [--page N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf as pymupdf

from raster_parser.parsing.parser import PdfRasterParser
from raster_parser.evaluation.native_vs_ocr.text_to_vector import TextToVector, extract_text_from_page, text_bbox
from raster_parser.rendering.pdf_render.reconstruct import reconstruct


def run_native_text_to_vector(
    pdf_path: Path,
    page: int,
    *,
    out_pdf: Path,
    out_onlyfill: Path,
    out_svg: Path | None = None,
) -> tuple[Path, Path]:
    """Native text → SVG paths → replayed vector PDF.

    Returns ``(out_pdf, out_onlyfill)`` — text-only PDF and full vectorized text PDF.
    """
    out_pdf = Path(out_pdf)
    out_onlyfill = Path(out_onlyfill)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_onlyfill.parent.mkdir(parents=True, exist_ok=True)

    text_only = extract_text_from_page(str(pdf_path), page, str(out_pdf))

    svg_path = out_svg or out_pdf.with_suffix(".svg")
    with pymupdf.open(str(text_only)) as doc:
        svg_path.write_text(doc[0].get_svg_image(text_as_path=True), encoding="utf-8")

    text_to_vector = TextToVector(str(svg_path))
    vectors = text_to_vector.get_vectors()

    with pymupdf.open(str(text_only)) as text_doc:
        reconstruct(
            vectors,
            text_doc[0],
            out_path=out_onlyfill,
            start=0,
            interval=len(vectors),
            svg_coordinates=True,
        )

    return out_pdf.resolve(), out_onlyfill.resolve()


def dump(
    pdf_path: Path,
    page: int,
    out_svg: Path,
    out_pdf: Path,
    out_txt: Path,
    out_reconstruct: Path,
) -> None:
    """Write SVG/PDF/text dumps for native text vectorized on one page."""
    out_pdf, out_onlyfill = run_native_text_to_vector(
        pdf_path,
        page,
        out_pdf=out_pdf,
        out_onlyfill=out_reconstruct,
        out_svg=out_svg,
    )
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_svg.resolve()}")

    source_parser = PdfRasterParser(str(pdf_path), page)
    source_parser.extract_text()

    text_to_vector = TextToVector(str(out_svg), text_words=source_parser.text)
    words, word_vectors = text_to_vector.get_native_mapping()
    print(f"Words: {len(words)}, glyphs: {len(text_to_vector.get_vectors())}")

    mapping = text_to_vector.get_mapping()
    lines_text: list[str] = []
    for word, char_vectors in zip(words, word_vectors):
        chars = "".join(mapping[v.seqno][0] for v in char_vectors)
        lines_text.append(
            f"Word {word.word!r} bbox={text_bbox(word)} "
            f"({len(char_vectors)} glyphs, svg={chars!r})"
        )
        for vector in char_vectors:
            ch = mapping[vector.seqno][0]
            lines_text.append(f"  [{ch!r}] seqno={vector.seqno} bbox={vector.rect}")
    out_txt.write_text("\n".join(lines_text))
    print(f"Wrote {out_txt.resolve()}")
    print(f"Wrote {out_onlyfill}")


def main() -> None:
    """CLI entry for native-text → vector dump."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument(
        "--out_svg",
        type=Path,
        default=Path("out.svg"),
    )
    ap.add_argument(
        "--out_pdf",
        type=Path,
        default=Path("out.pdf"),
    )
    ap.add_argument(
        "--out_txt",
        type=Path,
        default=Path("out.txt"),
    )
    ap.add_argument(
        "--out_reconstruct",
        type=Path,
        default=Path("onlyfill.pdf"),
        help="PDF replaying parsed native-text vectors",
    )
    args = ap.parse_args()

    if not args.pdf.is_file():
        ap.error(f"PDF not found: {args.pdf}")

    dump(
        args.pdf,
        args.page,
        args.out_svg,
        args.out_pdf,
        args.out_txt,
        args.out_reconstruct,
    )


if __name__ == "__main__":
    main()
