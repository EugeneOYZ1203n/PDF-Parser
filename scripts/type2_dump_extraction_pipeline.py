"""Dump native text, vectors, and assets (Type-2 fill OCR path).

Usage:
    uv run scripts/type2_dump_extraction_pipeline.py PDF_PATH [--page N] [--no-ocr]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from raster_parser.models import Text
from raster_parser.parsing.parser import PdfRasterParser
from raster_parser.rendering.pdf_render.reconstruct import (
    ocr_rotate_for_target,
    reconstruct,
    render_text_objects_to_pdf,
    filter_text_vectors,
    vector_per_page,
)
from raster_parser.ocr.wordgrouping import (
    Orientation,
    cluster_by_seqno,
    convert_vectors_to_glyphs,
    get_vectors,
)

logger = logging.getLogger(__name__)


def run_ocr_extraction(
    pdf_path: Path,
    page: int,
    *,
    out_txt_ocr: Path,
    out_ocr_pdf: Path,
    ocr_dpi: int = 300,
) -> tuple[Path, Path, list[tuple[Text, list[int]]]]:
    """OCR vector glyphs on one page; write OCR text dump and ``ocr.pdf``.

    Returns ``(out_txt_ocr, out_ocr_pdf, ocred_text_with_mapping)``.
    """
    parser = PdfRasterParser(str(pdf_path), page=page)
    all_vectors = parser.extract_vectors()
    filtered_vectors = filter_text_vectors(all_vectors)

    glyphs = convert_vectors_to_glyphs(filtered_vectors)
    page_rotation = int(parser.page.rotation)
    word_groups = cluster_by_seqno(glyphs, page_rotation)
    group_orientations = [wg.orientation for wg in word_groups]

    ocrlines_text: list[str] = [
        f"=== TEXT (PaddleOCR per-word-group @ {ocr_dpi} dpi) ===",
    ]
    ocred_text_with_mapping: list = []

    try:
        group_vectors = [get_vectors(wg) for wg in word_groups]
        group_boxes = [(wg.bbox.x0, wg.bbox.y0, wg.bbox.x1, wg.bbox.y1) for wg in word_groups]
        raw_pairs = parser.extract_text_from_paddle_word_groups(
            group_vectors,
            group_boxes,
            dpi=ocr_dpi,
            use_angle_cls=True,
        )
        by_index = {t.block_no: (t, vector_indices) for t, vector_indices in raw_pairs}
        for i, _wg in enumerate(word_groups):
            hit = by_index.get(i)
            if hit is None:
                continue
            text, vector_indices = hit
            if i < len(group_orientations):
                ori = group_orientations[i]
                text.glyph_orientation = ori
                target = 90 if ori == Orientation.VERTICAL else 0
                text.rotate = ocr_rotate_for_target(target, page_rotation)
            ocred_text_with_mapping.append((text, vector_indices))
        ocred_text_with_mapping.sort(key=lambda pair: pair[0].block_no)
        for t, _ in ocred_text_with_mapping:
            ocrlines_text.append(str(t))
    except Exception as e:
        ocrlines_text.append(f"(OCR failed) {type(e).__name__}: {e}")
        logger.exception("OCR failed")

    out_txt_ocr = Path(out_txt_ocr)
    out_ocr_pdf = Path(out_ocr_pdf)
    out_txt_ocr.parent.mkdir(parents=True, exist_ok=True)
    out_txt_ocr.write_text("\n".join(ocrlines_text) + "\n", encoding="utf-8")

    if ocred_text_with_mapping:
        render_text_objects_to_pdf(
            parser.page,
            [t for t, _ in ocred_text_with_mapping],
            out_ocr_pdf.resolve(),
            dpi=ocr_dpi,
            glyph_boxes=True,
        )

    return out_txt_ocr.resolve(), out_ocr_pdf.resolve(), ocred_text_with_mapping


def dump(
    pdf_path: Path,
    page: int,
    out_txt_text: Path,
    out_txt_vectors: Path,
    out_txt_words: Path,
    out_txt_ocr: Path,
    *,
    ocr_dpi: int = 300,
) -> None:
    """Write native text / vector / word dumps and optional OCR outputs for one page."""
    parser = PdfRasterParser(str(pdf_path), page=page)
    parser.extract_text()
    all_vectors = parser.extract_vectors()

    text_vectors_total = 0
    word_groups_total = 0
    filtered_vectors = filter_text_vectors(all_vectors)
    reconstruct(filtered_vectors, parser.page, out_path="onlyfill.pdf")
    vector_per_page(filtered_vectors, parser.page, per_page=100)

    lines_text: list[str] = []
    lines_vectors: list[str] = []
    words: list[str] = []

    glyphs = convert_vectors_to_glyphs(filtered_vectors)
    word_groups = cluster_by_seqno(glyphs, int(parser.page.rotation or 0))
    for word_group in word_groups:
        words.append(f"Group {word_groups_total}: {word_group}")
        word_groups_total += 1

    for t in parser.text:
        lines_text.append(str(t))

    for v in filtered_vectors:
        lines_vectors.append(f"Vector {text_vectors_total}: {v}")
        text_vectors_total += 1

    out_txt_ocr_path = Path(out_txt_ocr)
    ocr_pdf_path = Path("ocr.pdf")
    run_ocr_extraction(
        pdf_path,
        page,
        out_txt_ocr=out_txt_ocr_path,
        out_ocr_pdf=ocr_pdf_path,
        ocr_dpi=ocr_dpi,
    )

    out_txt_text.write_text("\n".join(lines_text) + "\n", encoding="utf-8")
    out_txt_vectors.write_text("\n".join(lines_vectors) + "\n", encoding="utf-8")
    out_txt_words.write_text("\n".join(words) + "\n", encoding="utf-8")

    drawing_vectors_len = len(all_vectors) - len(filtered_vectors)

    print(f"Drawing Vectors: {drawing_vectors_len}")
    print(f"Text vectors: {text_vectors_total}")
    print(f"Word groups: {word_groups_total}")
    print(f"Wrote {out_txt_text}")
    print(f"Wrote {out_txt_vectors}")
    print(f"Wrote {out_txt_ocr}")
    print(f"Wrote {out_txt_words}")


def main() -> None:
    """CLI entry for Type-2 dump extraction."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--out_text", type=Path, default=Path("out_text.txt"))
    ap.add_argument("--out_vectors", type=Path, default=Path("out_vectors.txt"))
    ap.add_argument("--out_words", type=Path, default=Path("out_words.txt"))
    ap.add_argument("--out_ocr", type=Path, default=Path("out_ocr.txt"))
    ap.add_argument("--ocr-dpi", type=int, default=300, metavar="N")
    args = ap.parse_args()

    if not args.pdf.is_file():
        ap.error(f"PDF not found: {args.pdf}")

    logging.basicConfig(level=logging.INFO)
    dump(
        args.pdf,
        args.page,
        args.out_text,
        args.out_vectors,
        args.out_words,
        args.out_ocr,
        ocr_dpi=args.ocr_dpi,
    )


if __name__ == "__main__":
    main()
