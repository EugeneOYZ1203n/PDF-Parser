"""Full pipeline: native text → vector PDF → PaddleOCR → OCR PDF → compare.

Step 1 — ``type2_nativetext_to_vector_pipeline``:
  - ``out.pdf`` — page with native text only
  - ``onlyfill.pdf`` — native text replayed as vector paths (OCR input)

Step 2 — ``type2_dump_extraction_pipeline`` OCR on ``onlyfill.pdf``:
  - ``out_ocr.txt`` — PaddleOCR per word group
  - ``ocr.pdf`` — OCR text rendered back onto the page

Step 3 — ``native_vs_ocr``:
  - ``out_results.txt`` — bbox IOU, CER, match per OCR word vs SVG glyphs

Usage:
    uv run scripts/type2_full_native_to_ocr_pipeline.py PDF_PATH [--page N]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

from type2_dump_extraction_pipeline import run_ocr_extraction
from type2_nativetext_to_vector_pipeline import run_native_text_to_vector
from raster_parser.models import Text
from raster_parser.evaluation.native_vs_ocr import NativeVsOcr, format_stats_report
from raster_parser.evaluation.native_vs_ocr.text_to_vector import TextToVector

logger = logging.getLogger(__name__)


def run_comparison(
    out_pdf: Path,
    ocr_texts_with_mapping: List[Tuple[Text, List[int]]],
    *,
    out_results: Path,
) -> Path:
    """Compare SVG native glyphs to OCR word groups; write stats file."""
    svg_path = Path(out_pdf).with_suffix(".svg")
    if not svg_path.is_file():
        raise FileNotFoundError(f"SVG not found for comparison: {svg_path}")

    native_text = TextToVector(str(svg_path)).get_mapping()
    checker = NativeVsOcr(native_text, ocr_texts_with_mapping)
    stats = checker.compare_words()
    not_ocred = checker.native_glyphs_not_ocred()

    out_results = Path(out_results)
    out_results.parent.mkdir(parents=True, exist_ok=True)
    out_results.write_text(
        format_stats_report(
            stats,
            not_ocred=not_ocred,
            total_native_characters=len(native_text),
        ),
        encoding="utf-8",
    )
    return out_results.resolve()


def run_pipeline(
    pdf_path: Path,
    page: int,
    *,
    out_pdf: Path,
    out_onlyfill: Path,
    out_ocr_txt: Path,
    out_ocr_pdf: Path,
    out_results: Path,
    ocr_dpi: int = 300,
    ocr_page: int = 0,
) -> dict[str, Path]:
    """Run native-text vectorization, OCR, then native-vs-OCR comparison."""
    logger.info("Step 1: native text → vectors (%s page %d)", pdf_path, page)
    out_pdf, out_onlyfill = run_native_text_to_vector(
        pdf_path,
        page,
        out_pdf=out_pdf,
        out_onlyfill=out_onlyfill,
        out_svg=Path(out_pdf).with_suffix(".svg"),
    )
    logger.info("  out.pdf: %s", out_pdf)
    logger.info("  onlyfill.pdf: %s", out_onlyfill)

    logger.info("Step 2: OCR on vectorized PDF (page %d)", ocr_page)
    out_ocr_txt, out_ocr_pdf, ocr_texts_with_mapping = run_ocr_extraction(
        out_onlyfill,
        ocr_page,
        out_txt_ocr=out_ocr_txt,
        out_ocr_pdf=out_ocr_pdf,
        ocr_dpi=ocr_dpi,
    )
    logger.info("  out_ocr.txt: %s", out_ocr_txt)
    logger.info("  ocr.pdf: %s", out_ocr_pdf)

    logger.info("Step 3: native vs OCR comparison")
    out_results_path = run_comparison(
        out_pdf,
        ocr_texts_with_mapping,
        out_results=out_results,
    )
    logger.info("  out_results.txt: %s", out_results_path)

    return {
        "out_pdf": out_pdf,
        "onlyfill_pdf": out_onlyfill,
        "out_ocr_txt": out_ocr_txt,
        "ocr_pdf": out_ocr_pdf,
        "out_results": out_results_path,
    }


def main() -> None:
    """CLI entry for native → vector → OCR → compare."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path, help="Source PDF")
    ap.add_argument("--page", type=int, default=0, help="Source page for native text")
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default="native-to-ocr-pipeline/out.pdf",
        help="Text-only PDF (native text)",
    )
    ap.add_argument(
        "--out-onlyfill",
        type=Path,
        default="native-to-ocr-pipeline/onlyfill.pdf",
        help="Vectorized native text PDF (OCR input)",
    )
    ap.add_argument(
        "--out-ocr",
        type=Path,
        default="native-to-ocr-pipeline/out_ocr.txt",
        help="OCR text dump",
    )
    ap.add_argument(
        "--ocr-pdf",
        type=Path,
        default="native-to-ocr-pipeline/ocr.pdf",
        help="PDF with OCR text rendered",
    )
    ap.add_argument(
        "--out-results",
        type=Path,
        default="native-to-ocr-pipeline/out_results.txt",
        help="Native vs OCR stats (IOU, CER, match)",
    )
    ap.add_argument("--ocr-dpi", type=int, default=300, metavar="N")
    ap.add_argument(
        "--ocr-page",
        type=int,
        default=0,
        help="Page index inside onlyfill.pdf (usually 0)",
    )
    args = ap.parse_args()

    if not args.pdf.is_file():
        ap.error(f"PDF not found: {args.pdf}")

    logging.basicConfig(level=logging.INFO)

    outputs = run_pipeline(
        args.pdf,
        args.page,
        out_pdf=args.out_pdf,
        out_onlyfill=args.out_onlyfill,
        out_ocr_txt=args.out_ocr,
        out_ocr_pdf=args.ocr_pdf,
        out_results=args.out_results,
        ocr_dpi=args.ocr_dpi,
        ocr_page=args.ocr_page,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
