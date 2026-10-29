"""Save Paddle-prep word-group PNGs from native text (not a full-page native PNG).

Same render path as ``scripts/debug/dump_debug_png_pipeline.py`` (render →
rotate → normalise), but glyphs come from native text extraction first
(``extract_text_from_page`` → SVG → ``TextToVector``) instead of
``page.get_drawings()``.

Usage:
    uv run scripts/type2_native_wordgroup_png_pipeline.py PDF [--max N] [--start N]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import pymupdf as pymupdf

from raster_parser.parsing.parser import PdfRasterParser
from raster_parser.evaluation.native_vs_ocr.text_to_vector import TextToVector, extract_text_from_page
from raster_parser.rendering.png_render.png_renderer import PngRenderer, save_rgb
from raster_parser.ocr.wordgrouping import cluster_by_seqno, convert_vectors_to_glyphs, get_vectors


def main() -> None:
    """CLI: write Paddle-prep word-group crop/norm PNGs from native text."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-n", "--max", type=int, default=100000, help="Max vectors (default 100000)")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=500)
    ap.add_argument("-o", "--out", type=Path, default=Path("native_png"))
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    if not args.pdf.is_file():
        ap.error(f"PDF not found: {args.pdf}")

    with tempfile.TemporaryDirectory(prefix="raster_native_png_") as tmp:
        text_only_pdf = Path(tmp) / "native_text.pdf"
        extract_text_from_page(str(args.pdf), args.page, str(text_only_pdf))

        with pymupdf.open(str(text_only_pdf)) as doc:
            svg_str = doc[0].get_svg_image(text_as_path=True)

        vectors = TextToVector.from_svg_string(svg_str).get_vectors()
        count = max(0, args.max - args.start)
        vectors = vectors[args.start : args.start + count]

        parser = PdfRasterParser(str(text_only_pdf), page=0)
        glyphs = convert_vectors_to_glyphs(vectors)
        word_groups = cluster_by_seqno(glyphs, int(parser.page.rotation or 0))

        args.out.mkdir(parents=True, exist_ok=True)
        for p in args.out.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

        renderer = PngRenderer(dpi=args.dpi, page=parser.page)
        written = 0
        for i, wg in enumerate(word_groups):
            group_vectors = get_vectors(wg)
            box = (wg.bbox.x0, wg.bbox.y0, wg.bbox.x1, wg.bbox.y1)
            prep, _ = parser._prepare_paddle_word_group(
                i, group_vectors, box, renderer, wg.orientation
            )
            if prep is None:
                continue
            save_rgb(prep.crop, args.out / f"group_{i:04d}_crop.png")
            save_rgb(prep.norm, args.out / f"group_{i:04d}_norm.png")
            written += 1

    print(f"Wrote {written} word groups ({written * 2} PNGs) to {args.out.resolve()}")


if __name__ == "__main__":
    main()
