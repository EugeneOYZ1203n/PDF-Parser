"""Standalone utility: convert a PDF into a purely-raster PDF.

Every page is rasterized to an image at the given DPI, then a new PDF is
rebuilt from just those images -- no vector/text content survives. Useful
as test input for the Raster half of the pipeline, independent of the
Vector half.

Not part of the raster_parser package: this is a one-off input-prep utility,
not a pipeline stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raster_parser.commons.helpers.pixmap import zoom_matrix
from raster_parser.commons.paths import output_dir


def rasterize_pdf(src_path: str, dst_path: str, dpi: int = 300) -> None:
    doc = pymupdf.open(src_path)
    out = pymupdf.open()
    matrix = zoom_matrix(dpi)

    for page in doc:
        # Rotated display space, matching the visible page -- correct
        # since the output is meant to be a flat raster of what's seen.
        pix = page.get_pixmap(matrix=matrix)
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=pix.tobytes("png"))

    out.save(dst_path)
    out.close()
    doc.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="Path to the source PDF.")
    parser.add_argument(
        "dst",
        nargs="?",
        default=None,
        help="Path to write the raster-only PDF to "
        "(default: outputs/rasterize/<src stem>_raster.pdf).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Render resolution (default: 300).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dst = args.dst or str(output_dir("rasterize") / f"{Path(args.src).stem}_raster.pdf")
    rasterize_pdf(args.src, dst, dpi=args.dpi)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
