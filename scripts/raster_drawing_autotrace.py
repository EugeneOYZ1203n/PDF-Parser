"""CLI: PNG -> autotrace SVG -> PDF -> list[Vector].

Expects an already binary (or otherwise ready) PNG — no soft/adaptive preprocess.

Usage::

    uv run scripts/raster_drawing_autotrace.py page_drawing_final.png
    uv run scripts/raster_drawing_autotrace.py page_drawing_final.png --svg out.svg --pdf out.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from raster_parser.autotrace.drawing_autotrace import (
    _DEFAULT_ERROR_THRESHOLD,
    _DEFAULT_FILTER_ITERATIONS,
    _DEFAULT_NOISE_REMOVAL,
    autotrace_drawing,
)


def main() -> None:
    """CLI: trace a PNG and print how many Vector paths were produced."""
    p = argparse.ArgumentParser(
        description="PNG -> autotrace SVG -> PDF -> list[Vector] (no preprocess)."
    )
    p.add_argument("png", help="input drawing PNG (e.g. *_drawing_final.png or a mask)")
    p.add_argument("--svg", default=None, help="keep the intermediate SVG at this path")
    p.add_argument("--pdf", default=None, help="keep the intermediate PDF at this path")
    p.add_argument(
        "--dpi", type=int, default=None, help="scale Vector coords by 72/dpi into PDF points"
    )
    p.add_argument(
        "--centerline",
        dest="centerline",
        action="store_true",
        help="trace centerlines instead of outlines (default: outline)",
    )
    p.add_argument(
        "--no-preserve-width",
        dest="preserve_width",
        action="store_false",
        help="disable --preserve-width under centerline",
    )
    p.add_argument("--color-count", type=int, default=2)
    p.add_argument(
        "--error-threshold",
        type=float,
        default=_DEFAULT_ERROR_THRESHOLD,
        help="curve fit tolerance in px; lower = more detail",
    )
    p.add_argument("--corner-always-threshold", type=float, default=80.0)
    p.add_argument("--corner-surround", type=int, default=4)
    p.add_argument(
        "--noise-removal",
        type=float,
        default=_DEFAULT_NOISE_REMOVAL,
        help="autotrace default 0.99 drops small marks; lower keeps detail",
    )
    p.add_argument(
        "--filter-iterations",
        type=int,
        default=_DEFAULT_FILTER_ITERATIONS,
        help="pre-fit smoothing passes; autotrace default 4 over-smooths",
    )
    p.add_argument("--despeckle-level", type=int, default=0)
    args = p.parse_args()

    png = Path(args.png)
    # CLI keeps SVG/PDF beside the input by default so paths on disk are always printable.
    svg_out = Path(args.svg) if args.svg else png.with_name(png.stem + "_autotrace.svg")
    pdf_out = Path(args.pdf) if args.pdf else png.with_name(png.stem + "_autotrace.pdf")

    vectors = autotrace_drawing(
        png,
        centerline=args.centerline,
        color_count=args.color_count,
        error_threshold=args.error_threshold,
        corner_always_threshold=args.corner_always_threshold,
        corner_surround=args.corner_surround,
        noise_removal=args.noise_removal,
        filter_iterations=args.filter_iterations,
        despeckle_level=args.despeckle_level,
        preserve_width=args.preserve_width,
        dpi=args.dpi,
        svg_out=svg_out,
        pdf_out=pdf_out,
    )
    print(f"vectors: {len(vectors)}")
    print(f"svg: {svg_out.resolve()}")
    print(f"pdf: {pdf_out.resolve()}")


if __name__ == "__main__":
    main()
