"""CLI: per-colour mask autotrace + combine.

Usage::

    uv run scripts/raster_drawing_colour_fill_autotrace.py page_drawing_final.png
    uv run scripts/raster_drawing_colour_fill_autotrace.py page_drawing_final.png --out-dir out/ --dpi 300
"""

from __future__ import annotations

import argparse

from raster_parser.autotrace.drawing_colour_fill_autotrace import (
    _DEFAULT_BG_THRESH,
    _DEFAULT_GROUP_DIST,
    _DEFAULT_MIN_GROUP_PIXELS,
    colour_fill_autotrace,
)


def main() -> None:
    """Parse CLI args and run ``colour_fill_autotrace`` on one PNG."""
    p = argparse.ArgumentParser(
        description="Per-colour mask autotrace + combine (standalone experiment).",
    )
    p.add_argument("png", help="colour drawing PNG (e.g. *_drawing_final.png)")
    p.add_argument(
        "--out-dir", default=None, help="write QA / SVG / PDF here (default: beside PNG)"
    )
    p.add_argument("--dpi", type=int, default=300, help="scale Vector coords by 72/dpi")
    p.add_argument(
        "--bg-thresh",
        type=int,
        default=_DEFAULT_BG_THRESH,
        help="RGB channels >= this count as background white",
    )
    p.add_argument(
        "--group-dist",
        type=float,
        default=_DEFAULT_GROUP_DIST,
        help=f"max RGB Euclidean distance to merge colours (default {_DEFAULT_GROUP_DIST:g})",
    )
    p.add_argument(
        "--min-group-pixels",
        type=int,
        default=_DEFAULT_MIN_GROUP_PIXELS,
        help=f"drop colour groups smaller than this (default {_DEFAULT_MIN_GROUP_PIXELS})",
    )
    p.add_argument(
        "--no-keep-intermediates",
        dest="keep_intermediates",
        action="store_false",
        help="delete per-colour mask / isolate PNGs after tracing",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="ProcessPool size for per-colour-group autotrace (1 = serial)",
    )
    args = p.parse_args()

    result = colour_fill_autotrace(
        args.png,
        out_dir=args.out_dir,
        dpi=args.dpi,
        bg_thresh=args.bg_thresh,
        group_dist=args.group_dist,
        min_group_pixels=args.min_group_pixels,
        keep_intermediates=args.keep_intermediates,
        workers=args.workers,
    )
    print(f"colour groups: {len(result.groups)} (group_dist={args.group_dist})")
    for group in result.groups:
        n_vec = len(result.group_vectors.get(group.label, []))
        print(f"  [{group.tag}] rgb={group.rgb} pixels={group.n_pixels} vectors={n_vec}")
    print(f"combined vectors: {len(result.combined_vectors)}")
    if result.colours_dir is not None:
        print(f"colours dir: {result.colours_dir.resolve()}")
    if result.combined_pdf_path is not None:
        print(f"combined pdf: {result.combined_pdf_path.resolve()}")
    if result.vectors_txt_path is not None:
        print(f"vectors txt: {result.vectors_txt_path.resolve()}")
    if result.vectors_json_path is not None:
        print(f"vectors json: {result.vectors_json_path.resolve()}")
    if result.palette_path is not None:
        print(f"palette: {result.palette_path.resolve()}")
    if result.centroid_map_path is not None:
        print(f"centroid map: {result.centroid_map_path.resolve()}")
    if result.summary_path is not None:
        print(f"summary: {result.summary_path.resolve()}")


if __name__ == "__main__":
    main()
