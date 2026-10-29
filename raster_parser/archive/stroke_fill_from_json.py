"""Debug: stroke vs fill split from a colour autotrace JSON dump.

Reads ``{stem}_colours/{stem}_colour_vectors.json`` (from ``drawing_colour_fill_autotrace``),
runs ``stroke_fill_parser``, and writes separate strokes / fills PDFs
(+ optional JSON dumps). Paths are left as-is (typically ``type="f"`` ribbons).

Usage::

    uv run scripts/debug/stroke_fill_from_json.py \\
        out/page_p0_colours/page_p0_colour_vectors.json
    uv run scripts/debug/stroke_fill_from_json.py VECTORS.json --out-dir out/ --q-thresh 0.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raster_parser.serialisation.deserialization import deserialize_vector
from raster_parser.models import VectorDTO
from raster_parser.autotrace.drawing_colour_fill_autotrace import (
    dump_colour_vectors_json,
)
from raster_parser.archive.stroke_fill_parser import stroke_fill_parser


def load_colour_vectors_json(path: Path | str) -> tuple[list, float | None, float | None, str | None]:
    """Load vectors + page size from a colour vectors JSON dump."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vectors = [
        deserialize_vector(VectorDTO.model_validate(row))
        for row in data.get("vectors") or []
    ]
    page_w = data.get("page_width")
    page_h = data.get("page_height")
    source = data.get("source")
    return (
        vectors,
        float(page_w) if page_w is not None else None,
        float(page_h) if page_h is not None else None,
        str(source) if source is not None else None,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stroke/fill split from colour autotrace vectors JSON.",
    )
    p.add_argument(
        "json_path",
        type=Path,
        help="colour vectors JSON (e.g. *_colour_vectors.json)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: beside the JSON)",
    )
    p.add_argument(
        "--stem",
        default=None,
        help="output filename stem (default: JSON stem with _colour_vectors stripped)",
    )
    p.add_argument(
        "--q-thresh",
        type=float,
        default=0.25,
        help="isoperimetric Q below this -> stroke (default 0.25)",
    )
    p.add_argument(
        "--dump-split-json",
        action="store_true",
        help="also write *_strokes.json and *_fills.json",
    )
    args = p.parse_args()

    json_path = args.json_path.resolve()
    if not json_path.is_file():
        raise SystemExit(f"JSON not found: {json_path}")

    vectors, page_w, page_h, source = load_colour_vectors_json(json_path)
    out_dir = (args.out_dir or json_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem
    if stem is None:
        stem = json_path.stem
        for suffix in ("_colour_vectors", "_vectors"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

    print(f"loaded {len(vectors)} vectors from {json_path.name}")
    if source:
        print(f"source: {source}")
    print(f"page: {page_w} x {page_h}")

    sf = stroke_fill_parser(
        vectors,
        out_dir=out_dir,
        stem=stem,
        page_width=page_w,
        page_height=page_h,
        q_thresh=args.q_thresh,
        write_pdfs=True,
    )

    print(
        f"Q<{sf.q_thresh:g}: strokes={len(sf.stroke_vectors)} "
        f"fills={len(sf.fill_vectors)}"
    )
    if sf.stroke_pdf_path is not None:
        print(f"strokes pdf: {sf.stroke_pdf_path}")
    if sf.fill_pdf_path is not None:
        print(f"fills pdf: {sf.fill_pdf_path}")

    if args.dump_split_json:
        if sf.stroke_vectors:
            sp = dump_colour_vectors_json(
                sf.stroke_vectors,
                out_dir / f"{stem}_colour_strokes.json",
                page_width=page_w or 0.0,
                page_height=page_h or 0.0,
                source=source,
            )
            print(f"strokes json: {sp}")
        if sf.fill_vectors:
            fp = dump_colour_vectors_json(
                sf.fill_vectors,
                out_dir / f"{stem}_colour_fills.json",
                page_width=page_w or 0.0,
                page_height=page_h or 0.0,
                source=source,
            )
            print(f"fills json: {fp}")


if __name__ == "__main__":
    main()
