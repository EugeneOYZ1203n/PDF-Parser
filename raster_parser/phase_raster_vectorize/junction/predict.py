"""Try Junction on a small sample and look at it in the pipeline report
viewer.

    python -m raster_parser.phase_raster_vectorize.junction.predict --pdf X.pdf \\
        [--pages 0,1] [--clip x0,y0,x1,y1] [--dpi 300] [--no-ocr] [--out DIR] [-v]

Per page, the region `--clip` (PDF points, unrotated page space -- the
coordinates the inspector and every extraction use; default: the whole page)
is rendered at `--dpi`, handed to the real P2 backend (`adapter.extract`) as
one embedded image, and every debug layer the backend draws (the shared
raster prelude's included) is written, plus:

  original / vectors       the page's own vector paths touching the clip, in
                           their real colors -- what Junction should find
  original / clip region   the sampled region's outline
  detected / vectors       Junction's output vectors, in their layer colors

The raster prelude's OCR + text erasing runs by default (Junction has no
training data to stay consistent with); `--no-ocr` skips it, so text glyphs
are traced like any other ink.

Output: `<out>/manifest.json` + one multi-page PDF per layer, the folder
`scripts/pipeline_report_viewer.py` opens, plus one hue/lightness histogram
PNG per page in `<out>/color_layer_histograms/`; the viewer command is
printed at the end. Default `<out>`: `outputs/junction_predict/<timestamp>__<pdf stem>/`.

`LayerSet` / `parse_clip` / `render_region` are copied from
`DeepVectoriser/predict.py` (sibling P2 backends share no code)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pymupdf
from raster_parser.commons.logging_setup import configure_logging, get_logger
from raster_parser.models.vector import Vector
from raster_parser.commons.paths import REPO_ROOT, output_dir
from raster_parser.commons.renderer.pdf import render_boxes_pdf, render_vectors_pdf
from raster_parser.phase_native_read.reader import Reader
from raster_parser.phase_native_read.vector_extract import extract_vectors
from raster_parser.phase_raster_vectorize.junction import adapter
from raster_parser.commons.raster_text.paddle_engine import OcrBox
from raster_parser.commons.raster_text.debug_tools import (  # noqa: F401 -- shared with the other raster backends
    HISTOGRAM_DIR,
    LayerSet,
    parse_clip,
    render_region,
    save_histograms,
)

_LOG = get_logger("junction.predict")

# OCR that finds nothing: the prelude then erases nothing
NO_OCR = {
    "detect_many": lambda images: [[] for _ in images],
    "recognize": lambda crops: [OcrBox("", 0.0) for _ in crops],
    "recognize_raw": lambda crops: [OcrBox("", 0.0) for _ in crops],
}
_CLIP_RGB = (1.0, 0.55, 0.0)


def _touches(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _paint(v: Vector) -> tuple[float, float, float]:
    rgb = v.color or v.fill or (0.0, 0.0, 0.0)
    return tuple(float(c) for c in rgb[:3])  # type: ignore[return-value]


def run(
    pdf: Path,
    pages: list[int],
    out: Path,
    *,
    clip=None,
    dpi: float = 300.0,
    ocr: bool = True,
    say=print,
) -> dict:
    """Run Junction on every page into `out`; returns the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    layers = LayerSet()
    summary = []
    with Reader(str(pdf)) as reader:
        for i in pages:
            page = reader.get_page(i)
            meta = page.meta
            layers.begin_page(meta.width, meta.height, int(page.pymupdf_page.rotation))
            region = clip or (0.0, 0.0, meta.width, meta.height)
            t0 = time.perf_counter()
            image = render_region(pdf, i, clip, dpi)
            originals = [v for v in extract_vectors(page) if _touches(v.bbox, region)]
            layers.add(
                "original",
                "vectors",
                "#000000",
                render_vectors_pdf(meta, originals, color_of=_paint),
            )
            layers.add(
                "original",
                "clip region",
                "#ff8c00",
                render_boxes_pdf(meta, [(image.bbox, _CLIP_RGB)]),
            )
            debug_out: dict = {}
            vectors, texts = adapter.extract(
                [image],
                page,
                on_debug_layer=layers.add,
                debug_out=debug_out,
                keep_debug_arrays=False,
                ocr_fns=None if ocr else NO_OCR,
            )
            # the sub-pixel RDP-thinned copies, as stroke batches: one vertex per
            # skeleton pixel through PyMuPDF's `Shape` takes minutes on a full page
            shown = debug_out.get("display_vectors") or vectors
            layers.add("detected", "vectors", "#0050ff", adapter.vectors_layer_pdf(meta, shown))
            histograms = save_histograms(out, debug_out)
            secs = time.perf_counter() - t0
            say(
                f"page {i}: region {tuple(round(v, 1) for v in image.bbox)} pt -> "
                f"{image.array.shape[1]}x{image.array.shape[0]} px | original {len(originals)} vector(s) | "
                f"detected {len(vectors)} vector(s){f', {len(texts)} OCR text(s)' if ocr else ''} | {secs:.1f}s"
            )
            summary.append(
                {
                    "page": i,
                    "region": list(image.bbox),
                    "original": len(originals),
                    "detected": len(vectors),
                    "texts": len(texts),
                    "histograms": histograms,
                    "seconds": round(secs, 2),
                }
            )
    manifest = {
        "source_pdf": str(Path(pdf).resolve()),
        "pages": pages,
        "variant": "junction-predict",
        "engine": "Junction",
        "dpi": dpi,
        "clip": list(clip) if clip else None,
        "ocr": ocr,
        "summary": summary,
        "layers": layers.save(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdf", required=True, help="source PDF")
    ap.add_argument("--pages", default="0", help="comma-separated page indices (default 0)")
    ap.add_argument(
        "--clip",
        default=None,
        help="x0,y0,x1,y1 in PDF points, unrotated page space (default: whole page)",
    )
    ap.add_argument("--dpi", type=float, default=300.0, help="render dpi (default 300)")
    ap.add_argument(
        "--no-ocr", action="store_true", help="skip the raster prelude's OCR + text erase"
    )
    ap.add_argument(
        "--out", default=None, help="output folder (default outputs/junction_predict/<ts>__<stem>)"
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    pdf = Path(args.pdf)
    if not pdf.is_file():
        raise SystemExit(f"PDF not found: {pdf}")
    with pymupdf.open(str(pdf)) as doc:
        n = doc.page_count
    pages = [int(p) for p in args.pages.split(",") if p.strip()]
    bad = [p for p in pages if not 0 <= p < n]
    if bad:
        raise SystemExit(f"--pages {bad} out of range (document has {n} pages)")
    out = (
        Path(args.out)
        if args.out
        else output_dir("junction_predict", f"{time.strftime('%Y%m%d_%H%M%S')}__{pdf.stem}")
    )
    run(pdf, pages, out, clip=parse_clip(args.clip), dpi=args.dpi, ocr=not args.no_ocr)
    viewer = REPO_ROOT / "scripts" / "pipeline_report_viewer.py"
    _LOG.info(
        'wrote %s -- view it with: python %s "%s"',
        out,
        viewer.relative_to(REPO_ROOT).as_posix(),
        out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
