"""Try a trained ImplicitSketchVec model on a small sample and look at it in
the pipeline report viewer.

    python -m raster_parser.phase_raster_vectorize.implicit_sketch_vec.predict --pdf X.pdf \\
        [--pages 0,1] [--clip x0,y0,x1,y1] [--dpi 300] [--weights W.pth] [--out DIR] [-v]

Per page, the region `--clip` (PDF points, unrotated page space -- the
coordinates the inspector and every extraction use; default: the whole page)
is rendered at `--dpi`, handed to the real backend (`adapter.extract`) as
one embedded image, and every debug layer the backend draws (the shared raster
prelude's included) is written, plus:

  original / vectors       the page's own vector paths touching the clip, in
                           their real colors -- what the model should find
  original / clip region   the sampled region's outline
  detected / vectors       the model's output vectors, in their layer colors

OCR + text erasing never run here: this script only tests the model, and
the shared training data (DeepVectoriser's `prep_dataset.py`) keeps text
glyphs as strokes to vectorize, so the model sees the same kind of input.

Output: `<out>/manifest.json` + one multi-page PDF per layer, the folder
`scripts/pipeline_report_viewer.py` opens, plus one hue/lightness histogram
PNG per page in `<out>/color_layer_histograms/`; the command is printed at
the end.
Default `<out>`: `outputs/implicitsketchvec_predict/<timestamp>__<pdf stem>/`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
from raster_parser.phase_raster_vectorize.implicit_sketch_vec import adapter
from raster_parser.phase_raster_vectorize.implicit_sketch_vec.config import WEIGHTS_ENV_VAR
from raster_parser.commons.raster_text.paddle_engine import OcrBox
from raster_parser.commons.raster_text.debug_tools import (
    LayerSet,
    parse_clip,
    render_region,
    save_histograms,
)

_LOG = get_logger("implicit_sketch_vec.predict")

# OCR that finds nothing: the backend then erases nothing (always, here)
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
    model_fn=None,
    say=print,
) -> dict:
    """Predict every page into `out`; returns the manifest. `model_fn`
    replaces the trained model (tests)."""
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
            vectors, _texts = adapter.extract(
                [image],
                page,
                on_debug_layer=layers.add,
                debug_out=debug_out,
                keep_debug_arrays=False,
                ocr_fns=NO_OCR,
                model_fn=model_fn,
            )
            histograms = save_histograms(out, debug_out)
            layers.add(
                "detected", "vectors", "#0050ff", render_vectors_pdf(meta, vectors, color_of=_paint)
            )
            secs = time.perf_counter() - t0
            say(
                f"page {i}: region {tuple(round(v, 1) for v in image.bbox)} pt -> "
                f"{image.array.shape[1]}x{image.array.shape[0]} px | original {len(originals)} vector(s) | "
                f"detected {len(vectors)} vector(s) | {secs:.1f}s"
            )
            summary.append(
                {
                    "page": i,
                    "region": list(image.bbox),
                    "original": len(originals),
                    "detected": len(vectors),
                    "histograms": histograms,
                    "seconds": round(secs, 2),
                }
            )
    manifest = {
        "source_pdf": str(Path(pdf).resolve()),
        "pages": pages,
        "variant": "implicitsketchvec-predict",
        "engine": "ImplicitSketchVec",
        "dpi": dpi,
        "clip": list(clip) if clip else None,
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
    ap.add_argument(
        "--dpi",
        type=float,
        default=300.0,
        help="render dpi (default 300 = the model's canonical scale)",
    )
    ap.add_argument(
        "--weights",
        default=None,
        help=f"model file (default: ${WEIGHTS_ENV_VAR} or raster_parser/weights)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output folder (default outputs/implicitsketchvec_predict/<ts>__<stem>)",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    pdf = Path(args.pdf)
    if not pdf.is_file():
        raise SystemExit(f"PDF not found: {pdf}")
    if args.weights:
        if not Path(args.weights).is_file():
            raise SystemExit(f"--weights not found: {args.weights}")
        os.environ[WEIGHTS_ENV_VAR] = str(Path(args.weights).resolve())
    with pymupdf.open(str(pdf)) as doc:
        n = doc.page_count
    pages = [int(p) for p in args.pages.split(",") if p.strip()]
    bad = [p for p in pages if not 0 <= p < n]
    if bad:
        raise SystemExit(f"--pages {bad} out of range (document has {n} pages)")
    out = (
        Path(args.out)
        if args.out
        else output_dir(
            "implicitsketchvec_predict", f"{time.strftime('%Y%m%d_%H%M%S')}__{pdf.stem}"
        )
    )
    try:
        run(pdf, pages, out, clip=parse_clip(args.clip), dpi=args.dpi)
    except FileNotFoundError as exc:
        _LOG.error("%s", exc)
        return 2
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
