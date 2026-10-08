"""Try a trained ImplicitSketchVec model on a small sample and look at it in
the pipeline report viewer.

    python -m rastervec.P2_Raster_To_Vec.ImplicitSketchVec.predict --pdf X.pdf \\
        [--pages 0,1] [--clip x0,y0,x1,y1] [--dpi 300] [--weights W.pth] [--out DIR] [-v]

Per page, the region `--clip` (PDF points, unrotated page space -- the
coordinates the inspector and every extraction use; default: the whole page)
is rendered at `--dpi`, handed to the real P2 backend (`adapter.extract`) as
one embedded image, and every debug layer the backend draws is written, plus:

  original / vectors       the page's own vector paths touching the clip, in
                           their real colors -- what the model should find
  original / clip region   the sampled region's outline
  detected / vectors       the model's output vectors, in their layer colors

OCR + text erasing never run here: this script only tests the model, and
the shared training data (DeepVectoriser's `prep_dataset.py`) keeps text
glyphs as strokes to vectorize, so the model sees the same kind of input.

Output: `<out>/manifest.json` + one multi-page PDF per layer, the folder
`scripts/pipeline_report_viewer.py` opens; the command is printed at the end.
Default `<out>`: `outputs/implicitsketchvec_predict/<timestamp>__<pdf stem>/`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import fitz
import numpy as np

from rastervec.commons.logging_setup import configure_logging, get_logger
from rastervec.commons.models import Image, Vector
from rastervec.commons.paths import REPO_ROOT, output_dir
from rastervec.commons.renderer import render_boxes_pdf, render_vectors_pdf
from rastervec.P1_Reading_Native.reader import Reader
from rastervec.P1_Reading_Native.vector_extract import extract_vectors
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import adapter
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec.config import WEIGHTS_ENV_VAR
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec.paddle_engine import OcrBox

_LOG = get_logger("P2.ImplicitSketchVec.predict")

# OCR that finds nothing: the backend then erases nothing (always, here)
NO_OCR = {
    "detect_many": lambda images: [[] for _ in images],
    "recognize": lambda crops: [OcrBox("", 0.0) for _ in crops],
    "recognize_raw": lambda crops: [OcrBox("", 0.0) for _ in crops],
}
_CLIP_RGB = (1.0, 0.55, 0.0)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "layer"


class LayerSet:
    """Page-aligned multi-page layer PDFs for the viewer: one document per
    (stage, label); a page where a layer never appears gets a blank page,
    and a layer added twice on one page is overlaid onto that page."""

    def __init__(self) -> None:
        self.docs: dict[str, fitz.Document] = {}
        self.meta: dict[str, dict] = {}
        self.geometry: list[tuple[float, float, int]] = []

    def begin_page(self, width: float, height: float, rotation: int) -> None:
        self.geometry.append((width, height, rotation))

    def add(self, stage: str, label: str, hex_color: str, pdf_bytes: bytes) -> None:
        fname = f"{_slug(stage)}__{_slug(label)}.pdf"
        doc = self.docs.setdefault(fname, fitz.open())
        self.meta.setdefault(fname, {"stage": stage, "layer": label, "file": fname, "color": hex_color})
        pos = len(self.geometry) - 1
        self._pad(doc, pos)
        with fitz.open("pdf", pdf_bytes) as src:
            if doc.page_count == pos:
                doc.insert_pdf(src)
                return
            page = doc[pos]  # second add on this page: overlay at rotation 0
            rotation = page.rotation
            page.set_rotation(0)
            src[0].set_rotation(0)
            page.show_pdf_page(page.rect, src, 0)
            if rotation:
                page.set_rotation(rotation)

    def _pad(self, doc: fitz.Document, upto: int) -> None:
        while doc.page_count < upto:
            width, height, rotation = self.geometry[doc.page_count]
            page = doc.new_page(width=width, height=height)
            if rotation:
                page.set_rotation(rotation)

    def save(self, out: Path) -> list[dict]:
        for fname, doc in self.docs.items():
            self._pad(doc, len(self.geometry))
            doc.save(str(out / fname), garbage=3, deflate=True)
            doc.close()
        return [self.meta[f] for f in self.docs]


def parse_clip(spec: "str | None") -> "tuple[float, float, float, float] | None":
    if not spec:
        return None
    parts = [float(v) for v in spec.replace(" ", "").split(",")]
    if len(parts) != 4 or parts[2] <= parts[0] or parts[3] <= parts[1]:
        raise SystemExit(f"--clip wants x0,y0,x1,y1 with x1>x0, y1>y0 (got {spec!r})")
    return tuple(parts)  # type: ignore[return-value]


def render_region(pdf: Path, page_index: int, clip, dpi: float) -> Image:
    """The clip (unrotated page space) rendered at `dpi`, as the embedded
    `Image` the backend expects: `transform` maps its unit square onto the
    exact page area the pixmap covers (MuPDF rounds the clip outwards to
    whole pixels)."""
    zoom = dpi / 72.0
    with fitz.open(str(pdf)) as doc:
        page = doc[page_index]
        page.set_rotation(0)  # render in unrotated page space (in memory only)
        rect = fitz.Rect(clip) if clip is not None else page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
        x0, y0 = pix.x / zoom, pix.y / zoom
    w_pt, h_pt = arr.shape[1] / zoom, arr.shape[0] / zoom
    return Image(array=arr, bbox=(x0, y0, x0 + w_pt, y0 + h_pt), dpi=dpi, source="embedded",
                 transform=(w_pt, 0.0, 0.0, h_pt, x0, y0))


def _touches(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _paint(v: Vector) -> tuple[float, float, float]:
    rgb = v.color or v.fill or (0.0, 0.0, 0.0)
    return tuple(float(c) for c in rgb[:3])  # type: ignore[return-value]


def run(pdf: Path, pages: list[int], out: Path, *, clip=None, dpi: float = 300.0,
        model_fn=None, say=print) -> dict:
    """Predict every page into `out`; returns the manifest. `model_fn`
    replaces the trained model (tests)."""
    out.mkdir(parents=True, exist_ok=True)
    layers = LayerSet()
    summary = []
    with Reader(str(pdf)) as reader:
        for i in pages:
            page = reader.get_page(i)
            meta = page.meta
            layers.begin_page(meta.width, meta.height, int(page.fitz_page.rotation))
            region = clip or (0.0, 0.0, meta.width, meta.height)
            t0 = time.perf_counter()
            image = render_region(pdf, i, clip, dpi)
            originals = [v for v in extract_vectors(page) if _touches(v.bbox, region)]
            layers.add("original", "vectors", "#000000",
                       render_vectors_pdf(meta, originals, color_of=_paint))
            layers.add("original", "clip region", "#ff8c00",
                       render_boxes_pdf(meta, [(image.bbox, _CLIP_RGB)]))
            vectors, _texts = adapter.extract(
                [image], page, on_debug_layer=layers.add, ocr_fns=NO_OCR, model_fn=model_fn,
            )
            layers.add("detected", "vectors", "#0050ff", render_vectors_pdf(meta, vectors, color_of=_paint))
            secs = time.perf_counter() - t0
            say(f"page {i}: region {tuple(round(v, 1) for v in image.bbox)} pt -> "
                f"{image.array.shape[1]}x{image.array.shape[0]} px | original {len(originals)} vector(s) | "
                f"detected {len(vectors)} vector(s) | {secs:.1f}s")
            summary.append({"page": i, "region": list(image.bbox), "original": len(originals),
                            "detected": len(vectors), "seconds": round(secs, 2)})
    manifest = {
        "source_pdf": str(Path(pdf).resolve()), "pages": pages, "variant": "implicitsketchvec-predict",
        "engine": "ImplicitSketchVec", "dpi": dpi, "clip": list(clip) if clip else None,
        "summary": summary, "layers": layers.save(out),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdf", required=True, help="source PDF")
    ap.add_argument("--pages", default="0", help="comma-separated page indices (default 0)")
    ap.add_argument("--clip", default=None, help="x0,y0,x1,y1 in PDF points, unrotated page space (default: whole page)")
    ap.add_argument("--dpi", type=float, default=300.0, help="render dpi (default 300 = the model's canonical scale)")
    ap.add_argument("--weights", default=None, help=f"model file (default: ${WEIGHTS_ENV_VAR} or rastervec/weights)")
    ap.add_argument("--out", default=None, help="output folder (default outputs/implicitsketchvec_predict/<ts>__<stem>)")
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
    with fitz.open(str(pdf)) as doc:
        n = doc.page_count
    pages = [int(p) for p in args.pages.split(",") if p.strip()]
    bad = [p for p in pages if not 0 <= p < n]
    if bad:
        raise SystemExit(f"--pages {bad} out of range (document has {n} pages)")
    out = Path(args.out) if args.out else output_dir("implicitsketchvec_predict",
                                                     f"{time.strftime('%Y%m%d_%H%M%S')}__{pdf.stem}")
    try:
        run(pdf, pages, out, clip=parse_clip(args.clip), dpi=args.dpi)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    viewer = REPO_ROOT / "scripts" / "pipeline_report_viewer.py"
    print(f"\nwrote {out}\nview it with:\n  .venv/Scripts/python.exe {viewer.relative_to(REPO_ROOT).as_posix()} \"{out}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
