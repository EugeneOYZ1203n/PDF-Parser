"""Debug and sample-run plumbing shared by every raster_vectorize backend built on the raster prelude
(`junction`): page-aligned layer PDFs for the viewer (`LayerSet`), rendering a
page clip as the embedded `Image` a backend expects (`render_region`), the prelude's histogram PNGs,
diff canvases, and the raster items / page documents their debug layers are built from."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from raster_parser.models.source import Source
from raster_parser.commons.helpers.geometry import PDF_POINTS_PER_INCH
from raster_parser.commons.helpers.pixmap import zoom_matrix
from raster_parser.models.image import Image
from raster_parser.models.raster_text import RasterItem
from raster_parser.commons.renderer.draw import (
    dot_spec,
    draw_image,
    hex_to_rgb01,
    image_spec,
    new_page_doc,
    polyline_spec,
    render_specs_pdf,
)

HISTOGRAM_DIR = "color_layer_histograms"
DebugLayer = "tuple[str, str, str, bytes]"


def place_raster(page, array: np.ndarray, image: Image) -> None:
    draw_image(page, image_spec(array, image.bbox, transform=image.transform))


def traced_vector_kwargs(page_index: int, seqno: int, color) -> dict:
    """The `Vector` fields every traced stroke shares: an opaque, unclipped, solid stroke in
    `color` (the caller adds type, items, width and rect)."""
    return dict(
        color=color,
        fill=None,
        dashes=None,
        closePath=False,
        lineCap=0,
        lineJoin=0,
        even_odd=False,
        stroke_opacity=1.0,
        fill_opacity=1.0,
        layer=None,
        scissor=None,
        blendmode=None,
        isolated=False,
        knockout=False,
        opacity=1.0,
        page_index=page_index,
        seqno=seqno,
        source=Source.FROM_IMAGE,
    )


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "layer"


class LayerSet:
    """Page-aligned multi-page layer PDFs for the viewer: one document per
    (stage, label); a page where a layer never appears gets a blank page,
    and a layer added twice on one page is overlaid onto that page."""

    def __init__(self) -> None:
        self.docs: dict[str, pymupdf.Document] = {}
        self.meta: dict[str, dict] = {}
        self.geometry: list[tuple[float, float, int]] = []

    def begin_page(self, width: float, height: float, rotation: int) -> None:
        self.geometry.append((width, height, rotation))

    def add(self, stage: str, label: str, hex_color: str, pdf_bytes: bytes) -> None:
        fname = f"{_slug(stage)}__{_slug(label)}.pdf"
        doc = self.docs.setdefault(fname, pymupdf.open())
        self.meta.setdefault(
            fname, {"stage": stage, "layer": label, "file": fname, "color": hex_color}
        )
        pos = len(self.geometry) - 1
        self._pad(doc, pos)
        with pymupdf.open("pdf", pdf_bytes) as src:
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

    def _pad(self, doc: pymupdf.Document, upto: int) -> None:
        while doc.page_count < upto:
            width, height, rotation = self.geometry[doc.page_count]
            page = doc.new_page(width=width, height=height)
            if rotation:
                page.set_rotation(rotation)

    def save(self, out: Path) -> list[dict]:
        for fname, doc in self.docs.items():
            self._pad(doc, len(self.geometry))
            # garbage=3: drop unused objects and merge duplicate ones (PyMuPDF's cleanup level)
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
    zoom = dpi / PDF_POINTS_PER_INCH
    with pymupdf.open(str(pdf)) as doc:
        page = doc[page_index]
        page.set_rotation(0)  # render in unrotated page space (in memory only)
        rect = pymupdf.Rect(clip) if clip is not None else page.rect
        pix = page.get_pixmap(matrix=zoom_matrix(dpi), clip=rect, alpha=False)
        arr = (
            np.frombuffer(pix.samples, np.uint8)
            .reshape(pix.height, pix.width, pix.n)[:, :, :3]
            .copy()
        )
        x0, y0 = pix.x / zoom, pix.y / zoom
    w_pt, h_pt = arr.shape[1] / zoom, arr.shape[0] / zoom
    return Image(
        array=arr,
        bbox=(x0, y0, x0 + w_pt, y0 + h_pt),
        dpi=dpi,
        source="embedded",
        transform=(w_pt, 0.0, 0.0, h_pt, x0, y0),
    )


def save_histograms(out: Path, debug_out: dict) -> list[str]:
    """Write the raster prelude's hue/lightness histogram PNGs; returns their
    file names."""
    prelude = debug_out.get("raster_prelude")
    if prelude is None or not prelude.debug.histograms:
        return []
    folder = out / HISTOGRAM_DIR
    folder.mkdir(parents=True, exist_ok=True)
    for name, png in prelude.debug.histograms:
        (folder / name).write_bytes(png)
    return [name for name, _png in prelude.debug.histograms]


def paste_codes(canvas: np.ndarray, codes: np.ndarray, x0: int, y0: int) -> None:
    """Paste `codes` into `canvas` at `(x0, y0)`, never overwriting a
    non-background canvas pixel with background."""
    region = canvas[y0 : y0 + codes.shape[0], x0 : x0 + codes.shape[1]]
    np.copyto(
        region,
        codes[: region.shape[0], : region.shape[1]],
        where=codes[: region.shape[0], : region.shape[1]] > 0,
    )


def rendered_to_rgba(rendered: np.ndarray) -> np.ndarray:
    """Vector render as black-on-transparent RGBA."""
    out = np.zeros((*rendered.shape, 4), dtype=np.uint8)
    out[rendered, 3] = 255
    return out


def finish_doc(doc_page, page_meta) -> bytes:
    """Close out a `draw.new_page_doc` page: `/Rotate` last, so every raster
    was placed in unrotated page space."""
    doc, page = doc_page
    try:
        page.set_rotation(page_meta.rotation)
        return doc.tobytes(deflate=True)
    finally:
        doc.close()


# diff-code canvas values
BACKGROUND, MATCHED, MISSED, SPURIOUS = 0, 1, 2, 3


def render_geometry(shape: tuple[int, int], polylines) -> np.ndarray:
    """Polylines drawn at their own (rounded) width -- always 1.0 now -- onto a bool canvas."""
    canvas = np.zeros(shape[:2], np.uint8)
    for pl in polylines:
        pts = np.round(np.asarray(pl.points, float)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, 255, max(1, int(round(pl.width))))
    return canvas > 0


def diff_codes(ink: np.ndarray, polylines, tol_px: int) -> "tuple[np.ndarray, np.ndarray]":
    """`(codes uint8, rendered bool)` in `ink`'s own pixel frame."""
    ink = np.asarray(ink, dtype=bool)
    rendered = render_geometry(ink.shape, polylines)
    if tol_px > 0:
        k = np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8)
        rendered_near = cv2.dilate(rendered.astype(np.uint8), k) > 0
        ink_near = cv2.dilate(ink.astype(np.uint8), k) > 0
    else:
        rendered_near, ink_near = rendered, ink
    codes = np.zeros(ink.shape, dtype=np.uint8)
    codes[ink & rendered_near] = MATCHED
    codes[ink & ~rendered_near] = MISSED
    codes[rendered & ~ink_near] = SPURIOUS
    return codes, rendered


class RasterSink:
    """Receives full-resolution debug rasters. Streaming: PNG-encode into one open single-page PDF
    per (stage, label) immediately, drop the array. Keep: retain
    `RasterItem`s for `debug_out`. Neither: ignore (`active` tells the
    caller not to build the array at all)."""

    def __init__(self, *, stream: bool, keep: bool) -> None:
        self.stream = stream
        self.keep = keep
        self.kept: list[RasterItem] = []
        self._docs: dict[tuple[str, str], tuple[str, object]] = {}

    @property
    def active(self) -> bool:
        return self.stream or self.keep

    def add(
        self, stage: str, label: str, hexcolor: str, array: np.ndarray, image: Image, page_meta
    ) -> None:
        if self.keep:
            self.kept.append(RasterItem(stage, label, hexcolor, array, image, array.shape[:2]))
        if self.stream:
            key = (stage, label)
            if key not in self._docs:
                self._docs[key] = (hexcolor, new_page_doc(page_meta))
            place_raster(self._docs[key][1][1], array, image)

    def finish_stream(self, page_meta) -> "list[DebugLayer]":
        out: "list[DebugLayer]" = []
        for (stage, label), (hexcolor, doc) in self._docs.items():
            out.append((stage, label, hexcolor, finish_doc(doc, page_meta)))
        self._docs.clear()
        return out


def geometry_layer(
    page_meta,
    stage: str,
    label: str,
    hexcolor: str,
    *,
    polylines=(),
    points=(),
    width: float = 0.6,
    radius: float = 0.8,
) -> DebugLayer:
    rgb = hex_to_rgb01(hexcolor)
    specs = [polyline_spec(pl, rgb, width=width) for pl in polylines if len(pl) >= 2]
    specs += [dot_spec(p, rgb, radius) for p in points]
    return (stage, label, hexcolor, render_specs_pdf(page_meta, specs))


def compose_raster_layers(page_meta, items: list[RasterItem]) -> "list[DebugLayer]":
    docs: dict[tuple[str, str], tuple[str, object]] = {}
    for it in items:
        key = (it.stage, it.label)
        if key not in docs:
            docs[key] = (it.hexcolor, new_page_doc(page_meta))
        place_raster(docs[key][1][1], it.array, it.image)
    return [
        (s, l, hexcolor, finish_doc(doc, page_meta)) for (s, l), (hexcolor, doc) in docs.items()
    ]
