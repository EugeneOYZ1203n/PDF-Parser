"""Artifact writing for `scripts/generate_pipeline_report.py`.

- `LayerWriter`: one multi-page PDF per debug layer, page-aligned with the report's pages,
  spilled to disk as layers arrive (backends stream them during a run).
- `debug_layer_sink` / `debug_image_sink`: the `RunSettings.on_debug_layer` / `on_debug_image`
  callbacks the report hands to the backends.
- `ImageReservoir`: a capped, uniform random sample of debug images per folder.
- `reconstructed_layer`: the final output as a page (Phase 4's `render_output_pdf`).
- `finalize_doc_dir`: `dump.json` + `manifest.json` (the viewer reads the manifest).
- `write_label_overlays` / `add_extra_prediction_layers`: benchmark-mode overlays.
"""
from __future__ import annotations

import json
import random
import re
import shutil
import tempfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf as fitz

from raster_parser.commons.logging_setup import get_logger
from raster_parser.commons.renderer.pdf import render_boxes_pdf, render_text_pdf, render_vectors_pdf
from raster_parser.evaluation import dump_io
from raster_parser.evaluation.evaluate import adapters, label_overlays, metrics
from raster_parser.evaluation.labelling.label_schema import LabelSet, path_signature
from raster_parser.evaluation.report.report_config import layer_slug
from raster_parser.P4_Output_Organization import render_output_pdf

if TYPE_CHECKING:
    from PIL import Image

    from raster_parser.commons.models import PageMeta, Text, Vector
    from raster_parser.core.result import PipelineResult
    from raster_parser.core.settings import RunSettings
    from raster_parser.evaluation.report.report_config import ReportConfig

_LOG = get_logger("generate_pipeline_report")

RECONSTRUCTED_COLOR = "#111827"


# ---------------------------------------------------------------------------
# Layer PDFs
# ---------------------------------------------------------------------------
class LayerWriter:
    """Incremental per-layer multi-page PDF writer.

    `add` spills one page's bytes straight to disk (`<spill_dir>/<n>/<page>_<seq>.pdf`) and keeps
    only the file list in memory. `finalize` then builds each layer's PDF one layer at a time and
    removes the spill folder.

    Page-aligned: `begin_page(pos, ...)` (called before each report page's run, since backends
    stream layers mid-run) tags every later `add` with that page, and `finalize` writes exactly
    one page per `begin_page`. A page where the layer never appeared is blank; a layer added
    several times on one page is overlaid onto that page. A layer that is blank on every page
    (no content stream at all) is never written."""

    def __init__(self, spill_dir: "Path | None" = None) -> None:
        self._spill_root = Path(spill_dir) if spill_dir is not None else Path(tempfile.mkdtemp(prefix="layers_"))
        self._parts: dict[str, dict[int, list[Path]]] = {}
        self._folder_index: dict[str, int] = {}
        self._has_content: set[str] = set()
        self._geometry: dict[int, tuple[float, float, int]] = {}
        self._pos = 0
        self.meta: dict[str, dict] = {}

    def begin_page(self, pos: int, width: float, height: float, rotation: int = 0) -> None:
        """Start report page `pos`; `width`/`height` (unrotated) + `rotation` size its blank page
        in any layer that skips it."""
        self._pos = pos
        self._geometry[pos] = (width, height, rotation)

    def add(self, fname: str, meta: dict, pdf_bytes: bytes) -> None:
        parts = self._parts.setdefault(fname, {}).setdefault(self._pos, [])
        if fname not in self._has_content:
            with fitz.open("pdf", pdf_bytes) as src:
                if _has_content(src):
                    self._has_content.add(fname)
        folder = self._spill_root / f"{self._folder_index.setdefault(fname, len(self._folder_index)):04d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"p{self._pos:04d}_{len(parts):05d}.pdf"
        path.write_bytes(pdf_bytes)
        parts.append(path)
        self.meta.setdefault(fname, meta)

    def filenames(self) -> list[str]:
        return [f for f in self._parts if f in self._has_content]

    def finalize(self, doc_dir: Path) -> None:
        try:
            for fname, by_pos in self._parts.items():
                if fname not in self._has_content:
                    continue
                n_pages = max([*self._geometry, *by_pos]) + 1
                out = fitz.open()
                try:
                    for pos in range(n_pages):
                        parts = by_pos.get(pos)
                        if parts:
                            _insert_merged(out, parts)
                        else:
                            self._insert_blank(out, pos, by_pos)
                    out.save(str(doc_dir / fname), garbage=3, deflate=True)
                finally:
                    out.close()
        finally:
            self._parts.clear()
            shutil.rmtree(self._spill_root, ignore_errors=True)

    def _insert_blank(self, out: "fitz.Document", pos: int, by_pos: dict[int, list[Path]]) -> None:
        geom = self._geometry.get(pos)
        if geom is None:  # no begin_page for it: borrow any part's page size
            with fitz.open(str(next(iter(by_pos.values()))[0])) as src:
                p = src[0]
                geom = (p.cropbox.width, p.cropbox.height, p.rotation)
        width, height, rotation = geom
        page = out.new_page(width=width, height=height)
        if rotation:
            page.set_rotation(rotation)


def _insert_merged(out: "fitz.Document", parts: "list[Path]") -> None:
    """Append `parts[0]`'s page to `out`, then overlay every further part on it. Both sides are
    overlaid at rotation 0 (the parts share the page's unrotated geometry) and the target's
    `/Rotate` is restored afterwards."""
    with fitz.open(str(parts[0])) as first:
        out.insert_pdf(first)
    if len(parts) == 1:
        return
    page = out[-1]
    rotation = page.rotation
    page.set_rotation(0)
    for part in parts[1:]:
        with fitz.open(str(part)) as src:
            src[0].set_rotation(0)
            page.show_pdf_page(page.rect, src, 0)
    if rotation:
        page.set_rotation(rotation)


def _has_content(doc: "fitz.Document") -> bool:
    """True if any page draws anything (a non-empty content stream)."""
    return any(any((doc.xref_stream(x) or b"").strip() for x in page.get_contents()) for page in doc)


def page_geometry(pdf_path: "str | Path", page_index: int) -> tuple[float, float, int]:
    """`(width, height, rotation)` of one page -- the unrotated CropBox size (= `PageMeta.width /
    height`) + `/Rotate` -- for `LayerWriter.begin_page`, read before the run."""
    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_index]
        return page.cropbox.width, page.cropbox.height, page.rotation


def merge_pdfs(page_bytes: list[bytes], out_path: Path) -> None:
    with fitz.open() as out:
        for b in page_bytes:
            with fitz.open("pdf", b) as src:
                out.insert_pdf(src)
        out.save(str(out_path))


_PAGE_COUNT_SUFFIX = re.compile(r"\s*\(\d[^()]*\)$")


def strip_page_count(label: str) -> str:
    """Drop a trailing per-page count (`"word groups (25)"`) from a layer label: it differs per
    page, and a different label is a different layer file."""
    return _PAGE_COUNT_SUFFIX.sub("", label)


def debug_layer_sink(writer: LayerWriter) -> "Callable[[str, str, str, bytes], None]":
    """The `RunSettings.on_debug_layer` callback: each layer goes to `writer` the moment a backend
    renders it."""

    def _sink(stage: str, label: str, hexc: str, pdf_bytes: bytes) -> None:
        label = strip_page_count(label)
        fname = f"{stage}__{layer_slug(label)}.pdf"
        writer.add(fname, {"stage": stage, "layer": label, "file": fname, "color": hexc}, pdf_bytes)

    return _sink


def reconstructed_layer(writer: LayerWriter, res: "PipelineResult") -> None:
    """The final output as a page, through Phase 4's `render_output_pdf`."""
    fname = "reconstructed__page.pdf"
    writer.add(
        fname, {"stage": "reconstructed", "layer": "page", "file": fname, "color": RECONSTRUCTED_COLOR},
        render_output_pdf(res.page.meta, list(res.texts), list(res.vectors)),
    )


# ---------------------------------------------------------------------------
# Debug images
# ---------------------------------------------------------------------------
class ImageReservoir:
    """Disk-backed reservoir sample (Algorithm R) of at most `cap` images into `folder`: the n-th
    image offered is kept with probability cap/n, replacing (and deleting) a random earlier pick.
    `make_image` is only called for an image that is kept. `cap=None` keeps everything. The
    folder is created on the first kept image."""

    def __init__(self, folder: Path, cap: "int | None" = 100, seed: int = 0) -> None:
        self.folder = Path(folder)
        self.cap = cap
        self.seen = 0
        self.kept: list[Path] = []
        self._rng = random.Random(seed)

    def offer(self, name: str, make_image: "Callable[[], Image.Image]") -> bool:
        self.seen += 1
        slot = None
        if self.cap is not None and len(self.kept) >= self.cap:
            slot = self._rng.randrange(self.seen)
            if slot >= self.cap:
                return False
        image = make_image()
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / name
        image.save(path)
        if slot is None:
            self.kept.append(path)
        else:
            if self.kept[slot] != path:
                self.kept[slot].unlink(missing_ok=True)
            self.kept[slot] = path
        return True


class ImageReservoirs:
    """One `ImageReservoir` per debug-image folder of one document, created when a backend first
    offers an image to that folder. Seeded from the document name + folder name, so a rerun keeps
    the same images."""

    def __init__(self, doc_dir: Path, cap: "int | None") -> None:
        self.doc_dir = doc_dir
        self.cap = cap
        self._by_folder: dict[str, ImageReservoir] = {}

    def __getitem__(self, folder: str) -> ImageReservoir:
        if folder not in self._by_folder:
            seed = zlib.crc32(f"{self.doc_dir.name}/{folder}".encode("utf-8"))
            self._by_folder[folder] = ImageReservoir(self.doc_dir / folder, self.cap, seed)
        return self._by_folder[folder]


def debug_image_sink(reservoirs: ImageReservoirs, page_index: int):
    """The `RunSettings.on_debug_image` callback: `(folder, name, make_image)` goes straight to
    that folder's reservoir, prefixed with the page (a run on a converted one-page PDF only
    learns its real page index from here)."""

    def _sink(folder: str, name: str, make_image) -> None:
        reservoirs[folder].offer(f"p{page_index}_{name}", make_image)

    return _sink


# ---------------------------------------------------------------------------
# Run folder
# ---------------------------------------------------------------------------
def write_hyperparams(path: Path, config: "ReportConfig", settings: "RunSettings", config_path: Path) -> None:
    run_settings = {
        k: str(v) for k, v in vars(settings).items() if not k.startswith("on_")
    }
    path.write_text(
        "# Source config\n" + str(config_path.resolve())
        + "\n\n# Run config\n" + config.model_dump_json(indent=2)
        + "\n\n# RunSettings\n" + json.dumps(run_settings, indent=2) + "\n",
        encoding="utf-8",
    )


def restamp_page(res: "PipelineResult", page_index: int) -> None:
    """A run on a per-page converted PDF reports page 0; stamp the real source page back on so
    dump.json, overlays and scores key off the right page."""
    res.page.meta.index = page_index
    res.page.meta.number = page_index + 1


def finalize_doc_dir(
    doc_dir: Path, source_pdf: Path, pages: list[int], config: "ReportConfig",
    writer: LayerWriter, dumps: list[dump_io.PageDump],
    *, extra_layers: tuple[dict, ...] = (), doc_durations: "dict | None" = None,
) -> None:
    """Writes the layer PDFs, `dump.json` and `manifest.json`. `doc_durations` gains
    `layer_save`."""
    import time

    doc_durations = dict(doc_durations or {})
    layer_filenames = writer.filenames()
    start = time.perf_counter()
    writer.finalize(doc_dir)
    doc_durations["layer_save"] = time.perf_counter() - start

    dump_io.write_dump(doc_dir / "dump.json", str(source_pdf), dumps, doc_durations)
    (doc_dir / "manifest.json").write_text(json.dumps({
        "source_pdf": str(source_pdf),
        "pages": pages,
        "variant": f"{config.p2}:{config.p3}",
        "p2": config.p2,
        "p3": config.p3,
        "final_stage": config.final_stage,
        "vectorised": config.vectorise or config.benchmark,
        "layers": [writer.meta[f] for f in layer_filenames] + list(extra_layers),
    }, indent=2), encoding="utf-8")
    _LOG.info("wrote %s", doc_dir)


# ---------------------------------------------------------------------------
# Benchmark overlays
# ---------------------------------------------------------------------------
def write_label_overlays(
    doc_dir: Path, text_type: str, gt: LabelSet, dumps: list[dump_io.PageDump],
    cfg: metrics.MetricConfig,
) -> None:
    """`<text_type>_bbox.pdf` (GT boxes, green = reached by a prediction / red = missed) and
    `<text_type>_text.pdf` (GT text, per word green / yellow / red by read accuracy), one page
    per report page."""
    gt_regions = adapters.gt_regions_from_labelset(gt)
    bbox_pages: list[bytes] = []
    text_pages: list[bytes] = []
    for pd in dumps:
        pi = pd.page_meta.index
        regions = [g for g in gt_regions if g.page_index == pi]
        preds = adapters.predictions_from_texts([t for t in pd.texts if t.source == "ocr"])
        graph = metrics.build_overlap_graph(regions, preds, cfg)
        bbox_pages.append(render_boxes_pdf(pd.page_meta, label_overlays.gt_bbox_overlay(graph)))
        text_pages.append(render_text_pdf(pd.page_meta, label_overlays.gt_word_overlay(graph)))
    merge_pdfs(bbox_pages, doc_dir / f"{text_type}_bbox.pdf")
    merge_pdfs(text_pages, doc_dir / f"{text_type}_text.pdf")


_EXTRA_LAYERS = (
    ("extra text", "#dc2626"),
    ("extra vectors", "#f97316"),
    ("missed vectors", "#9333ea"),
)


def _rgb01(hexc: str) -> tuple[float, float, float]:
    h = hexc.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def text_routed_vectors(input_vectors: "list[Vector]", output_vectors: "list[Vector]") -> "list[Vector]":
    """The input vectors Phase 3 did not keep as drawing (it classified them as text), matched
    by `path_signature`. Phase 2's traced vectors have no input counterpart and never count."""
    kept = {path_signature(v) for v in output_vectors}
    return [v for v in input_vectors if path_signature(v) not in kept]


def add_extra_prediction_layers(
    writer: LayerWriter, page_meta: "PageMeta", texts: "list[Text]", routed: "list[Vector]",
    drawing: "list[Vector]", gt_boxes: list, manual_boxes: list,
) -> None:
    """This page's `benchmark__extra_text` / `extra_vectors` / `missed_vectors` layers (always
    one page each, so every layer stays page-aligned; an all-blank layer is dropped)."""
    extra_texts, extra_vectors, missed_vectors = label_overlays.extra_predictions(
        texts, routed, drawing, gt_boxes, manual_boxes,
    )
    rendered = (
        render_text_pdf(page_meta, extra_texts, color_of=lambda _t: _rgb01(_EXTRA_LAYERS[0][1])),
        render_vectors_pdf(page_meta, extra_vectors, color_of=lambda _v: _rgb01(_EXTRA_LAYERS[1][1])),
        render_vectors_pdf(page_meta, missed_vectors, color_of=lambda _v: _rgb01(_EXTRA_LAYERS[2][1])),
    )
    for (label, hexc), pdf_bytes in zip(_EXTRA_LAYERS, rendered):
        fname = f"benchmark__{layer_slug(label)}.pdf"
        writer.add(fname, {"stage": "benchmark", "layer": label, "file": fname, "color": hexc}, pdf_bytes)
