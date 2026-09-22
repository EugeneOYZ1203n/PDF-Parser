"""Artifact-writing for `generate_pipeline_report.py`: which stages render
what (`_active_artifacts`), the per-page/per-layer PDF + stats accumulation
(`_accumulate_page`, `_LayerWriter`, `_debug_layer_sink`), the run-folder
finalization (`_finalize_doc_dir`), and the GT-overlay writer
(`_write_label_overlays`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pymupdf as fitz

import rastervec.config as rvconfig
from rastervec.Evaluation import dump_io
from rastervec.Evaluation.Evaluate import adapters, label_overlays, metrics
from rastervec.Evaluation.Labelling.label_schema import LabelSet
from rastervec.Evaluation.Report import stage_stats
from rastervec.commons.logging_setup import get_logger
from rastervec.commons.renderer import render_boxes_pdf, render_reconstructed_pdf, stages

from scripts.debug_image_savers import (
    _save_fastintopaddle_detect_images,
    _save_fastintopaddle_recog_images,
    _save_fastintopaddle_tile_images,
    _save_legacyrecreation_ocr_images,
    _save_vectorclassification_detect_images,
    _save_vectorclassification_recog_images,
    _save_vectorclassification_tile_images,
)
from scripts.report_config import NEW_STEP_NAMES, _layer_slug

if TYPE_CHECKING:
    from scripts.report_config import ReportConfig

_LOG = get_logger("generate_pipeline_report")

# The old engine's fixed 9-step name list (`pipelines/current.py`), still
# used ONLY for `pipeline: "legacy"`'s hardcoded reconstructed-only
# artifact row (which never actually gates on a step name). The new
# `core.pipeline` engine (`pipeline: "current"`) uses its own short,
# generic step list instead -- `NEW_STEP_NAMES` (`report_config.py`).
_ARTIFACTS: list[tuple[str, str, str | None, str]] = [
    ("native_text", "native", "native", "native"),
    ("vector_extraction", "vectors", "vectors", "vectors"),
    ("similarity", "similarity", "similarity", "similarity"),
    ("fast_heatmap", "fast", "fast", "fast"),
    ("reclassify", "reclassify", "reclassify", "reclassify"),
    ("separation", "separation", "separation", "separation"),
    ("clusters", "clusters", "clusters", "clusters"),
    ("paddle_detect", "paddle_detect", "paddle_detect", "paddle_detect"),
    ("assignment", "assignment", "assignment", "assignment"),
    ("rotate", "rotate", "rotate", "rotate"),
    ("paddle_ocr", "ocr", "ocr", "ocr"),
    ("drawing_vectors", "drawing", None, "drawing"),
    ("reconstructed", "reconstructed", None, "drawing"),
]

# The new `core.pipeline` engine's artifact set -- deliberately small and
# generic (see `commons/renderer/stages.py`'s "phase1"/"phase2"/"final"
# branches): no per-backend stats files (stats_key=None throughout), no
# partial-run support (the new orchestrator always runs phase1->p2->p3 in
# full; `final_stage` here only trims which of these 4 rows get rendered).
_NEW_ARTIFACTS: list[tuple[str, str, str | None, str]] = [
    ("phase1", "phase1", None, "phase1"),
    ("phase2", "phase2", None, "phase2"),
    ("final", "final", None, "phase3"),
    ("reconstructed", "reconstructed", None, "phase3"),
]

RASTER_TEXT_TYPES = ("vector_to_raster", "original_raster", "native_to_raster")


def _reached(step: str, final_stage: str | None) -> bool:
    if final_stage is None or final_stage not in NEW_STEP_NAMES:
        return True
    if step not in NEW_STEP_NAMES:
        return True
    return NEW_STEP_NAMES.index(step) <= NEW_STEP_NAMES.index(final_stage)


def _merge_pdfs(page_bytes: list[bytes], out_path: Path) -> None:
    out = fitz.open()
    try:
        for b in page_bytes:
            src = fitz.open("pdf", b)
            try:
                out.insert_pdf(src)
            finally:
                src.close()
        out.save(str(out_path))
    finally:
        out.close()


class _LayerWriter:
    """Incremental per-layer multi-page PDF writer, replacing the old
    "accumulate every page's rendered bytes for the whole document in a
    `dict[str, list[bytes]]`, then merge everything in one pass at the
    end" shape (`_merge_pdfs`, now used only for the smaller GT-overlay
    outputs). Each call to `add` inserts that one page's bytes into its
    layer's own already-open `fitz.Document` immediately and drops the
    bytes right after -- so a layer's pages never sit duplicated in both a
    growing Python list and a second, later re-parse pass. Combined with
    each backend's own `on_debug_layer` streaming (see `core/registry.py`),
    a debug layer's underlying heavy source data (render crops, masks) is
    never held any longer than that one page/step's own rendering needs
    it."""

    def __init__(self) -> None:
        self._docs: dict[str, "fitz.Document"] = {}
        self.meta: dict[str, dict] = {}

    def add(self, fname: str, meta: dict, pdf_bytes: bytes) -> None:
        doc = self._docs.get(fname)
        if doc is None:
            doc = fitz.open()
            self._docs[fname] = doc
        src = fitz.open("pdf", pdf_bytes)
        try:
            doc.insert_pdf(src)
        finally:
            src.close()
        self.meta.setdefault(fname, meta)

    def filenames(self) -> list[str]:
        return list(self._docs)

    def finalize(self, doc_dir: Path) -> None:
        for fname, doc in self._docs.items():
            doc.save(str(doc_dir / fname))
            doc.close()
        self._docs.clear()


def _write_hyperparams(path: Path, config: "ReportConfig", variant, config_path: Path) -> None:
    lines = [
        "# Source config\n", str(config_path.resolve()), "\n\n",
        "# Run config\n", config.model_dump_json(indent=2), "\n\n# Variant\n",
    ]
    lines.append(json.dumps(
        {
            "name": variant.name, "engine": variant.engine,
            "p2": variant.p2, "p3": variant.p3, "enable_fast": variant.enable_fast,
        },
        indent=2,
    ))
    lines.append("\n\n# rastervec.config constants\n")
    for name in sorted(vars(rvconfig)):
        if name.isupper():
            lines.append(f"{name} = {getattr(rvconfig, name)!r}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _restamp_page(res, page_index: int) -> None:
    """A run on a per-page *converted* PDF reports `page_meta.index == 0`.
    Stamp the real source page index back on so every `dump.json` PageDump /
    label overlay / benchmark score keys off the right page."""
    if res.page is not None and res.page.meta is not None:
        res.page.meta.index = page_index
        res.page.meta.number = page_index + 1


def _active_artifacts(config: "ReportConfig", variant) -> list[tuple]:
    if variant.engine == "legacy":
        return [row for row in _ARTIFACTS if row[0] == "reconstructed"]
    return [row for row in _NEW_ARTIFACTS if _reached(row[3], config.final_stage)]


def _accumulate_page(
    res, page_index: int, active: list[tuple],
    writer: _LayerWriter,
    stats_pages: dict[str, list[tuple[int, dict]]],
    detect_dir: Path, recog_dir: Path, fast_tile_dir: Path, ocr_dir: Path,
    *, is_legacy: bool = False, p3: str = "",
) -> None:
    """Render every active stage's fixed layer PDFs (phase1/phase2/final/
    reconstructed -- these need the whole, finished `res`, so they're
    necessarily rendered post-hoc rather than streamed) + numeric stats for
    one page, writing each layer into `writer` immediately; also dump each
    backend's own pre-OCR debug images (what PaddleOCR's own text detector /
    recognizer / FAST saw), reading them from `res.extra["p3_debug"]` --
    each P3 backend's own folder set differs, see
    `generate_pipeline_report.py`'s module docstring.
    Per-backend debug *layers* (the heavier, genuinely streamable PDF
    overlays) are NOT handled here -- see `_debug_layer_sink` / the
    `on_debug_layer` callback passed straight into `run_pipeline`."""
    for stem, stage_key, stats_key, _gate in active:
        try:
            layers = stages.render_stage_layers(res, stage_key)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("%s render failed for page %d: %s", stem, page_index, exc)
            layers = []
        for label, hexc, pdf_bytes in layers:
            fname = f"{stem}__{_layer_slug(label)}.pdf"
            writer.add(fname, {"stage": stem, "layer": label, "file": fname, "color": hexc}, pdf_bytes)
        if stats_key is not None:
            stats_pages[stem].append(
                (page_index, stage_stats.stats_for_stage(res, stats_key))
            )
    if is_legacy:
        return
    p3_debug = (res.extra or {}).get("p3_debug") or {}
    if p3 == "FastIntoPaddle":
        _save_fastintopaddle_tile_images(p3_debug, fast_tile_dir, page_index)
        _save_fastintopaddle_detect_images(p3_debug, detect_dir, page_index)
        _save_fastintopaddle_recog_images(p3_debug, recog_dir, page_index)
    elif p3 == "VectorClassification":
        _save_vectorclassification_tile_images(p3_debug, fast_tile_dir, page_index)
        _save_vectorclassification_detect_images(p3_debug, detect_dir, page_index)
        _save_vectorclassification_recog_images(p3_debug, recog_dir, page_index)
    elif p3 == "LegacyRecreation":
        _save_legacyrecreation_ocr_images(p3_debug, ocr_dir, page_index)


def _debug_layer_sink(writer: _LayerWriter):
    """Builds the `on_debug_layer` callback passed straight into
    `run_pipeline` for the `current` engine: each backend calls this the
    moment it renders one of its own debug layers (interleaved with its
    normal computation -- see `core/registry.py`'s `P2_RENDER_DEBUG`/
    `P3_RENDER_DEBUG` docstring), so a layer reaches `writer` -- and the
    backend's own heavier step-local data (render crops, masks) can be
    dropped -- well before the rest of that page's pipeline run finishes,
    let alone the whole document's page loop."""

    def _sink(stage: str, label: str, hexc: str, pdf_bytes: bytes) -> None:
        fname = f"{stage}__{_layer_slug(label)}.pdf"
        writer.add(fname, {"stage": stage, "layer": label, "file": fname, "color": hexc}, pdf_bytes)

    return _sink


def _finalize_doc_dir(
    doc_dir: Path, source_pdf: Path, pages: list[int], config: "ReportConfig", variant,
    active: list[tuple], writer: _LayerWriter,
    stats_pages: dict[str, list[tuple[int, dict]]], dumps: list[dump_io.PageDump],
    *, extra_layers: tuple[dict, ...] = (),
) -> None:
    layer_filenames = writer.filenames()
    writer.finalize(doc_dir)

    for stem, _sk, stats_key, _gate in active:
        if stats_key is None:
            continue
        body = "".join(
            f"\n## page {pi}\n{stage_stats.format_stats(stats_key, data)}"
            for pi, data in stats_pages[stem]
        )
        (doc_dir / f"{stem}.txt").write_text(
            f"# {stats_key} stats for {Path(source_pdf).name}\n{body}", encoding="utf-8"
        )

    dump_io.write_dump(doc_dir / "dump.json", str(source_pdf), dumps)
    (doc_dir / "manifest.json").write_text(json.dumps({
        "source_pdf": str(source_pdf),
        "pages": pages,
        "engine": variant.engine,
        "variant": variant.name,
        "final_stage": config.final_stage,
        "vectorised": config.vectorise or config.benchmark,
        "layers": [writer.meta[f] for f in layer_filenames] + list(extra_layers),
    }, indent=2), encoding="utf-8")
    _LOG.info("wrote %s", doc_dir)


def _write_label_overlays(
    doc_dir: Path, source: str, gt: LabelSet, dumps: list[dump_io.PageDump],
    cfg: metrics.MetricConfig,
) -> None:
    """`<source>_bbox.pdf` (GT boxes green=covered by a prediction / red=missed)
    and `<source>_text.pdf` (GT text, per word green/yellow/red by read
    accuracy) -- one page per report page. `source` is a text-type name;
    `vector_to_raster`/`original_raster`/`native_to_raster` are scored
    against that page's `raster_texts` (the SEPARATE rasterised-PDF run),
    every other type against `texts` (the vectorised-PDF run) -- never the
    other's predictions."""
    gt_regions = adapters.gt_regions_from_labelset(gt)
    bbox_pages: list[bytes] = []
    text_pages: list[bytes] = []
    use_raster = source in RASTER_TEXT_TYPES
    for pd in dumps:
        pi = pd.page_meta.index
        regions = [g for g in gt_regions if g.page_index == pi]
        run_texts = pd.raster_texts if use_raster else pd.texts
        preds = adapters.predictions_from_texts(
            [t for t in run_texts if t.source == "ocr"]
        )
        graph = metrics.build_overlap_graph(regions, preds, cfg)
        bbox_pages.append(
            render_boxes_pdf(pd.page_meta, label_overlays.gt_bbox_overlay(graph))
        )
        text_pages.append(
            render_reconstructed_pdf(
                pd.page_meta, text_boxes=label_overlays.gt_word_overlay(graph)
            )
        )
    _merge_pdfs(bbox_pages, doc_dir / f"{source}_bbox.pdf")
    _merge_pdfs(text_pages, doc_dir / f"{source}_text.pdf")
