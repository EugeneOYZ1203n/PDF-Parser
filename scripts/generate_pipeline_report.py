"""Config-driven pipeline stage report generator.

Reads a JSON config, runs the pipeline once per (pdf, page), and writes a
timestamped run folder:

    outputs/pipeline_report/<ts>__<config-stem>/
        config_and_hyperparameters.txt
        <pdf-stem>/
            manifest.json
            <stage>__<layer>.pdf                       (one multi-page PDF per
                                                       visual layer, e.g.
                                                       vector_classification__group_bbox.pdf;
                                                       the viewer toggles each)
            native_text.txt ...                       (one stats file per stage)
            dump.json                                 (every Text + Vector, reloadable)
            paddle_detect_images/  (one PNG per cluster: exactly what
                                   PaddleOCR's text *detector* saw -- the
                                   cluster's own single render -- with every
                                   detected box drawn on top)
            paddle_recog_images/   (one PNG per rotated detection crop: exactly
                                   what PaddleOCR's text *recognizer* saw, no
                                   dedup/election, recognised text in the
                                   filename)

Replaces `rastervec/notebooks/pipeline_stage_visualization.ipynb`.

With `benchmark: true` the same `<pdf-stem>/` folder is also a scoring
artifact for `pipeline_report_benchmark.py`: one `convert_page_to_vector_text`
run per page (predictions for `native_to_vector`/`original_vector`) plus,
when the input is a master_label.py folder with its own `rasterised.pdf`, a
SEPARATE pipeline run directly on that rasterised page (predictions for
`vector_to_raster`/`original_raster`/`native_to_raster` -- these 3 types are
never scored from the vectorised run). Writes `ground_truth_<text_type>.json`
(only the non-empty of the 5 text types) and the matching
`<text_type>_{bbox,text}.pdf` overlays (registered in the manifest under
stage `benchmark`), and a run-root `benchmark.json` marker.
`input_files`/`input_dir` entries may be a `.pdf`
(auto-only), a `.json` label sidecar, or a directory (a
`scripts/label/master_label.py` output folder, merging its
`native_labels.json`/`vector_labels.json`/`raster_labels.json`). See
`docs/SCRIPTS.md`.

    .venv/Scripts/python.exe scripts/generate_pipeline_report.py --config run.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, field_validator, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rastervec.config as rvconfig
from rastervec.Evaluation import conversion, dump_io
from rastervec.Evaluation.Evaluate import adapters, label_overlays, metrics
from rastervec.Evaluation.Evaluate.variants import PipelineVariant, resolve_variant
from rastervec.Evaluation.Labelling import native_label
from rastervec.Evaluation.Evaluate.metrics import TEXT_TYPES
from rastervec.Evaluation.Labelling.label_schema import (
    LabelEntry,
    LabelSet,
    load_labels,
    load_labels_from_master_folder,
    save_labels,
)
from rastervec.commons.helpers.geometry import union_bbox
from rastervec.Evaluation.Report import stage_stats
from rastervec.commons.logging_setup import configure_logging, get_logger
from rastervec.commons.paths import output_dir
from rastervec.core.registry import P2_REGISTRY, P3_REGISTRY, DEFAULT_P2, DEFAULT_P3
from rastervec.commons.renderer import render_boxes_pdf, render_reconstructed_pdf, stages

# The old engine's fixed 9-step name list (`pipelines/current.py`), still
# used ONLY for `pipeline: "legacy"`'s hardcoded reconstructed-only
# artifact row (which never actually gates on a step name). The new
# `core.pipeline` engine (`pipeline: "current"`) uses its own short,
# generic step list instead -- see `NEW_STEP_NAMES` below.
NEW_STEP_NAMES = ["phase1", "phase2", "phase3"]

_LOG = get_logger("generate_pipeline_report")

_CONVERT = {
    "text_only": conversion.convert_page_text_only,
    "drawings_only": conversion.convert_page_drawings_only,
    "to_vector_text": conversion.convert_page_to_vector_text,
}

# stage stem -> (stage_key for stages.render_stage_layers, stats-key or
# None, gate STEP_NAME). Each stage emits one single-purpose PDF per visual
# layer: `<stem>__<layer-slug>.pdf`. Only ever used for `pipeline: "legacy"`
# now (`_active_artifacts` hardcodes it to the "reconstructed" row alone) --
# kept as-is/dead otherwise since the old engine it fully described
# (`pipelines/current.py`) is no longer what `pipeline: "current"` runs.
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


def _layer_slug(label: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in label.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_") or "layer"


class ReportConfig(BaseModel):
    # `pipeline` is the top-level engine axis: "current" (the new pluggable
    # core.pipeline, P1 -> p2 -> p3) or "legacy" (archive/raster_parser,
    # unchanged, p2/p3 ignored). `p2`/`p3` only apply when
    # `pipeline == "current"` -- they select a `core.registry.P2_REGISTRY`/
    # `P3_REGISTRY` backend by name.
    pipeline: Literal["current", "legacy"] = "current"
    p2: str = DEFAULT_P2
    p3: str = DEFAULT_P3
    enable_fast: bool = True
    final_stage: str | None = None
    input_dir: Path | None = None
    input_files: list[Path] = Field(default_factory=list)
    label_files: dict[str, Path] = Field(default_factory=dict)
    pages: list[int] | dict[str, list[int]] | None = None
    vectorise: bool = False
    vectorise_mode: str = "to_vector_text"
    benchmark: bool = False
    iou_edge_min: float = metrics.MetricConfig().iou_edge_min
    dpi: int = 300
    output_root: Path | None = None

    @field_validator("p2")
    @classmethod
    def _known_p2(cls, v: str) -> str:
        if v not in P2_REGISTRY:
            raise ValueError(f"p2 must be one of {sorted(P2_REGISTRY)}")
        return v

    @field_validator("p3")
    @classmethod
    def _known_p3(cls, v: str) -> str:
        if v not in P3_REGISTRY:
            raise ValueError(f"p3 must be one of {sorted(P3_REGISTRY)}")
        return v

    @field_validator("final_stage")
    @classmethod
    def _known_stage(cls, v: str | None) -> str | None:
        # Validated against the new engine's short step list regardless of
        # `pipeline` (legacy's `_active_artifacts` never consults
        # `final_stage` at all, so it's a harmless no-op there too).
        if v is not None and v not in NEW_STEP_NAMES:
            raise ValueError(f"final_stage must be one of {NEW_STEP_NAMES}")
        return v

    @field_validator("vectorise_mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        if v not in _CONVERT:
            raise ValueError(f"vectorise_mode must be one of {list(_CONVERT)}")
        return v

    @model_validator(mode="after")
    def _benchmark_excludes_vectorise(self) -> "ReportConfig":
        if self.benchmark and self.vectorise:
            raise ValueError(
                "benchmark mode runs its own text-only / drawings-only conversions; "
                "set vectorise=false"
            )
        return self

    def resolved_pdfs(self) -> list[Path]:
        pdfs: list[Path] = list(self.input_files)
        if self.input_dir is not None:
            pdfs += sorted(Path(self.input_dir).glob("*.pdf"))
        seen, out = set(), []
        for p in pdfs:
            p = Path(p).resolve()
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def pages_for(self, stem: str) -> list[int]:
        if self.pages is None:
            return [0]
        if isinstance(self.pages, dict):
            return list(self.pages.get(stem, self.pages.get("*", [0])))
        return list(self.pages)

    def benchmark_inputs(self) -> list["BenchInput"]:
        """One `BenchInput` per config input, benchmark mode only. A `.json`
        input (or a pdf with a `label_files` entry) carries ground truth and
        is keyed `labels:<json-stem>`; a directory (a `scripts/label/
        master_label.py` output folder) is keyed `labels:<folder-stem>`,
        `pdf_path` from the folder's own `original.pdf`, `labels_path` the
        directory itself; a bare `.pdf` is auto-only and keyed
        `pdf:<pdf-stem>`. The key is what the benchmark script matches
        shared inputs on."""
        out: list[BenchInput] = []
        seen: set[str] = set()

        def _add(key: str, pdf: Path, labels: Path | None, rasterised: Path | None = None) -> None:
            if key not in seen:
                seen.add(key)
                out.append(BenchInput(
                    key=key, pdf_path=pdf.resolve(), labels_path=labels,
                    rasterised_pdf_path=rasterised.resolve() if rasterised else None,
                ))

        raw: list[Path] = list(self.input_files)
        if self.input_dir is not None:
            raw += sorted(Path(self.input_dir).glob("*.pdf"))
        for item in raw:
            item = Path(item)
            if item.is_dir():
                folder = item.resolve()
                labels = load_labels_from_master_folder(folder)
                pdf = Path(labels.pdf_path)
                rasterised = folder / "rasterised.pdf"
                _add(f"labels:{folder.stem}", pdf, folder, rasterised if rasterised.is_file() else None)
                continue
            if item.suffix.lower() == ".json":
                labels = item.resolve()
                pdf = Path(load_labels(str(labels)).pdf_path)
                _add(f"labels:{labels.stem}", pdf, labels)
                continue
            paired = self.label_files.get(item.stem)
            if paired is not None:
                paired = Path(paired).resolve()
                _add(f"labels:{paired.stem}", item, paired)
            else:
                _add(f"pdf:{item.stem}", item, None)
        return out


class BenchInput(NamedTuple):
    key: str
    pdf_path: Path
    labels_path: Path | None
    # A master_label.py folder's own `rasterised.pdf` (per-page flattened-
    # to-image renders, 1:1 page-index-aligned with `original.pdf`) -- when
    # present, `_process_pdf_benchmark` runs the pipeline a SECOND time per
    # page directly on this file, and vector_to_raster/original_raster/
    # native_to_raster are scored from that run's own OCR output instead of
    # the vectorised run's. `None` for a bare `.pdf`/`.json` input (no
    # rasterised counterpart known) -- those 3 types stay GT-only.
    rasterised_pdf_path: Path | None = None


ReportConfig.model_rebuild()


def _reached(step: str, final_stage: str | None) -> bool:
    if final_stage is None or final_stage not in NEW_STEP_NAMES:
        return True
    if step not in NEW_STEP_NAMES:
        return True
    return NEW_STEP_NAMES.index(step) <= NEW_STEP_NAMES.index(final_stage)


def _safe_slug(text: str, limit: int = 40) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in (text or "")).strip("_")
    return keep[:limit] or "blank"


def _draw_boxes(img: Image.Image, boxes, outline=(220, 30, 30), width=2) -> Image.Image:
    out = img.convert("RGB")
    d = ImageDraw.Draw(out)
    for b in boxes:
        if b is None:
            continue
        x0, y0, x1, y1 = b
        d.rectangle(
            [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
            outline=outline, width=width,
        )
    return out


def _save_paddle_detect_inputs(res, folder: Path, page_index: int) -> int:
    """One PNG per cluster -- exactly what `PaddleDetectBackend.
    detect_on_cluster` saw (`ClusterDetection.image`, already white-padded),
    with every detected box drawn on top. Each `PaddleDetection.bbox` is
    page space, so it must be mapped back into that render's own pixel
    space via `page_points_to_pixel` using the *same* `padding` the render
    itself used (`cluster_render_padding`), then shifted by the render's
    own white-pad offset (`cd.pad_x_px`/`cd.pad_y_px`) -- getting either
    wrong silently misaligns the overlay boxes."""
    clusters = getattr(res, "spatial_clusters", None) or []
    cluster_detections = getattr(res, "cluster_detections", None) or []
    if not clusters or not cluster_detections:
        return 0
    from rastervec.pipelines._steps import cluster_render_padding
    from rastervec.commons.renderer import page_points_to_pixel

    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, cd in enumerate(cluster_detections):
        if cd is None or cd.image is None:
            continue
        cluster = clusters[i] if i < len(clusters) else []
        padding = cluster_render_padding(cluster) if cluster else 0.0
        img = Image.fromarray(np.asarray(cd.image)[..., ::-1])  # BGR -> RGB
        boxes = []
        for det in cd.detections:
            (px0, py0), (px1, py1) = page_points_to_pixel(
                cluster, cd.dpi, [(det.bbox[0], det.bbox[1]), (det.bbox[2], det.bbox[3])],
                padding=padding,
            )
            boxes.append((px0 + cd.pad_x_px, py0 + cd.pad_y_px, px1 + cd.pad_x_px, py1 + cd.pad_y_px))
        img = _draw_boxes(img, boxes, outline=(220, 30, 30))
        img.save(folder / f"p{page_index}_cluster_{i:03d}.png")
        n += 1
    return n


def _save_paddle_recog_inputs(res, folder: Path, page_index: int) -> int:
    """One PNG per rotated detection crop -- the exact crop handed to
    PaddleOCR recognition (`Segment.image`), no election/dedup, recognised
    text in the filename."""
    segs = getattr(res, "rotated_segments", None) or []
    if not segs:
        return 0
    texts = getattr(res, "restored_texts", None) or []
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, seg in enumerate(segs):
        if seg.image is None:
            continue
        img = Image.fromarray(np.asarray(seg.image))
        rec = texts[i].text if i < len(texts) else ""
        img.save(folder / f"p{page_index}_word_{i:03d}__{_safe_slug(rec)}.png")
        n += 1
    return n


def _save_fast_tile_images(res, folder: Path, page_index: int) -> int:
    """One PNG per FAST tile -- the exact crop of the whole-page FAST render
    (`FastPageResult.page_image`, which is `debug_image_scale`-downsampled
    from the full-res image `FastDetector.detect`/`_detect_job` actually
    saw for that tile) matching that tile's page-space rect (`all_tiles`)."""
    fr = getattr(res, "fast_result", None)
    tiles = getattr(fr, "all_tiles", None) if fr is not None else None
    if fr is None or fr.page_image is None or not tiles:
        return 0
    from rastervec.config import FAST_PAGE_RENDER_DPI, FAST_TILE_SCALE_FACTOR
    from rastervec.commons.helpers.geometry import PDF_POINTS_PER_INCH

    debug_scale = getattr(fr, "debug_image_scale", 1.0)
    zoom = (FAST_PAGE_RENDER_DPI * FAST_TILE_SCALE_FACTOR) / PDF_POINTS_PER_INCH * debug_scale
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, rect in enumerate(tiles):
        x0, y0, x1, y1 = (c * zoom for c in rect)
        crop = fr.page_image.crop((int(x0), int(y0), int(x1), int(y1)))
        crop.save(folder / f"p{page_index}_tile_{i:03d}.png")
        n += 1
    return n


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


def _write_hyperparams(path: Path, config: ReportConfig, variant) -> None:
    lines = ["# Run config\n", config.model_dump_json(indent=2), "\n\n# Variant\n"]
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


def _active_artifacts(config: ReportConfig, variant) -> list[tuple]:
    if variant.engine == "legacy":
        return [row for row in _ARTIFACTS if row[0] == "reconstructed"]
    return [row for row in _NEW_ARTIFACTS if _reached(row[3], config.final_stage)]


def _accumulate_page(
    res, page_index: int, active: list[tuple],
    writer: _LayerWriter,
    stats_pages: dict[str, list[tuple[int, dict]]],
    detect_dir: Path, recog_dir: Path, fast_tile_dir: Path,
    *, is_legacy: bool = False,
) -> None:
    """Render every active stage's fixed layer PDFs (phase1/phase2/final/
    reconstructed -- these need the whole, finished `res`, so they're
    necessarily rendered post-hoc rather than streamed) + numeric stats for
    one page, writing each layer into `writer` immediately; also dump the
    paddle detect/recog PNG debug images (what PaddleOCR's own text
    detector / recognizer saw). Per-backend debug layers (the heavier,
    genuinely streamable ones) are NOT handled here -- see
    `_debug_layer_sink` / the `on_debug_layer` callback passed straight
    into `run_pipeline`."""
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
    if not is_legacy:
        # These read old-engine-only verbose fields via `getattr(..., None)`
        # defaults, so they no-op harmlessly (return 0) for the new
        # core.pipeline engine's `PipelineResult`, which has none of them.
        _save_paddle_detect_inputs(res, detect_dir, page_index)
        _save_paddle_recog_inputs(res, recog_dir, page_index)
        _save_fast_tile_images(res, fast_tile_dir, page_index)


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
    doc_dir: Path, source_pdf: Path, pages: list[int], config: ReportConfig, variant,
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


def _image_dirs(doc_dir: Path) -> tuple[Path, Path, Path]:
    """(detect-input dir, recog-input dir, fast-tile dir): what PaddleOCR's
    own text *detector* saw vs. what its text *recognizer* saw vs. what FAST
    saw per tile."""
    return (
        doc_dir / "paddle_detect_images", doc_dir / "paddle_recog_images",
        doc_dir / "fast_tile_images",
    )


def _process_pdf(pdf_path: Path, config: ReportConfig, variant, run_dir: Path) -> None:
    from rastervec.core.pipeline import run_pipeline as run_current
    from rastervec.pipelines.legacy import run_pipeline as run_legacy

    is_legacy = variant.engine == "legacy"
    doc_dir = run_dir / pdf_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    detect_dir, recog_dir, fast_tile_dir = _image_dirs(doc_dir)

    pages = config.pages_for(pdf_path.stem)
    active = _active_artifacts(config, variant)

    writer = _LayerWriter()
    stats_pages: dict[str, list[tuple[int, dict]]] = {row[0]: [] for row in active}
    dumps: list[dump_io.PageDump] = []

    for page_index in pages:
        run_input, run_page = str(pdf_path), page_index
        if config.vectorise:
            conv_path = doc_dir / f"converted_p{page_index}.pdf"
            _CONVERT[config.vectorise_mode](str(pdf_path), page_index, str(conv_path))
            run_input, run_page = str(conv_path), 0

        _LOG.info("running %s page %d (%s)", pdf_path.name, page_index, variant.name)
        if is_legacy:
            res = run_legacy(run_input, run_page, verbose=True)
        else:
            res = run_current(
                run_input, run_page, p2=variant.p2, p3=variant.p3,
                enable_fast=variant.enable_fast, verbose=True,
                on_debug_layer=_debug_layer_sink(writer),
            )
        if run_page != page_index:
            _restamp_page(res, page_index)

        _accumulate_page(res, page_index, active, writer, stats_pages,
                         detect_dir, recog_dir, fast_tile_dir, is_legacy=is_legacy)
        dumps.append(dump_io.PageDump(
            page_meta=res.page.meta, texts=list(res.texts or []),
            vectors=list(res.vectors or []), engine=variant.engine,
            step_durations=dict(res.step_durations or {}),
        ))

    _finalize_doc_dir(doc_dir, pdf_path, pages, config, variant, active,
                      writer, stats_pages, dumps)


_OVERLAY_COLORS = {
    "native_to_vector": "#22c55e",
    "original_vector": "#2563eb",
    "vector_to_raster": "#f97316",
    "original_raster": "#a855f7",
    "native_to_raster": "#eab308",
}


def _bench_ground_truth_by_type(bench: BenchInput, pages: list[int]) -> "dict[str, LabelSet]":
    """One `LabelSet` per text type (all `TEXT_TYPES` keys always present,
    entries + geometry_entries filtered to `pages`).

    `native_to_vector` is live-derived via `native_label.native_label_pdf`
    per page -- UNLESS `bench.labels_path` is a master_label.py folder that
    already has its own `native_labels.json`, in which case that file's
    entries are used instead (avoids redundant re-derivation).

    `original_vector`/`vector_to_raster`/`original_raster`/`native_to_raster`
    come from `bench.labels_path`: `None` -> all empty; a `.json` file ->
    `load_labels` + `entries_by_text_type` (single-file convention); a
    directory -> `load_labels_from_master_folder` + `entries_by_text_type`
    (+ geometry_entries, needed for vector_to_raster/original_raster
    vector-geometry scoring).

    `native_to_raster` ground truth is the SAME text as `native_to_vector`
    (the same native region, just scored against a rasterised-PDF pipeline
    run instead of the vectorised one) -- taken from the label folder's own
    `"natsync:"`-prefixed raster_labels.json entries
    (`raster_label.sync_native_text_from_native_labels`) when present, else
    synthesized on the fly from `native_entries` (a label folder predating
    that sync step).
    """
    pdf_str = str(bench.pdf_path)
    by_type: "dict[str, LabelSet]" = {t: LabelSet(pdf_path=pdf_str, entries=[]) for t in TEXT_TYPES}

    native_from_folder = (
        bench.labels_path is not None
        and bench.labels_path.is_dir()
        and (bench.labels_path / "native_labels.json").is_file()
    )
    if native_from_folder:
        merged = load_labels_from_master_folder(bench.labels_path)
        native_entries = adapters.entries_by_text_type(merged)["native_to_vector"]
    else:
        native_entries = []
        for p in pages:
            native_entries += native_label.native_label_pdf(str(bench.pdf_path), p).entries
    native_entries = [e for e in native_entries if e.page_index in pages]
    by_type["native_to_vector"] = LabelSet(pdf_path=pdf_str, entries=native_entries)

    if bench.labels_path is not None:
        if bench.labels_path.is_dir():
            merged = load_labels_from_master_folder(bench.labels_path)
        else:
            merged = load_labels(str(bench.labels_path))
        buckets = adapters.entries_by_text_type(merged)
        # geometry_entries (vector-geometry GT, distinct from these text
        # `entries`) only ever belong on vector_to_raster ("auto"-source)
        # and original_raster ("manual"-source) -- split the same way
        # `adapters.gt_geometry_by_vector_type` does, and only there, so
        # `_merge_gt` (pipeline_report_benchmark.py) doesn't double them up
        # by finding them duplicated across several ground_truth_*.json.
        geometry_by_source = {
            "vector_to_raster": "auto", "original_raster": "manual",
        }
        for t in ("original_vector", "vector_to_raster", "original_raster", "native_to_raster"):
            geom_source = geometry_by_source.get(t)
            by_type[t] = LabelSet(
                pdf_path=pdf_str,
                entries=[e for e in buckets[t] if e.page_index in pages],
                geometry_entries=(
                    [g for g in merged.geometry_entries
                     if g.page_index in pages and g.source == geom_source]
                    if geom_source else []
                ),
            )

    if not by_type["native_to_raster"].entries:
        by_type["native_to_raster"] = LabelSet(
            pdf_path=pdf_str,
            entries=[
                LabelEntry(
                    page_index=e.page_index, cluster_bbox=e.cluster_bbox,
                    cluster_signature=e.cluster_signature,
                    label_id=f"natsync:{e.label_id}", text=e.text, source="raster",
                    expected_rotation=e.expected_rotation,
                )
                for e in native_entries
            ],
        )

    return by_type


RASTER_TEXT_TYPES = ("vector_to_raster", "original_raster", "native_to_raster")


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


def _extract_single_page(src_pdf: Path, page_index: int, out_path: Path) -> None:
    doc = fitz.open(str(src_pdf))
    try:
        out = fitz.open()
        try:
            out.insert_pdf(doc, from_page=page_index, to_page=page_index)
            out.save(str(out_path))
        finally:
            out.close()
    finally:
        doc.close()


def _process_pdf_benchmark(
    bench: BenchInput, config: ReportConfig, variant, run_dir: Path,
) -> dict:
    """One benchmark input -> `run_dir/<pdf-stem>/`: the full per-stage
    report for a `convert_page_to_vector_text` run per page (predictions for
    `native_to_vector`/`original_vector`), plus -- when `bench.rasterised_
    pdf_path` is set -- a SEPARATE pipeline run directly on that rasterised
    PDF's own page (predictions for `vector_to_raster`/`original_raster`/
    `native_to_raster`; that page has no vector paths at all, so this run
    currently always yields empty predictions -- see `PageDump.raster_
    texts`'s docstring). Writes `ground_truth_<type>.json` (only the
    non-empty types) and `<type>_{bbox,text}.pdf` overlays for whichever
    types have labels. Returns the `benchmark.json` entry."""
    from rastervec.core.pipeline import run_pipeline as run_current
    from rastervec.pipelines.legacy import run_pipeline as run_legacy

    is_legacy = variant.engine == "legacy"
    doc_dir = run_dir / bench.pdf_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    detect_dir, recog_dir, fast_tile_dir = _image_dirs(doc_dir)
    pages = config.pages_for(bench.pdf_path.stem)
    cfg = metrics.MetricConfig(iou_edge_min=config.iou_edge_min)
    active = _active_artifacts(config, variant)

    writer = _LayerWriter()
    stats_pages: dict[str, list[tuple[int, dict]]] = {row[0]: [] for row in active}
    dumps: list[dump_io.PageDump] = []

    for p in pages:
        conv_path = doc_dir / f"converted_p{p}.pdf"
        conversion.convert_page_to_vector_text(str(bench.pdf_path), p, str(conv_path))
        _LOG.info("benchmark %s page %d (%s)", bench.key, p, variant.name)
        if is_legacy:
            res = run_legacy(str(conv_path), 0, verbose=True)
        else:
            res = run_current(
                str(conv_path), 0, p2=variant.p2, p3=variant.p3,
                enable_fast=variant.enable_fast, verbose=True,
                on_debug_layer=_debug_layer_sink(writer),
            )
        _restamp_page(res, p)
        _accumulate_page(res, p, active, writer, stats_pages,
                         detect_dir, recog_dir, fast_tile_dir, is_legacy=is_legacy)

        raster_texts = []
        if bench.rasterised_pdf_path is not None:
            raster_page_path = doc_dir / f"rasterised_p{p}.pdf"
            try:
                _extract_single_page(bench.rasterised_pdf_path, p, raster_page_path)
                if is_legacy:
                    res_raster = run_legacy(str(raster_page_path), 0, verbose=True)
                else:
                    res_raster = run_current(
                        str(raster_page_path), 0, p2=variant.p2, p3=variant.p3,
                        enable_fast=variant.enable_fast, verbose=True,
                    )
                _restamp_page(res_raster, p)
                raster_texts = list(res_raster.texts or [])
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "%s: rasterised-PDF run failed for page %d: %s", bench.key, p, exc,
                )

        dumps.append(dump_io.PageDump(
            page_meta=res.page.meta, texts=list(res.texts or []),
            vectors=list(res.vectors or []), engine=variant.engine,
            step_durations=dict(res.step_durations or {}),
            raster_texts=raster_texts,
        ))

    sources: list[str] = []
    extra_layers: list[dict] = []
    gt_by_type = _bench_ground_truth_by_type(bench, pages)
    for text_type in TEXT_TYPES:
        gt = gt_by_type[text_type]
        if not gt.entries and not gt.geometry_entries:
            if text_type != "native_to_vector":
                _LOG.info("%s: no %s labels for pages %s", bench.key, text_type, pages)
            continue
        save_labels(gt, str(doc_dir / f"ground_truth_{text_type}.json"))
        if gt.entries:
            _write_label_overlays(doc_dir, text_type, gt, dumps, cfg)
        sources.append(text_type)

    for s in sources:
        for kind in ("bbox", "text"):
            extra_layers.append({
                "stage": "benchmark", "layer": f"{s} label {kind}",
                "file": f"{s}_{kind}.pdf", "color": _OVERLAY_COLORS[s],
            })

    _finalize_doc_dir(doc_dir, bench.pdf_path, pages, config, variant, active,
                      writer, stats_pages, dumps,
                      extra_layers=tuple(extra_layers))

    (doc_dir / "benchmark_meta.json").write_text(json.dumps({
        "key": bench.key,
        "source_pdf": str(bench.pdf_path),
        "labels": str(bench.labels_path) if bench.labels_path else None,
        "pages": pages,
        "sources": sources,
        "final_stage": config.final_stage,
        "variant": variant.name,
    }, indent=2), encoding="utf-8")
    return {
        "key": bench.key,
        "pdf_stem": bench.pdf_path.stem,
        "dir": bench.pdf_path.stem,
        "sources": sources,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="Path to the run config JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging()
    config = ReportConfig.model_validate_json(Path(args.config).read_text(encoding="utf-8"))
    # `config.pipeline` picks the engine; `config.p2`/`config.p3` (only
    # meaningful for "current") come straight from the run config, not from
    # a named `variants.VARIANTS` preset -- a report run wants exactly the
    # backend combo the config asks for, not a fixed default combo.
    if config.pipeline == "legacy":
        variant = resolve_variant("legacy")
    else:
        variant = PipelineVariant(
            name=f"current:{config.p2}:{config.p3}", engine="current",
            p2=config.p2, p3=config.p3, enable_fast=config.enable_fast,
        )

    root = config.output_root or output_dir("pipeline_report")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{ts}__{Path(args.config).stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_hyperparams(run_dir / "config_and_hyperparameters.txt", config, variant)

    if config.benchmark:
        inputs = config.benchmark_inputs()
        if not inputs:
            _LOG.error("no benchmark inputs (set input_dir and/or input_files)")
            return 1
        entries = [
            _process_pdf_benchmark(bench, config, variant, run_dir) for bench in inputs
        ]
        (run_dir / "benchmark.json").write_text(json.dumps({
            "benchmark": True,
            "config_stem": Path(args.config).stem,
            "variant": variant.name,
            "final_stage": config.final_stage,
            "iou_edge_min": config.iou_edge_min,
            "entries": entries,
        }, indent=2), encoding="utf-8")
        print(f"wrote {run_dir}")
        return 0

    pdfs = config.resolved_pdfs()
    if not pdfs:
        _LOG.error("no input PDFs (set input_dir and/or input_files)")
        return 1
    for pdf_path in pdfs:
        _process_pdf(pdf_path, config, variant, run_dir)

    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
