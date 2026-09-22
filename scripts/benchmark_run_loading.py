"""Loading a `generate_pipeline_report.py --benchmark` run folder
(`RunEntry`/`_merge_gt`/`_load_run`) and scoring one shared input's text
(`_score_text`) / vector (`_score_vectors`) metrics against it."""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from rastervec.Evaluation import dump_io
from rastervec.Evaluation.Evaluate import adapters, vector_metrics
from rastervec.Evaluation.Evaluate.metrics import (
    TEXT_TYPES,
    MetricConfig,
    OverlapGraph,
    TextMetricSuiteResult,
    aggregate_text_metrics,
    build_overlap_graphs_by_type,
    combine_text_metrics_by_type,
    evaluate_text_metrics,
)
from rastervec.Evaluation.Evaluate.vector_metrics import (
    VectorMetricConfig,
    VectorMetricSuiteResult,
    aggregate_vector_metrics,
)
from rastervec.Evaluation.Labelling.label_schema import LabelSet, load_labels


class RunEntry(NamedTuple):
    run_name: str
    run_dir: Path
    key: str
    pdf_stem: str
    doc_dir: Path
    dump_path: Path
    gt: LabelSet  # merged native/vector/raster labels for this input


def _merge_gt(doc: Path) -> LabelSet:
    """Merges whichever `ground_truth_*.json` files this run wrote (every
    name in `TEXT_TYPES`, falling back to the pre-rework `auto`/`manual`
    names for an older report folder) into one `LabelSet`."""
    entries = []
    geometry_entries = []
    pdf_path = ""
    for name in (*TEXT_TYPES, "auto", "manual"):
        p = doc / f"ground_truth_{name}.json"
        if p.is_file():
            ls = load_labels(str(p))
            pdf_path = pdf_path or ls.pdf_path
            entries.extend(ls.entries)
            geometry_entries.extend(ls.geometry_entries)
    return LabelSet(pdf_path=pdf_path, entries=entries, geometry_entries=geometry_entries)


def _load_run(run_dir: Path, parser) -> dict[str, RunEntry]:
    marker = run_dir / "benchmark.json"
    if not marker.is_file():
        parser.error(f"{run_dir} has no benchmark.json (not a benchmark run)")
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if not meta.get("benchmark"):
        parser.error(f"{run_dir}: benchmark.json does not mark this as a benchmark run")

    entries: dict[str, RunEntry] = {}
    for e in meta.get("entries", []):
        doc = run_dir / e["dir"]
        entries[e["key"]] = RunEntry(
            run_name=run_dir.name,
            run_dir=run_dir.resolve(),
            key=e["key"],
            pdf_stem=e["pdf_stem"],
            doc_dir=doc,
            dump_path=doc / "dump.json",
            gt=_merge_gt(doc),
        )
    return entries


# native_to_vector/original_vector are scored from the vectorised-PDF run's
# own OCR predictions; vector_to_raster/original_raster/native_to_raster are
# scored from the SEPARATE rasterised-PDF run's own OCR predictions -- never
# each other's. Mirrors `generate_pipeline_report.py`'s `RASTER_TEXT_TYPES`.
_VECTORISED_TEXT_TYPES = ("native_to_vector", "original_vector")
_RASTER_TEXT_TYPES = ("vector_to_raster", "original_raster", "native_to_raster")


def _score_text(
    entry: RunEntry, cfg: MetricConfig,
) -> "tuple[list[tuple[int, TextMetricSuiteResult, dict[str, OverlapGraph]]], TextMetricSuiteResult | None]":
    dump = dump_io.load_dump(entry.dump_path)
    gt_by_type_all = adapters.gt_regions_by_text_type(entry.gt)
    entries_by_type_all = adapters.entries_by_text_type(entry.gt)
    per_page: "list[tuple[int, TextMetricSuiteResult, dict[str, OverlapGraph]]]" = []
    for page in dump.pages:
        pi = page.page_meta.index
        page_area = page.page_meta.width * page.page_meta.height
        gt_by_type = {t: [g for g in gt_by_type_all[t] if g.page_index == pi] for t in TEXT_TYPES}

        def _restricted(types, src=gt_by_type):
            return {t: (src[t] if t in types else []) for t in TEXT_TYPES}

        vec_preds = adapters.predictions_from_texts([t for t in page.texts if t.source == "ocr"])
        vec_gt = _restricted(_VECTORISED_TEXT_TYPES)
        vec_entries = _restricted(_VECTORISED_TEXT_TYPES, entries_by_type_all)
        res_vec = evaluate_text_metrics(vec_gt, vec_entries, vec_preds, cfg=cfg, page_area=page_area)
        vec_graphs = build_overlap_graphs_by_type(vec_gt, vec_preds, cfg)

        raster_preds = adapters.predictions_from_texts(
            [t for t in page.raster_texts if t.source == "ocr"]
        )
        raster_gt = _restricted(_RASTER_TEXT_TYPES)
        raster_entries = _restricted(_RASTER_TEXT_TYPES, entries_by_type_all)
        res_raster = evaluate_text_metrics(raster_gt, raster_entries, raster_preds, cfg=cfg, page_area=page_area)
        raster_graphs = build_overlap_graphs_by_type(raster_gt, raster_preds, cfg)

        res = combine_text_metrics_by_type({
            **{t: res_vec for t in _VECTORISED_TEXT_TYPES},
            **{t: res_raster for t in _RASTER_TEXT_TYPES},
        })
        graphs_by_type = {
            **{t: vec_graphs[t] for t in _VECTORISED_TEXT_TYPES},
            **{t: raster_graphs[t] for t in _RASTER_TEXT_TYPES},
        }
        per_page.append((pi, res, graphs_by_type))
    return per_page, aggregate_text_metrics([r for _pi, r, _g in per_page])


def _score_vectors(
    entry: RunEntry, cfg: VectorMetricConfig,
) -> "tuple[list[tuple[int, VectorMetricSuiteResult]], VectorMetricSuiteResult | None]":
    """Vector metrics from the run's own ground-truth labels alone --
    `vector_to_raster`/`original_raster` only (`original_vector` is a
    text-provenance type, not scored at the vector-geometry level; see
    `adapters.build_vector_eval_inputs`). Neither has a prediction
    population (no raster-tracing pipeline stage exists), so this reports
    GT-only stats."""
    dump = dump_io.load_dump(entry.dump_path)
    per_page: "list[tuple[int, VectorMetricSuiteResult]]" = []

    for page in dump.pages:
        pi = page.page_meta.index
        page_gt = LabelSet(
            pdf_path=entry.gt.pdf_path,
            entries=[e for e in entry.gt.entries if e.page_index == pi],
            geometry_entries=[g for g in entry.gt.geometry_entries if g.page_index == pi],
        )
        inputs = adapters.build_vector_eval_inputs(page_gt)
        res = vector_metrics.evaluate_vector_metrics(
            inputs.gt_by_type, inputs.preds_by_type, inputs.label_counts, cfg=cfg,
        )
        per_page.append((pi, res))
    return per_page, aggregate_vector_metrics([r for _pi, r in per_page])
