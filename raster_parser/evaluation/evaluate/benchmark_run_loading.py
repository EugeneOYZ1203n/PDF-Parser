"""Loading a `generate_pipeline_report.py` benchmark run folder (`RunEntry`, `load_run`),
scoring one input's text against its ground truth (`score_text`), and reading its timings
(`load_timings`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from raster_parser.evaluation import dump_io
from raster_parser.evaluation.evaluate import adapters, timing
from raster_parser.evaluation.evaluate.metrics import (
    TEXT_TYPES,
    MetricConfig,
    TextMetricSuiteResult,
    aggregate_text_metrics,
    evaluate_text_metrics,
)
from raster_parser.evaluation.labelling.label_schema import LabelSet, load_labels


class RunEntry(NamedTuple):
    run_name: str
    run_dir: Path
    key: str  # `pdf:<stem>` / `labels:<stem>`, the identity runs are matched on
    source_pdf: str
    labels: "str | None"
    pages: list[int]
    doc_dir: Path
    dump_path: Path
    gt: LabelSet  # every ground_truth_<type>.json of this input, merged


class RunMeta(NamedTuple):
    name: str
    path: Path
    p2: str
    p3: str
    final_stage: "str | None"
    config_path: "str | None"


def merge_gt(doc: Path) -> LabelSet:
    """Every `ground_truth_<type>.json` the run wrote for this input, merged into one set."""
    entries = []
    pdf_path = ""
    for text_type in TEXT_TYPES:
        p = doc / f"ground_truth_{text_type}.json"
        if p.is_file():
            ls = load_labels(str(p))
            pdf_path = pdf_path or ls.pdf_path
            entries.extend(ls.entries)
    return LabelSet(pdf_path=pdf_path, entries=entries)


def load_run(run_dir: Path, parser) -> "tuple[RunMeta, dict[str, RunEntry]]":
    """The run's own settings and `{key: RunEntry}`; `parser.error` on a folder that is not a
    benchmark run."""
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
            source_pdf=e.get("source_pdf", ""),
            labels=e.get("labels"),
            pages=list(e.get("pages", [])),
            doc_dir=doc,
            dump_path=doc / "dump.json",
            gt=merge_gt(doc),
        )
    run_meta = RunMeta(
        name=run_dir.name,
        path=run_dir.resolve(),
        p2=meta.get("p2", "?"),
        p3=meta.get("p3", "?"),
        final_stage=meta.get("final_stage"),
        config_path=meta.get("config_path"),
    )
    return run_meta, entries


class TextScore(NamedTuple):
    per_page: "list[tuple[int, TextMetricSuiteResult]]"
    aggregate: "TextMetricSuiteResult | None"
    n_ocr: int  # OCR readings, blank included (`Text.source == "ocr"`)
    n_ocr_touching_gt: int  # ... that overlap at least one ground-truth region


def _touches(bbox, regions) -> bool:
    x0, y0, x1, y1 = bbox
    return any(min(x1, g[2]) > max(x0, g[0]) and min(y1, g[3]) > max(y0, g[1]) for g in regions)


def score_text(entry: RunEntry, cfg: MetricConfig) -> TextScore:
    """Every page of the run scored against this input's ground truth (one prediction list, the
    run's OCR text, scored against each text type)."""
    dump = dump_io.load_dump(entry.dump_path)
    gt_by_type_all = adapters.gt_regions_by_text_type(entry.gt)
    entries_by_type_all = adapters.entries_by_text_type(entry.gt)
    per_page: "list[tuple[int, TextMetricSuiteResult]]" = []
    n_ocr = n_touching = 0
    for page in dump.pages:
        pi = page.result.page.index
        gt_by_type = {t: [g for g in gt_by_type_all[t] if g.page_index == pi] for t in TEXT_TYPES}
        entries_by_type = {
            t: [e for e in entries_by_type_all[t] if e.page_index == pi] for t in TEXT_TYPES
        }
        ocr = [t for t in page.result.texts if t.source == "ocr"]
        res = evaluate_text_metrics(
            gt_by_type,
            entries_by_type,
            adapters.predictions_from_texts(ocr),
            cfg=cfg,
            page_area=page.result.page.width * page.result.page.height,
        )
        per_page.append((pi, res))
        regions = [g.bbox for t in TEXT_TYPES for g in gt_by_type[t]]
        n_ocr += len(ocr)
        n_touching += sum(_touches(t.bbox, regions) for t in ocr)
    return TextScore(
        per_page, aggregate_text_metrics([r for _pi, r in per_page]), n_ocr, n_touching
    )


def load_timings(entry: RunEntry) -> "dict[str, list[dict]]":
    """`{timing.RUN_KIND: [per-page timing row], timing.DOC_KIND: [this document's row]}` from
    the input's `dump.json`. A page that recorded nothing contributes no row."""
    dump = dump_io.load_dump(entry.dump_path)
    out: "dict[str, list[dict]]" = {timing.RUN_KIND: [], timing.DOC_KIND: []}
    for page in dump.pages:
        row = timing.flatten_page_timing(page.result.step_durations, page.debug_durations)
        if row:
            out[timing.RUN_KIND].append(row)
    doc_row = timing.flatten_doc_timing(dump.doc_durations)
    if doc_row:
        out[timing.DOC_KIND].append(doc_row)
    return out
