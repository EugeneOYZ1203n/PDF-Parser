"""Benchmark OCR text from one or more pipeline dump JSONs against ground
truth -- the final aggregate-metric portion of the old
`benchmark_vector_classification.ipynb`.

Ground truth is either auto-derived from a target PDF (`--pdf`, native-text
lines) or read from a label JSON (`--labels`, auto + manual scored
separately). Miss-attribution metrics are absent: a dump carries no
classification state, so `evaluate_metrics(clustering=None)`.

    .venv/Scripts/python.exe scripts/pipeline_report_benchmark.py \
        --dump run/<stem>/dump.json --pdf references/<stem>.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rastervec.Evaluation import dump_io
from rastervec.Evaluation.Evaluate import adapters
from rastervec.Evaluation.Evaluate.benchmark import (
    aggregate_results,
    format_aggregate_comparison,
    format_report,
)
from rastervec.Evaluation.Evaluate.metrics import MetricConfig, evaluate_metrics
from rastervec.Evaluation.Labelling.auto_label import auto_label_pdf
from rastervec.Evaluation.Labelling.label_schema import load_labels, split_labelset_by_source
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.paths import output_dir

_LOG = get_logger("pipeline_report_benchmark")


def _gt_for_page(source, page_index: int):
    """`(auto_regions, manual_regions)` for one page. `source` is a Path to a
    target PDF (auto only) or to a label JSON (both)."""
    source = Path(source)
    if source.suffix.lower() == ".pdf":
        regions = adapters.gt_regions_from_labelset(auto_label_pdf(str(source), page_index))
        return regions, []
    by_src = split_labelset_by_source(load_labels(str(source)))
    auto = [
        r for r in adapters.gt_regions_from_labelset(by_src["auto"])
        if r.page_index == page_index
    ] if "auto" in by_src else []
    manual = [
        r for r in adapters.gt_regions_from_labelset(by_src["manual"])
        if r.page_index == page_index
    ] if "manual" in by_src else []
    return auto, manual


def _score_dump(dump_path: Path, source, cfg: MetricConfig, report_lines: list[str]):
    dump = dump_io.load_dump(dump_path)
    auto_suites, manual_suites = [], []
    for page in dump.pages:
        pi = page.page_meta.index
        ocr_texts = [t for t in page.texts if t.source == "ocr"]
        preds = adapters.predictions_from_texts(ocr_texts)
        cand = adapters.text_candidate_boxes(ocr_texts)
        auto_gt, manual_gt = _gt_for_page(source, pi)

        auto_suite = evaluate_metrics(auto_gt, preds, cand, clustering=None, cfg=cfg)
        auto_suites.append(auto_suite)
        report_lines.append(format_report(f"{dump_path.parent.name} [auto]", pi, auto_suite))
        if manual_gt:
            manual_suite = evaluate_metrics(manual_gt, preds, cand, clustering=None, cfg=cfg)
            manual_suites.append(manual_suite)
            report_lines.append(format_report(f"{dump_path.parent.name} [manual]", pi, manual_suite))
    return aggregate_results(auto_suites), aggregate_results(manual_suites)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", action="append", required=True, type=Path, help="dump.json (repeatable).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", type=Path, help="Target PDF for auto ground truth.")
    src.add_argument("--labels", type=Path, help="Label JSON (auto + manual).")
    p.add_argument("--iou-threshold", type=float, default=MetricConfig().iou_edge_min)
    p.add_argument("--out", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging()
    cfg = MetricConfig(iou_edge_min=args.iou_threshold)
    source = args.pdf or args.labels

    report_lines: list[str] = []
    auto_by_dump, manual_by_dump = {}, {}
    for dump_path in args.dump:
        name = dump_path.parent.name or dump_path.stem
        auto_agg, manual_agg = _score_dump(dump_path, source, cfg, report_lines)
        auto_by_dump[name] = auto_agg
        manual_by_dump[name] = manual_agg

    blocks = report_lines + [
        format_aggregate_comparison(auto_by_dump, title="Aggregate metrics -- AUTO ground truth"),
    ]
    if any(v is not None for v in manual_by_dump.values()):
        blocks.append(format_aggregate_comparison(
            manual_by_dump, title="Aggregate metrics -- MANUAL ground truth"
        ))
    text = "\n\n".join(blocks) + "\n"

    out = args.out or (output_dir("pipeline_report") / "benchmark.txt")
    Path(out).write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
