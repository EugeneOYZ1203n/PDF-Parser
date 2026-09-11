"""Compare 1-2 benchmark report folders.

Each folder is a `generate_pipeline_report.py` run with `benchmark: true`.
Such a run scores **one** combined pipeline run per page
(`convert_page_to_vector_text`) against both auto and manual ground truth,
and embeds `<pdf-stem>/dump.json` + `ground_truth_auto.json`
(+ `ground_truth_manual.json`) + a `benchmark.json` marker.

This script matches the folders on the inputs they **share** (by input
identity: `pdf:<stem>` = auto-only, `labels:<stem>` = auto + manual) and
re-scores each with `metrics.evaluate_multiclass` -- so a prediction that
matched the *other* label class is not counted as a false positive. Output
(timestamped folder under `outputs/pipeline_report_benchmark/`):

    benchmark.txt          per-key AUTO / MANUAL tables + confusion matrices
    runs.json              the compared folders + threshold
    viewer_commands.txt    a viewer CLI line per shared input
    charts/                <key>__p<N>__{auto,manual,confusion}.png,
                           <key>__aggregate__*.png, aggregate__*.png

    .venv/Scripts/python.exe scripts/pipeline_report_benchmark.py \
        --run outputs/pipeline_report/<ts1>__benchmark \
        --run outputs/pipeline_report/<ts2>__benchmark_legacy
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rastervec.Evaluation import dump_io
from rastervec.Evaluation.Evaluate import adapters, charts
from rastervec.Evaluation.Evaluate.benchmark import (
    format_aggregate_comparison,
    format_confusion,
)
from rastervec.Evaluation.Evaluate.metrics import (
    MetricConfig,
    MulticlassResult,
    aggregate_multiclass,
    evaluate_multiclass,
)
from rastervec.Evaluation.Labelling.label_schema import LabelSet, load_labels
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.paths import output_dir

_LOG = get_logger("pipeline_report_benchmark")

_VENV_PY = ".venv/Scripts/python.exe"


class RunEntry(NamedTuple):
    run_name: str
    run_dir: Path
    key: str
    pdf_stem: str
    doc_dir: Path
    dump_path: Path
    auto_gt: LabelSet
    manual_gt: LabelSet | None


def _load_run(run_dir: Path, parser: argparse.ArgumentParser) -> dict[str, RunEntry]:
    marker = run_dir / "benchmark.json"
    if not marker.is_file():
        parser.error(f"{run_dir} has no benchmark.json (not a benchmark run)")
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if not meta.get("benchmark"):
        parser.error(f"{run_dir}: benchmark.json does not mark this as a benchmark run")

    entries: dict[str, RunEntry] = {}
    for e in meta.get("entries", []):
        doc = run_dir / e["dir"]
        manual_path = doc / "ground_truth_manual.json"
        entries[e["key"]] = RunEntry(
            run_name=run_dir.name,
            run_dir=run_dir.resolve(),
            key=e["key"],
            pdf_stem=e["pdf_stem"],
            doc_dir=doc,
            dump_path=doc / "dump.json",
            auto_gt=load_labels(str(doc / "ground_truth_auto.json")),
            manual_gt=load_labels(str(manual_path)) if manual_path.is_file() else None,
        )
    return entries


def _score(entry: RunEntry, cfg: MetricConfig) -> tuple[list[tuple[int, MulticlassResult]], MulticlassResult | None]:
    dump = dump_io.load_dump(entry.dump_path)
    auto_regions = adapters.gt_regions_from_labelset(entry.auto_gt)
    manual_regions = (
        adapters.gt_regions_from_labelset(entry.manual_gt) if entry.manual_gt else []
    )
    per_page: list[tuple[int, MulticlassResult]] = []
    for page in dump.pages:
        pi = page.page_meta.index
        ocr = [t for t in page.texts if t.source == "ocr"]
        preds = adapters.predictions_from_texts(ocr)
        cand = adapters.text_candidate_boxes(ocr)
        res = evaluate_multiclass(
            [g for g in auto_regions if g.page_index == pi],
            [g for g in manual_regions if g.page_index == pi],
            preds, cand, cfg=cfg,
        )
        per_page.append((pi, res))
    return per_page, aggregate_multiclass([r for _pi, r in per_page])


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_") or "key"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, type=Path,
                   help="Benchmark run folder (repeatable; 1 or 2).")
    p.add_argument("--iou-threshold", type=float, default=MetricConfig().iou_edge_min)
    p.add_argument("--out-root", type=Path, default=None,
                   help="Parent for the timestamped output folder "
                   "(default: outputs/pipeline_report_benchmark/).")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging()
    if len(args.run) > 2:
        parser.error("at most 2 --run folders (benchmark compares two pipelines)")
    cfg = MetricConfig(iou_edge_min=args.iou_threshold)

    runs = [_load_run(Path(d), parser) for d in args.run]
    shared_keys = set(runs[0])
    for r in runs[1:]:
        shared_keys &= set(r)
    if not shared_keys:
        parser.error("the run folders share no benchmark input")

    root = args.out_root or output_dir("pipeline_report_benchmark")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / ts
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    blocks: list[str] = []
    # (source -> list[MulticlassResult]) for the grand aggregate charts
    grand: dict[str, list[MulticlassResult]] = {}

    for key in sorted(shared_keys):
        scored = {r[key].run_name: _score(r[key], cfg) for r in runs}
        has_manual = all(r[key].manual_gt is not None for r in runs)

        # ---- text ----
        for src in ("auto", "manual"):
            if src == "manual" and not has_manual:
                continue
            by_run = {
                name: (agg.auto if src == "auto" else agg.manual) if agg else None
                for name, (_pp, agg) in scored.items()
            }
            blocks.append(format_aggregate_comparison(
                by_run, title=f"{key}  --  {src.upper()} ground truth"))
        blocks.append(format_confusion(
            {name: (agg.confusion if agg else {}) for name, (_pp, agg) in scored.items()},
            title=f"{key}  --  confusion matrix"))
        blocks.append("combined candidate precision: " + "  ".join(
            f"{name}={('n/a' if not agg or not agg.combined_candidate_precision.applicable else f'{agg.combined_candidate_precision.value:.3f}')}"
            for name, (_pp, agg) in scored.items()))

        # ---- charts ----
        pages = sorted({pi for (pp, _agg) in scored.values() for pi, _r in pp})
        kslug = _slug(key)
        for pi in pages:
            per_run = {
                name: next((r for p, r in pp if p == pi), None)
                for name, (pp, _agg) in scored.items()
            }
            charts.metric_comparison_chart(
                {n: (r.auto if r else None) for n, r in per_run.items()},
                title=f"{key} p{pi} auto", path=charts_dir / f"{kslug}__p{pi}__auto.png")
            if has_manual:
                charts.metric_comparison_chart(
                    {n: (r.manual if r else None) for n, r in per_run.items()},
                    title=f"{key} p{pi} manual", path=charts_dir / f"{kslug}__p{pi}__manual.png")
            charts.confusion_heatmap(
                {n: (r.confusion if r else {}) for n, r in per_run.items()},
                title=f"{key} p{pi} confusion", path=charts_dir / f"{kslug}__p{pi}__confusion.png")

        agg_by_run = {n: a for n, (_pp, a) in scored.items()}
        charts.metric_comparison_chart(
            {n: (a.auto if a else None) for n, a in agg_by_run.items()},
            title=f"{key} aggregate auto", path=charts_dir / f"{kslug}__aggregate__auto.png")
        if has_manual:
            charts.metric_comparison_chart(
                {n: (a.manual if a else None) for n, a in agg_by_run.items()},
                title=f"{key} aggregate manual", path=charts_dir / f"{kslug}__aggregate__manual.png")
        charts.confusion_heatmap(
            {n: (a.confusion if a else {}) for n, a in agg_by_run.items()},
            title=f"{key} aggregate confusion", path=charts_dir / f"{kslug}__aggregate__confusion.png")

        for name, (pp, _agg) in scored.items():
            grand.setdefault(name, []).extend(r for _pi, r in pp)

    # ---- grand aggregate over every shared key ----
    grand_agg = {name: aggregate_multiclass(rs) for name, rs in grand.items()}
    if grand_agg:
        charts.metric_comparison_chart(
            {n: (a.auto if a else None) for n, a in grand_agg.items()},
            title="aggregate auto (all inputs)", path=charts_dir / "aggregate__auto.png")
        charts.metric_comparison_chart(
            {n: (a.manual if a else None) for n, a in grand_agg.items()},
            title="aggregate manual (all inputs)", path=charts_dir / "aggregate__manual.png")
        charts.confusion_heatmap(
            {n: (a.confusion if a else {}) for n, a in grand_agg.items()},
            title="aggregate confusion (all inputs)", path=charts_dir / "aggregate__confusion.png")

    header = "compared runs:\n" + "\n".join(f"  {Path(d).name}" for d in args.run)
    (out_dir / "benchmark.txt").write_text(
        header + "\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    (out_dir / "runs.json").write_text(json.dumps({
        "runs": [str(Path(d).resolve()) for d in args.run],
        "iou_edge_min": args.iou_threshold,
        "shared_keys": sorted(shared_keys),
    }, indent=2), encoding="utf-8")

    # ---- viewer_commands.txt ----
    lines = ["# Open each benchmarked input in the viewer (pipelines side by side):"]
    for key in sorted(shared_keys):
        dirs = " ".join(f'"{r[key].doc_dir.resolve()}"' for r in runs)
        lines.append(f"{_VENV_PY} scripts/pipeline_report_viewer.py {dirs}")
    (out_dir / "viewer_commands.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print((out_dir / "benchmark.txt").read_text(encoding="utf-8"))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
