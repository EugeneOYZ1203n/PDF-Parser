"""Score and compare 1-2 benchmark report folders.

Each folder is a `generate_pipeline_report.py` run with `benchmark: true`: per input, the
pipeline ran on a `convert_page_to_vector_text` copy of each page (native text drawn as vector
paths over the untouched drawings), and the folder holds that run's `dump.json` and the
`ground_truth_<type>.json` it is scored against (`native_to_vector`, `original_vector`).

The folders are matched on the inputs they share (`pdf:<stem>` / `labels:<stem>`). Each shared
input's OCR text is scored per text type (`metrics.evaluate_text_metrics`), and everything goes
into one self-contained `report.html`:

1. **Inputs**: the run folders (backends, config) and every file scored (source PDF, labels,
   pages, ground-truth types).
2. **Summary**: the aggregate over every shared input -- char / word F1, bbox IoU, rotation,
   candidate precision and time per page -- one column per run.
3. **Per input**: the text tables (labels, char / word overlap, bbox, rotation, per-character
   accuracy, extra characters) with charts, the timing tables, and a ready-to-paste
   `pipeline_report_viewer.py` command (individual errors are inspected there, incl. the
   `benchmark__extra_*` layers).

Output (`outputs/pipeline_report_benchmark/<ts>/`):

    report.html             the deliverable (above)
    benchmark.txt           the same numbers as plain text (grep/diff-friendly)
    runs.json               the compared folders, threshold and shared inputs
    viewer_commands.txt     a viewer command per shared input
    charts/                 the chart PNGs report.html links

    python scripts/pipeline_report_benchmark.py --run outputs/pipeline_report/<ts1>__a \\
        [--run outputs/pipeline_report/<ts2>__b]

Needs matplotlib (`uv sync --group report`).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raster_parser.commons.logging_setup import configure_logging, get_logger  # noqa: E402
from raster_parser.commons.paths import output_dir  # noqa: E402
from raster_parser.evaluation.evaluate import charts, timing  # noqa: E402
from raster_parser.evaluation.evaluate.benchmark_format import (  # noqa: E402
    format_aggregate_comparison,
    format_confusion_table,
    format_run_timing_comparison,
)
from raster_parser.evaluation.evaluate.benchmark_report_sections import (  # noqa: E402
    add_inputs_header,
    add_summary,
    add_text_sections,
    add_timing_sections,
)
from raster_parser.evaluation.evaluate.benchmark_run_loading import load_run, load_timings, score_text  # noqa: E402
from raster_parser.evaluation.evaluate.html_report import ReportBuilder  # noqa: E402
from raster_parser.evaluation.evaluate.metrics import MetricConfig, aggregate_text_metrics  # noqa: E402

_LOG = get_logger("pipeline_report_benchmark")


def short_slug(text: str, max_len: int = 16) -> str:
    """A filename-safe slug capped at `max_len` chars (+ a short hash when cut), so several can
    share one path component without hitting Windows' path-length limit."""
    full = "".join(c if c.isalnum() else "_" for c in text).strip("_") or "key"
    if len(full) <= max_len:
        return full
    return f"{full[:max_len]}_{hashlib.sha1(text.encode()).hexdigest()[:8]}"


def _text_charts(agg_by_run: dict, label: str, slug: str, charts_dir: Path, out_dir: Path) -> "dict[str, list[Path]]":
    """The per-text-type charts for one input (or the aggregate), as report-relative paths."""
    paths = {name: charts_dir / f"{slug}__{name}.png" for name in ("labels", "char", "word", "bbox", "rotation")}
    charts.label_description_chart(agg_by_run, title=f"{label}: labels", path=paths["labels"])
    charts.char_word_overlap_chart(agg_by_run, field="char_overlap", title=f"{label}: char recall", path=paths["char"])
    charts.char_word_overlap_chart(agg_by_run, field="word_overlap", title=f"{label}: word recall", path=paths["word"])
    charts.bbox_accuracy_chart(agg_by_run, title=f"{label}: bbox mean IoU", path=paths["bbox"])
    charts.rotation_chart(agg_by_run, title=f"{label}: rotation correct", path=paths["rotation"])
    return {name: [p.relative_to(out_dir)] for name, p in paths.items() if p.is_file()}


def _timing(rows_by_run: "dict[str, dict[str, list[dict]]]", label: str, slug: str,
            charts_dir: Path, out_dir: Path, blocks: list[str]) -> "tuple[dict, Path | None]":
    """`({run: {kind: summary}}, report-relative chart path)`; appends the text tables."""
    summaries = {run: {kind: timing.summarize_timings(rows) for kind, rows in by_kind.items()}
                 for run, by_kind in rows_by_run.items()}
    pages = {run: s.get(timing.RUN_KIND, {}) for run, s in summaries.items()}
    chart = None
    if any(pages.values()):
        path = charts_dir / f"{slug}__timing.png"
        charts.timing_chart(pages, title=f"{label}: mean s/page", path=path)
        chart = path.relative_to(out_dir)
        for run, summary in pages.items():
            blocks.append(timing.format_timing_table(summary, title=f"[{label} / {run}] wall-clock per page (s)"))
        if len(pages) > 1:
            blocks.append(format_run_timing_comparison(pages, title=f"[{label}] median s/page by run"))
    return summaries, chart


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, type=Path, help="Benchmark run folder (1 or 2).")
    p.add_argument("--iou-threshold", type=float, default=MetricConfig().iou_edge_min)
    p.add_argument("--out-root", type=Path, default=None,
                   help="Parent of the timestamped output folder (default: outputs/pipeline_report_benchmark/).")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging()
    if len(args.run) > 2:
        parser.error("at most 2 --run folders")
    cfg = MetricConfig(iou_edge_min=args.iou_threshold)

    loaded = [load_run(Path(d), parser) for d in args.run]
    metas = [m for m, _e in loaded]
    runs = [e for _m, e in loaded]
    if len({m.name for m in metas}) != len(metas):
        parser.error("the run folders have the same name; compare folders with different names")
    shared_keys = sorted(set(runs[0]).intersection(*runs[1:]))
    if not shared_keys:
        parser.error("the run folders share no benchmark input")

    root = args.out_root or output_dir("pipeline_report_benchmark")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(root) / ts
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Score everything first: the summary goes at the top of the report.
    scored = {key: {r[key].run_name: score_text(r[key], cfg) for r in runs} for key in shared_keys}
    timings = {key: {r[key].run_name: load_timings(r[key]) for r in runs} for key in shared_keys}
    run_names = [m.name for m in metas]
    grand_agg = {
        name: aggregate_text_metrics([res for key in shared_keys for _pi, res in scored[key][name].per_page])
        for name in run_names
    }
    grand_candidates = {
        name: (sum(scored[k][name].n_ocr for k in shared_keys), sum(scored[k][name].n_ocr_touching_gt for k in shared_keys))
        for name in run_names
    }
    grand_rows = {name: {kind: [row for key in shared_keys for row in timings[key][name][kind]]
                         for kind in (timing.RUN_KIND, timing.DOC_KIND)} for name in run_names}

    builder = ReportBuilder(f"Pipeline benchmark ({ts})")
    blocks: list[str] = ["compared runs:\n" + "\n".join(
        f"  {m.name}  (p2={m.p2}, p3={m.p3})  {m.path}" for m in metas)]

    add_inputs_header(builder, metas, {key: [r[key] for r in runs] for key in shared_keys})
    grand_timing, grand_chart = _timing(grand_rows, "all inputs", "aggregate", charts_dir, out_dir, blocks)
    add_summary(builder, grand_agg, grand_candidates,
                {run: s.get(timing.RUN_KIND, {}) for run, s in grand_timing.items()})
    add_text_sections(builder, grand_agg, _text_charts(grand_agg, "all inputs", "aggregate", charts_dir, out_dir))
    add_timing_sections(builder, grand_timing, grand_chart)
    blocks.append(format_aggregate_comparison(grand_agg, title="all inputs -- text"))

    viewer_lines = ["# Open each benchmarked input in the viewer (runs side by side):"]
    for key in shared_keys:
        kslug = short_slug(key)
        agg_by_run = {name: score.aggregate for name, score in scored[key].items()}
        blocks.append(format_aggregate_comparison(agg_by_run, title=f"{key} -- text"))
        for name, agg in agg_by_run.items():
            if agg is not None:
                blocks.append(f"[{key} / {name}] " + format_confusion_table(agg))

        builder.add_key_section(key)
        add_text_sections(builder, agg_by_run, _text_charts(agg_by_run, key, kslug, charts_dir, out_dir))
        key_timing, key_chart = _timing(timings[key], key, kslug, charts_dir, out_dir, blocks)
        add_timing_sections(builder, key_timing, key_chart)
        dirs = " ".join(f'"{r[key].doc_dir.resolve()}"' for r in runs)
        command = f"python scripts/pipeline_report_viewer.py {dirs}"
        builder.add_viewer_command(key, command)
        viewer_lines.append(command)

    (out_dir / "report.html").write_text(builder.render(), encoding="utf-8")
    (out_dir / "benchmark.txt").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    (out_dir / "runs.json").write_text(json.dumps({
        "runs": [str(m.path) for m in metas],
        "iou_edge_min": args.iou_threshold,
        "shared_keys": shared_keys,
    }, indent=2), encoding="utf-8")
    (out_dir / "viewer_commands.txt").write_text("\n".join(viewer_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
