"""Compare 1-2 benchmark report folders.

Each folder is a `generate_pipeline_report.py` run with `benchmark: true`.
Such a run scores **one** combined pipeline run per page
(`convert_page_to_vector_text`) against ground truth for up to 4 text types
and 2 vector types, and embeds `<pdf-stem>/dump.json` +
`ground_truth_<text_type>.json` (only the non-empty types) + a
`benchmark.json` marker.

This script matches the folders on the inputs they **share** (by input
identity: `pdf:<stem>` = native_to_vector-only, `labels:<stem>` = whatever
label sources the run had), scores each shared input's text metrics
(`metrics.evaluate_text_metrics`) and vector metrics
(`vector_metrics.evaluate_vector_metrics`, GT-only for both
`vector_to_raster`/`original_raster` -- no raster-tracing pipeline stage
exists; `original_vector` is scored as a text-provenance type only, not
at the vector-geometry level), and writes one
self-contained `report.html` -- one section per shared key, broken into
Text (Char/Word/Font size/Rotation/Bbox/Vector classification/Reading
order/Confusion) and Vector (Count/Endpoint/Property) subsections, each
with linked chart PNGs, illustrated error-example galleries (extra
predictions / missed GT / confusion misreads, cropped from the run's own
`converted_p<N>.pdf`), the run's own diagnostic image galleries
(`*_images/` folders), and a ready-to-paste `pipeline_report_viewer.py`
command. Output (timestamped folder under `outputs/pipeline_report_benchmark/`):

    report.html             the primary deliverable (see above)
    benchmark.txt           supplementary plain-text dump (grep/diff-friendly)
    runs.json               the compared folders + threshold
    viewer_commands.txt     a viewer CLI line per shared input
    charts/                 <key>__aggregate__*.png, aggregate__*.png
    examples/                <key>__<run>__<type>__<category>__<n>.png crops

This module's own run-loading + scoring (`RunEntry`/`_merge_gt`/`_load_run`/
`_score_text`/`_score_vectors`) lives in `benchmark_run_loading.py`, the
illustrated-example collection (`_collect_examples`/`_crop_to_png`) in
`benchmark_examples.py`, and the HTML section builders
(`_add_text_sections`/`_add_vector_sections`) in
`benchmark_report_sections.py` -- all re-exported here since some names are
imported directly from this module's own path.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rastervec.Evaluation.Evaluate import charts
from rastervec.Evaluation.Evaluate.benchmark import (
    format_aggregate_comparison,
    format_confusion_table,
    format_vector_aggregate_comparison,
)
from rastervec.Evaluation.Evaluate.html_report import ReportBuilder
from rastervec.Evaluation.Evaluate.metrics import (
    TEXT_TYPES,
    MetricConfig,
    TextMetricSuiteResult,
    aggregate_text_metrics,
)
from rastervec.Evaluation.Evaluate.vector_metrics import (
    VectorMetricConfig,
    VectorMetricSuiteResult,
    aggregate_vector_metrics,
)
from rastervec.commons.logging_setup import configure_logging, get_logger
from rastervec.commons.paths import output_dir

from scripts.benchmark_examples import (  # noqa: F401 -- re-exported for callers/tests
    _EXAMPLE_CAP,
    _collect_examples,
    _crop_to_png,
    _short_slug,
    _slug,
)
from scripts.benchmark_report_sections import (  # noqa: F401 -- re-exported for callers/tests
    _add_text_sections,
    _add_vector_sections,
    _fmt_property_rows,
)
from scripts.benchmark_run_loading import (  # noqa: F401 -- re-exported for callers/tests
    RunEntry,
    _load_run,
    _merge_gt,
    _score_text,
    _score_vectors,
)

_LOG = get_logger("pipeline_report_benchmark")

_VENV_PY = ".venv/Scripts/python.exe"


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
    vcfg = VectorMetricConfig()

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
    examples_dir = out_dir / "examples"
    charts_dir.mkdir(parents=True, exist_ok=True)

    builder = ReportBuilder(f"Pipeline benchmark report ({ts})")
    blocks: list[str] = []
    grand_text: "dict[str, list[TextMetricSuiteResult]]" = {}
    grand_vector: "dict[str, list[VectorMetricSuiteResult]]" = {}

    for key in sorted(shared_keys):
        kslug = _short_slug(key)
        text_scored = {r[key].run_name: _score_text(r[key], cfg) for r in runs}
        vector_scored = {r[key].run_name: _score_vectors(r[key], vcfg) for r in runs}

        text_by_run = {name: agg for name, (_pp, agg) in text_scored.items()}
        vector_by_run = {name: agg for name, (_pp, agg) in vector_scored.items()}
        blocks.append(format_aggregate_comparison(text_by_run, title=f"{key} -- text"))
        for name, agg in text_by_run.items():
            if agg is not None:
                blocks.append(f"[{key} / {name}] " + format_confusion_table(agg))
        blocks.append(format_vector_aggregate_comparison(vector_by_run, title=f"{key} -- vectors"))

        builder.add_key_section(key)
        _add_text_sections(builder, text_by_run)
        _add_vector_sections(builder, vector_by_run)

        charts.label_description_chart(
            text_by_run, title=f"{key} aggregate labels", path=charts_dir / f"{kslug}__aggregate__labels.png")
        charts.char_word_overlap_chart(
            text_by_run, field="char_overlap", title=f"{key} aggregate char recall",
            path=charts_dir / f"{kslug}__aggregate__char.png")
        charts.char_word_overlap_chart(
            text_by_run, field="word_overlap", title=f"{key} aggregate word recall",
            path=charts_dir / f"{kslug}__aggregate__word.png")
        charts.bbox_accuracy_chart(
            text_by_run, title=f"{key} aggregate bbox IoU", path=charts_dir / f"{kslug}__aggregate__bbox.png")
        charts.rotation_chart(
            text_by_run, title=f"{key} aggregate rotation", path=charts_dir / f"{kslug}__aggregate__rotation.png")
        charts.funnel_chart(
            text_by_run, title=f"{key} aggregate funnel", path=charts_dir / f"{kslug}__aggregate__funnel.png")
        charts.reading_order_chart(
            text_by_run, title=f"{key} aggregate reading order",
            path=charts_dir / f"{kslug}__aggregate__reading_order.png")

        chart_paths_by_subsection: "dict[str, list[Path]]" = {
            "labels": [charts_dir / f"{kslug}__aggregate__labels.png"],
            "char": [charts_dir / f"{kslug}__aggregate__char.png"],
            "word": [charts_dir / f"{kslug}__aggregate__word.png"],
            "bbox": [charts_dir / f"{kslug}__aggregate__bbox.png"],
            "rotation": [charts_dir / f"{kslug}__aggregate__rotation.png"],
            "funnel": [charts_dir / f"{kslug}__aggregate__funnel.png"],
            "reading_order": [charts_dir / f"{kslug}__aggregate__reading_order.png"],
        }
        for group, paths in chart_paths_by_subsection.items():
            existing = [p for p in paths if p.is_file()]
            if existing:
                builder.add_raw_html(
                    '<div class="charts">' + "".join(
                        f'<img src="{p.relative_to(out_dir).as_posix()}">' for p in existing
                    ) + "</div>"
                )

        font_size_paths: "dict[str, list[Path]]" = {t: [] for t in TEXT_TYPES}
        for name, agg in text_by_run.items():
            nslug = _short_slug(name)
            for text_type in TEXT_TYPES:
                path = charts_dir / f"{kslug}__{nslug}__{text_type}__font_size.png"
                charts.font_size_histogram_chart(
                    agg, text_type, title=f"{key} {name} {text_type} font size", path=path)
                font_size_paths[text_type].append(path)
            charts.extra_chars_table_image(
                agg, title=f"{key} {name} extra predicted chars",
                path=charts_dir / f"{kslug}__{nslug}__extra_chars.png")

        font_size_html = ["<h4>Font size distribution</h4>"]
        for text_type in TEXT_TYPES:
            imgs = [p for p in font_size_paths[text_type] if p.is_file()]
            if imgs:
                font_size_html.append(f"<p>{text_type}</p>")
                font_size_html.append(
                    '<div class="charts">' + "".join(
                        f'<img src="{p.relative_to(out_dir).as_posix()}">' for p in imgs
                    ) + "</div>"
                )
        builder.add_raw_html("".join(font_size_html))

        if any(v is not None for v in vector_by_run.values()):
            charts.vector_count_chart(
                vector_by_run, title=f"{key} aggregate vector count",
                path=charts_dir / f"{kslug}__aggregate__vector_count.png")
            charts.endpoint_accuracy_chart(
                vector_by_run, title=f"{key} aggregate endpoint accuracy",
                path=charts_dir / f"{kslug}__aggregate__vector_endpoint.png")
            builder.add_raw_html(
                '<div class="charts">'
                f'<img src="{(charts_dir / f"{kslug}__aggregate__vector_count.png").relative_to(out_dir).as_posix()}">'
                f'<img src="{(charts_dir / f"{kslug}__aggregate__vector_endpoint.png").relative_to(out_dir).as_posix()}">'
                '</div>'
            )

        # ---- illustrated error examples (one run's pages at a time) ----
        for r in runs:
            entry = r[key]
            per_page, _agg = text_scored[entry.run_name]
            examples = _collect_examples(entry, per_page, examples_dir, kslug)
            for text_type in TEXT_TYPES:
                for category, label in (
                    ("extra_prediction", "Extra predictions"),
                    ("missed_gt", "Missed GT"),
                    ("confusion", "Confusion misreads"),
                ):
                    cards = examples.get((text_type, category), [])
                    if cards:
                        builder.add_error_examples(f"{label} [{entry.run_name}]", text_type, cards)

        # ---- pipeline diagnostic image galleries ----
        gallery_dir = out_dir / "gallery"
        for r in runs:
            entry = r[key]
            for folder in sorted(entry.doc_dir.glob("*_images")):
                files = sorted(folder.iterdir())[:5]
                if not files:
                    continue
                rslug = _short_slug(entry.run_name)
                gallery_dir.mkdir(parents=True, exist_ok=True)
                copied: "list[Path]" = []
                for f in files:
                    dst = gallery_dir / f"{kslug}__{rslug}__{folder.name}__{f.name}"
                    dst.write_bytes(f.read_bytes())
                    copied.append(Path("gallery") / dst.name)
                builder.add_image_gallery(entry.run_name, folder.name, copied)

        for name, (pp, _agg) in text_scored.items():
            grand_text.setdefault(name, []).extend(r for _pi, r, _g in pp)
        for name, (pp, _agg) in vector_scored.items():
            grand_vector.setdefault(name, []).extend(r for _pi, r in pp)

        dirs = " ".join(f'"{r[key].doc_dir.resolve()}"' for r in runs)
        builder.add_viewer_command(key, f"{_VENV_PY} scripts/pipeline_report_viewer.py {dirs}")

    # ---- grand aggregate over every shared key ----
    grand_text_agg = {name: aggregate_text_metrics(rs) for name, rs in grand_text.items()}
    grand_vector_agg = {name: aggregate_vector_metrics(rs) for name, rs in grand_vector.items()}
    if grand_text_agg:
        charts.label_description_chart(
            grand_text_agg, title="aggregate labels (all inputs)", path=charts_dir / "aggregate__labels.png")
        charts.char_word_overlap_chart(
            grand_text_agg, field="char_overlap", title="aggregate char recall (all inputs)",
            path=charts_dir / "aggregate__char.png")
        charts.bbox_accuracy_chart(
            grand_text_agg, title="aggregate bbox IoU (all inputs)", path=charts_dir / "aggregate__bbox.png")

        builder.add_key_section("(grand aggregate over every shared input)")
        _add_text_sections(builder, grand_text_agg)
        _add_vector_sections(builder, grand_vector_agg)
        builder.add_raw_html(
            '<div class="charts">' + "".join(
                f'<img src="{p.relative_to(out_dir).as_posix()}">'
                for p in (charts_dir / "aggregate__labels.png", charts_dir / "aggregate__char.png", charts_dir / "aggregate__bbox.png")
                if p.is_file()
            ) + "</div>"
        )

        grand_font_size_html = ["<h4>Font size distribution</h4>"]
        for name, agg in grand_text_agg.items():
            nslug = _short_slug(name)
            for text_type in TEXT_TYPES:
                path = charts_dir / f"aggregate__{nslug}__{text_type}__font_size.png"
                charts.font_size_histogram_chart(
                    agg, text_type, title=f"(all inputs) {name} {text_type} font size", path=path)
        for text_type in TEXT_TYPES:
            imgs = [
                charts_dir / f"aggregate__{_short_slug(name)}__{text_type}__font_size.png"
                for name in grand_text_agg
            ]
            imgs = [p for p in imgs if p.is_file()]
            if imgs:
                grand_font_size_html.append(f"<p>{text_type}</p>")
                grand_font_size_html.append(
                    '<div class="charts">' + "".join(
                        f'<img src="{p.relative_to(out_dir).as_posix()}">' for p in imgs
                    ) + "</div>"
                )
        builder.add_raw_html("".join(grand_font_size_html))

    header = "compared runs:\n" + "\n".join(f"  {Path(d).name}" for d in args.run)
    (out_dir / "benchmark.txt").write_text(
        header + "\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    (out_dir / "runs.json").write_text(json.dumps({
        "runs": [str(Path(d).resolve()) for d in args.run],
        "iou_edge_min": args.iou_threshold,
        "shared_keys": sorted(shared_keys),
    }, indent=2), encoding="utf-8")

    lines = ["# Open each benchmarked input in the viewer (pipelines side by side):"]
    for key in sorted(shared_keys):
        dirs = " ".join(f'"{r[key].doc_dir.resolve()}"' for r in runs)
        lines.append(f"{_VENV_PY} scripts/pipeline_report_viewer.py {dirs}")
    (out_dir / "viewer_commands.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / "report.html").write_text(builder.render(), encoding="utf-8")

    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
