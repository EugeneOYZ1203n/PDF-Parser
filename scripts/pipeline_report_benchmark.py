"""Compare 1-2 benchmark report folders.

Each folder is a `generate_pipeline_report.py` run with `benchmark: true`:
per input, `ground_truth_<text_type>.json` (only the non-empty types) and
one sub-folder per input form it ran (`original/`, `vectorised/`,
`rasterised/`, each with its own `dump.json`), plus a `benchmark.json`
marker. A pre-`inputs` folder (one `<input>/dump.json` holding the
vectorised run and the rasterised run's `raster_*` fields) still loads.
Each form is scored on its own types (`rastervec/Evaluation/forms.py`), and
the report shows, per input, a one-line-per-form summary followed by one
group per form (Text / Vector / Counts); comparing two runs matches inputs
and forms.

This script matches the folders on the inputs they **share** (by input
identity: `pdf:<stem>` = native_to_vector-only, `labels:<stem>` = whatever
label sources the run had), scores each shared input's text metrics
(`metrics.evaluate_text_metrics`) and vector metrics
(`vector_metrics.evaluate_vector_metrics`: the rasterised-PDF run's
vectors -- P2's output -- against the union of the
`vector_to_raster`/`original_raster` geometry labels, by the Liu & Dori
(1997) protocol, see `docs/RasterToVectorEvaluation.md`; `original_vector`
is scored as a text-provenance type only, not at the vector-geometry level),
and writes one
self-contained `report.html` -- one section per shared key, broken into
Text (Char/Word/Font size/Rotation/Bbox/Vector classification/Reading
order/Confusion), Vector (Liu-Dori headline/Q_v components/splits/
fragmentation/pixel level/Length/Direction) and Timing (per-page
wall-clock per phase + P3 sub-step and report-generation cost, vectorised
and rasterised runs, with a stacked-bar chart) subsections, each with
linked chart PNGs, a per-character OCR accuracy table per text type
(detected / dropped / unreached / misclassified / inserted), the run's own
diagnostic image galleries (`*_images/` folders, linked in place), and a
ready-to-paste `pipeline_report_viewer.py` command. No example crops are
rendered -- `pipeline_report_viewer.py` over the run folders (incl. the
`benchmark__extra_*` layers) is where individual errors are inspected. Output (timestamped folder under `outputs/pipeline_report_benchmark/`):

    report.html             the primary deliverable (see above)
    benchmark.txt           supplementary plain-text dump (grep/diff-friendly)
    runs.json               the compared folders + threshold
    viewer_commands.txt     a viewer CLI line per shared input and form
    charts/                 <key>__<form>__*.png, aggregate__<form>__*.png

This module's own run-loading + scoring (`RunEntry`/`_merge_gt`/`_load_run`/
`_score_text`/`_score_vectors`) lives in
`rastervec/Evaluation/Evaluate/benchmark_run_loading.py`, and the HTML
section builders (`_add_text_sections`/`_add_vector_sections`) in
`rastervec/Evaluation/Evaluate/benchmark_report_sections.py` -- all
re-exported here since some names are
imported directly from this module's own path.

    .venv/Scripts/python.exe scripts/pipeline_report_benchmark.py \
        --run outputs/pipeline_report/<ts1>__benchmark \
        --run outputs/pipeline_report/<ts2>__benchmark_legacy
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rastervec.Evaluation.Evaluate import charts, timing
from rastervec.Evaluation.Evaluate.benchmark import (
    format_aggregate_comparison,
    format_confusion_table,
    format_fast_cluster_comparison,
    format_retry_stats_comparison,
    format_variant_timing_comparison,
    format_vector_aggregate_comparison,
)
from rastervec.Evaluation.Evaluate.html_report import ReportBuilder
from rastervec.Evaluation.Evaluate.metrics import (
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

from rastervec.Evaluation.Evaluate.benchmark_report_sections import (  # noqa: F401 -- re-exported for callers/tests
    _add_counts_section,
    _add_fast_cluster_section,
    _add_form_summary,
    _add_retry_stats_section,
    _add_text_sections,
    _add_timing_sections,
    _add_vector_sections,
)
from rastervec.Evaluation.Evaluate.benchmark_run_loading import (  # noqa: F401 -- re-exported for callers/tests
    RunEntry,
    _label_counts,
    _load_counts,
    _load_fast_cluster_stats,
    _load_retry_stats,
    _load_run,
    _load_timings,
    _merge_gt,
    _score_text,
    _score_vectors,
    _sum_counts,
    entry_forms,
    form_dir,
)
from rastervec.Evaluation.forms import FORM_TEXT_TYPES, FORM_TITLES, FORM_VECTOR_GT, FORMS, type_label

_LOG = get_logger("pipeline_report_benchmark")

_VENV_PY = ".venv/Scripts/python.exe"


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_") or "key"


def _short_slug(text: str, max_len: int = 16) -> str:
    """A filename-safe slug capped to `max_len` chars (+ a short hash
    suffix for uniqueness) -- unlike `_slug`, safe to compose several of
    into one path component without hitting Windows' ~260-char MAX_PATH."""
    full = _slug(text)
    if len(full) <= max_len:
        return full
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    return f"{full[:max_len]}_{h}"


def _link_from(out_dir: Path, target: Path) -> str:
    """`target` as an `<img src>` relative to `out_dir` (the report.html
    folder), or a `file://` URI when no relative path exists (a different
    Windows drive)."""
    try:
        return Path(os.path.relpath(target.resolve(), out_dir.resolve())).as_posix()
    except ValueError:
        return target.resolve().as_uri()


def _word_f1(result: "TextMetricSuiteResult | None", types: "tuple[str, ...]") -> float:
    """Word-overlap F1 pooled over `types` (micro: numerators and
    denominators summed) -- the per-form summary's text score."""
    if result is None:
        return math.nan
    p_num = p_den = r_num = r_den = 0.0
    for t in types:
        wo = result.by_type[t].word_overlap
        if wo.precision.applicable:
            p_num += wo.precision.numerator
            p_den += wo.precision.denominator
        if wo.recall.applicable:
            r_num += wo.recall.numerator
            r_den += wo.recall.denominator
    if not p_den or not r_den:
        return math.nan
    p, r = p_num / p_den, r_num / r_den
    return 2 * p * r / (p + r) if p + r else 0.0


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
    charts_dir.mkdir(parents=True, exist_ok=True)

    builder = ReportBuilder(f"Pipeline benchmark report ({ts})")
    blocks: list[str] = []
    # {run: {kind: [flattened per-page timing row]}} over every shared key
    grand_timing: "dict[str, dict[str, list[dict]]]" = {}
    # {run: {"total": N, "passed": P}} FAST cluster counts, summed over every shared key
    grand_fast_cluster: "dict[str, dict]" = {}
    # {run: {"0": N, "1": N, "2": N, "3": N, "failed": N}} retry counts, summed over every shared key
    grand_retry_stats: "dict[str, dict]" = {}

    def _timing_block(
        rows_by_run: "dict[str, dict[str, list[dict]]]", label: str, cslug: str,
    ) -> None:
        """Summarise, chart, add the HTML Timing section, and append the
        benchmark.txt timing tables for one key (or the grand aggregate)."""
        summaries = {
            run: {kind: timing.summarize_timings(rows) for kind, rows in by_kind.items()}
            for run, by_kind in rows_by_run.items()
        }
        chart_paths: "dict[str, Path]" = {}
        for kind in timing.RUN_KINDS:
            per_run = {run: s_.get(kind, {}) for run, s_ in summaries.items()}
            if not any(per_run.values()):
                continue
            path = charts_dir / f"{cslug}__timing__{kind}.png"
            charts.timing_chart(per_run, title=f"{label} {kind} run: mean s/page", path=path)
            chart_paths[kind] = path.relative_to(out_dir)
            for run, summary in per_run.items():
                blocks.append(timing.format_timing_table(
                    summary, title=f"[{label} / {run}] {kind} run wall-clock per page (seconds)"))
            if len(per_run) > 1:
                blocks.append(format_variant_timing_comparison(
                    per_run, title=f"[{label}] {kind} run median seconds per page by run"))
        for run, s_ in summaries.items():
            if s_.get(timing.DOC_KIND):
                blocks.append(timing.format_timing_table(
                    s_[timing.DOC_KIND],
                    title=f"[{label} / {run}] report generation per document (seconds)"))
        _add_timing_sections(builder, summaries, chart_paths)

    def _img_html(paths: "list[Path]") -> str:
        existing = [p for p in paths if p.is_file()]
        if not existing:
            return ""
        return '<div class="charts">' + "".join(
            f'<img src="{p.relative_to(out_dir).as_posix()}">' for p in existing
        ) + "</div>"

    def _text_block(
        text_by_run: "dict[str, TextMetricSuiteResult | None]", form: str, label: str, slug: str, *,
        full: bool = True,
    ) -> None:
        """One form's Text group (+ charts) for one key or the grand aggregate."""
        types = FORM_TEXT_TYPES[form]
        labels = {t: type_label(form, t) for t in types}
        if not any(v is not None for v in text_by_run.values()):
            return
        _add_text_sections(builder, text_by_run, types=types, labels=labels,
                           group=f"{FORM_TITLES[form]} -- Text")
        kw = dict(types=types, labels=labels)
        paths = [charts_dir / f"{slug}__labels.png", charts_dir / f"{slug}__char.png",
                 charts_dir / f"{slug}__bbox.png"]
        charts.label_description_chart(text_by_run, title=f"{label} labels", path=paths[0], **kw)
        charts.char_word_overlap_chart(text_by_run, field="char_overlap", title=f"{label} char recall",
                                       path=paths[1], **kw)
        charts.bbox_accuracy_chart(text_by_run, title=f"{label} bbox IoU", path=paths[2], **kw)
        if full:
            more = [charts_dir / f"{slug}__{n}.png" for n in ("word", "rotation", "funnel", "reading_order")]
            charts.char_word_overlap_chart(text_by_run, field="word_overlap", title=f"{label} word recall",
                                           path=more[0], **kw)
            charts.rotation_chart(text_by_run, title=f"{label} rotation", path=more[1], **kw)
            if any(t in types for t in ("native_to_vector", "original_vector")):
                charts.funnel_chart(text_by_run, title=f"{label} funnel", path=more[2], **kw)
            charts.reading_order_chart(text_by_run, title=f"{label} reading order", path=more[3], **kw)
            paths += more
        builder.add_raw_html(_img_html(paths))

        font_html = ["<h4>Font size distribution</h4>"]
        for text_type in types:
            imgs = []
            for name, agg in text_by_run.items():
                path = charts_dir / f"{slug}__{_short_slug(name)}__{text_type}__font_size.png"
                charts.font_size_histogram_chart(
                    agg, text_type, title=f"{label} {name} {labels[text_type]} font size", path=path)
                imgs.append(path)
            html = _img_html(imgs)
            if html:
                font_html.append(f"<p>{labels[text_type]}</p>" + html)
        builder.add_raw_html("".join(font_html))

    def _vector_block(
        vector_by_run: "dict[str, VectorMetricSuiteResult | None]", form: str, label: str, slug: str,
    ) -> None:
        if not any(v is not None for v in vector_by_run.values()):
            return
        _add_vector_sections(builder, vector_by_run, vector_types=FORM_VECTOR_GT[form],
                             group=f"{FORM_TITLES[form]} -- Vector")
        names = ("headline", "components", "fragmentation", "consolidation", "quality", "direction")
        paths = {n: charts_dir / f"{slug}__vector_{n}.png" for n in names}
        charts.vector_headline_chart(vector_by_run, title=f"{label} vector detection (Liu-Dori + pixel)",
                                     path=paths["headline"])
        charts.vector_components_chart(vector_by_run, title=f"{label} Q_v components", path=paths["components"])
        charts.fragmentation_chart(vector_by_run, side="gt", title=f"{label} predicted lines per GT line |D(g)|",
                                   path=paths["fragmentation"])
        charts.fragmentation_chart(vector_by_run, side="pred", title=f"{label} GT lines per predicted line |G(k)|",
                                   path=paths["consolidation"])
        charts.vector_quality_chart(vector_by_run, title=f"{label} vector quality", path=paths["quality"])
        charts.direction_histogram_chart(vector_by_run, title=f"{label} direction histogram",
                                         path=paths["direction"])
        builder.add_raw_html(_img_html(list(paths.values())))

    def _scores_cell(text_res, vec_res, form: str) -> dict:
        return {
            "text_f1": _word_f1(text_res, FORM_TEXT_TYPES[form]),
            "d_v": vec_res.liu_dori.d_v if vec_res is not None else math.nan,
            "vri": vec_res.liu_dori.vri if vec_res is not None else math.nan,
        }

    # {form: {run: [per-page result]}} over every shared key
    grand_text: "dict[str, dict[str, list[TextMetricSuiteResult]]]" = {}
    grand_vector: "dict[str, dict[str, list[VectorMetricSuiteResult]]]" = {}
    grand_counts: "dict[str, dict[str, dict]]" = {}
    viewer_lines: list[str] = []

    for key in sorted(shared_keys):
        kslug = _short_slug(key)
        entries = {r[key].run_name: r[key] for r in runs}
        forms = [f for f in FORMS if any(f in entry_forms(e) for e in entries.values())]
        builder.add_key_section(key)

        scored = {}
        for form in forms:
            ts = {n: _score_text(e, form, cfg) for n, e in entries.items()}
            vs = {n: _score_vectors(e, form, vcfg) for n, e in entries.items()}
            scored[form] = (
                {n: (r[1] if r else None) for n, r in ts.items()},
                {n: (r[1] if r else None) for n, r in vs.items()},
            )
            for n, r in ts.items():
                if r:
                    grand_text.setdefault(form, {}).setdefault(n, []).extend(res for _pi, res, _g in r[0])
            for n, r in vs.items():
                if r:
                    grand_vector.setdefault(form, {}).setdefault(n, []).extend(res for _pi, res in r[0])
        _add_form_summary(builder, {
            FORM_TITLES[form]: {
                n: _scores_cell(scored[form][0][n], scored[form][1][n], form)
                for n in entries if form in entry_forms(entries[n])
            }
            for form in forms
        })

        for form in forms:
            title = FORM_TITLES[form]
            types = FORM_TEXT_TYPES[form]
            labels = {t: type_label(form, t) for t in types}
            fslug = f"{kslug}__{form}"
            text_by_run, vector_by_run = scored[form]
            blocks.append(format_aggregate_comparison(
                text_by_run, title=f"{key} -- {title} -- text", types=types, labels=labels))
            for name, agg in text_by_run.items():
                if agg is not None:
                    blocks.append(f"[{key} / {title} / {name}] "
                                  + format_confusion_table(agg, types=types, labels=labels))
            if FORM_VECTOR_GT[form]:
                blocks.append(format_vector_aggregate_comparison(
                    vector_by_run, title=f"{key} -- {title} -- vectors"))

            counts_by_run = {n: _load_counts(e, form) for n, e in entries.items()}
            for n, c in counts_by_run.items():
                if c:
                    _sum_counts(grand_counts.setdefault(form, {}).setdefault(n, {}), c)
            fast_by_run = {n: _load_fast_cluster_stats(e, form) for n, e in entries.items()}
            retry_by_run = {n: _load_retry_stats(e, form) for n, e in entries.items()}
            for name, stats in fast_by_run.items():
                if stats is not None:
                    acc = grand_fast_cluster.setdefault(name, {"total": 0, "passed": 0})
                    acc["total"] += stats["total"]
                    acc["passed"] += stats["passed"]
            for name, stats in retry_by_run.items():
                if stats is not None:
                    acc = grand_retry_stats.setdefault(name, {"0": 0, "1": 0, "2": 0, "3": 0, "failed": 0})
                    for k in acc:
                        acc[k] += stats.get(k, 0)
            if any(v is not None for v in fast_by_run.values()):
                blocks.append(format_fast_cluster_comparison(
                    fast_by_run, title=f"{key} -- {title} -- clusters dropped by FAST"))
            if any(v is not None for v in retry_by_run.values()):
                blocks.append(format_retry_stats_comparison(
                    retry_by_run, title=f"{key} -- {title} -- blank-recognition retries"))

            _text_block(text_by_run, form, f"{key} {title}", fslug)
            _vector_block(vector_by_run, form, f"{key} {title}", fslug)
            _add_counts_section(
                builder, counts_by_run,
                label_counts={n: _label_counts(e, form) for n, e in entries.items()},
                group=f"{title} -- Counts",
            )
            _add_fast_cluster_section(builder, fast_by_run)
            _add_retry_stats_section(builder, retry_by_run)

            # pipeline diagnostic image galleries, linked in place, never copied
            for name, entry in entries.items():
                if form not in entry_forms(entry):
                    continue
                for folder in sorted(form_dir(entry, form).glob("*_images")):
                    files = sorted(folder.iterdir())[:5]
                    if files:
                        builder.add_image_gallery(f"{name} / {title}", folder.name,
                                                  [_link_from(out_dir, f) for f in files])

            dirs = " ".join(
                f'"{form_dir(e, form).resolve()}"' for e in entries.values() if form in entry_forms(e)
            )
            command = f"{_VENV_PY} scripts/pipeline_report_viewer.py {dirs}"
            builder.add_viewer_command(f"{key} / {title}", command)
            viewer_lines.append(command)

        timing_rows = {n: _load_timings(e) for n, e in entries.items()}
        _timing_block(timing_rows, key, kslug)
        for name, by_kind in timing_rows.items():
            for kind, rows in by_kind.items():
                grand_timing.setdefault(name, {}).setdefault(kind, []).extend(rows)

    # ---- grand aggregate over every shared key ----
    grand_header_done = False

    def _grand_header() -> None:
        nonlocal grand_header_done
        if not grand_header_done:
            builder.add_key_section("(grand aggregate over every shared input)")
            grand_header_done = True

    for form in FORMS:
        text_agg = {n: aggregate_text_metrics(rs) for n, rs in grand_text.get(form, {}).items()}
        vector_agg = {n: aggregate_vector_metrics(rs) for n, rs in grand_vector.get(form, {}).items()}
        if not text_agg and not vector_agg:
            continue
        _grand_header()
        _text_block(text_agg, form, f"(all inputs) {FORM_TITLES[form]}", f"aggregate__{form}", full=False)
        _vector_block(vector_agg, form, f"(all inputs) {FORM_TITLES[form]}", f"aggregate__{form}")
        _add_counts_section(builder, grand_counts.get(form, {}), group=f"{FORM_TITLES[form]} -- Counts")

    if grand_fast_cluster:
        _grand_header()
        blocks.append(format_fast_cluster_comparison(
            grand_fast_cluster, title="(all inputs) -- clusters dropped by FAST",
        ))
        _add_fast_cluster_section(builder, grand_fast_cluster)

    if grand_retry_stats:
        _grand_header()
        blocks.append(format_retry_stats_comparison(
            grand_retry_stats, title="(all inputs) -- blank-recognition retries",
        ))
        _add_retry_stats_section(builder, grand_retry_stats)

    if grand_timing:
        _grand_header()
        _timing_block(grand_timing, "(all inputs)", "aggregate")

    header = "compared runs:\n" + "\n".join(f"  {Path(d).name}" for d in args.run)
    (out_dir / "benchmark.txt").write_text(
        header + "\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    (out_dir / "runs.json").write_text(json.dumps({
        "runs": [str(Path(d).resolve()) for d in args.run],
        "iou_edge_min": args.iou_threshold,
        "shared_keys": sorted(shared_keys),
    }, indent=2), encoding="utf-8")

    lines = ["# Open each benchmarked input / form in the viewer (pipelines side by side):", *viewer_lines]
    (out_dir / "viewer_commands.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / "report.html").write_text(builder.render(), encoding="utf-8")

    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
