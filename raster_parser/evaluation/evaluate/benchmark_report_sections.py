"""The sections of `scripts/pipeline_report_benchmark.py`'s `report.html`:

- `add_inputs_header`: what was run -- each run folder (backends, config) and each input
  (source PDF, labels, pages, ground-truth types);
- `add_summary`: the aggregate over every shared input, one row per (text type, metric) and one
  column per run, plus time per page;
- `add_text_sections`: per input, the full text tables (labels, char / word overlap, bbox,
  rotation, per-character accuracy, extra characters);
- `add_timing_sections`: wall-clock per page and report generation per document.
"""
from __future__ import annotations

import math
from pathlib import Path

from raster_parser.evaluation.evaluate import timing
from raster_parser.evaluation.evaluate.adapters import text_type_of
from raster_parser.evaluation.evaluate.benchmark_format import fmt_ratio
from raster_parser.evaluation.evaluate.html_report import ReportBuilder
from raster_parser.evaluation.evaluate.metrics import TEXT_TYPES, TextMetricSuiteResult, f1_from


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "n/a"


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Header + summary
# ---------------------------------------------------------------------------
def add_inputs_header(builder: ReportBuilder, runs: list, entries_by_key: "dict[str, list]") -> None:
    """`runs`: `RunMeta` per run folder; `entries_by_key`: `{key: [RunEntry per run]}` for every
    input the runs share."""
    builder.add_key_section("Inputs")
    builder.add_subsection(
        "Runs", ["run", "p2", "p3", "final_stage", "folder", "config"],
        [[r.name, r.p2, r.p3, r.final_stage or "-", str(r.path), r.config_path or "-"] for r in runs],
    )
    rows = []
    for key, entries in entries_by_key.items():
        first = entries[0]
        types = ", ".join(t for t in TEXT_TYPES if any(text_type_of(e) == t for e in first.gt.entries)) or "-"
        rows.append([key, first.source_pdf, first.labels or "(native text only)",
                     ", ".join(str(p) for p in first.pages), types])
    builder.add_subsection("Files scored", ["input", "source PDF", "labels", "pages", "ground truth"], rows)


def add_summary(
    builder: ReportBuilder, agg_by_run: "dict[str, TextMetricSuiteResult | None]",
    candidates_by_run: "dict[str, tuple[int, int]]", timing_by_run: "dict[str, dict]",
) -> None:
    """`candidates_by_run`: `{run: (n_ocr, n_ocr_touching_gt)}`; `timing_by_run`: `{run:
    timing.summarize_timings(...)}` of the per-page rows."""
    builder.add_key_section("Summary (all inputs)")
    runs = list(agg_by_run)
    rows: list[list[str]] = []
    for t in TEXT_TYPES:
        for name, value in (
            ("labels", lambda r: str(r.label_stats.label_count)),
            ("char F1", lambda r: _fmt(f1_from(r.char_overlap.recall, r.char_overlap.precision))),
            ("char recall", lambda r: _fmt(r.char_overlap.recall.value)),
            ("word F1", lambda r: _fmt(f1_from(r.word_overlap.recall, r.word_overlap.precision))),
            ("bbox mean IoU", lambda r: _fmt(r.bbox_accuracy.mean_iou.value)),
            ("rotation correct", lambda r: _pct(r.rotation.buckets.correct, r.rotation.n_localized)),
        ):
            rows.append([t, name] + [
                "n/a" if agg_by_run[run] is None else value(agg_by_run[run].by_type[t]) for run in runs
            ])
    rows.append(["all", "candidate precision"] + [
        f"{_fmt(touch / n if n else math.nan)} ({touch}/{n} OCR readings touch a label)"
        for n, touch in (candidates_by_run[run] for run in runs)
    ])
    for key, name in ((timing.TOTAL, "pipeline s/page (median)"), (timing.TOTAL_INCL_DEBUG, "incl. report s/page (median)")):
        rows.append(["all", name] + [
            timing.fmt_seconds(timing_by_run.get(run, {}).get(key, {}).get("median", math.nan)) for run in runs
        ])
    builder.add_subsection("Aggregate", ["ground truth", "metric"] + runs, rows)


# ---------------------------------------------------------------------------
# Per-input text tables
# ---------------------------------------------------------------------------
def _fmt_rotation(rot) -> str:
    b, n = rot.buckets, rot.n_localized
    return (
        f"correct {b.correct} ({_pct(b.correct, n)}) | off90 {b.off_90} ({_pct(b.off_90, n)}) | "
        f"off180 {b.off_180} ({_pct(b.off_180, n)}) | (n={n}) | "
        f"mean_err={rot.mean_error_deg:.1f} | rmse={rot.rmse_deg:.1f}"
    )


def _fmt_char_stats(cs, *, top_n: int = 5) -> str:
    """One per-char cell: every gt-char outcome as `n (x% of total)`, the top-`top_n` misread
    replacements, and the inserted count."""
    t = cs.total
    if not t:
        return f"total 0 | inserted {cs.inserted}"
    top = ", ".join(f"{r!r}: {c}" for r, c in cs.replacements.most_common(top_n))
    return (
        f"total {t} | detected {cs.detected} ({_pct(cs.detected, t)}) | "
        f"dropped {cs.dropped} ({_pct(cs.dropped, t)}) | "
        f"unreached {cs.unreached} ({_pct(cs.unreached, t)}) | "
        f"misclassified {cs.misclassified} ({_pct(cs.misclassified, t)})"
        + (f" → {top}" if top else "")
        + f" | inserted {cs.inserted}"
    )


def char_order(agg_by_run: "dict[str, TextMetricSuiteResult | None]", text_type: str) -> "list[str]":
    """Every char in any run's `char_stats` for `text_type`, worst error rate first (ties: more
    gt occurrences first); insertion-only chars last."""
    stats_by_char: dict = {}
    for res in agg_by_run.values():
        if res is None:
            continue
        for ch, cs in res.by_type[text_type].char_stats.items():
            prev = stats_by_char.get(ch)
            if prev is None or (not prev.total and cs.total):
                stats_by_char[ch] = cs

    def key(ch: str):
        cs = stats_by_char[ch]
        rate = cs.error_rate
        return (math.isnan(rate), -(0.0 if math.isnan(rate) else rate), -cs.total, ch)

    return sorted(stats_by_char, key=key)


def add_text_sections(
    builder: ReportBuilder, agg_by_run: "dict[str, TextMetricSuiteResult | None]",
    chart_paths: "dict[str, list[Path]] | None" = None,
) -> None:
    """`chart_paths`: `{"labels" | "char" | "word" | "bbox" | "rotation": [report-relative PNG]}`."""
    charts = chart_paths or {}
    builder.add_group_header("Text")
    headers = ["text_type"] + list(agg_by_run)

    def _rows(cell) -> list[list[str]]:
        return [[t] + ["n/a" if res is None else cell(res.by_type[t]) for res in agg_by_run.values()]
                for t in TEXT_TYPES]

    builder.add_subsection("Labels", headers, _rows(
        lambda r: f"{r.label_stats.label_count} labels | {r.label_stats.char_count} chars | "
                  f"{r.label_stats.word_count} words"), charts.get("labels", []))
    for field, name, chart in (("char_overlap", "Char overlap", "char"), ("word_overlap", "Word overlap", "word")):
        builder.add_subsection(name, headers, _rows(
            lambda r, field=field: (lambda co: (
                f"matched {co.matched}/{co.total_gt} | unclassified {co.unclassified} | missing {co.missing} | "
                f"P {fmt_ratio(co.precision)} | R {fmt_ratio(co.recall)}"))(getattr(r, field))),
            charts.get(chart, []))

    rows = _rows(lambda r: f"mean IoU {fmt_ratio(r.bbox_accuracy.mean_iou)} "
                           f"(n_gt={r.bbox_accuracy.n_gt} n_localized={r.bbox_accuracy.n_localized})")
    rows.append(["no label"] + [
        "n/a" if res is None else
        f"spurious OCR readings {res.bbox_unclassified.spurious_pred_count} "
        f"(area frac {_fmt(res.bbox_unclassified.spurious_pred_area_frac)})"
        for res in agg_by_run.values()
    ])
    builder.add_subsection("Bbox accuracy", headers, rows, charts.get("bbox", []))
    builder.add_subsection("Rotation accuracy", headers, _rows(lambda r: _fmt_rotation(r.rotation)),
                           charts.get("rotation", []))

    for t in TEXT_TYPES:
        c_rows = []
        for ch in char_order(agg_by_run, t):
            c_rows.append([repr(ch)] + [
                "" if res is None or (cs := res.by_type[t].char_stats.get(ch)) is None else _fmt_char_stats(cs)
                for res in agg_by_run.values()
            ])
        builder.add_subsection(f"Per-character OCR accuracy (confusion) -- {t}", ["gt_char"] + list(agg_by_run), c_rows)

    chars: set[str] = set()
    for res in agg_by_run.values():
        if res is not None:
            chars.update(res.extra_chars)
    ec_rows = []
    for ch in sorted(chars):
        row = [repr(ch)]
        for res in agg_by_run.values():
            c = 0 if res is None else res.extra_chars.get(ch, 0)
            total = 1 if res is None else (sum(res.extra_chars.values()) or 1)
            row.append(f"{c} ({100 * c / total:.0f}%)" if c else "")
        ec_rows.append(row)
    builder.add_subsection("Extra predicted characters (no label overlap at all)", ["char"] + list(agg_by_run), ec_rows)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _timing_label(row: str) -> str:
    if row.startswith(timing.DEBUG_PARENT + "."):
        return f"{timing.DEBUG_PARENT} › {row.split('.', 1)[1]}"
    return row


def _timing_cell(st: "dict | None") -> str:
    return "" if not st else (
        f"median {timing.fmt_seconds(st['median'])} | mean {timing.fmt_seconds(st['mean'])} | "
        f"min {timing.fmt_seconds(st['min'])} | max {timing.fmt_seconds(st['max'])} | "
        f"Σ {timing.fmt_seconds(st['sum'])} (n={st['n']})"
    )


def add_timing_sections(
    builder: ReportBuilder, timings_by_run: "dict[str, dict[str, dict]]", chart_path: "Path | None" = None,
) -> None:
    """A table of per-page seconds (a row per phase, `total`, then the report-generation
    `debug › *` rows; a column per run, plus a median delta vs the first run), and a table of
    per-document report generation. `timings_by_run` = `{run: {kind: summarize_timings(...)}}`."""
    builder.add_group_header("Timing")
    runs = list(timings_by_run)
    pages = {run: by_kind.get(timing.RUN_KIND, {}) for run, by_kind in timings_by_run.items()}
    if any(pages.values()):
        headers = ["step"] + runs + [f"Δ median vs {runs[0]}: {r}" for r in runs[1:]]
        rows = []
        for key in timing.row_order(list(pages.values())):
            row = [_timing_label(key)] + [_timing_cell(pages[run].get(key)) for run in runs]
            base = pages[runs[0]].get(key)
            for run in runs[1:]:
                st = pages[run].get(key)
                if not st or not base:
                    row.append("")
                else:
                    delta = st["median"] - base["median"]
                    pct = f" ({100 * delta / base['median']:+.0f}%)" if base["median"] else ""
                    row.append(f"{delta:+.3f}s{pct}")
            rows.append(row)
        builder.add_subsection(
            "Wall-clock per page (total = pipeline; debug › * = report generation)",
            headers, rows, [chart_path] if chart_path else [],
        )
    docs = {run: by_kind.get(timing.DOC_KIND, {}) for run, by_kind in timings_by_run.items()}
    if any(docs.values()):
        builder.add_subsection(
            "Report generation per document (label overlays, saving layer PDFs)", ["step"] + runs,
            [[key] + [_timing_cell(docs[run].get(key)) for run in runs]
             for key in timing.row_order(list(docs.values()))],
        )
    if not any(pages.values()) and not any(docs.values()):
        builder.add_raw_html('<p class="empty">(no timing recorded in these runs)</p>')
