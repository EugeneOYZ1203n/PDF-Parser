"""Wall-clock timing summaries for `scripts/pipeline_report_benchmark.py`.

Pure -- no pipeline import. Input is what `dump.json` records per page (`evaluation/dump_io.
PageDump`): the pipeline's phase seconds (`step_durations`, `phase1`..`phase4`) and the report
generator's own per-page `debug_durations` (input conversion, the reconstructed layer,
extra-prediction layers).

`flatten_page_timing` turns one page into one flat `{row: seconds}` dict: every phase as-is,
`total` = their sum (pipeline only), then report-generation cost as `debug.<name>`,
`debug.total` and `total_incl_debug`. Backend debug layers are rendered inside their phase, so
a report run's phase times include that rendering.

`flatten_doc_timing` does the same for a dump's document-level `doc_durations` (GT overlays,
saving the layer PDFs), summarised in its own `DOC_KIND` table. `summarize_timings` reduces a
list of rows into per-row `distribution_stats` (+ `sum`) in a fixed display order.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

PHASE_ORDER = ("phase1", "phase2", "phase3", "phase4")
TOTAL = "total"
DEBUG_PARENT = "debug"
DEBUG_ORDER = ("conversion", "reconstructed", "extra_predictions")
DEBUG_TOTAL = "debug.total"
TOTAL_INCL_DEBUG = "total_incl_debug"
# Document-level report-generation rows (one per document, not per page).
DOC_KIND = "document"
# The per-page timing block a dump carries.
RUN_KIND = "pages"

_STAT_COLUMNS = ("n", "min", "median", "mean", "max", "sum")


def distribution_stats(values: Sequence[float]) -> dict:
    """`n`, `min`, `q1`, `median`, `mean`, `q3`, `max` of `values` (NaN when empty)."""
    if not values:
        return {k: (0 if k == "n" else math.nan) for k in ("n", "min", "q1", "median", "mean", "q3", "max")}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "q3": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def debug_row(name: str) -> str:
    return f"{DEBUG_PARENT}.{name}"


def _is_debug(key: str) -> bool:
    return key.startswith(DEBUG_PARENT + ".") or key == TOTAL_INCL_DEBUG


def flatten_page_timing(step_durations: dict, debug_durations: "dict | None" = None) -> dict:
    """One page's `{row: seconds}` -- see the module docstring. `{}` if the page recorded no
    phase timings."""
    if not step_durations:
        return {}
    row = {k: float(v) for k, v in step_durations.items()}
    row[TOTAL] = sum(row.values())
    debug = {k: float(v) for k, v in (debug_durations or {}).items()}
    if debug:
        for name, secs in debug.items():
            row[debug_row(name)] = secs
        row[DEBUG_TOTAL] = sum(debug.values())
        row[TOTAL_INCL_DEBUG] = row[TOTAL] + row[DEBUG_TOTAL]
    return row


def flatten_doc_timing(doc_durations: "dict | None") -> dict:
    """One document's report-generation `{row: seconds}` plus `total`; `{}` when nothing was
    recorded."""
    if not doc_durations:
        return {}
    row = {k: float(v) for k, v in doc_durations.items()}
    row[TOTAL] = sum(row.values())
    return row


def row_order(rows: "Sequence[dict]") -> list[str]:
    """Display order over every key present in `rows`: phases in `PHASE_ORDER`, then any other
    pipeline key, then `total`, then the `debug.*` group (`DEBUG_ORDER` first), `debug.total`
    and `total_incl_debug`."""
    seen: list[str] = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    ordered = [p for p in PHASE_ORDER if p in seen]
    ordered += [k for k in seen if k not in ordered and k != TOTAL and not _is_debug(k)]
    if TOTAL in seen:
        ordered.append(TOTAL)
    debug = [debug_row(n) for n in DEBUG_ORDER if debug_row(n) in seen]
    debug += [k for k in seen if k.startswith(DEBUG_PARENT + ".") and k not in debug and k != DEBUG_TOTAL]
    ordered += debug
    ordered += [k for k in (DEBUG_TOTAL, TOTAL_INCL_DEBUG) if k in seen]
    return ordered


def summarize_timings(rows: "Sequence[dict]") -> dict:
    """`{row: stats}` in `row_order` -- `distribution_stats` over the pages that recorded that
    row, plus `sum`. `{}` for no pages."""
    rows = [r for r in rows if r]
    if not rows:
        return {}
    out: dict = {}
    for key in row_order(rows):
        values = [r[key] for r in rows if key in r]
        stats = distribution_stats(values)
        stats["sum"] = float(sum(values))
        out[key] = stats
    return out


def leaf_rows(summary: dict) -> list[str]:
    """Rows that partition a page's pipeline total without overlap (for a stacked chart)."""
    return [k for k in summary if k != TOTAL and not _is_debug(k)]


def fmt_seconds(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3f}s"


def format_timing_table(summary: dict, *, title: str) -> str:
    """Fixed-width text table of one `summarize_timings` result."""
    if not summary:
        return f"{title}\n  (no timing data)"
    name_w = max(len(k) for k in summary) + 2
    header = f"  {'step':<{name_w}}" + "".join(f"{c:>10}" for c in _STAT_COLUMNS)
    lines = [title, header, "  " + "-" * (len(header) - 2)]
    for key, stats in summary.items():
        label = f"debug {key.split('.', 1)[1]}" if key.startswith(DEBUG_PARENT + ".") else key
        cells = "".join(
            f"{int(stats.get(c, 0)):>10}" if c == "n" else f"{stats.get(c, math.nan):>10.3f}"
            for c in _STAT_COLUMNS
        )
        lines.append(f"  {label:<{name_w}}{cells}")
    return "\n".join(lines)
