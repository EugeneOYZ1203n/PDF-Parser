"""Wall-clock timing summaries for `scripts/pipeline_report_benchmark.py`.

Pure -- no pipeline import. Input is what `dump.json` records per page
(`Evaluation/dump_io.PageDump`): phase-level `step_durations`
(`phase1`..`phase4` for the current engine, a single `legacy` for the
legacy engine) plus the P3 backend's own `substep_durations`, plus the
report generator's own per-page `debug_durations` (conversion, stage
layers, debug images, extra-prediction layers).

`flatten_page_timing` turns one page into one flat `{row: seconds}` dict:

- every phase key as-is, except that the backend's `debug_render`
  sub-step (streaming its debug layers -- report-only work) is taken back
  out of `phase3`, so every pipeline row is production cost only;
- every other P3 sub-step as `phase3.<name>`, plus `phase3.other` =
  phase3 minus the sub-steps' sum (untimed glue; clamped at 0);
- `total` = the sum of the phase-level keys only (sub-steps are already
  inside `phase3`, so they never double-count);
- report-generation cost as `debug.<name>` (`debug.backend_layers` = the
  `debug_render` sub-step), then `debug.total` and `total_incl_debug` =
  `total + debug.total`.

`flatten_doc_timing` does the same for a dump's document-level
`doc_durations` (GT overlays, saving the layer PDFs) -- one row per
document, summarised in its own `DOC_KIND` table.

`summarize_timings` then reduces a list of those into per-row
`benchmark.distribution_stats` (+ `sum`), in a fixed display order.
"""
from __future__ import annotations

import math
from typing import Sequence

from rastervec.Evaluation.Evaluate.benchmark import distribution_stats

PHASE_ORDER = ("phase1", "phase2", "phase3", "phase4", "legacy")
SUBSTEP_PARENT = "phase3"
OTHER_SUFFIX = "other"
TOTAL = "total"
DEBUG_PARENT = "debug"
DEBUG_RENDER_SUBSTEP = "debug_render"
DEBUG_ORDER = ("backend_layers", "stage_layers", "debug_images", "extra_predictions", "conversion")
DEBUG_TOTAL = "debug.total"
TOTAL_INCL_DEBUG = "total_incl_debug"
# Document-level report-generation rows (one per document, not per page).
DOC_KIND = "document"

# Timing blocks a dump can carry: the vectorised-PDF run and (benchmark mode
# with a rasterised input) the separate rasterised-PDF run.
RUN_KINDS = ("vectorised", "rasterised")

_STAT_COLUMNS = ("n", "min", "median", "mean", "max", "sum")


def substep_row(name: str) -> str:
    return f"{SUBSTEP_PARENT}.{name}"


def debug_row(name: str) -> str:
    return f"{DEBUG_PARENT}.{name}"


def _is_debug(key: str) -> bool:
    return key.startswith(DEBUG_PARENT + ".") or key == TOTAL_INCL_DEBUG


def flatten_page_timing(
    step_durations: dict, substep_durations: "dict | None" = None,
    debug_durations: "dict | None" = None,
) -> dict:
    """One page's `{row: seconds}` -- see the module docstring. `{}` if the
    page recorded no phase timings at all."""
    if not step_durations:
        return {}
    phases = {k: float(v) for k, v in step_durations.items()}
    subs = {k: float(v) for k, v in (substep_durations or {}).items()}
    debug = {k: float(v) for k, v in (debug_durations or {}).items()}
    backend_debug = subs.pop(DEBUG_RENDER_SUBSTEP, None)
    if backend_debug is not None:
        debug["backend_layers"] = backend_debug
        if SUBSTEP_PARENT in phases:
            phases[SUBSTEP_PARENT] = max(0.0, phases[SUBSTEP_PARENT] - backend_debug)
    row = dict(phases)
    for name, secs in subs.items():
        row[substep_row(name)] = secs
    if subs and SUBSTEP_PARENT in phases:
        row[substep_row(OTHER_SUFFIX)] = max(0.0, phases[SUBSTEP_PARENT] - sum(subs.values()))
    row[TOTAL] = sum(phases.values())
    if debug:
        for name, secs in debug.items():
            row[debug_row(name)] = secs
        row[DEBUG_TOTAL] = sum(debug.values())
        row[TOTAL_INCL_DEBUG] = row[TOTAL] + row[DEBUG_TOTAL]
    return row


def flatten_doc_timing(doc_durations: "dict | None") -> dict:
    """One document's report-generation `{row: seconds}` plus `total`;
    `{}` when nothing was recorded (older dumps)."""
    if not doc_durations:
        return {}
    row = {k: float(v) for k, v in doc_durations.items()}
    row[TOTAL] = sum(row.values())
    return row


def row_order(rows: "Sequence[dict]") -> list[str]:
    """Display order over every key present in `rows`: phases in
    `PHASE_ORDER` with phase3's sub-steps (first-seen order, `other` last)
    directly after `phase3`, then any unknown phase-level key, then
    `total`, then the `debug.*` group (`DEBUG_ORDER` first), `debug.total`
    and `total_incl_debug`."""
    seen: list[str] = []
    for r in rows:
        for k in r:
            if k not in seen:
                seen.append(k)
    other = substep_row(OTHER_SUFFIX)
    subs = [k for k in seen if k.startswith(SUBSTEP_PARENT + ".") and k != other]
    if other in seen:
        subs.append(other)
    ordered: list[str] = []
    for phase in PHASE_ORDER:
        if phase in seen:
            ordered.append(phase)
        if phase == SUBSTEP_PARENT:
            ordered.extend(subs)
    ordered += [
        k for k in seen
        if k not in ordered and k != TOTAL and k not in subs and not _is_debug(k)
    ]
    if TOTAL in seen:
        ordered.append(TOTAL)
    debug = [debug_row(n) for n in DEBUG_ORDER if debug_row(n) in seen]
    debug += [
        k for k in seen
        if k.startswith(DEBUG_PARENT + ".") and k not in debug and k != DEBUG_TOTAL
    ]
    ordered += debug
    ordered += [k for k in (DEBUG_TOTAL, TOTAL_INCL_DEBUG) if k in seen]
    return ordered


def summarize_timings(rows: "Sequence[dict]") -> dict:
    """`{row: stats}` in `row_order` -- stats are `distribution_stats` over
    the pages that recorded that row, plus `sum` (total seconds across
    those pages). `{}` for no pages."""
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
    """Rows that partition a page's total without overlap (for a stacked
    chart): phase3 is replaced by its sub-steps when it has any."""
    has_subs = any(k.startswith(SUBSTEP_PARENT + ".") for k in summary)
    return [
        k for k in summary
        if k != TOTAL and not _is_debug(k) and not (has_subs and k == SUBSTEP_PARENT)
    ]


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
        if key.startswith(SUBSTEP_PARENT + "."):
            label = f"  {key.split('.', 1)[1]}"
        elif key.startswith(DEBUG_PARENT + "."):
            label = f"debug {key.split('.', 1)[1]}"
        else:
            label = key
        cells = "".join(
            f"{int(stats.get(c, 0)):>10}" if c == "n" else f"{stats.get(c, math.nan):>10.3f}"
            for c in _STAT_COLUMNS
        )
        lines.append(f"  {label:<{name_w}}{cells}")
    return "\n".join(lines)
