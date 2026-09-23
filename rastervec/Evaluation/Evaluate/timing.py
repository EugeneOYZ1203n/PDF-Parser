"""Wall-clock timing summaries for `scripts/pipeline_report_benchmark.py`.

Pure -- no pipeline import. Input is what `dump.json` records per page
(`Evaluation/dump_io.PageDump`): phase-level `step_durations`
(`phase1`..`phase4` for the current engine, a single `legacy` for the
legacy engine) plus the P3 backend's own `substep_durations`.

`flatten_page_timing` turns one page into one flat `{row: seconds}` dict:

- every phase key as-is;
- every P3 sub-step as `phase3.<name>`, plus `phase3.other` = phase3 minus
  the sub-steps' sum (debug-layer rendering + glue the backend doesn't
  time; clamped at 0);
- `total` = the sum of the phase-level keys only (sub-steps are already
  inside `phase3`, so they never double-count).

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

# Timing blocks a dump can carry: the vectorised-PDF run and (benchmark mode
# with a rasterised input) the separate rasterised-PDF run.
RUN_KINDS = ("vectorised", "rasterised")

_STAT_COLUMNS = ("n", "min", "median", "mean", "max", "sum")


def substep_row(name: str) -> str:
    return f"{SUBSTEP_PARENT}.{name}"


def flatten_page_timing(step_durations: dict, substep_durations: "dict | None" = None) -> dict:
    """One page's `{row: seconds}` -- see the module docstring. `{}` if the
    page recorded no phase timings at all."""
    if not step_durations:
        return {}
    row = {k: float(v) for k, v in step_durations.items()}
    subs = {k: float(v) for k, v in (substep_durations or {}).items()}
    for name, secs in subs.items():
        row[substep_row(name)] = secs
    if subs and SUBSTEP_PARENT in step_durations:
        row[substep_row(OTHER_SUFFIX)] = max(0.0, float(step_durations[SUBSTEP_PARENT]) - sum(subs.values()))
    row[TOTAL] = sum(float(v) for v in step_durations.values())
    return row


def row_order(rows: "Sequence[dict]") -> list[str]:
    """Display order over every key present in `rows`: phases in
    `PHASE_ORDER` with phase3's sub-steps (first-seen order, `other` last)
    directly after `phase3`, then any unknown phase-level key, then
    `total`."""
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
    ordered += [k for k in seen if k not in ordered and k != TOTAL and k not in subs]
    if TOTAL in seen:
        ordered.append(TOTAL)
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
        if k != TOTAL and not (has_subs and k == SUBSTEP_PARENT)
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
        label = f"  {key.split('.', 1)[1]}" if key.startswith(SUBSTEP_PARENT + ".") else key
        cells = "".join(
            f"{int(stats.get(c, 0)):>10}" if c == "n" else f"{stats.get(c, math.nan):>10.3f}"
            for c in _STAT_COLUMNS
        )
        lines.append(f"  {label:<{name_w}}{cells}")
    return "\n".join(lines)
