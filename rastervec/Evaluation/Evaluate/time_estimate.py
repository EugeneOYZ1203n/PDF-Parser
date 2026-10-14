"""Estimate pipeline time for a whole scanned dataset from a sampled run.

Pure -- no fitz/pipeline import. Inputs are what
`rastervec/notebooks/pdf_dataset_stats.ipynb` writes:

- the scan cache `outputs/pdf_dataset_stats/_scan_cache/<scan_key>/
  page_stats_vectorised.txt` (CSV, one row per page with
  `num_vectors_vectorised`), keyed by `scan_key(INPUT_FOLDER)`;
- a run folder `outputs/pdf_dataset_stats/<run_tag>/` holding
  `sample_pages.json` and `pipeline_reports/<ts>__generated_report_config/
  <stem>/dump.json` per sampled page.

Each sample's time is `timing.flatten_page_timing(...)["total"]` (pipeline
only: phase1-4, no conversion/report cost). A page with `n` vectors is
estimated by `estimate_seconds`: piecewise-linear interpolation between the
two samples whose vector counts bracket `n`; below the smallest sample its
time is used as-is (per-page overhead dominates tiny pages); above the
largest, `t_max * n / n_max`.
"""
from __future__ import annotations

import bisect
import csv
import dataclasses
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from rastervec.commons.logging_setup import get_logger
from rastervec.Evaluation.Evaluate.timing import TOTAL, flatten_page_timing

_LOG = get_logger("time_estimate")

SCAN_STATS_FILENAME = "page_stats_vectorised.txt"
SAMPLE_PAGES_FILENAME = "sample_pages.json"
GENERATED_CONFIG_STEM = "generated_report_config"

EXACT = "exact"
INTERP = "interp"
CLAMP_LOW = "clamp_low"
RATIO_HIGH = "ratio_high"
METHODS = (EXACT, INTERP, CLAMP_LOW, RATIO_HIGH)


@dataclasses.dataclass(frozen=True)
class TimedSample:
    num_vectors: int
    seconds: float
    pdf_path: str = ""
    page_index: int = 0


@dataclasses.dataclass(frozen=True)
class PageEstimate:
    pdf_path: str
    page_index: int
    num_vectors: int
    seconds: float
    method: str
    lower_sample_n: "int | None"
    upper_sample_n: "int | None"


def scan_key(input_folder: "str | Path") -> str:
    """The notebook's `_SCAN_KEY`: sha1 of the folder's resolved path, 16 hex."""
    return hashlib.sha1(str(Path(input_folder).resolve()).encode("utf-8")).hexdigest()[:16]


def load_scan_records(path: "str | Path") -> list[dict]:
    """Rows of a `page_stats_vectorised.txt`, typed as the notebook's cache loader does."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{
            "pdf_path": row["pdf_path"],
            "page_index": int(row["page_index"]),
            "width": float(row["width"]),
            "height": float(row["height"]),
            "num_native_text": int(row["num_native_text"]),
            "num_vectors": int(row["num_vectors"]),
            "num_vectors_vectorised": int(row["num_vectors_vectorised"]),
        } for row in csv.DictReader(f)]


def find_report_run_dir(run_dir: "str | Path") -> Path:
    """Newest `pipeline_reports/*__generated_report_config/` of a notebook run."""
    reports = Path(run_dir) / "pipeline_reports"
    candidates = sorted(reports.glob(f"*__{GENERATED_CONFIG_STEM}"))
    if not candidates:
        raise FileNotFoundError(f"no report run folder under {reports}")
    return candidates[-1]


def _page_total(dump_path: Path, page_index: int) -> "float | None":
    # Raw JSON rather than dump_io.load_dump: only the timing dicts are
    # needed, not every Text/Vector of the page.
    payload = json.loads(dump_path.read_text(encoding="utf-8"))
    for page in payload.get("pages", []):
        if page.get("page_meta", {}).get("index") == page_index:
            row = flatten_page_timing(
                page.get("step_durations", {}), page.get("substep_durations"),
                page.get("debug_durations"),
            )
            return row.get(TOTAL)
    return None


def load_timed_samples(run_dir: "str | Path") -> list[TimedSample]:
    """Every sampled page of a notebook run that has a recorded pipeline
    time; samples whose dump/page/timing is missing are logged and skipped
    (an interrupted run leaves only the cheaper samples)."""
    run_dir = Path(run_dir)
    samples = json.loads((run_dir / SAMPLE_PAGES_FILENAME).read_text(encoding="utf-8"))
    report_dir = find_report_run_dir(run_dir)
    out: list[TimedSample] = []
    for s in samples:
        stem = Path(s["pdf_path"]).stem
        dump_path = report_dir / stem / "dump.json"
        if not dump_path.is_file():
            _LOG.warning("no dump.json for %s (page %d)", stem, s["page_index"])
            continue
        seconds = _page_total(dump_path, s["page_index"])
        if seconds is None:
            _LOG.warning("no timing for %s page %d", stem, s["page_index"])
            continue
        out.append(TimedSample(int(s["num_vectors_vectorised"]), seconds, s["pdf_path"], s["page_index"]))
    return out


def _curve(samples: Sequence[TimedSample]) -> tuple[list[int], list[float]]:
    # Sorted (n, t) knots; duplicate counts averaged.
    by_n: dict[int, list[float]] = defaultdict(list)
    for s in samples:
        by_n[s.num_vectors].append(s.seconds)
    ns = sorted(by_n)
    return ns, [sum(by_n[n]) / len(by_n[n]) for n in ns]


def _estimate(n: int, ns: list[int], ts: list[float]) -> tuple[float, str, "int | None", "int | None"]:
    i = bisect.bisect_left(ns, n)
    if i < len(ns) and ns[i] == n:
        return ts[i], EXACT, n, n
    if i == 0:
        return ts[0], CLAMP_LOW, None, ns[0]
    if i == len(ns):
        n_max, t_max = ns[-1], ts[-1]
        # n_max can only be 0 here if n < 0, which a count never is.
        return (t_max * n / n_max if n_max else t_max), RATIO_HIGH, n_max, None
    n0, n1, t0, t1 = ns[i - 1], ns[i], ts[i - 1], ts[i]
    return t0 + (t1 - t0) * (n - n0) / (n1 - n0), INTERP, n0, n1


def estimate_seconds(n: int, samples: Sequence[TimedSample]) -> tuple[float, str]:
    """Estimated seconds for a page with `n` vectors + the method used."""
    if not samples:
        raise ValueError("no timed samples")
    seconds, method, _, _ = _estimate(n, *_curve(samples))
    return seconds, method


def estimate_pages(records: Sequence[dict], samples: Sequence[TimedSample]) -> list[PageEstimate]:
    """One `PageEstimate` per scan record, in input order."""
    if not samples:
        raise ValueError("no timed samples")
    ns, ts = _curve(samples)
    out = []
    for r in records:
        n = r["num_vectors_vectorised"]
        seconds, method, lo, hi = _estimate(n, ns, ts)
        out.append(PageEstimate(r["pdf_path"], r["page_index"], n, seconds, method, lo, hi))
    return out


def per_pdf_totals(estimates: Sequence[PageEstimate]) -> list[tuple[str, int, float]]:
    """`(pdf_path, n_pages, seconds)` per PDF, most expensive first."""
    pages: dict[str, int] = defaultdict(int)
    secs: dict[str, float] = defaultdict(float)
    for e in estimates:
        pages[e.pdf_path] += 1
        secs[e.pdf_path] += e.seconds
    return sorted(((p, pages[p], secs[p]) for p in pages), key=lambda row: -row[2])


def format_duration(seconds: float) -> str:
    """`3h 12m 05s` / `4m 09s` / `12.3s`."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"
