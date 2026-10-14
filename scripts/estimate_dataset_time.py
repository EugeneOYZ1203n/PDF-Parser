"""Estimate pipeline time over a whole scanned dataset.

Takes one `rastervec/notebooks/pdf_dataset_stats.ipynb` run folder (its
sampled pages' measured pipeline times) plus that notebook's scan cache
(every page's `num_vectors_vectorised`) and estimates every page's time by
interpolating between the sampled times by vector count -- see
`rastervec/Evaluation/Evaluate/time_estimate.py` for the rule.

    python scripts/estimate_dataset_time.py --run outputs/pdf_dataset_stats/<tag> \\
        --input-folder references [--scan PATH] [--out DIR]

`--input-folder` is the notebook's `INPUT_FOLDER`: the scan cache lives at
`outputs/pdf_dataset_stats/_scan_cache/<sha1(resolved path)[:16]>/`.
`--scan` points straight at a `page_stats_vectorised.txt` instead. Output
(default `<run>/time_estimate/`):

    summary.txt            total serial time, counts, method breakdown
    pages_estimate.csv     one row per page
    pdf_totals.csv         estimated seconds per PDF, most expensive first
    time_vs_vectors.png    every page's estimate, samples overlaid
    cumulative_time.png    cumulative estimated hours over pages by vector count
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rastervec.commons.logging_setup import configure_logging, get_logger  # noqa: E402
from rastervec.commons.paths import output_dir  # noqa: E402
from rastervec.Evaluation.Evaluate.time_estimate import (  # noqa: E402
    METHODS, RATIO_HIGH, SCAN_STATS_FILENAME, PageEstimate, TimedSample, estimate_pages,
    format_duration, load_scan_records, load_timed_samples, per_pdf_totals, scan_key,
)

_LOG = get_logger("estimate_dataset_time")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path, help="a outputs/pdf_dataset_stats/<run_tag>/ folder")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-folder", type=Path, help="the notebook's INPUT_FOLDER (locates the scan cache)")
    src.add_argument("--scan", type=Path, help=f"a {SCAN_STATS_FILENAME} to use directly")
    p.add_argument("--out", type=Path, help="output folder (default <run>/time_estimate/)")
    return p


def _scan_path(args: argparse.Namespace) -> Path:
    if args.scan is not None:
        return args.scan
    cache = output_dir("pdf_dataset_stats", "_scan_cache", scan_key(args.input_folder), create=False)
    return cache / SCAN_STATS_FILENAME


def _write_pages_csv(path: Path, estimates: list[PageEstimate]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdf_path", "page_index", "num_vectors_vectorised", "est_seconds", "method",
                    "lower_sample_n", "upper_sample_n"])
        for e in estimates:
            w.writerow([e.pdf_path, e.page_index, e.num_vectors, f"{e.seconds:.4f}", e.method,
                        "" if e.lower_sample_n is None else e.lower_sample_n,
                        "" if e.upper_sample_n is None else e.upper_sample_n])


def _write_pdf_csv(path: Path, totals: list[tuple[str, int, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdf_path", "num_pages", "est_seconds"])
        for pdf, n_pages, secs in totals:
            w.writerow([pdf, n_pages, f"{secs:.4f}"])


def _summary(estimates: list[PageEstimate], samples: list[TimedSample],
             totals: list[tuple[str, int, float]], scan: Path, run: Path) -> str:
    total = sum(e.seconds for e in estimates)
    by_method = Counter(e.method for e in estimates)
    secs_by_method = Counter()
    for e in estimates:
        secs_by_method[e.method] += e.seconds
    lines = [
        f"run: {run}",
        f"scan cache: {scan}",
        f"samples with timing: {len(samples)} "
        f"(vectors {min(s.num_vectors for s in samples)}..{max(s.num_vectors for s in samples)})",
        f"pages: {len(estimates)} across {len(totals)} PDF(s)",
        "",
        f"ESTIMATED TOTAL (serial, pipeline only): {format_duration(total)} ({total:.0f}s)",
        f"mean per page: {total / len(estimates):.2f}s" if estimates else "mean per page: n/a",
        "",
        "by method (pages / share of estimated time):",
    ]
    for m in METHODS:
        share = secs_by_method[m] / total if total else 0.0
        lines.append(f"  {m:<11} {by_method[m]:>7}  {share:6.1%}")
    if secs_by_method[RATIO_HIGH]:
        lines.append(f"  ({RATIO_HIGH} = beyond the largest sample, linear extrapolation -- least certain)")
    lines += ["", "top 10 PDFs by estimated time:"]
    for pdf, n_pages, secs in totals[:10]:
        lines.append(f"  {format_duration(secs):>12}  {n_pages:>4} page(s)  {pdf}")
    return "\n".join(lines) + "\n"


def _plot_time_vs_vectors(path: Path, estimates: list[PageEstimate], samples: list[TimedSample]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter([e.num_vectors for e in estimates], [e.seconds for e in estimates],
               s=6, color="#94a3b8", alpha=0.5, label=f"estimated pages ({len(estimates)})")
    ordered = sorted(samples, key=lambda s: s.num_vectors)
    ax.plot([s.num_vectors for s in ordered], [s.seconds for s in ordered],
            color="#2563eb", linewidth=1, marker="o", markersize=5, label=f"timed samples ({len(samples)})")
    ax.set_xlabel("vectors after vectorisation")
    ax.set_ylabel("pipeline time (s)")
    ax.set_title("Estimated pipeline time per page")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_cumulative(path: Path, estimates: list[PageEstimate]) -> None:
    ordered = sorted(estimates, key=lambda e: e.num_vectors)
    xs, ys, acc = [], [], 0.0
    for e in ordered:
        acc += e.seconds
        xs.append(e.num_vectors)
        ys.append(acc / 3600)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, color="#2563eb")
    ax.set_xlabel("vectors after vectorisation (pages sorted ascending)")
    ax.set_ylabel("cumulative estimated time (h)")
    ax.set_title(f"Cumulative estimated time ({ys[-1]:.1f} h total)" if ys else "Cumulative estimated time")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging()
    scan = _scan_path(args)
    if not scan.is_file():
        print(f"scan cache not found: {scan}", file=sys.stderr)
        return 1
    records = load_scan_records(scan)
    samples = load_timed_samples(args.run)
    if not samples:
        print(f"no sampled page in {args.run} has timing data", file=sys.stderr)
        return 1

    estimates = estimate_pages(records, samples)
    totals = per_pdf_totals(estimates)

    out = args.out or (args.run / "time_estimate")
    out.mkdir(parents=True, exist_ok=True)
    _write_pages_csv(out / "pages_estimate.csv", estimates)
    _write_pdf_csv(out / "pdf_totals.csv", totals)
    summary = _summary(estimates, samples, totals, scan, args.run)
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    _plot_time_vs_vectors(out / "time_vs_vectors.png", estimates, samples)
    _plot_cumulative(out / "cumulative_time.png", estimates)
    print(summary)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
