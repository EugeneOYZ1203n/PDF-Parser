"""Helpers behind `pdf_dataset_timing.ipynb` -- scan a folder of PDFs for how many vectors each page has
once its text is drawn as glyph paths (`conversion.convert_page_to_vector_text`, what a benchmark-style
run sees), sample pages evenly across that distribution, and write the timing of a vector-classification
only report run next to those counts."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import pymupdf
from raster_parser.evaluation import conversion
from raster_parser.phase_native_read.reader import Reader

STATS_FIELDNAMES = [
    "pdf_path",
    "page_index",
    "width",
    "height",
    "num_native_text",
    "num_vectors",
    "num_vectors_vectorised",
]
HIST_BINS = 30
OUTLIER_SD = 3.0
BLUE, ORANGE = "#2563eb", "#f97316"

# get_cdrawings() skips get_drawings()'s per-item Rect/Point/Quad wrapping; not on every pymupdf install.
_HAS_CDRAWINGS = hasattr(pymupdf.Page, "get_cdrawings")


def count_vectors(pymupdf_page) -> int:
    getter = pymupdf_page.get_cdrawings if _HAS_CDRAWINGS else pymupdf_page.get_drawings
    return len(getter(extended=False))


def count_vectorised(pdf_path: Path, page_index: int) -> int:
    pdf_bytes = conversion.convert_page_to_vector_text(str(pdf_path), page_index)
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return count_vectors(doc[0])


def scan_pdf(pdf_path: Path) -> list[dict]:
    """One record per page: the original counts off PyMuPDF plus the vector count of the page's
    vectorised version. Pages that fail to convert are skipped (logged by `conversion`)."""
    records = []
    with Reader(str(pdf_path)) as reader:
        for page in reader.iter_pages():
            pymupdf_page = page.pymupdf_page
            try:
                vectorised = count_vectorised(pdf_path, page.meta.index)
            except Exception:  # noqa: BLE001 -- keep this PDF's other pages
                continue
            records.append(
                {
                    "pdf_path": str(pdf_path),
                    "page_index": page.meta.index,
                    "width": page.meta.width,
                    "height": page.meta.height,
                    "num_native_text": len(pymupdf_page.get_text("words")),
                    "num_vectors": count_vectors(pymupdf_page),
                    "num_vectors_vectorised": vectorised,
                }
            )
    return records


def save_page_records(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def load_page_records(path: Path) -> list[dict]:
    casts = {"pdf_path": str, "width": float, "height": float}
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {k: casts.get(k, int)(row[k]) for k in STATS_FIELDNAMES} for row in csv.DictReader(f)
        ]


def plot_histogram_with_outliers(values: list[int], ax, *, label: str = "vectors") -> None:
    """30 equal bins over [mean - 3sd, mean + 3sd] (clipped to the data), plus one same-width overflow
    bin per side holding every value beyond it."""
    mu, sd = statistics.fmean(values), statistics.pstdev(values)
    hi = min(max(values), mu + OUTLIER_SD * sd)
    lo = max(min(values), mu - OUTLIER_SD * sd)
    high = [v for v in values if v > hi]
    low = [v for v in values if v < lo]
    if sd == 0 or hi <= lo or not (high or low):
        ax.hist(values, bins=HIST_BINS, color=BLUE, edgecolor="white")
        ax.set_title(f"Distribution of {label} ({len(values)} pages)")
        return
    width = (hi - lo) / HIST_BINS
    edges = [lo + i * width for i in range(HIST_BINS + 1)]
    ax.hist([v for v in values if lo <= v <= hi], bins=edges, color=BLUE, edgecolor="white")
    ticks, labels = [], []
    for group, x0, text, extreme in (
        (high, hi + width, f"> {hi:.0f}", max(values)),
        (low, lo - 2 * width, f"< {lo:.0f}", min(values)),
    ):
        if not group:
            continue
        ax.bar(x0, len(group), width=width, align="edge", color=ORANGE, edgecolor="white")
        ax.annotate(
            f"{len(group)} pages\n{'max' if group is high else 'min'} {extreme:.0f}",
            (x0 + width / 2, len(group)),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ticks.append(x0 + width / 2)
        labels.append(text)
    ax.set_xlim(lo - (3 if low else 1) * width, hi + (3 if high else 1) * width)
    base = [
        t for t in ax.get_xticks() if lo + (width if low else 0) <= t <= hi - (width if high else 0)
    ]
    ax.set_xticks(base + ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in base] + labels)
    ax.set_title(
        f"Distribution of {label} ({len(values)} pages)\n"
        f"outliers beyond mean ± {OUTLIER_SD:g} sd grouped",
        fontsize=10,
    )


def sample_distinct(records: list[dict], metric_fn, n_samples: int) -> list[dict]:
    """Evenly by rank over `records` sorted by `metric_fn`; each pick is the nearest page to its target
    rank whose PDF stem and metric value are both still unused. Ascending by metric."""
    ordered = sorted(records, key=metric_fn)
    n_avail = len(ordered)
    used_stems: set[str] = set()
    used_values: set = set()
    picked = []
    if n_avail == 0 or n_samples <= 0:
        return picked
    for k in range(n_samples):
        target = 0 if n_samples == 1 else round(k / (n_samples - 1) * (n_avail - 1))
        chosen = None
        for offset in range(n_avail):
            for idx in (target - offset, target + offset):
                if 0 <= idx < n_avail:
                    rec = ordered[idx]
                    if (
                        Path(rec["pdf_path"]).stem not in used_stems
                        and metric_fn(rec) not in used_values
                    ):
                        chosen = rec
                        break
            if chosen is not None:
                break
        if chosen is None:
            break
        used_stems.add(Path(chosen["pdf_path"]).stem)
        used_values.add(metric_fn(chosen))
        picked.append({**chosen, "rank_target": target})
    return sorted(picked, key=metric_fn)


def cap_vectors(records: list[dict], max_vectors: int | None) -> list[dict]:
    """Drop pages whose vectorised count exceeds `max_vectors` (None keeps all) -- the biggest pages need
    several GB for classification."""
    return (
        records
        if max_vectors is None
        else [r for r in records if r["num_vectors_vectorised"] <= max_vectors]
    )


def pages_by_stem(samples: list[dict]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in samples:
        pages = out.setdefault(Path(r["pdf_path"]).stem, [])
        if r["page_index"] not in pages:
            pages.append(r["page_index"])
    return out


def write_timing_results(samples: list[dict], path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for r in samples:
            print(f"pdf: {r['pdf_path']}", file=f)
            print(f"  page_index: {r['page_index']}", file=f)
            print(f"  width x height: {r['width']:.1f} x {r['height']:.1f}", file=f)
            print(
                f"  original: num_native_text: {r['num_native_text']}, num_vectors: {r['num_vectors']}",
                file=f,
            )
            print(f"  num_vectors_vectorised: {r['num_vectors_vectorised']}", file=f)
            if r["timing"]:
                print(f"  total time: {r['timing']['total']:.4f}s", file=f)
                print("  timing:", file=f)
                for k, v in r["timing"].items():
                    print(f"    {k}: {v:.4f}s", file=f)
            else:
                print("  timing: MISSING", file=f)
            print(file=f)
    return path
