"""Load pipeline comparison stats from disk and plot histograms."""

from __future__ import annotations

import string
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

from raster_parser.evaluation.native_vs_ocr import (
    Stats,
    _char_outcomes,
    _text_for_compare,
    _union_rect,
)
from raster_parser.evaluation.page_stats import (
    PageStats,
    RunStats,
    load_native_mapping_for_page,
    load_stats_from_out_results,
)


def stats_from_page(page: PageStats) -> tuple[list[Stats], list[tuple[int, str]]]:
    """Return comparison results from memory or ``out_results.txt`` on disk."""
    if page.error:
        raise RuntimeError(page.error)
    if page.stats:
        return page.stats, page.not_ocred
    if page.out_results.is_file():
        return load_stats_from_out_results(page.out_results)
    raise ValueError(
        f"No stats for {page.pdf_path} page {page.page_index} "
        f"(expected {page.out_results})"
    )


def combined_stats_from_run(run: RunStats) -> tuple[list[Stats], list[tuple[int, str]]]:
    """Concatenate stats from all successful pages in a batch run."""
    stats: list[Stats] = []
    not_ocred: list[tuple[int, str]] = []
    for page in run.pages:
        if page.ok:
            page_stats, page_not_ocred = stats_from_page(page)
            stats.extend(page_stats)
            not_ocred.extend(page_not_ocred)
    return stats, not_ocred


def glyph_outcome_counts(
    stats: list[Stats],
    not_ocred: list[tuple[int, str]] | None = None,
) -> Counter[str]:
    """Count glyphs as pass / fail / failed_ocr (same rules as ``format_stats_report``)."""
    outcomes = _char_outcomes(stats, not_ocred or [])
    return Counter(outcomes.values())


def print_run_total_statistics(run: RunStats) -> None:
    """Print ``total_statistics.txt`` for a completed run."""
    path = run.total_statistics
    if path is None or not path.is_file():
        print("No total_statistics.txt for this run")
        return
    print(path.read_text(encoding="utf-8"))


def print_summary(
    stats: list[Stats],
    not_ocred: list[tuple[int, str]] | None = None,
    *,
    native_mapping: dict | None = None,
    run: RunStats | None = None,
) -> None:
    """Print OCR row count, missed glyphs, and exact match rate."""
    if run is not None and run.total_statistics and run.total_statistics.is_file():
        print("--- run total (from total_statistics.txt) ---")
        print_run_total_statistics(run)
        print("--- parsed stats (from out_results.txt per page) ---")
    outcomes = glyph_outcome_counts(stats, not_ocred)
    total = sum(outcomes.values())
    pass_n = outcomes.get("pass", 0)
    fail_n = outcomes.get("fail", 0)
    failed_n = outcomes.get("failed_ocr", 0)
    rate = (100.0 * pass_n / total) if total else 0.0
    print(f"OCR rows (Stats): {len(stats)}")
    print(f"Glyphs not OCR'd: {len(not_ocred or [])}")
    print(f"Glyphs: pass={pass_n} fail={fail_n} failed_ocr={failed_n} total={total}")
    print(f"Exact match rate: {rate:.2f}% ({pass_n}/{total} glyphs)")
    if run is not None:
        print(f"Pages: {run.pages_ok} ok, {run.pages_failed} failed, {len(run.pages)} total")
        if run.run_dir:
            print(f"Run dir: {run.run_dir}")
    if native_mapping:
        print(f"native_mapping entries: {len(native_mapping)}")
    elif run is not None:
        n = sum(len(load_native_mapping_for_page(p)) for p in run.pages if p.ok)
        print(f"native_mapping entries (all ok pages): {n}")


def _glyph_size(
    idx: int,
    native_mapping: dict | None,
) -> tuple[float | None, float | None, float | None]:
    if not native_mapping or idx not in native_mapping:
        return None, None, None
    _ch, vec = native_mapping[idx]
    x0, y0, x1, y1 = vec.rect
    w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
    return w, h, max(w, h)


def _is_large_group(
    native_nums: list[int],
    native_mapping: dict | None,
) -> bool | None:
    """Mirror ``parser._prepare_paddle_word_group`` using PDF glyph union bbox."""
    if not native_nums:
        return None
    if native_mapping:
        rects = [native_mapping[i][1].rect for i in native_nums if i in native_mapping]
        if not rects:
            return None
        x0, y0, x1, y1 = _union_rect(rects)
        w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
        return w > h and (w / max(h, 1) >= 8 or len(native_nums) >= 20)
    return len(native_nums) >= 20


def char_category(ch: str) -> str:
    if not ch or ch == "?":
        return "other"
    if len(ch) == 1 and ch.isdigit():
        return "digit"
    if len(ch) == 1 and ch.isalpha():
        return "alpha"
    if len(ch) == 1 and ch in string.punctuation:
        return "punctuation"
    return "other"


def _ordered_native_nums(
    native_nums: list[int],
    native_mapping: dict | None,
) -> list[int]:
    if native_mapping:
        return sorted(native_nums, key=lambda i: native_mapping[i][1].seqno)
    return list(native_nums)


def glyph_positional_records(
    stats: list[Stats],
    not_ocred: list[tuple[int, str]] | None = None,
    *,
    native_mapping: dict | None = None,
) -> list[dict]:
    """One row per native glyph with positional correct / wrong / failed_ocr."""
    records: list[dict] = []

    for s in stats:
        ordered = _ordered_native_nums(s.native_nums, native_mapping)
        native = _text_for_compare(s.native_char)
        ocr = _text_for_compare(s.ocr_char)
        large = _is_large_group(ordered, native_mapping)
        for i, idx in enumerate(ordered):
            if native_mapping and idx in native_mapping:
                ch = native_mapping[idx][0]
            elif i < len(native):
                ch = native[i]
            else:
                ch = "?"
            correct = i < len(native) and i < len(ocr) and native[i] == ocr[i]
            w, h, side = _glyph_size(idx, native_mapping)
            records.append({
                "char": ch,
                "category": char_category(ch),
                "correct": correct,
                "mode": "correct" if correct else "wrong",
                "orientation": s.glyph_orientation.name,
                "native_len": len(native),
                "cer": s.CER,
                "large_group": large,
                "glyph_width": w,
                "glyph_height": h,
                "glyph_max_side": side,
            })

    for idx, ch in not_ocred or []:
        w, h, side = _glyph_size(idx, native_mapping)
        records.append({
            "char": ch,
            "category": char_category(ch),
            "correct": False,
            "mode": "failed_ocr",
            "orientation": None,
            "native_len": None,
            "cer": None,
            "large_group": None,
            "glyph_width": w,
            "glyph_height": h,
            "glyph_max_side": side,
        })

    return records


def glyph_records_from_run(run: RunStats) -> list[dict]:
    """Per-glyph records across a batch run (loads mapping from disk per page)."""
    records: list[dict] = []
    for page in run.pages:
        if page.ok:
            stats, not_ocred = stats_from_page(page)
            mapping = load_native_mapping_for_page(page)
            records.extend(glyph_positional_records(stats, not_ocred, native_mapping=mapping))
    return records


def _resolve_glyph_records(
    stats: list[Stats],
    not_ocred: list[tuple[int, str]] | None,
    *,
    native_mapping: dict | None = None,
    run: RunStats | None = None,
) -> list[dict]:
    if run is not None:
        return glyph_records_from_run(run)
    return glyph_positional_records(stats, not_ocred, native_mapping=native_mapping)


def row_records_from_stats(
    stats: list[Stats],
    *,
    native_mapping: dict | None = None,
) -> list[dict]:
    """One row per OCR word-group with CER, length, orientation, large_group."""
    rows: list[dict] = []
    for s in stats:
        if not s.native_char:
            continue
        rows.append({
            "cer": s.CER,
            "native_len": len(_text_for_compare(s.native_char)),
            "orientation": s.glyph_orientation.name,
            "match": s.match,
            "large_group": _is_large_group(s.native_nums, native_mapping),
            "glyph_count": len(s.native_nums),
        })
    return rows


ACCURACY_YLABEL = "accuracy (%)  correct / (correct + wrong + failed_ocr)"


def _accuracy_pct(correct: int, total: int) -> float:
    return 100.0 * correct / total if total else 0.0


def _accuracy_bins(
    values: list[float],
    correct: list[bool],
    *,
    bins: int | list[float],
    xlabel: str,
    title: str,
) -> None:
    if not values:
        print(f"no data for {title}")
        return

    if isinstance(bins, int):
        edges = np.histogram(values, bins=bins)[1].tolist()
    else:
        edges = list(bins)

    mids: list[float] = []
    rates: list[float] = []
    counts: list[int] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        idxs = [j for j, v in enumerate(values) if lo <= v < hi]
        n_correct = sum(1 for j in idxs if correct[j])
        n = len(idxs)
        if n == 0:
            continue
        mids.append((lo + hi) / 2)
        rates.append(_accuracy_pct(n_correct, n))
        counts.append(n)

    if not mids:
        print(f"no bins with data for {title}")
        return

    print(f"  {title}: {len(mids)} bins, {sum(counts)} glyphs")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(mids)), rates, color="#1f77b4", edgecolor="white")
    ax.set_xticks(range(len(mids)))
    ax.set_xticklabels([f"{m:.1f}" for m in mids], rotation=45, ha="right")
    ax.set_ylabel(ACCURACY_YLABEL)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    for i, (r, n) in enumerate(zip(rates, counts)):
        ax.text(i, min(r + 1, 98), f"n={n}", ha="center", fontsize=8)
    fig.tight_layout()
    plt.show()


def _mean_cer_bins(
    values: list[float],
    cers: list[float],
    *,
    bins: int | list[float],
    xlabel: str,
    title: str,
) -> None:
    if not values:
        print(f"no data for {title}")
        return

    if isinstance(bins, int):
        edges = np.histogram(values, bins=bins)[1].tolist()
    else:
        edges = list(bins)

    mids: list[float] = []
    means: list[float] = []
    counts: list[int] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        bucket = [c for v, c in zip(values, cers) if lo <= v < hi]
        if not bucket:
            continue
        mids.append((lo + hi) / 2)
        means.append(float(np.mean(bucket)))
        counts.append(len(bucket))

    if not mids:
        print(f"no bins with data for {title}")
        return

    print(f"  {title}: {len(mids)} bins, {sum(counts)} OCR rows")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(mids)), means, color="#ff7f0e", edgecolor="white")
    ax.set_xticks(range(len(mids)))
    ax.set_xticklabels([f"{m:.1f}" for m in mids], rotation=45, ha="right")
    ax.set_ylabel("mean CER")
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, min(1.05, max(means) * 1.15 + 0.05))
    ax.set_title(title)
    for i, (m, n) in enumerate(zip(means, counts)):
        ax.text(i, m + 0.01, f"n={n}", ha="center", fontsize=8)
    fig.tight_layout()
    plt.show()


def _int_bin_edges(values: list[int], *, max_bins: int) -> int | list[float]:
    if not values:
        return max_bins
    lo, hi = min(values), max(values)
    if hi - lo + 1 <= max_bins + 1:
        return [x - 0.5 for x in range(lo, hi + 2)]
    return max_bins


def plot_character_vs_accuracy(
    stats: list[Stats],
    not_ocred: list[tuple[int, str]] | None = None,
    *,
    native_mapping: dict | None = None,
    run: RunStats | None = None,
    min_count: int = 5,
    max_chars: int = 30,
    title: str = "Native character vs accuracy",
) -> None:
    """Bar chart: correct / (correct + wrong + failed_ocr) per native character."""
    records = _resolve_glyph_records(stats, not_ocred, native_mapping=native_mapping, run=run)
    by_char: dict[str, list[int]] = {}
    for rec in records:
        ch = rec["char"]
        if not ch or ch == "?":
            continue
        bucket = by_char.setdefault(ch, [0, 0])
        bucket[1] += 1
        if rec["correct"]:
            bucket[0] += 1

    items = [(ch, p, t) for ch, (p, t) in by_char.items() if t >= min_count]
    if not items:
        print(f"no characters with at least {min_count} glyphs")
        return

    items.sort(key=lambda x: -x[2])
    items = items[:max_chars]
    labels = [ch for ch, _, _ in items]
    rates = [_accuracy_pct(p, t) for _, p, t in items]

    print(f"Plotting character vs accuracy ({len(items)} chars with count>={min_count})")
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 4))
    ax.bar(range(len(labels)), rates, color="#1f77b4", edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ACCURACY_YLABEL)
    ax.set_xlabel("native character")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_orientation_vs_cer(
    stats: list[Stats],
    *,
    title: str = "Orientation vs mean CER",
) -> None:
    """Bar chart: mean CER per word-group orientation."""
    rows = row_records_from_stats(stats)
    if not rows:
        print("no OCR rows for orientation vs CER")
        return

    by_ori: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_ori[r["orientation"]].append(r["cer"])

    labels = sorted(by_ori)
    means = [float(np.mean(by_ori[k])) for k in labels]
    counts = [len(by_ori[k]) for k in labels]

    print(f"Plotting orientation vs CER ({sum(counts)} OCR rows)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, color="#ff7f0e", edgecolor="white")
    ax.set_ylabel("mean CER")
    ax.set_xlabel("orientation")
    ax.set_ylim(0, min(1.05, max(means) * 1.15 + 0.05))
    ax.set_title(title)
    for i, (m, n) in enumerate(zip(means, counts)):
        ax.text(i, m + 0.01, f"n={n}", ha="center", fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_size_vs_accuracy(
    stats: list[Stats],
    not_ocred: list[tuple[int, str]] | None = None,
    *,
    native_mapping: dict | None = None,
    run: RunStats | None = None,
    bins: int = 12,
    size_key: str = "glyph_max_side",
    title: str = "Glyph size vs accuracy (max side, pt)",
) -> None:
    """Binned glyph max-side vs correct / (correct + wrong + failed_ocr)."""
    records = _resolve_glyph_records(stats, not_ocred, native_mapping=native_mapping, run=run)
    values = []
    correct = []
    for rec in records:
        side = rec.get(size_key)
        if side is None:
            continue
        values.append(float(side))
        correct.append(rec["correct"])

    if not values:
        print("glyph size vs accuracy needs native_mapping (single page) or run= (batch)")
        return

    print(f"Plotting size vs accuracy ({len(values)} glyphs with bbox)")
    _accuracy_bins(
        values,
        correct,
        bins=bins,
        xlabel="glyph max side (pt)",
        title=title,
    )


def plot_text_length_vs_cer(
    stats: list[Stats],
    *,
    bins: int | list[float] = 15,
    title: str = "Native text length vs mean CER",
) -> None:
    """Binned len(native_char) per OCR row vs mean CER."""
    rows = row_records_from_stats(stats)
    lengths = [float(r["native_len"]) for r in rows]
    cers = [r["cer"] for r in rows]
    if not lengths:
        print("no data for text length vs CER")
        return

    edges = bins if isinstance(bins, list) else _int_bin_edges([int(x) for x in lengths], max_bins=bins)
    print(
        f"Plotting text length vs CER "
        f"({len(lengths)} rows, len range {int(min(lengths))}–{int(max(lengths))})"
    )
    _mean_cer_bins(lengths, cers, bins=edges, xlabel="len(native_char)", title=title)


def plot_category_vs_accuracy(
    stats: list[Stats],
    not_ocred: list[tuple[int, str]] | None = None,
    *,
    native_mapping: dict | None = None,
    run: RunStats | None = None,
    title: str = "Character category vs accuracy",
) -> None:
    """Bar chart: digit / alpha / punctuation / other — correct / (correct + wrong + failed_ocr)."""
    records = _resolve_glyph_records(stats, not_ocred, native_mapping=native_mapping, run=run)
    by_cat: dict[str, list[int]] = {}
    for rec in records:
        cat = rec["category"]
        bucket = by_cat.setdefault(cat, [0, 0])
        bucket[1] += 1
        if rec["correct"]:
            bucket[0] += 1

    order = ["digit", "alpha", "punctuation", "other"]
    labels = [c for c in order if c in by_cat]
    if not labels:
        print("no categorized glyphs")
        return

    rates = [_accuracy_pct(by_cat[c][0], by_cat[c][1]) for c in labels]
    counts = [by_cat[c][1] for c in labels]

    print(f"Plotting category vs accuracy ({sum(counts)} glyphs)")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, rates, color="#2ca02c", edgecolor="white")
    ax.set_ylabel(ACCURACY_YLABEL)
    ax.set_xlabel("category")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    for i, (r, n) in enumerate(zip(rates, counts)):
        ax.text(i, r + 1, f"n={n}", ha="center", fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_failure_modes(
    stats: list[Stats],
    not_ocred: list[tuple[int, str]] | None = None,
    *,
    native_mapping: dict | None = None,
    run: RunStats | None = None,
    title: str = "Failure mode breakdown",
) -> None:
    """Enumerate failures by type, large_group, orientation, length, size, category."""
    records = _resolve_glyph_records(stats, not_ocred, native_mapping=native_mapping, run=run)
    failures = [r for r in records if r["mode"] in ("wrong", "failed_ocr")]
    if not failures:
        print("no failures to plot")
        return

    print(f"Plotting failure modes ({len(failures)} failed glyphs)")

    type_counts = Counter(r["mode"] for r in failures)
    large_counts = Counter(
        "large" if r["large_group"] else "normal"
        for r in failures
        if r["mode"] == "wrong" and r["large_group"] is not None
    )
    ori_counts = Counter(
        r["orientation"] for r in failures if r["mode"] == "wrong" and r["orientation"]
    )

    def _len_bucket(n: int | None) -> str:
        if n is None:
            return "n/a"
        if n <= 3:
            return "1-3"
        if n <= 8:
            return "4-8"
        if n <= 15:
            return "9-15"
        return "16+"

    len_counts = Counter(_len_bucket(r["native_len"]) for r in failures if r["mode"] == "wrong")

    def _size_bucket(side: float | None) -> str:
        if side is None:
            return "unknown"
        if side < 4:
            return "<4pt"
        if side < 8:
            return "4-8pt"
        if side < 12:
            return "8-12pt"
        if side < 20:
            return "12-20pt"
        return "20+pt"

    size_counts = Counter(_size_bucket(r["glyph_max_side"]) for r in failures)
    cat_counts = Counter(r["category"] for r in failures)

    panels: list[tuple[str, Counter]] = [
        ("failure type", type_counts),
        ("large_group (wrong only)", large_counts),
        ("orientation (wrong only)", ori_counts),
        ("native len bucket (wrong only)", len_counts),
        ("glyph size (all failures)", size_counts),
        ("char category (all failures)", cat_counts),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=12)
    for ax, (subtitle, counts) in zip(axes.flat, panels):
        if not counts:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(subtitle)
            ax.axis("off")
            continue
        labels, values = zip(*sorted(counts.items(), key=lambda x: -x[1]))
        ax.bar(range(len(labels)), values, color="#d62728", edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("count")
        ax.set_title(subtitle)
        for i, v in enumerate(values):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    plt.show()


def plot_all(
    stats: list[Stats],
    not_ocred: list[tuple[int, str]] | None = None,
    *,
    native_mapping: dict | None = None,
    run: RunStats | None = None,
) -> None:
    """Run the six requested histograms."""
    print("--- 1/6 character vs accuracy ---")
    plot_character_vs_accuracy(
        stats, not_ocred, native_mapping=native_mapping, run=run
    )
    print("--- 2/6 orientation vs CER ---")
    plot_orientation_vs_cer(stats)
    print("--- 3/6 size vs accuracy ---")
    plot_size_vs_accuracy(
        stats, not_ocred, native_mapping=native_mapping, run=run
    )
    print("--- 4/6 text length vs CER ---")
    plot_text_length_vs_cer(stats)
    print("--- 5/6 category vs accuracy ---")
    plot_category_vs_accuracy(
        stats, not_ocred, native_mapping=native_mapping, run=run
    )
    print("--- 6/6 failure modes ---")
    plot_failure_modes(
        stats, not_ocred, native_mapping=native_mapping, run=run
    )
    print("--- done plotting ---")


plt.style.use("ggplot")
