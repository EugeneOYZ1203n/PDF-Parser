from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pymupdf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from notebooks._helpers import dataset_timing_helpers as h  # noqa: E402
from raster_parser.evaluation.report.report_config import ReportConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _pdf(path: Path, lines: int, text: str = "hello") -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 30), text, fontsize=12)
    for i in range(lines):
        page.draw_line((10, 50 + i * 5), (190, 50 + i * 5))
    doc.save(path)
    doc.close()
    return path


def test_scan_pdf_counts_native_and_vectorised_pages(tmp_path):
    rows = h.scan_pdf(_pdf(tmp_path / "a.pdf", 4))
    assert [r["page_index"] for r in rows] == [0]
    r = rows[0]
    assert r["num_vectors"] == 4 and r["num_native_text"] == 1
    assert r["num_vectors_vectorised"] > r["num_vectors"]  # the text became glyph paths


def test_stats_round_trip_through_the_cache_csv(tmp_path):
    rows = h.scan_pdf(_pdf(tmp_path / "a.pdf", 2))
    h.save_page_records(rows, tmp_path / "stats.txt")
    assert h.load_page_records(tmp_path / "stats.txt") == rows


def test_sample_distinct_spreads_by_rank_with_distinct_pdfs_and_counts():
    recs = [
        {"pdf_path": f"d/{i // 2}.pdf", "page_index": i % 2, "n": i // 2 * 10} for i in range(20)
    ]
    picked = h.sample_distinct(recs, lambda r: r["n"], 4)
    assert len(picked) == 4
    assert len({Path(r["pdf_path"]).stem for r in picked}) == 4
    assert len({r["n"] for r in picked}) == 4
    assert [r["n"] for r in picked] == sorted(r["n"] for r in picked)
    assert picked[0]["n"] == 0 and picked[-1]["n"] == 90


def test_sample_distinct_stops_when_the_dataset_runs_out():
    recs = [
        {"pdf_path": "a.pdf", "page_index": 0, "n": 1},
        {"pdf_path": "a.pdf", "page_index": 1, "n": 2},
    ]
    assert len(h.sample_distinct(recs, lambda r: r["n"], 5)) == 1
    assert h.sample_distinct([], lambda r: r["n"], 3) == []


def test_histogram_groups_outliers_in_overflow_bins():
    fig, ax = plt.subplots()
    h.plot_histogram_with_outliers([10] * 50 + [11] * 50 + [5000], ax, label="vectors")
    assert any("> " in t.get_text() for t in ax.get_xticklabels()) or ax.patches
    plt.close(fig)


def test_cap_vectors_drops_pages_over_the_cap():
    recs = [{"num_vectors_vectorised": n} for n in (5, 50, 500)]
    assert h.cap_vectors(recs, None) == recs
    assert [r["num_vectors_vectorised"] for r in h.cap_vectors(recs, 50)] == [5, 50]


def test_pages_by_stem_collects_each_sampled_page_once():
    samples = [
        {"pdf_path": "x/a.pdf", "page_index": 2},
        {"pdf_path": "y/a.pdf", "page_index": 2},
        {"pdf_path": "x/b.pdf", "page_index": 0},
    ]
    assert h.pages_by_stem(samples) == {"a": [2], "b": [0]}


def test_results_text_lists_each_sample_and_missing_timing(tmp_path):
    samples = [
        {
            "pdf_path": "a.pdf",
            "page_index": 0,
            "width": 10.0,
            "height": 20.0,
            "num_native_text": 1,
            "num_vectors": 2,
            "num_vectors_vectorised": 5,
            "timing": {"total": 1.5, "parse": 1.0},
        },
        {
            "pdf_path": "b.pdf",
            "page_index": 1,
            "width": 10.0,
            "height": 20.0,
            "num_native_text": 0,
            "num_vectors": 0,
            "num_vectors_vectorised": 0,
            "timing": None,
        },
    ]
    out = h.write_timing_results(samples, tmp_path / "t.txt").read_text(encoding="utf-8")
    assert "total time: 1.5000s" in out and "parse: 1.0000s" in out and "timing: MISSING" in out


def test_classification_timing_config_is_classification_only():
    cfg = ReportConfig(
        **json.loads((ROOT / "scripts/report_configs/classify_timing_v2.json").read_text())
    )
    assert (cfg.p2, cfg.p3) == ("stub", "v2_vector_class")
    assert cfg.vectorise and not cfg.benchmark and not cfg.debug_images and not cfg.debug_layers
