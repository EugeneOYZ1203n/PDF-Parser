import json

import pytest

from rastervec.Evaluation.Evaluate.time_estimate import (
    CLAMP_LOW, EXACT, INTERP, RATIO_HIGH, TimedSample, estimate_pages, estimate_seconds,
    format_duration, load_timed_samples, per_pdf_totals, scan_key,
)

SAMPLES = [TimedSample(100, 2.0), TimedSample(300, 6.0), TimedSample(1000, 20.0)]


def test_exact_hit():
    assert estimate_seconds(300, SAMPLES) == (6.0, EXACT)


def test_interpolates_between_neighbours():
    secs, method = estimate_seconds(200, SAMPLES)
    assert method == INTERP
    assert secs == pytest.approx(4.0)


def test_clamps_below_smallest_sample():
    assert estimate_seconds(0, SAMPLES) == (2.0, CLAMP_LOW)


def test_ratio_above_largest_sample():
    secs, method = estimate_seconds(2500, SAMPLES)
    assert method == RATIO_HIGH
    assert secs == pytest.approx(50.0)


def test_single_sample():
    one = [TimedSample(50, 1.0)]
    assert estimate_seconds(10, one) == (1.0, CLAMP_LOW)
    assert estimate_seconds(100, one) == (pytest.approx(2.0), RATIO_HIGH)


def test_duplicate_counts_averaged():
    assert estimate_seconds(10, [TimedSample(10, 1.0), TimedSample(10, 3.0)]) == (2.0, EXACT)


def test_no_samples_raises():
    with pytest.raises(ValueError):
        estimate_seconds(10, [])


def test_estimate_pages_and_pdf_totals():
    records = [
        {"pdf_path": "a.pdf", "page_index": 0, "num_vectors_vectorised": 100},
        {"pdf_path": "a.pdf", "page_index": 1, "num_vectors_vectorised": 200},
        {"pdf_path": "b.pdf", "page_index": 0, "num_vectors_vectorised": 2000},
    ]
    est = estimate_pages(records, SAMPLES)
    assert [e.method for e in est] == [EXACT, INTERP, RATIO_HIGH]
    assert (est[1].lower_sample_n, est[1].upper_sample_n) == (100, 300)
    totals = per_pdf_totals(est)
    assert totals[0] == ("b.pdf", 1, pytest.approx(40.0))
    assert totals[1] == ("a.pdf", 2, pytest.approx(6.0))


def test_format_duration():
    assert format_duration(12.34) == "12.3s"
    assert format_duration(249) == "4m 09s"
    assert format_duration(3 * 3600 + 12 * 60 + 5) == "3h 12m 05s"


def test_scan_key_matches_notebook_rule(tmp_path):
    import hashlib
    assert scan_key(tmp_path) == hashlib.sha1(str(tmp_path.resolve()).encode("utf-8")).hexdigest()[:16]


def _write_dump(path, page_index, step_durations, substeps=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pages": [{
        "page_meta": {"index": page_index},
        "step_durations": step_durations,
        "substep_durations": substeps or {},
    }]}), encoding="utf-8")


def test_load_timed_samples(tmp_path):
    run = tmp_path / "run"
    report = run / "pipeline_reports" / "20261014_000000__generated_report_config"
    (run).mkdir()
    (run / "sample_pages.json").write_text(json.dumps([
        {"pdf_path": "/x/a.pdf", "page_index": 2, "num_vectors_vectorised": 10},
        {"pdf_path": "/x/b.pdf", "page_index": 0, "num_vectors_vectorised": 20},  # no dump
        {"pdf_path": "/x/c.pdf", "page_index": 5, "num_vectors_vectorised": 30},  # wrong page
    ]), encoding="utf-8")
    # debug_render is taken back out of phase3 -> total = 1 + (3 - 0.5) = 3.5
    _write_dump(report / "a" / "dump.json", 2, {"phase1": 1.0, "phase3": 3.0}, {"debug_render": 0.5})
    _write_dump(report / "c" / "dump.json", 0, {"phase1": 1.0})

    samples = load_timed_samples(run)
    assert len(samples) == 1
    assert samples[0].num_vectors == 10
    assert samples[0].seconds == pytest.approx(3.5)
