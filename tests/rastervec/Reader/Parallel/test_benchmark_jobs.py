from __future__ import annotations

import pickle

from rastervec.Evaluation.Evaluate.metrics import MetricConfig, MetricSuiteResult, Ratio
from rastervec.Reader.Parallel import benchmark_jobs as bj
from rastervec.Reader.Parallel.benchmark_jobs import (
    PageResult,
    PageTask,
    ShowcaseSample,
    run_benchmark,
    run_page_task,
)


def test_pagetask_pickle_round_trip():
    task = PageTask(pdf_path="x.pdf", page_index=2, iou_edge_min=0.2)
    assert pickle.loads(pickle.dumps(task)) == task


def test_pageresult_pickle_round_trip():
    r = PageResult(
        pdf_path="x.pdf", page_index=0, variant="current",
        auto=MetricSuiteResult(ratios={"page_char_multiset_recall": Ratio(1.0, 2.0)}),
        showcase=[ShowcaseSample(png=b"\x89PNG", text="HI", passed=True)],
        stage_durations={"reader": 0.1},
    )
    back = pickle.loads(pickle.dumps(r))
    assert back.showcase[0].text == "HI"
    assert back.auto.ratios["page_char_multiset_recall"] == Ratio(1.0, 2.0)


def test_run_page_task_missing_pdf_captures_error():
    task = PageTask(pdf_path="does_not_exist.pdf", page_index=0, variant="current")
    result = run_page_task(task)
    assert result.error is not None
    assert result.auto is None
    assert result.variant == "current"


def test_run_page_task_unknown_variant_captures_error():
    result = run_page_task(PageTask(pdf_path="x.pdf", page_index=0, variant="nonsense"))
    assert result.error is not None
    assert "unknown pipeline variant" in result.error


def test_pagetask_defaults():
    task = PageTask(pdf_path="a.pdf", page_index=0)
    assert task.variant == "current"
    assert task.iou_edge_min == MetricConfig().iou_edge_min
    assert task.manual_entries == []


def test_run_page_task_forwards_compute_to_run_current(tmp_pdf_path, monkeypatch):
    import pymupdf as fitz

    doc = fitz.open()
    doc.new_page(width=100, height=100)
    path = tmp_pdf_path(doc)

    captured: dict = {}

    def fake_run_current(task, gt, cfg, variant, compute=None):
        captured["compute"] = compute
        return PageResult(pdf_path=task.pdf_path, page_index=task.page_index, variant=task.variant)

    monkeypatch.setattr(bj, "_run_current", fake_run_current)
    sentinel = object()
    result = run_page_task(PageTask(pdf_path=path, page_index=0), compute=sentinel)

    assert captured["compute"] is sentinel
    assert result.error is None


def test_run_page_task_legacy_ignores_compute(tmp_pdf_path, monkeypatch):
    import pymupdf as fitz

    doc = fitz.open()
    doc.new_page(width=100, height=100)
    path = tmp_pdf_path(doc)

    def fake_run_legacy(task, gt, cfg):
        return PageResult(pdf_path=task.pdf_path, page_index=task.page_index, variant=task.variant)

    called = {"run_current": False}

    def fake_run_current(*args, **kwargs):
        called["run_current"] = True
        raise AssertionError("legacy variant must not call _run_current")

    monkeypatch.setattr(bj, "_run_legacy", fake_run_legacy)
    monkeypatch.setattr(bj, "_run_current", fake_run_current)

    result = run_page_task(
        PageTask(pdf_path=path, page_index=0, variant="legacy"), compute=object(),
    )
    assert not called["run_current"]
    assert result.error is None


def test_run_benchmark_compute_workers_wires_a_starmap_capable_proxy(monkeypatch):
    captured: dict = {}

    def fake_run_page_task(task, compute=None):
        captured["compute"] = compute
        return PageResult(pdf_path=task.pdf_path, page_index=task.page_index, variant=task.variant)

    from rastervec.Reader.Parallel import pool as pool_mod

    monkeypatch.setattr(bj, "run_page_task", fake_run_page_task)
    monkeypatch.setattr(pool_mod, "warmup", lambda: None)
    tasks = [PageTask(pdf_path="x.pdf", page_index=0)]

    results = run_benchmark(tasks, workers=1, compute_workers=1, desc="")

    assert len(results) == 1
    assert results[0].error is None
    assert hasattr(captured["compute"], "starmap")
    assert hasattr(captured["compute"], "apply")


def test_run_benchmark_compute_workers_zero_passes_no_compute(monkeypatch):
    captured: dict = {}

    def fake_run_page_task(task, compute=None):
        captured["compute"] = compute
        return PageResult(pdf_path=task.pdf_path, page_index=task.page_index, variant=task.variant)

    monkeypatch.setattr(bj, "run_page_task", fake_run_page_task)
    tasks = [PageTask(pdf_path="x.pdf", page_index=0)]

    run_benchmark(tasks, workers=1, desc="")
    assert captured["compute"] is None
