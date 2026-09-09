from __future__ import annotations

import pickle

from rastervec.Evaluation.Evaluate.metrics import MetricSuiteResult, Ratio
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


def test_run_benchmark_compute_workers_wires_a_starmap_capable_proxy(monkeypatch):
    captured: dict = {}

    def fake_run_page_task(task, compute=None, progress_counter=None):
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

    def fake_run_page_task(task, compute=None, progress_counter=None):
        captured["compute"] = compute
        return PageResult(pdf_path=task.pdf_path, page_index=task.page_index, variant=task.variant)

    monkeypatch.setattr(bj, "run_page_task", fake_run_page_task)
    tasks = [PageTask(pdf_path="x.pdf", page_index=0)]

    run_benchmark(tasks, workers=1, desc="")
    assert captured["compute"] is None


def test_run_benchmark_forwards_a_progress_counter(monkeypatch):
    # The counter proxy is only reachable while run_benchmark's own Manager
    # is alive (it's shut down before run_benchmark returns), so read
    # .value from inside the fake job, not after the call returns.
    captured: dict = {}

    def fake_run_page_task(task, compute=None, progress_counter=None):
        captured["had_counter"] = progress_counter is not None
        captured["initial_value"] = progress_counter.value if progress_counter is not None else None
        return PageResult(pdf_path=task.pdf_path, page_index=task.page_index, variant=task.variant)

    monkeypatch.setattr(bj, "run_page_task", fake_run_page_task)
    tasks = [PageTask(pdf_path="x.pdf", page_index=0)]

    run_benchmark(tasks, workers=1, desc="")

    assert captured["had_counter"] is True
    assert captured["initial_value"] == 0
