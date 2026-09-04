from __future__ import annotations

import pickle

from rastervec.Evaluation.Evaluate.metrics import MetricConfig, MetricSuiteResult, Ratio
from rastervec.Reader.Parallel.benchmark_jobs import (
    PageResult,
    PageTask,
    ShowcaseSample,
    run_page_task,
)


def test_pagetask_pickle_round_trip():
    task = PageTask(pdf_path="x.pdf", page_index=2, iou_edge_min=0.2)
    assert pickle.loads(pickle.dumps(task)) == task


def test_pageresult_pickle_round_trip():
    r = PageResult(
        pdf_path="x.pdf", page_index=0, variant="current_light",
        auto=MetricSuiteResult(ratios={"page_char_multiset_recall": Ratio(1.0, 2.0)}),
        showcase=[ShowcaseSample(png=b"\x89PNG", text="HI", passed=True)],
        stage_durations={"reader": 0.1},
    )
    back = pickle.loads(pickle.dumps(r))
    assert back.showcase[0].text == "HI"
    assert back.auto.ratios["page_char_multiset_recall"] == Ratio(1.0, 2.0)


def test_run_page_task_missing_pdf_captures_error():
    task = PageTask(pdf_path="does_not_exist.pdf", page_index=0, variant="current_light")
    result = run_page_task(task)
    assert result.error is not None
    assert result.auto is None
    assert result.variant == "current_light"


def test_run_page_task_unknown_variant_captures_error():
    result = run_page_task(PageTask(pdf_path="x.pdf", page_index=0, variant="nonsense"))
    assert result.error is not None
    assert "unknown pipeline variant" in result.error


def test_pagetask_defaults():
    task = PageTask(pdf_path="a.pdf", page_index=0)
    assert task.variant == "current_light"
    assert task.iou_edge_min == MetricConfig().iou_edge_min
    assert task.manual_entries == []
