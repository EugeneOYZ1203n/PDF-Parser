"""charts: metric bar chart + confusion heatmap PNG output."""
from __future__ import annotations

from rastervec.Evaluation.Evaluate.charts import confusion_heatmap, metric_comparison_chart
from rastervec.Evaluation.Evaluate.metrics import (
    GtRegion,
    Prediction,
    evaluate_multiclass,
)


def _result():
    auto = [GtRegion(0, (0, 0, 50, 10), "hello world", 0)]
    manual = [GtRegion(0, (0, 100, 50, 110), "foo bar", 0)]
    preds = [Prediction("HELLO WORLD", (0, 0, 50, 10), 0),
             Prediction("ZZZ", (300, 300, 320, 310), 0)]
    return evaluate_multiclass(auto, manual, preds, [p.bbox for p in preds])


def test_metric_comparison_chart_writes_png(tmp_path):
    r = _result()
    path = tmp_path / "m.png"
    metric_comparison_chart({"current": r.auto, "legacy": None}, title="t", path=path)
    assert path.is_file() and path.stat().st_size > 1000


def test_confusion_heatmap_writes_png(tmp_path):
    r = _result()
    path = tmp_path / "sub" / "c.png"
    confusion_heatmap({"current": r.confusion, "legacy": r.confusion}, title="c", path=path)
    assert path.is_file() and path.stat().st_size > 1000
