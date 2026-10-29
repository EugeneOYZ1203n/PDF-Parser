"""evaluation/evaluate/charts.py: every chart writes a PNG, with a missing run as a gap."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")  # the `report` dependency group

from raster_parser.evaluation.evaluate import charts, timing  # noqa: E402
from raster_parser.evaluation.evaluate.metrics import (  # noqa: E402
    TEXT_TYPES,
    GtRegion,
    Prediction,
    evaluate_text_metrics,
)


def _result():
    gt = {t: [] for t in TEXT_TYPES}
    gt["native_to_vector"] = [
        GtRegion(page_index=0, bbox=(0, 0, 50, 12), text="HELLO", text_type="native_to_vector")
    ]
    pred = Prediction(
        text="HELLO",
        bbox=(0, 0, 50, 12),
        rotation=0,
        reached_ocr=True,
        ocr_blank=False,
        source_cluster_id=1,
    )
    return evaluate_text_metrics(gt, {t: [] for t in TEXT_TYPES}, [pred])


@pytest.mark.parametrize(
    "chart, kwargs",
    [
        (charts.label_description_chart, {}),
        (charts.char_word_overlap_chart, {"field": "char_overlap"}),
        (charts.char_word_overlap_chart, {"field": "word_overlap"}),
        (charts.bbox_accuracy_chart, {}),
        (charts.rotation_chart, {}),
    ],
)
def test_text_charts_write_png(tmp_path, chart, kwargs):
    path = tmp_path / "c.png"
    chart({"a": _result(), "b": None}, title="t", path=path, **kwargs)
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_timing_chart_writes_png(tmp_path):
    summary = timing.summarize_timings(
        [timing.flatten_page_timing({"native_read": 0.5, "vector_classify": 1.0})]
    )
    path = tmp_path / "t.png"
    charts.timing_chart({"a": summary, "b": {}}, title="t", path=path)
    assert path.read_bytes()[:4] == b"\x89PNG"
