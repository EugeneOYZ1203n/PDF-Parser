"""charts: one smoke test per new chart function -- PNG written, non-trivial size."""
from __future__ import annotations

from rastervec.Evaluation.Evaluate import charts
from rastervec.Evaluation.Evaluate.metrics import (
    TEXT_TYPES,
    GtRegion,
    evaluate_text_metrics,
)
from rastervec.Evaluation.Evaluate.vector_metrics import (
    GeometryEntry,
    evaluate_vector_metrics,
)


def _text_result():
    gt_by_type = {t: [] for t in TEXT_TYPES}
    gt_by_type["native_to_vector"] = [GtRegion(0, (0, 0, 50, 10), "hello world", 0, "native_to_vector")]
    entries_by_type = {t: [] for t in TEXT_TYPES}
    return evaluate_text_metrics(gt_by_type, entries_by_type, [])


def _vector_result():
    line = GeometryEntry(kind="l", points=((0, 0), (10, 0)))
    gt_by_type = {"original_vector": [line], "vector_to_raster": [], "original_raster": []}
    preds_by_type = {"original_vector": [line], "vector_to_raster": [], "original_raster": []}
    return evaluate_vector_metrics(gt_by_type, preds_by_type, {"original_vector": 1, "vector_to_raster": 0, "original_raster": 0})


def _assert_png(path):
    assert path.is_file() and path.stat().st_size > 500


def test_label_description_chart(tmp_path):
    charts.label_description_chart({"current": _text_result(), "legacy": None}, title="t", path=tmp_path / "a.png")
    _assert_png(tmp_path / "a.png")


def test_char_word_overlap_chart(tmp_path):
    charts.char_word_overlap_chart({"current": _text_result()}, field="char_overlap", title="t", path=tmp_path / "b.png")
    _assert_png(tmp_path / "b.png")


def test_bbox_accuracy_chart(tmp_path):
    charts.bbox_accuracy_chart({"current": _text_result()}, title="t", path=tmp_path / "c.png")
    _assert_png(tmp_path / "c.png")


def test_rotation_chart(tmp_path):
    charts.rotation_chart({"current": _text_result()}, title="t", path=tmp_path / "d.png")
    _assert_png(tmp_path / "d.png")


def test_funnel_chart(tmp_path):
    charts.funnel_chart({"current": _text_result()}, title="t", path=tmp_path / "e.png")
    _assert_png(tmp_path / "e.png")


def test_reading_order_chart(tmp_path):
    charts.reading_order_chart({"current": _text_result()}, title="t", path=tmp_path / "f.png")
    _assert_png(tmp_path / "f.png")


def test_font_size_histogram_chart(tmp_path):
    charts.font_size_histogram_chart(_text_result(), "native_to_vector", title="t", path=tmp_path / "fs.png")
    _assert_png(tmp_path / "fs.png")


def test_extra_chars_table_image(tmp_path):
    charts.extra_chars_table_image(_text_result(), title="t", path=tmp_path / "ec.png")
    _assert_png(tmp_path / "ec.png")


def test_confusion_char_table_image(tmp_path):
    charts.confusion_char_table_image(_text_result(), "native_to_vector", title="t", path=tmp_path / "g.png")
    _assert_png(tmp_path / "g.png")


def test_vector_count_chart(tmp_path):
    charts.vector_count_chart({"current": _vector_result()}, title="t", path=tmp_path / "h.png")
    _assert_png(tmp_path / "h.png")


def test_endpoint_accuracy_chart(tmp_path):
    charts.endpoint_accuracy_chart({"current": _vector_result()}, title="t", path=tmp_path / "i.png")
    _assert_png(tmp_path / "i.png")


def test_property_accuracy_chart(tmp_path):
    charts.property_accuracy_chart({"current": _vector_result()}, property_name="width", title="t", path=tmp_path / "j.png")
    _assert_png(tmp_path / "j.png")
