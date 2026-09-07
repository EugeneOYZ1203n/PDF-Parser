from __future__ import annotations

import pytest

from rastervec.models import TextVectorResult, VectorPath
from rastervec.pipelines.sub_pipelines import ocr as ocr_mod
from rastervec.pipelines.sub_pipelines.radon import ClusterSegmentation


class _FakePage:
    def __init__(self, index: int = 0):
        self.meta = type("Meta", (), {"index": index})()


def _path(seq, bbox) -> VectorPath:
    return VectorPath(
        seq=seq, item_index=0, kind="l", fill_rule="s",
        points=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], bbox=bbox,
        stroke_color=(0, 0, 0), fill_color=None, stroke_opacity=None,
        fill_opacity=None, stroke_width=1.0, dashes=None, closed=False,
        layer=None, page_index=0,
    )


class _CountingRenderOCR:
    calls: list = []

    def __init__(self, backend=None):
        self.backend = backend

    def recognize_segmented(self, seg, cluster, page):
        type(self).calls.append(cluster)
        return TextVectorResult(
            paths=cluster, text=f"CALL{len(type(self).calls)}", confidence=0.7,
            bbox=(0.0, 0.0, 1.0, 1.0), ocr_bbox=(0.0, 0.0, 1.0, 1.0),
            rotation_used=0, page_index=page.meta.index, words=None,
        )


def _seg():
    return ClusterSegmentation(0.0, 0.0, [], [], render_dpi=150)


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    _CountingRenderOCR.calls = []
    monkeypatch.setattr(ocr_mod, "RenderOCR", _CountingRenderOCR)


def test_recognize_reuses_reading_across_similarity_group():
    a = [_path(0, (0, 0, 10, 10))]
    b = [_path(1, (100, 100, 110, 110))]
    res = ocr_mod.recognize(
        [_seg(), _seg()], [a, b], _FakePage(), similarity_id={id(a): 0, id(b): 0},
    )
    assert _CountingRenderOCR.calls == [a]
    assert [r.resolved.text for r in res.cluster_results] == ["CALL1", "CALL1"]
    assert res.cluster_results[1].resolved.bbox == pytest.approx((100, 100, 110, 110))
    assert res.cluster_results[1].resolved.words is None
    assert res.cluster_results[1].ocr_seconds == 0.0


def test_recognize_no_reuse_across_different_groups():
    a = [_path(0, (0, 0, 10, 10))]
    b = [_path(1, (100, 100, 110, 110))]
    res = ocr_mod.recognize(
        [_seg(), _seg()], [a, b], _FakePage(), similarity_id={id(a): 0, id(b): 1},
    )
    assert _CountingRenderOCR.calls == [a, b]
    assert [r.resolved.text for r in res.cluster_results] == ["CALL1", "CALL2"]


def test_recognize_no_group_ocrs_each():
    a = [_path(0, (0, 0, 10, 10))]
    b = [_path(1, (100, 100, 110, 110))]
    res = ocr_mod.recognize([_seg(), _seg()], [a, b], _FakePage(), similarity_id={})
    assert _CountingRenderOCR.calls == [a, b]


def test_recognize_blank_reading_folds_into_failed(monkeypatch):
    class _BlankRenderOCR(_CountingRenderOCR):
        def recognize_segmented(self, seg, cluster, page):
            return TextVectorResult(
                paths=cluster, text="", confidence=0.0, bbox=(0, 0, 1, 1),
                ocr_bbox=None, rotation_used=0, page_index=0, words=None,
            )

    monkeypatch.setattr(ocr_mod, "RenderOCR", _BlankRenderOCR)
    a = [_path(0, (0, 0, 10, 10))]
    res = ocr_mod.recognize([_seg()], [a], _FakePage(), similarity_id={})
    assert res.failed == [a]
