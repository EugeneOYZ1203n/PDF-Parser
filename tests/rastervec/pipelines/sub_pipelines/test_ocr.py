from __future__ import annotations

import numpy as np
import pytest

from rastervec.models import OcrWord, TextVectorResult, VectorPath
from rastervec.pipelines.sub_pipelines import ocr as ocr_mod
from rastervec.OCR.radon import ClusterSegmentation


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


def test_recognize_reuse_carries_translated_and_scaled_word_boxes(monkeypatch):
    class _WordyRenderOCR(_CountingRenderOCR):
        def recognize_segmented(self, seg, cluster, page):
            type(self).calls.append(cluster)
            return TextVectorResult(
                paths=cluster, text="AB", confidence=0.9,
                bbox=(0.0, 0.0, 10.0, 10.0), ocr_bbox=(1.0, 1.0, 9.0, 9.0),
                rotation_used=0, page_index=page.meta.index,
                words=[
                    OcrWord(text="A", confidence=0.9, bbox=(0.0, 2.0, 4.0, 8.0)),
                    OcrWord(text="B", confidence=0.8, bbox=(6.0, 2.0, 10.0, 8.0)),
                ],
            )

    monkeypatch.setattr(ocr_mod, "RenderOCR", _WordyRenderOCR)
    a = [_path(0, (0, 0, 10, 10))]
    # b is a's bbox translated by (100, 200) and scaled 2x in each axis.
    b = [_path(1, (100, 200, 120, 220))]
    res = ocr_mod.recognize(
        [_seg(), _seg()], [a, b], _FakePage(), similarity_id={id(a): 0, id(b): 0},
    )

    reused = res.cluster_results[1].resolved
    assert reused.text == "AB"
    assert reused.words is not None
    assert [w.text for w in reused.words] == ["A", "B"]
    assert reused.words[0].bbox == pytest.approx((100.0, 204.0, 108.0, 216.0))
    assert reused.words[1].bbox == pytest.approx((112.0, 204.0, 120.0, 216.0))
    assert reused.ocr_bbox == pytest.approx((102.0, 202.0, 118.0, 218.0))


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


class _FakeCompute:
    """Stands in for a `multiprocessing.managers.SyncManager`-hosted `Pool`
    proxy: `.apply` runs inline so the test stays deterministic."""

    def __init__(self):
        self.calls = 0

    def apply(self, fn, args):
        self.calls += 1
        return fn(*args)


def test_recognize_passes_recognize_fn_only_when_compute_given(monkeypatch):
    captured_kwargs: list[dict] = []

    class _CapturingRenderOCR:
        def __init__(self, **kwargs):
            captured_kwargs.append(kwargs)

        def recognize_segmented(self, seg, cluster, page):
            return TextVectorResult(
                paths=cluster, text="X", confidence=0.5, bbox=(0.0, 0.0, 1.0, 1.0),
                ocr_bbox=None, rotation_used=0, page_index=page.meta.index, words=None,
            )

    monkeypatch.setattr(ocr_mod, "RenderOCR", _CapturingRenderOCR)
    a = [_path(0, (0, 0, 10, 10))]

    ocr_mod.recognize([_seg()], [a], _FakePage(), similarity_id={})
    assert "recognize_fn" not in captured_kwargs[-1]

    ocr_mod.recognize([_seg()], [a], _FakePage(), similarity_id={}, compute=_FakeCompute())
    assert "recognize_fn" in captured_kwargs[-1]


def test_recognize_fn_closure_dispatches_through_compute_apply(monkeypatch):
    captured: dict = {}

    class _CapturingRenderOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def recognize_segmented(self, seg, cluster, page):
            return TextVectorResult(
                paths=cluster, text="X", confidence=0.5, bbox=(0.0, 0.0, 1.0, 1.0),
                ocr_bbox=None, rotation_used=0, page_index=page.meta.index, words=None,
            )

    monkeypatch.setattr(ocr_mod, "RenderOCR", _CapturingRenderOCR)
    compute = _FakeCompute()
    a = [_path(0, (0, 0, 10, 10))]
    ocr_mod.recognize([_seg()], [a], _FakePage(), similarity_id={}, compute=compute)

    crops = [np.zeros((4, 4), dtype=np.uint8)]
    monkeypatch.setattr(
        ocr_mod, "_recognize_crops_job", lambda c, *a, **k: [f"got {len(c)}"],
    )
    result = captured["recognize_fn"](crops)
    assert compute.calls == 1
    assert result == ["got 1"]


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
