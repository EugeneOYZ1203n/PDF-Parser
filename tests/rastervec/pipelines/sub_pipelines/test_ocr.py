from __future__ import annotations

from rastervec.models import SegmentMeta, UniqueSegment
from rastervec.pipelines.sub_pipelines import ocr as ocr_mod


def test_recognize_delegates_to_recognize_unique_segments(monkeypatch):
    captured: dict = {}

    def fake_recognize_unique_segments(uniques, *, recognize_fn=None):
        captured["uniques"] = uniques
        captured["recognize_fn"] = recognize_fn
        return ["a text per unique"]

    monkeypatch.setattr(ocr_mod, "_recognize_unique_segments", fake_recognize_unique_segments)
    uniques = [UniqueSegment(vectors=[])]

    result = ocr_mod.recognize(uniques)

    assert result == ["a text per unique"]
    assert captured["uniques"] is uniques
    assert captured["recognize_fn"] is None


class _FakeCompute:
    """Stands in for a `multiprocessing.managers.SyncManager`-hosted `Pool`
    proxy: `.apply` runs inline so the test stays deterministic."""

    def __init__(self):
        self.calls = 0

    def apply(self, fn, args):
        self.calls += 1
        return fn(*args)


def test_recognize_passes_recognize_fn_only_when_compute_given(monkeypatch):
    captured: list = []

    def fake_recognize_unique_segments(uniques, *, recognize_fn=None):
        captured.append(recognize_fn)
        return []

    monkeypatch.setattr(ocr_mod, "_recognize_unique_segments", fake_recognize_unique_segments)

    ocr_mod.recognize([])
    assert captured[-1] is None

    ocr_mod.recognize([], compute=_FakeCompute())
    assert captured[-1] is not None


def test_recognize_fn_closure_dispatches_through_compute_apply(monkeypatch):
    captured: dict = {}

    def fake_recognize_unique_segments(uniques, *, recognize_fn=None):
        captured["recognize_fn"] = recognize_fn
        return []

    monkeypatch.setattr(ocr_mod, "_recognize_unique_segments", fake_recognize_unique_segments)
    monkeypatch.setattr(ocr_mod, "_recognize_crops_job", lambda c, *a, **k: [f"got {len(c)}"])

    compute = _FakeCompute()
    ocr_mod.recognize([], compute=compute)

    result = captured["recognize_fn"]([1, 2, 3])
    assert compute.calls == 1
    assert result == ["got 3"]


class _FakeCounter:
    def __init__(self, value: int = 0) -> None:
        self.value = value


def test_recognize_fn_closure_increments_progress_counter_by_crop_count(monkeypatch):
    captured: dict = {}

    def fake_recognize_unique_segments(uniques, *, recognize_fn=None):
        captured["recognize_fn"] = recognize_fn
        return []

    monkeypatch.setattr(ocr_mod, "_recognize_unique_segments", fake_recognize_unique_segments)
    monkeypatch.setattr(ocr_mod, "_recognize_crops_job", lambda c, *a, **k: [f"got {len(c)}"])

    counter = _FakeCounter()
    ocr_mod.recognize([], compute=_FakeCompute(), progress_counter=counter)

    captured["recognize_fn"]([1, 2, 3])
    assert counter.value == 3
    captured["recognize_fn"]([1])
    assert counter.value == 4


def test_recognize_no_progress_counter_without_compute(monkeypatch):
    # progress_counter alone (no compute) has nothing to hook into --
    # recognize_fn stays None, same as without progress_counter at all.
    captured: list = []

    def fake_recognize_unique_segments(uniques, *, recognize_fn=None):
        captured.append(recognize_fn)
        return []

    monkeypatch.setattr(ocr_mod, "_recognize_unique_segments", fake_recognize_unique_segments)
    ocr_mod.recognize([], progress_counter=_FakeCounter())
    assert captured[-1] is None


# --------------------------------------------------------------------------
# restore_segment_texts
# --------------------------------------------------------------------------
def test_restore_segment_texts_transforms_canonical_text_by_meta(text):
    canonical = text(text="Hi", bbox=(0.0, 0.0, 10.0, 5.0), direction=(1.0, 0.0))
    meta = SegmentMeta(unique_index=0, offset=(100.0, 200.0), rotation=90.0, page_index=3, seqno=7)

    [restored] = ocr_mod.restore_segment_texts([canonical], [meta])

    assert restored.text == "Hi"
    assert restored.page_index == 3
    assert restored.seqno == 7
    # A 90-degree rotation turns the (1, 0) direction into ~(0, 1).
    assert round(restored.direction[0], 6) == 0.0
    assert round(restored.direction[1], 6) == 1.0


def test_restore_segment_texts_identity_transform_leaves_geometry_unchanged(text):
    canonical = text(text="Hi", bbox=(1.0, 2.0, 11.0, 7.0), direction=(1.0, 0.0))
    meta = SegmentMeta(unique_index=0, offset=(0.0, 0.0), rotation=0.0, page_index=0, seqno=0)

    [restored] = ocr_mod.restore_segment_texts([canonical], [meta])

    assert restored.bbox == canonical.bbox
    assert restored.direction == canonical.direction
    assert restored.origin == canonical.origin


def test_restore_segment_texts_duplicates_one_unique_across_many_metas(text):
    canonical = text(text="Dup", bbox=(0.0, 0.0, 10.0, 5.0), direction=(1.0, 0.0))
    metas = [
        SegmentMeta(unique_index=0, offset=(0.0, 0.0), rotation=0.0, page_index=0, seqno=i)
        for i in range(5)
    ]

    restored = ocr_mod.restore_segment_texts([canonical], metas)

    assert len(restored) == 5
    assert all(t.text == "Dup" for t in restored)
    assert [t.seqno for t in restored] == [0, 1, 2, 3, 4]
