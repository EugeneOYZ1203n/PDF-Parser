from __future__ import annotations

import os

import numpy as np
import pytest

from rastervec.config import OCR_LANG, OCR_VERSION
from rastervec.models import Segment
from rastervec.OCR.Paddle_OCR.ocr_backend import (
    OcrBox,
    PaddleDetectBackend,
    PaddleRecBackend,
    _recognize_crops_job,
    recognize_segments,
)

_RUN_OCR_TESTS = os.environ.get("RASTERVEC_RUN_OCR_TESTS") == "1"


class _FakeEngine:
    """`text_classifier` flags every crop past index 0 as upright (label
    "0"); crop 0 is flagged "180" -- exercises the flip-detection path
    without needing a real model. `text_recognizer` returns fixed text per
    (already-flip-corrected) crop, in input order."""

    def __init__(self, texts: list[str], confidences: list[float] | None = None) -> None:
        self.texts = texts
        self.confidences = confidences or [0.9] * len(texts)
        self.classifier_calls = 0
        self.recognizer_calls = 0

    def text_classifier(self, crops):
        self.classifier_calls += 1
        labels = [("180" if i == 0 else "0", 0.99) for i in range(len(crops))]
        return crops, labels, 0.0

    def text_recognizer(self, crops):
        self.recognizer_calls += 1
        rows = list(zip(self.texts[: len(crops)], self.confidences[: len(crops)]))
        return rows, 0.0


def test_backend_key_uses_config_defaults():
    assert PaddleRecBackend().key == (OCR_VERSION, OCR_LANG)


def test_recognize_crops_maps_engine_results_to_boxes(monkeypatch):
    backend = PaddleRecBackend()
    engine = _FakeEngine(["AB", ""])
    monkeypatch.setattr(backend, "_engine", lambda: engine)

    boxes = backend.recognize_crops([np.zeros((8, 10, 3), np.uint8), np.zeros((8, 10, 3), np.uint8)])

    assert [b.text for b in boxes] == ["AB", ""]
    assert isinstance(boxes[0], OcrBox)
    assert boxes[0].confidence == 0.9
    assert boxes[1].confidence == 0.0
    assert engine.classifier_calls == 1
    assert engine.recognizer_calls == 1


def test_recognize_crops_flags_180_from_classifier(monkeypatch):
    backend = PaddleRecBackend()
    engine = _FakeEngine(["FLIPPED", "UPRIGHT"])
    monkeypatch.setattr(backend, "_engine", lambda: engine)

    boxes = backend.recognize_crops([np.zeros((8, 10, 3), np.uint8), np.zeros((8, 10, 3), np.uint8)])

    assert boxes[0].flip_deg == 180
    assert boxes[1].flip_deg == 0


def test_recognize_crops_accepts_bare_list_result(monkeypatch):
    class _BareEngine(_FakeEngine):
        def text_recognizer(self, crops):
            return list(zip(self.texts[: len(crops)], self.confidences[: len(crops)]))

    backend = PaddleRecBackend()
    monkeypatch.setattr(backend, "_engine", lambda: _BareEngine(["HI"]))
    boxes = backend.recognize_crops([np.zeros((8, 10, 3), np.uint8)])
    assert boxes[0].text == "HI"
    assert boxes[0].confidence == 0.9


def test_recognize_crops_empty_input():
    assert PaddleRecBackend().recognize_crops([]) == []


def test_recognize_crops_job_delegates_to_backend(monkeypatch):
    calls = []

    def fake_recognize_crops(self, crops):
        calls.append((self.key, len(crops)))
        return [OcrBox(text="OK", confidence=1.0)]

    monkeypatch.setattr(PaddleRecBackend, "recognize_crops", fake_recognize_crops)
    crops = [np.zeros((8, 10, 3), np.uint8)]
    boxes = _recognize_crops_job(crops, ocr_version="PP-OCRvX", lang="de")
    assert calls == [(("PP-OCRvX", "de"), 1)]
    assert boxes[0].text == "OK"


# --------------------------------------------------------------------------
# recognize_segments -- batching + Text construction from an already-
# rendered `Segment.image`, with a stubbed recognize_fn (no real model
# needed, and no render call to stub either).
# --------------------------------------------------------------------------
def _word_segment(vector, bbox=(0.0, 0.0, 40.0, 20.0), angle=0.0) -> Segment:
    return Segment(
        vectors=[vector(kind="re", bbox=bbox, fill=(0, 0, 0))],
        angle=angle,
        image=np.zeros((10, 10), dtype=np.uint8),
    )


def test_recognize_segments_returns_one_text_per_segment(vector):
    segments = [_word_segment(vector), _word_segment(vector, bbox=(0.0, 0.0, 60.0, 30.0))]

    def stub_recognize(crops):
        return [OcrBox(text=f"W{i}", confidence=0.8, flip_deg=0) for i, _c in enumerate(crops)]

    texts = recognize_segments(segments, recognize_fn=stub_recognize)

    assert [t.text for t in texts] == ["W0", "W1"]
    assert all(t.source == "ocr" for t in texts)
    assert all(t.confidence == 0.8 for t in texts)


def test_recognize_segments_respects_batch_size(vector):
    segments = [_word_segment(vector) for _ in range(5)]
    batch_sizes = []

    def stub_recognize(crops):
        batch_sizes.append(len(crops))
        return [OcrBox(text="X", confidence=0.5) for _ in crops]

    recognize_segments(segments, batch_size=2, recognize_fn=stub_recognize)

    assert batch_sizes == [2, 2, 1]


def test_recognize_segments_flip_deg_sets_direction(vector):
    segments = [_word_segment(vector)]

    def stub_recognize(crops):
        return [OcrBox(text="UPSIDE", confidence=0.9, flip_deg=180)]

    texts = recognize_segments(segments, recognize_fn=stub_recognize)

    assert round(texts[0].angle()) % 360 == 180


def test_recognize_segments_combines_own_angle_with_flip(vector):
    """`direction` folds in the word's own Radon residual `angle` (not just
    the classifier's flip) -- these vectors are still in the representative
    cluster's canonical frame, not re-zeroed per word."""
    segments = [_word_segment(vector, angle=10.0)]

    def stub_recognize(crops):
        return [OcrBox(text="TILTED", confidence=0.9, flip_deg=0)]

    texts = recognize_segments(segments, recognize_fn=stub_recognize)

    assert texts[0].angle() == pytest.approx(10.0)


def test_recognize_segments_empty_input():
    assert recognize_segments([]) == []


# --------------------------------------------------------------------------
# PaddleDetectBackend (test branch `test/paddle-detect-post-fast`) --
# detection-only, no recognition -- see `pipelines/_steps.py::detect_text_paddle`.
# --------------------------------------------------------------------------
class _FakeDetectEngine:
    def __init__(self, quads: list[np.ndarray]):
        self.quads = quads
        self.detector_calls = 0

    def text_detector(self, image, use_slice=False):
        self.detector_calls += 1
        return np.array(self.quads, dtype=np.float32), 0.0


def test_detect_backend_key_uses_config_defaults():
    assert PaddleDetectBackend().key == (OCR_VERSION, OCR_LANG)


def test_detect_maps_quads_to_axis_aligned_bboxes(monkeypatch):
    backend = PaddleDetectBackend()
    quad = [[2.0, 5.0], [12.0, 5.0], [12.0, 15.0], [2.0, 15.0]]
    engine = _FakeDetectEngine([quad])
    monkeypatch.setattr(backend, "_engine", lambda: engine)

    boxes = backend.detect(np.zeros((20, 20, 3), np.uint8))

    assert boxes == [(2.0, 5.0, 12.0, 15.0)]
    assert engine.detector_calls == 1


def test_detect_returns_no_boxes_when_nothing_found(monkeypatch):
    backend = PaddleDetectBackend()
    engine = _FakeDetectEngine([])
    monkeypatch.setattr(backend, "_engine", lambda: engine)

    assert backend.detect(np.zeros((20, 20, 3), np.uint8)) == []


def test_detect_backend_uses_its_own_engine_cache():
    assert PaddleDetectBackend._ENGINE_CACHE is not PaddleRecBackend._ENGINE_CACHE


@pytest.mark.skipif(
    not _RUN_OCR_TESTS,
    reason="real PaddleOCR round-trip; opt in via RASTERVEC_RUN_OCR_TESTS=1",
)
def test_detect_finds_box_around_real_rendered_text(tmp_pdf_path):
    """End-to-end smoke test with the real PaddleOCR detection model: a real
    vector-text page rendered whole-page, detected boxes should land
    somewhere over the rendered word."""
    from rastervec.Evaluation.conversion import convert_page_text_only
    from rastervec.Reader.reader import Reader
    from rastervec.renderer import render_page_paths
    from rastervec.Vector.vector import extract_vectors
    import pymupdf as fitz
    import tempfile
    from pathlib import Path

    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "HELLO", fontsize=28)
    src_path = tmp_pdf_path(doc)

    text_only_bytes = convert_page_text_only(src_path, 0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "text_only.pdf"
        path.write_bytes(text_only_bytes)
        with Reader(str(path)) as reader:
            page = reader.get_page(0)
            vectors = extract_vectors(page)
            image = render_page_paths(vectors, page.meta, 150)

    boxes = PaddleDetectBackend().detect(np.array(image))

    assert boxes, "expected at least one detected box over the rendered word"


@pytest.mark.skipif(
    not _RUN_OCR_TESTS,
    reason="real PaddleOCR round-trip; opt in via RASTERVEC_RUN_OCR_TESTS=1",
)
def test_recognize_segments_reads_real_rendered_text(tmp_pdf_path):
    """End-to-end smoke test with the real PaddleOCR engine: a real vector-
    text page, run through classification, then Radon segmentation directly
    on that surviving cluster's own vectors (the pipeline itself now Radon-
    segments every FAST-surviving cluster, not just elected representatives
    -- see `docs/PIPELINE.md`), recognizing the real word-level `Segment`s
    (with their real captured crop images) it produces."""
    from rastervec.Evaluation.conversion import convert_page_text_only
    from rastervec.OCR.radon import segment_clusters
    from rastervec.pipelines.sub_pipelines.vector_classification import classify_vectors
    from rastervec.Reader.reader import Reader
    from rastervec.Vector.vector import extract_vectors
    import pymupdf as fitz
    import tempfile
    from pathlib import Path

    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "HELLO", fontsize=28)
    src_path = tmp_pdf_path(doc)

    text_only_bytes = convert_page_text_only(src_path, 0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "text_only.pdf"
        path.write_bytes(text_only_bytes)
        with Reader(str(path)) as reader:
            page = reader.get_page(0)
            vectors = extract_vectors(page)
            cls = classify_vectors(vectors, page)
            flat_clusters = [[v for group in c for v in group] for c in cls.text_clusters]
            representative = flat_clusters[0]
            word_segments = segment_clusters([representative])

    assert word_segments, "expected at least one Radon segment from the rendered text"
    texts = recognize_segments(word_segments)

    assert len(texts) >= 1
    joined = "".join(t.text for t in texts).upper().replace(" ", "")
    # spacing/case varies by rec model; the point is the glyphs were read.
    assert "HELLO" in joined
