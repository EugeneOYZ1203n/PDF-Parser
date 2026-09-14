from __future__ import annotations

import os

import numpy as np
import pytest

from rastervec.config import OCR_LANG, OCR_VERSION
from rastervec.commons.models import Segment
from rastervec.OCR.Paddle_OCR.ocr_backend import (
    ClusterDetection,
    OcrBox,
    PaddleDetectBackend,
    PaddleDetection,
    PaddleRecBackend,
    _normalize_rotation,
    _quad_rotation_deg,
    _recognize_crops_job,
    _rotate_crop,
    crop_rotated_detection,
    dpi_for_cluster,
    pad_image,
    recognize_segments,
    rotation_transform,
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


@pytest.mark.skipif(
    not _RUN_OCR_TESTS,
    reason="real PaddleOCR round-trip; opt in via RASTERVEC_RUN_OCR_TESTS=1",
)
def test_run_pipeline_reads_real_rendered_text(tmp_pdf_path):
    """End-to-end smoke test with the real PaddleOCR engine: a real
    vector-text page, run through the full `current` pipeline (FAST off,
    since it needs its own model weights and isn't the point of this test) --
    PaddleOCR's own detector finds the text, `OCR/radon.py::sweep_rotation`
    refines the angle, and recognition reads the real crop."""
    from rastervec.Evaluation.conversion import convert_page_text_only
    from rastervec.pipelines.current import run_pipeline
    import tempfile
    from pathlib import Path
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "HELLO", fontsize=28)
    src_path = tmp_pdf_path(doc)

    text_only_bytes = convert_page_text_only(src_path, 0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "text_only.pdf"
        path.write_bytes(text_only_bytes)
        res = run_pipeline(str(path), 0, enable_fast=False)

    ocr_texts = [t for t in res.texts if t.source == "ocr"]
    assert ocr_texts, "expected at least one OCR-recognized text"
    joined = "".join(t.text for t in ocr_texts).upper().replace(" ", "")
    # spacing/case varies by rec model; the point is the glyphs were read.
    assert "HELLO" in joined


# --------------------------------------------------------------------------
# PaddleDetectBackend -- pure geometry helpers (no engine needed)
# --------------------------------------------------------------------------
def test_normalize_rotation_wraps_to_plus_minus_90():
    assert _normalize_rotation(0.0) == pytest.approx(0.0)
    # Exact multiples of 90 sit on the wrap boundary -- +/-90 are the same
    # line orientation (mod 180), and the formula consistently resolves the
    # boundary to -90.
    assert _normalize_rotation(90.0) == pytest.approx(-90.0)
    assert _normalize_rotation(91.0) == pytest.approx(-89.0)
    assert _normalize_rotation(-91.0) == pytest.approx(89.0)
    assert _normalize_rotation(180.0) == pytest.approx(0.0)


def test_quad_rotation_deg_axis_aligned_quad_is_zero():
    quad = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 2.0), (0.0, 2.0)])
    assert _quad_rotation_deg(quad) == pytest.approx(0.0, abs=1e-6)


def test_quad_rotation_deg_picks_longer_edge_for_vertical_text():
    # Taller than wide -- the left edge (0->3) is the dominant (longer) one,
    # already vertical, so the "bring to horizontal" rotation is +/-90.
    quad = np.array([(0.0, 0.0), (2.0, 0.0), (2.0, 10.0), (0.0, 10.0)])
    assert abs(_quad_rotation_deg(quad)) == pytest.approx(90.0, abs=1e-6)


def test_quad_rotation_deg_tilted_quad():
    # Top edge tilted 10 degrees below horizontal in image (y-down) coords.
    import math

    theta = math.radians(-10.0)
    dx, dy = 10.0 * math.cos(theta), 10.0 * math.sin(theta)
    quad = np.array([(0.0, 0.0), (dx, dy), (dx, dy + 2.0), (0.0, 2.0)])
    # Rotating by +10 (CCW) should bring this edge to horizontal.
    assert _quad_rotation_deg(quad) == pytest.approx(10.0, abs=1e-3)


def test_dpi_for_cluster_bumps_small_cluster_up(vector):
    v = vector(bbox=(0.0, 0.0, 1.0, 1.0))
    assert dpi_for_cluster([v], dpi=72, padding=0.0) > 72


def test_dpi_for_cluster_never_reduces_dpi(vector):
    v = vector(bbox=(0.0, 0.0, 1000.0, 1000.0))
    assert dpi_for_cluster([v], dpi=300, padding=0.0) == 300


def test_rotate_crop_axis_aligned_quad_is_a_plain_crop():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:10, 2:12] = 255
    quad = np.array([(2.0, 5.0), (12.0, 5.0), (12.0, 10.0), (2.0, 10.0)])

    crop = _rotate_crop(img, quad)

    assert crop.shape[:2] == (5, 10)
    assert crop.mean() > 200  # mostly the white region we carved out


class _FakeDetectEngine:
    """Stands in for a `paddleocr.PaddleOCR` engine's `text_detector` +
    `text_classifier` for `PaddleDetectBackend.detect_on_cluster` tests."""

    def __init__(self, dt_boxes, flip_labels=None):
        self.dt_boxes = dt_boxes
        self.flip_labels = flip_labels or ["0"] * len(dt_boxes)
        self._cls_call = 0

    def text_detector(self, image):
        return self.dt_boxes, 0.0

    def text_classifier(self, crops):
        label = self.flip_labels[self._cls_call]
        self._cls_call += 1
        return crops, [(label, 0.99)], 0.0


def test_detect_on_cluster_returns_none_for_empty_vectors():
    backend = PaddleDetectBackend()
    assert backend.detect_on_cluster([]) is None


def test_detect_on_cluster_builds_page_space_detections(monkeypatch, vector):
    from rastervec.OCR.Paddle_OCR.ocr_backend import dpi_for_cluster, pad_image
    from rastervec.commons.renderer import render_vector_cluster

    v = vector(bbox=(0.0, 0.0, 100.0, 20.0))
    # A quad expressed in the *padded* render's own pixel space: the real
    # pad offset (PADDLE_WHITE_PAD_FRACTION of the render's own dpi-bumped
    # size -- dpi_for_cluster bumps a small cluster's render dpi well past
    # the `dpi=72` requested here) plus a small quad well inside the
    # unpadded frame, so the detection maps back inside the cluster's own
    # page-space bbox once detect_on_cluster subtracts that same offset.
    dpi_used = dpi_for_cluster([v], dpi=72, padding=0.0)
    unpadded_shape = np.asarray(render_vector_cluster([v], dpi_used, 0.0)).shape
    _, (pad_x, pad_y) = pad_image(np.zeros(unpadded_shape[:2], dtype=np.uint8))
    quad = np.array([[[pad_x + 5.0, pad_y + 5.0], [pad_x + 50.0, pad_y + 5.0],
                       [pad_x + 50.0, pad_y + 15.0], [pad_x + 5.0, pad_y + 15.0]]])
    engine = _FakeDetectEngine(dt_boxes=quad, flip_labels=["0"])

    backend = PaddleDetectBackend()
    monkeypatch.setattr(backend, "_engine", lambda: engine)

    result = backend.detect_on_cluster([v], dpi=72)

    assert isinstance(result, ClusterDetection)
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert isinstance(detection, PaddleDetection)
    # Axis-aligned quad -> ~0 rotation (no flip).
    assert detection.rotation_deg == pytest.approx(0.0, abs=1e-3)
    # bbox/quad should land within the cluster's own page-space bbox.
    x0, y0, x1, y1 = detection.bbox
    assert 0.0 <= x0 < x1 <= 100.0
    assert 0.0 <= y0 < y1 <= 20.0


def test_detect_on_cluster_applies_180_flip(monkeypatch, vector):
    v = vector(bbox=(0.0, 0.0, 100.0, 20.0))
    quad = np.array([[[5.0, 5.0], [50.0, 5.0], [50.0, 15.0], [5.0, 15.0]]])
    engine = _FakeDetectEngine(dt_boxes=quad, flip_labels=["180"])

    backend = PaddleDetectBackend()
    monkeypatch.setattr(backend, "_engine", lambda: engine)

    result = backend.detect_on_cluster([v], dpi=72)

    # An axis-aligned quad (0 deg) + a 180 flip normalizes right back to 0
    # (mod 180) -- a line has no arrow, so 0 and 180 are indistinguishable
    # once wrapped by `_normalize_rotation`.
    assert result.detections[0].rotation_deg == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------
# pad_image -- the pipeline's two white-margin pixel operations share this
# --------------------------------------------------------------------------
def test_pad_image_adds_white_border_of_the_configured_fraction():
    img = np.zeros((50, 200), dtype=np.uint8)  # all-black, so the border stands out

    padded, (pad_x, pad_y) = pad_image(img, fraction=0.1)

    assert (pad_x, pad_y) == (20, 20)
    assert padded.shape == (50 + 2 * 20, 200 + 2 * 20)
    assert (padded[:pad_y, :] == 255).all()
    assert (padded[-pad_y:, :] == 255).all()
    assert (padded[:, :pad_x] == 255).all()
    assert (padded[:, -pad_x:] == 255).all()
    assert np.array_equal(padded[pad_y:pad_y + 50, pad_x:pad_x + 200], img)


def test_pad_image_defaults_to_the_config_fraction():
    from rastervec.config import PADDLE_WHITE_PAD_FRACTION

    padded, (pad_x, pad_y) = pad_image(np.zeros((30, 40), dtype=np.uint8))
    expected = round(max(30, 40) * PADDLE_WHITE_PAD_FRACTION)
    assert (pad_x, pad_y) == (expected, expected)
    assert padded.shape == (30 + 2 * pad_y, 40 + 2 * pad_x)


def test_pad_image_zero_area_returned_unchanged():
    empty = np.zeros((0, 5), dtype=np.uint8)
    padded, offset = pad_image(empty)
    assert padded is empty
    assert offset == (0, 0)


def test_pad_image_copies_rather_than_viewing_its_input():
    img = np.zeros((20, 20), dtype=np.uint8)
    padded, (pad_x, pad_y) = pad_image(img)
    img[:] = 128
    assert (padded[pad_y:pad_y + 20, pad_x:pad_x + 20] == 0).all()


def test_pad_image_works_on_3d_images():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    padded, (pad_x, pad_y) = pad_image(img, fraction=0.5)
    assert padded.shape == (10 + 2 * pad_y, 20 + 2 * pad_x, 3)
    assert (padded[0, :, :] == 255).all()


# --------------------------------------------------------------------------
# rotation_transform -- forward/inverse affine for an upright, resized-to-fit
# rotated canvas, shared by crop_rotated_detection.
# --------------------------------------------------------------------------
def test_rotation_transform_zero_angle_is_identity_shape():
    out_shape, forward, inverse = rotation_transform((50, 80), 0.0)
    assert out_shape == (50, 80)
    pts = np.array([(0.0, 0.0), (79.0, 49.0)])
    assert np.allclose(forward(pts), pts, atol=1e-6)


def test_rotation_transform_inverse_round_trips():
    out_shape, forward, inverse = rotation_transform((50, 80), 12.0)
    pts = np.array([(0.0, 0.0), (79.0, 0.0), (40.0, 25.0)])
    back = inverse(forward(pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_rotation_transform_output_canvas_fits_rotated_corners():
    # A 90-degree rotation of a wide rectangle swaps width/height.
    out_shape, _forward, _inverse = rotation_transform((10, 100), 90.0)
    oh, ow = out_shape
    assert ow == pytest.approx(10, abs=2)
    assert oh == pytest.approx(100, abs=2)


# --------------------------------------------------------------------------
# crop_rotated_detection -- the recognition-side crop out of an already
# white-padded cluster render, at the refined rotation angle.
# --------------------------------------------------------------------------
def test_crop_rotated_detection_no_vectors_returns_none():
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    assert crop_rotated_detection(img, 300, [], [], 0.0) is None


def test_crop_rotated_detection_crops_the_detections_own_region(vector):
    from rastervec.commons.renderer import render_vector_cluster

    v = vector(kind="re", bbox=(0.0, 0.0, 100.0, 40.0), fill=(0, 0, 0))
    dpi = 72
    image = np.asarray(render_vector_cluster([v], dpi, 0.0))

    crop = crop_rotated_detection(image, dpi, [v], [v], 0.0)

    assert crop is not None
    assert crop.ndim == image.ndim
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    # the recognition-side pad ran: outer border is white
    assert (crop[0, :] == 255).all()
    assert (crop[-1, :] == 255).all()


def test_crop_rotated_detection_respects_render_pad_offset(vector):
    from rastervec.commons.renderer import render_vector_cluster

    v = vector(kind="re", bbox=(0.0, 0.0, 100.0, 40.0), fill=(0, 0, 0))
    dpi = 72
    unpadded = np.asarray(render_vector_cluster([v], dpi, 0.0))
    padded, (pad_x, pad_y) = pad_image(unpadded)

    crop = crop_rotated_detection(
        padded, dpi, [v], [v], 0.0, render_pad=(pad_x, pad_y),
    )

    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0
