"""Tests for the memory/efficiency rewrites in LatestVectorClassification:
the single-warp `upright_crop`, the chunked crossing counts, the vectorised
ink clip and indexed quad ownership, the oversize step + cluster area caps,
`keep_steps`, the long-quad split, streamed debug images and the shared
PaddleOCR engine. Each rewrite is checked against a straightforward
reference implementation of the old behaviour."""
from __future__ import annotations

import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from rastervec.commons.helpers.clustering import cluster_spatial
from rastervec.commons.models import Page
from rastervec.P3_Vector_Parsing.LatestVectorClassification import classify_vectors as cv
from rastervec.P3_Vector_Parsing.LatestVectorClassification import line_geometry as lg
from rastervec.P3_Vector_Parsing.LatestVectorClassification import paddle_engine as pe
from rastervec.P3_Vector_Parsing.LatestVectorClassification import parse as lvc
from rastervec.P3_Vector_Parsing.LatestVectorClassification.group_filters import combine_overlapping_seq
from rastervec.P3_Vector_Parsing.LatestVectorClassification.paddle_engine import (
    OcrBox, PaddleDetectBackend, PaddleRecBackend,
)


# ---------------------------------------------------------------------------
# upright_crop: one warp straight into the crop == rotate the square region,
# then crop (up to OpenCV's 1/32-px fixed-point interpolation rounding).
# ---------------------------------------------------------------------------
def _reference_upright_crop(bgr, quad, angle_deg, expand=pe.CROP_EXPAND_FRACTION, border=pe.CROP_BORDER_PX):
    long_side, short_side = pe.quad_size(quad)
    region, rc = pe.quad_region(bgr, quad, expand=expand, border=border)
    if angle_deg % 360.0 != 0.0:
        m = cv2.getRotationMatrix2D(rc, angle_deg, 1.0)
        region = cv2.warpAffine(
            region, m, (region.shape[1], region.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )
    cw, ch = long_side * (1.0 + expand), short_side * (1.0 + expand)
    rx0 = max(0, int(math.floor(rc[0] - cw / 2.0)))
    ry0 = max(0, int(math.floor(rc[1] - ch / 2.0)))
    rx1 = min(region.shape[1], int(math.ceil(rc[0] + cw / 2.0)))
    ry1 = min(region.shape[0], int(math.ceil(rc[1] + ch / 2.0)))
    cropped = region[ry0:ry1, rx0:rx1] if rx1 > rx0 and ry1 > ry0 else np.full((1, 1, 3), 255, np.uint8)
    return np.pad(cropped, ((border, border), (border, border), (0, 0)), constant_values=255)


def test_upright_crop_matches_the_rotate_region_then_crop_reference():
    rng = np.random.default_rng(0)
    img = np.full((240, 320, 3), 255, dtype=np.uint8)
    for _ in range(40):  # text-like ink strokes
        p = tuple(int(v) for v in rng.integers(0, [320, 240]))
        q = tuple(int(v) for v in rng.integers(0, [320, 240]))
        cv2.line(img, p, q, (0, 0, 0), 2)
    for _ in range(60):
        c = rng.uniform([-10, -10], [330, 250])
        a = rng.uniform(-180, 180)
        L, S = rng.uniform(5, 200), rng.uniform(3, 40)
        u = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        w = np.array([-u[1], u[0]])
        quad = np.array([c - u * L / 2 - w * S / 2, c + u * L / 2 - w * S / 2,
                         c + u * L / 2 + w * S / 2, c - u * L / 2 + w * S / 2])
        for angle in (0.0, pe.quad_long_edge_angle(quad), float(rng.uniform(-200, 200))):
            ref = _reference_upright_crop(img, quad, angle)
            got = pe.upright_crop(img, quad, angle)
            assert got.shape == ref.shape
            diff = np.abs(got.astype(int) - ref.astype(int))
            assert (diff > 0).mean() < 0.02


def test_upright_crop_size_override_crops_along_the_given_axis():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    # A quad 10 wide x 40 tall, but cropped as a 10-long piece of a
    # horizontal line: the crop must stay 10 wide (plus border), not 40.
    quad = np.array([[45.0, 30.0], [55.0, 30.0], [55.0, 70.0], [45.0, 70.0]])
    crop = pe.upright_crop(img, quad, 0.0, size=(10.0, 40.0))
    b = pe.CROP_BORDER_PX
    assert crop.shape[1] - 2 * b < crop.shape[0] - 2 * b


# ---------------------------------------------------------------------------
# line_crossing_counts: same counts as the pairwise reference, any chunk size.
# ---------------------------------------------------------------------------
def _reference_crossing_counts(vectors, eps):
    pieces = [lg.item_pieces(v) for v in vectors]
    segs = [np.asarray(s, dtype=float).reshape(-1, 4) for s, _ in pieces]
    out = []
    for i, v in enumerate(vectors):
        if not lg.is_line_only(v) or not len(segs[i]):
            out.append(None)
            continue
        n = 0
        for j in range(len(vectors)):
            if j == i:
                continue
            if len(segs[j]):
                n += int(lg.crossing_matrix(segs[i], segs[j], eps).any(axis=0).sum())
            for cubic in pieces[j][1]:
                n += sum(lg.line_cubic_crossings(tuple(s), cubic, eps) for s in segs[i])
        out.append(n)
    return out


def _random_vectors(rng, n):
    def pt():
        return (float(rng.uniform(0, 60)), float(rng.uniform(0, 60)))

    out = []
    for _ in range(n):
        line_only = rng.random() < 0.7
        items = []
        for _ in range(int(rng.integers(1, 6))):
            kind = "l" if line_only else str(rng.choice(["l", "c", "re", "qu"]))
            if kind == "l":
                items.append(("l", pt(), pt()))
            elif kind == "c":
                items.append(("c", pt(), pt(), pt(), pt()))
            elif kind == "re":
                a, b = pt(), pt()
                items.append(("re", (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))))
            else:
                items.append(("qu", (pt(), pt(), pt(), pt())))
        out.append(SimpleNamespace(items=items, bbox=(0.0, 0.0, 60.0, 60.0)))
    return out


@pytest.mark.parametrize("chunk", [1, 5, 250_000])
def test_line_crossing_counts_match_the_pairwise_reference(chunk):
    rng = np.random.default_rng(7)
    for _ in range(40):
        vectors = _random_vectors(rng, int(rng.integers(2, 25)))
        assert lg.line_crossing_counts(vectors, eps=0.01, chunk=chunk) == _reference_crossing_counts(vectors, 0.01)


# ---------------------------------------------------------------------------
# Ownership: vectorised ink clip == per-segment Cyrus-Beck; indexed quad
# ownership == testing every quad.
# ---------------------------------------------------------------------------
def _random_quad(rng):
    c = rng.uniform(0, 60, 2)
    a = rng.uniform(0, math.pi)
    u = np.array([math.cos(a), math.sin(a)])
    w = np.array([-u[1], u[0]])
    L, S = rng.uniform(1, 30), rng.uniform(1, 15)
    return [tuple(map(float, p)) for p in (c - u * L - w * S, c + u * L - w * S, c + u * L + w * S, c - u * L + w * S)]


def test_ink_fraction_matches_the_per_segment_reference():
    rng = np.random.default_rng(3)
    for v in _random_vectors(rng, 300):
        quad = _random_quad(rng)
        poly = np.asarray(quad)
        segs = lg.ink_segments(v, 32)
        total = sum(math.hypot(s[2] - s[0], s[3] - s[1]) for s in segs)
        expected = min(1.0, sum(lg._clip_length_convex(s, poly) for s in segs) / total) if total else None
        if expected is None:
            continue
        assert lg.ink_fraction_in_quad(v, quad, curve_samples=32) == pytest.approx(expected, abs=1e-9)


def test_text_vectors_by_quad_matches_testing_every_quad(vector):
    rng = np.random.default_rng(5)
    clusters, quads_by_cluster = [], {}
    for ci in range(6):
        cluster = []
        for _ in range(30):
            x, y = rng.uniform(0, 60, 2)
            w, h = rng.uniform(0.5, 6, 2)
            cluster.append(vector(bbox=(x, y, x + w, y + h), items=[("l", (x, y), (x + w, y + h))]))
        clusters.append(cluster)
        quads_by_cluster[ci] = [_random_quad(rng) for _ in range(int(rng.integers(0, 8)))]
    text, drawing = lvc._text_vectors_by_quad(clusters, quads_by_cluster)
    expected_text = [
        v for ci, cluster in enumerate(clusters) for v in cluster
        if any(lvc._quad_owns(v, q, lvc._quad_envelope(q), lg.quad_area(q)) for q in quads_by_cluster[ci])
    ]
    assert [id(v) for v in text] == [id(v) for v in expected_text]
    assert len(text) + len(drawing) == sum(len(c) for c in clusters)


# ---------------------------------------------------------------------------
# Oversize step + cluster area caps + keep_steps.
# ---------------------------------------------------------------------------
def _page(page_meta, **kw) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(**kw), fitz_page=None)


def test_oversize_vectors_go_to_drawing(page_meta, vector):
    big = vector(kind="re", bbox=(0.0, 0.0, 150.0, 150.0), seqno=0)  # 56 % of a 200 x 200 page
    small = vector(bbox=(10.0, 10.0, 12.0, 14.0), seqno=1)
    res = cv.classify_vectors([big, small], _page(page_meta))
    assert big in res.drawing_vectors
    assert small not in res.drawing_vectors
    steps = next(iter(res.clustering.values())).steps
    assert steps[0].label == "Oversize"
    assert steps[0].categories[cv.OVERSIZE_CATEGORY].groups == [[big]]


def test_cluster_spatial_union_area_cap():
    boxes = [(0.0, 0.0, 10.0, 10.0), (12.0, 0.0, 22.0, 10.0), (24.0, 0.0, 34.0, 10.0)]
    assert len(cluster_spatial(boxes, lambda b: b, threshold=5.0)) == 1
    capped = cluster_spatial(boxes, lambda b: b, threshold=5.0, max_union_area=250.0)
    assert sorted(len(c) for c in capped) == [1, 2]
    for c in capped:
        x0 = min(b[0] for b in c); x1 = max(b[2] for b in c)
        assert (x1 - x0) * 10.0 < 250.0


def test_combine_overlapping_seq_area_cap(vector):
    vs = [vector(bbox=(10.0 * i, 0.0, 10.0 * i + 10.0, 10.0), seqno=i) for i in range(4)]
    assert len(combine_overlapping_seq([[v] for v in vs], 1.0)[0]) == 1
    groups, _ = combine_overlapping_seq([[v] for v in vs], 1.0, max_area=250.0)
    assert [len(g) for g in groups] == [2, 2]


def test_keep_steps_false_conserves_every_vector(page_meta, vector):
    rng = np.random.default_rng(11)
    vectors = []
    for i in range(80):
        x, y = rng.uniform(0, 180, 2)
        w, h = rng.uniform(0, 8, 2)
        vectors.append(vector(bbox=(x, y, x + w, y + h), items=[("l", (x, y), (x + w, y + h))], seqno=i))
    vectors.append(vector(kind="re", bbox=(0.0, 0.0, 190.0, 190.0), seqno=999))
    full = cv.classify_vectors(vectors, _page(page_meta), keep_steps=True)
    lean = cv.classify_vectors(vectors, _page(page_meta), keep_steps=False)
    flat = lambda res: sorted(id(v) for c in res.text_clusters for g in c for v in g)  # noqa: E731
    assert flat(lean) == flat(full)
    assert sorted(map(id, lean.drawing_vectors)) == sorted(map(id, full.drawing_vectors))
    assert sorted(flat(lean) + [id(v) for v in lean.drawing_vectors]) == sorted(map(id, vectors))
    for stage in lean.clustering.values():
        assert all(not s.categories["kept"].groups for s in stage.steps[:-1])


# ---------------------------------------------------------------------------
# Long-quad split.
# ---------------------------------------------------------------------------
def test_split_positions_prefer_gaps_and_fall_back_to_hard_cuts():
    # Ink at [0, 9] and [11, 20]: the one cut lands in the gap (t = 10).
    assert lvc._split_positions([(0.0, 9.0), (11.0, 20.0)], 20.0, 2) == [10.0]
    # Solid ink: hard cuts at the targets.
    assert lvc._split_positions([(0.0, 30.0)], 30.0, 3) == pytest.approx([10.0, 20.0])
    # A gap outside the window (half a piece) of the target is not used.
    assert lvc._split_positions([(0.0, 1.0), (2.0, 40.0)], 40.0, 2) == [20.0]
    assert lvc._split_positions([(0.0, 5.0)], 5.0, 1) == []


def test_sub_quads_tile_the_quad():
    quad = ((0.0, 0.0), (30.0, 0.0), (30.0, 2.0), (0.0, 2.0))
    pieces = lvc._sub_quads(quad, [10.0, 20.0], 30.0)
    assert [p[0][0] for p in pieces] == [0.0, 10.0, 20.0]
    assert [p[1][0] for p in pieces] == [10.0, 20.0, 30.0]
    assert all(p[2][1] == 2.0 and p[3][1] == 2.0 for p in pieces)


def _row_of_strokes(vector, n=15):
    """Vertical 4 pt strokes 10 pt apart along y = 10..14 -- one cluster."""
    return [
        vector(bbox=(x, 10.0, x, 14.0), items=[("l", (x, 10.0), (x, 14.0))], seqno=i)
        for i, x in enumerate(10.0 + 10.0 * k for k in range(n))
    ]


def _whole_image_quad(self, bgr):
    h, w = bgr.shape[:2]
    return [np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])]


def _patch_rec(monkeypatch, text="AB"):
    monkeypatch.setattr(PaddleRecBackend, "recognize_crops",
                        lambda self, crops: [OcrBox(text=text, confidence=0.99) for _ in crops])
    monkeypatch.setattr(PaddleRecBackend, "recognize_crops_raw",
                        lambda self, crops: [OcrBox(text=text, confidence=0.99) for _ in crops])


def test_parse_splits_a_long_quad_between_vectors(page_meta, vector, monkeypatch):
    monkeypatch.setattr(PaddleDetectBackend, "detect", _whole_image_quad)
    monkeypatch.setattr(lvc, "MAX_CROP_ASPECT", 2.0)
    _patch_rec(monkeypatch)
    strokes = _row_of_strokes(vector)
    drawing, texts = lvc.parse(strokes, [], _page(page_meta))
    assert len(texts) >= 2
    xs = sorted(t.bbox[0] for t in texts)[1:]
    # Every interior cut lies midway between two strokes (x = 15, 25, ...).
    assert all(abs(((x - 15.0) / 10.0) - round((x - 15.0) / 10.0)) < 1e-6 for x in xs)
    # Ownership uses the whole detected quad: every stroke is text.
    assert drawing == []


def test_parse_does_not_split_a_short_quad(page_meta, vector, monkeypatch):
    monkeypatch.setattr(PaddleDetectBackend, "detect", _whole_image_quad)
    _patch_rec(monkeypatch)
    _drawing, texts = lvc.parse(_row_of_strokes(vector, 3), [], _page(page_meta))
    assert len(texts) == 1


# ---------------------------------------------------------------------------
# Streamed debug images.
# ---------------------------------------------------------------------------
def test_on_debug_image_streams_images_and_keeps_no_arrays(page_meta, vector, monkeypatch):
    monkeypatch.setattr(PaddleDetectBackend, "detect", _whole_image_quad)
    _patch_rec(monkeypatch)
    seen: list = []

    def _sink(folder, name, make):
        img = make()
        seen.append((folder, name, img.size))

    debug_out: dict = {}
    lvc.parse(_row_of_strokes(vector, 3), [], _page(page_meta), verbose=True,
              debug_out=debug_out, on_debug_image=_sink, keep_debug_arrays=True)
    folders = {f for f, _n, _s in seen}
    assert folders == {"detect", "rotation_quad", "rotation_classifier", "recog_0"}
    assert all(w > 0 and h > 0 for _f, _n, (w, h) in seen)
    assert debug_out["cluster_detections"] == []
    assert debug_out["quad_rotation_regions"] == []
    assert debug_out["paddle_classifier_crops"] == []
    assert all(not v for v in debug_out["recog_bucket_crops"].values())


def test_without_on_debug_image_debug_out_keeps_rgb_arrays(page_meta, vector, monkeypatch):
    monkeypatch.setattr(PaddleDetectBackend, "detect", _whole_image_quad)
    _patch_rec(monkeypatch)
    debug_out: dict = {}
    lvc.parse(_row_of_strokes(vector, 3), [], _page(page_meta), verbose=True, debug_out=debug_out)
    assert len(debug_out["cluster_detections"]) == 1
    assert len(debug_out["paddle_classifier_crops"]) == 1
    assert len(debug_out["recog_bucket_crops"]["0"]) == 1


# ---------------------------------------------------------------------------
# One PaddleOCR engine per process.
# ---------------------------------------------------------------------------
def test_detect_and_recognition_share_one_engine(monkeypatch):
    sentinel = object()
    monkeypatch.setitem(pe._ENGINES, ("vX", "xx"), sentinel)
    assert PaddleRecBackend("vX", "xx")._engine() is sentinel
    assert PaddleDetectBackend("vX", "xx")._engine() is sentinel


def test_debug_image_time_is_not_counted_inside_computation_steps(page_meta, vector, monkeypatch):
    import time

    monkeypatch.setattr(PaddleDetectBackend, "detect", _whole_image_quad)
    _patch_rec(monkeypatch)
    steps: dict = {}
    lvc.parse(_row_of_strokes(vector, 3), [], _page(page_meta), step_durations=steps,
              on_debug_image=lambda _f, _n, _m: time.sleep(0.05))
    assert steps["debug_render"] >= 0.2  # 4+ images x 50 ms
    assert steps["ocr_crop"] < 0.05 and steps["ocr_assemble"] < 0.05 and steps["ocr_render"] < 0.05
