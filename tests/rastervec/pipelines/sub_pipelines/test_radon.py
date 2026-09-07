from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw
from skimage.transform import rotate as sk_rotate

from rastervec.pipelines.sub_pipelines import radon


def _text_image(lines: int = 3, w: int = 240, line_h: int = 16, gap: int = 14) -> np.ndarray:
    """White image with `lines` horizontal black bars (stand-in for text
    lines), each `line_h` tall, separated by `gap` white rows."""
    top = 20
    h = top * 2 + lines * line_h + (lines - 1) * gap
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    y = top
    for _ in range(lines):
        # 3 "words" per line: each a tight group of 3 short dashes (small
        # intra-word gaps), with a wide gap between words.
        for word_x in (20, 100, 170):
            for i in range(3):
                x = word_x + i * 12
                d.rectangle([x, y, x + 8, y + line_h], fill=0)
        y += line_h + gap
    return np.asarray(img)


# --------------------------------------------------------------------------
# 1-D gap rule (ported from the old test_ink_segment.py)
# --------------------------------------------------------------------------
def test_ink_runs_finds_contiguous_blocks():
    arr = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert radon._ink_runs(arr) == [(2, 4), (7, 7)]
    assert radon._ink_runs(np.zeros(5, dtype=bool)) == []


def test_split_on_gaps_single_run_is_one_span():
    arr = np.zeros(12, dtype=bool)
    arr[5:10] = True
    assert radon._split_on_gaps(arr, gap_factor=1.9, min_gap=2.0) == [(5, 9)]


def test_split_on_gaps_breaks_on_wide_gap():
    arr = np.zeros(50, dtype=bool)
    for s, e in [(0, 1), (3, 4), (6, 7), (40, 41)]:
        arr[s:e + 1] = True
    assert radon._split_on_gaps(arr, gap_factor=1.9, min_gap=2.0) == [(0, 7), (40, 41)]


# --------------------------------------------------------------------------
# skew estimate
# --------------------------------------------------------------------------
@pytest.mark.parametrize("true_skew", [-6.0, -2.0, 0.0, 3.0, 8.0])
def test_estimate_skew_recovers_small_rotation(true_skew):
    upright = _text_image()
    # rotate the upright image by -true_skew so estimate_skew should report
    # +true_skew as the correction.
    rotated = sk_rotate(upright, -true_skew, resize=True, cval=255, preserve_range=True)
    est = radon.estimate_skew(rotated.astype(np.uint8))
    assert est == pytest.approx(true_skew, abs=1.5)


def test_estimate_skew_blank_is_zero():
    assert radon.estimate_skew(np.full((30, 30), 255, np.uint8)) == 0.0


# --------------------------------------------------------------------------
# line bands / spacing
# --------------------------------------------------------------------------
def test_line_bands_counts_lines():
    img = _text_image(lines=4)
    prof = radon.row_profile(radon.to_ink(img))
    assert len(radon.line_bands(prof)) == 4
    assert radon.line_spacing(prof) > 0


# --------------------------------------------------------------------------
# segment_cluster end to end
# --------------------------------------------------------------------------
def test_segment_cluster_upright_splits_words_and_lines():
    img = _text_image(lines=3)
    seg = radon.segment_cluster(Image.fromarray(img))
    assert abs(seg.skew_deg) < 2.0
    # 3 lines x 3 words each
    assert len(seg.word_crops) >= 9
    assert len(seg.word_corners) == len(seg.word_crops)
    for corners in seg.word_corners:
        assert len(corners) == 4
        xs = [x for x, _ in corners]
        ys = [y for _, y in corners]
        assert 0 <= min(xs) and max(xs) <= img.shape[1] + 2
        assert 0 <= min(ys) and max(ys) <= img.shape[0] + 2


def test_segment_cluster_deskews_before_splitting():
    img = _text_image(lines=3)
    rotated = sk_rotate(img, -7.0, resize=True, cval=255, preserve_range=True).astype(np.uint8)
    seg = radon.segment_cluster(Image.fromarray(rotated))
    assert seg.skew_deg == pytest.approx(7.0, abs=1.5)
    assert len(seg.word_crops) >= 9


def test_segment_cluster_blank_is_empty():
    seg = radon.segment_cluster(Image.new("L", (40, 40), 255))
    assert seg.word_crops == [] and seg.word_corners == []


def test_rotation_inverse_round_trips():
    out_shape, forward, inverse = radon._rotation((50, 80), 12.0)
    pts = np.array([(0.0, 0.0), (79.0, 0.0), (40.0, 25.0)])
    back = inverse(forward(pts))
    assert np.allclose(back, pts, atol=1e-6)
