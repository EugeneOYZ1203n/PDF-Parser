from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw
from skimage.transform import rotate as sk_rotate

from rastervec.OCR import radon


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
    assert radon._split_on_gaps(arr, gap_threshold=2.0) == [(5, 9)]


def test_split_on_gaps_breaks_on_wide_gap():
    arr = np.zeros(50, dtype=bool)
    for s, e in [(0, 1), (3, 4), (6, 7), (40, 41)]:
        arr[s:e + 1] = True
    assert radon._split_on_gaps(arr, gap_threshold=2.0) == [(0, 7), (40, 41)]


def test_line_gaps_returns_inter_run_gaps():
    arr = np.zeros(50, dtype=bool)
    for s, e in [(0, 1), (3, 4), (6, 7), (40, 41)]:
        arr[s:e + 1] = True
    assert radon._line_gaps(arr) == [1.0, 1.0, 32.0]
    assert radon._line_gaps(np.array([True, True, True])) == []
    assert radon._line_gaps(np.zeros(5, dtype=bool)) == []


def test_cluster_gap_threshold_pools_and_floors():
    assert radon._cluster_gap_threshold([1.0, 1.0, 1.0, 9.0, 9.0, 9.0]) == 5.0
    assert radon._cluster_gap_threshold([]) == radon.RADON_MIN_GAP_PX
    assert radon._cluster_gap_threshold([0.1, 0.1]) == radon.RADON_MIN_GAP_PX
    assert radon._cluster_gap_threshold([1.0, 1.0, 1.0], min_gap=0.5) == 1.0


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


def test_segment_cluster_pools_gap_threshold_across_lines():
    """Line A has 3 ink runs with its own gaps [4, 20]px; judged on its own
    (median 12, no pooling) its 20px gap would exceed that and split it
    into two words. Line B has 4 runs with three 40px gaps, pulling the
    cluster-wide pooled median up to 40 -- above line A's 20px gap -- so
    with the shared threshold line A's runs merge into a single word
    instead. This is the behavioral difference between a per-line and a
    cluster-wide gap threshold."""
    w, h = 240, 86
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    # line A: 3 runs, gaps [4, 20]
    for x0 in (10, 24, 54):
        d.rectangle([x0, 20, x0 + 9, 35], fill=0)
    # line B: 4 runs, gaps [40, 40, 40]
    for x0 in (10, 60, 110, 160):
        d.rectangle([x0, 50, x0 + 9, 65], fill=0)

    seg = radon.segment_cluster(img)
    assert len(seg.word_crops) == 2


def test_rotation_inverse_round_trips():
    out_shape, forward, inverse = radon._rotation((50, 80), 12.0)
    pts = np.array([(0.0, 0.0), (79.0, 0.0), (40.0, 25.0)])
    back = inverse(forward(pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_split_words_gives_each_word_its_own_tight_y_extent():
    """Two words on one baseline with different glyph heights (a short
    word, then a tall one) must get different y-extents -- not the whole
    line's shared ink bbox repeated for both words. Each "word" is two
    small ink runs (like _text_image's dashes) with a small intra-word gap
    and a wide inter-word gap, since _split_on_gaps can't split apart a
    column profile with only one gap in it (as a single solid rectangle
    per word would produce)."""
    w, h = 200, 40
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    baseline = 30
    # short word: two ticks, no ascender
    d.rectangle([10, baseline - 6, 16, baseline], fill=0)
    d.rectangle([20, baseline - 6, 26, baseline], fill=0)
    # tall word: two ticks, with an ascender
    d.rectangle([80, baseline - 20, 86, baseline], fill=0)
    d.rectangle([90, baseline - 20, 96, baseline], fill=0)
    boxes = radon.split_words(np.asarray(img), gap_threshold=10.0)
    assert len(boxes) == 2
    (_, short_y0, _, short_y1), (_, tall_y0, _, tall_y1) = boxes
    assert (short_y0, short_y1) != (tall_y0, tall_y1)
    assert (short_y1 - short_y0) < (tall_y1 - tall_y0)
