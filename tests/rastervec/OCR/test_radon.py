from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw
from skimage.transform import rotate as sk_rotate

from rastervec.helpers.geometry import transform_vector
from rastervec.models import Vector
from rastervec.OCR import radon


def _word_vector(bbox: tuple[float, float, float, float], seqno: int) -> Vector:
    """A single filled rect Vector -- stands in for a "dash" of ink in a
    synthetic text-like cluster, the same role _text_image's PIL rectangles
    play for the pure-image tests above."""
    return Vector(
        type="f", items=[("re", bbox, 1)], color=None, fill=(0, 0, 0), width=None,
        dashes=None, closePath=True, lineCap=0, lineJoin=0, even_odd=False,
        stroke_opacity=None, fill_opacity=None, layer=None, rect=bbox,
        scissor=None, seqno=seqno, blendmode=None, isolated=False, knockout=False,
        opacity=None, page_index=0,
    )


def _two_word_cluster() -> list[Vector]:
    """Two words, each three short dashes close together, well separated
    horizontally -- mirrors _text_image's per-word dash pattern."""
    vectors: list[Vector] = []
    seq = 0
    for word_x in (20, 100):
        for i in range(3):
            x = word_x + i * 12
            vectors.append(_word_vector((x, 20, x + 8, 36), seq))
            seq += 1
    return vectors


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


def test_line_run_widths_returns_ink_run_widths():
    arr = np.zeros(50, dtype=bool)
    for s, e in [(0, 4), (10, 11), (20, 29)]:
        arr[s:e + 1] = True
    assert radon._line_run_widths(arr) == [5.0, 2.0, 10.0]
    assert radon._line_run_widths(np.zeros(5, dtype=bool)) == []


def test_cluster_gap_threshold_is_15pct_of_widest_run_and_floors():
    assert radon._cluster_gap_threshold([4.0, 8.0, 40.0]) == pytest.approx(6.0)
    assert radon._cluster_gap_threshold([]) == radon.RADON_MIN_GAP_PX
    assert radon._cluster_gap_threshold([1.0, 1.0]) == radon.RADON_MIN_GAP_PX
    assert radon._cluster_gap_threshold([10.0], min_gap=0.5) == pytest.approx(1.5)


def test_enforce_min_run_count_merges_short_spans_regardless_of_gap():
    # Span 0 (1 run) is short; merging it into span 1 (2 runs) reaches the
    # 3-run minimum, so the well-formed span 2 (5 runs) is left alone.
    spans = [(0, 1, 1), (10, 15, 2), (30, 39, 5)]
    merged = radon._enforce_min_run_count(spans, min_chars=3)
    assert merged == [(0, 15, 3), (30, 39, 5)]


def test_enforce_min_run_count_last_span_merges_backward():
    spans = [(0, 9, 3), (20, 21, 1)]
    assert radon._enforce_min_run_count(spans, min_chars=3) == [(0, 21, 4)]


def test_enforce_min_run_count_single_span_left_alone():
    # Nothing left to merge into -- stays under min_chars.
    assert radon._enforce_min_run_count([(0, 1, 1)], min_chars=3) == [(0, 1, 1)]


def test_split_columns_into_words_overrides_gap_threshold_for_short_words():
    """Two single-run "words" separated by a gap far wider than the gap
    threshold must still merge into one word -- the 3-character minimum
    outranks the gap split."""
    arr = np.zeros(60, dtype=bool)
    arr[0:2] = True    # run 1 (1 run)
    arr[40:42] = True  # run 2 (1 run) -- 38px gap, way over gap_threshold=2.0
    assert radon._split_columns_into_words(arr, gap_threshold=2.0, min_chars=3) == [(0, 41)]


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
# segment_clusters end to end (operates on Vector clusters, not raw images
# -- it renders each cluster itself via render_cluster_for_radon)
# --------------------------------------------------------------------------
def test_segment_clusters_splits_into_one_segment_per_word():
    cluster = _two_word_cluster()

    segments = radon.segment_clusters([cluster])

    assert len(segments) == 2
    # every input Vector accounted for, none lost or duplicated
    seen = {v.seqno for seg in segments for v in seg.vectors}
    assert seen == {v.seqno for v in cluster}
    for seg in segments:
        assert abs(seg.angle) < 2.0


def test_segment_clusters_angle_is_full_precision_not_quarter_turn():
    # A cluster rotated by an arbitrary, non-quarter-turn angle -- the
    # returned Segment.angle must reflect that precisely, never rounded to
    # the nearest 90 degrees (see OCR/radon.py's standing precision note).
    cluster = [transform_vector(v, offset=(0.0, 0.0), rotation_deg=6.0) for v in _two_word_cluster()]

    segments = radon.segment_clusters([cluster])

    assert segments
    for seg in segments:
        assert seg.angle == pytest.approx(6.0, abs=radon.RADON_ANGLE_STEP_DEG * 2)
        # explicitly not snapped to a quarter turn
        assert seg.angle not in (0.0, 90.0, 180.0, 270.0, -90.0)


def test_segment_clusters_blank_cluster_yields_no_segments():
    # A single, tiny Vector with a degenerate rect -- effectively no ink to
    # detect any line bands from.
    v = _word_vector((0.0, 0.0, 0.001, 0.001), 0)
    assert radon.segment_clusters([[v]]) == []


def test_segment_clusters_empty_input():
    assert radon.segment_clusters([]) == []


def test_segment_clusters_pools_widest_run_across_lines():
    """Line A's wide (40pt) runs push the cluster-wide pooled gap
    threshold (15% of the widest run anywhere in the cluster) up to 6pt.
    Line B's own runs are narrow (4pt) -- judged on its own, its threshold
    would be a mere 0.6pt, well under the 5pt gap separating its two
    3-run "words" (each already clearing the 3-character minimum on its
    own, so item 4's merge doesn't confound this), which would stay
    separate at that local threshold. With the cluster-wide pool, that
    same 5pt gap falls under the shared 6pt threshold, so line B's two
    words merge into one 6-run segment -- the cross-line pooling
    behavioral difference a per-line-only threshold would not produce."""
    vectors: list[Vector] = []
    seq = 0
    # line A: 3 runs, 40pt wide each, 2pt gaps -- sets the pooled max.
    for x0 in (10, 52, 94):
        vectors.append(_word_vector((x0, 20, x0 + 40, 35), seq))
        seq += 1
    # line B, word 1: 3 runs, 4pt wide each, 1pt gaps.
    for x0 in (10, 15, 20):
        vectors.append(_word_vector((x0, 50, x0 + 4, 65), seq))
        seq += 1
    # line B, word 2: same pattern, starting 5pt after word 1 ends (x=24).
    for x0 in (29, 34, 39):
        vectors.append(_word_vector((x0, 50, x0 + 4, 65), seq))
        seq += 1
    line_b_seqnos = {v.seqno for v in vectors[3:]}

    segments = radon.segment_clusters([vectors])

    line_b_segments = [seg for seg in segments if {v.seqno for v in seg.vectors} & line_b_seqnos]
    assert len(line_b_segments) == 1
    assert {v.seqno for v in line_b_segments[0].vectors} == line_b_seqnos


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
    # min_chars=1: each word here is only 2 ink runs, below the real
    # 3-character minimum -- irrelevant to what this test checks (y-extent
    # tightness), so disabled here to keep the two words from merging.
    boxes = radon.split_words(np.asarray(img), gap_threshold=10.0, min_chars=1)
    assert len(boxes) == 2
    (_, short_y0, _, short_y1), (_, tall_y0, _, tall_y1) = boxes
    assert (short_y0, short_y1) != (tall_y0, tall_y1)
    assert (short_y1 - short_y0) < (tall_y1 - tall_y0)
