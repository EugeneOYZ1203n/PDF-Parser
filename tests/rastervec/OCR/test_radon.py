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
    """Two words on one baseline, separated far enough horizontally that
    `group_by_aspect` keeps them two segments (combined width / line height
    clears RADON_MAX_SEGMENT_ASPECT). The first word is 3 dashes, the
    second 4, so their crops are genuinely distinct -- mirrors
    _text_image's per-word dash pattern."""
    vectors: list[Vector] = []
    seq = 0
    for word_x, n in ((20, 3), (160, 4)):
        for i in range(n):
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


def test_line_gaps_returns_inter_run_gaps():
    arr = np.zeros(50, dtype=bool)
    for s, e in [(0, 4), (10, 11), (20, 29)]:
        arr[s:e + 1] = True
    assert radon._line_gaps(arr) == [5.0, 8.0]
    # fewer than two runs -> no gaps to sample
    assert radon._line_gaps(np.zeros(5, dtype=bool)) == []
    single = np.zeros(10, dtype=bool)
    single[2:5] = True
    assert radon._line_gaps(single) == []


def test_cluster_gap_threshold_is_1_3x_median_gap_and_floors():
    # median([2, 4, 30]) == 4 -> 1.3 * 4
    assert radon._cluster_gap_threshold([2.0, 4.0, 30.0]) == pytest.approx(5.2)
    assert radon._cluster_gap_threshold([]) == radon.RADON_MIN_GAP_PX
    # a median at/near zero falls back to the floor
    assert radon._cluster_gap_threshold([1.0, 1.0]) == radon.RADON_MIN_GAP_PX
    assert radon._cluster_gap_threshold([10.0], min_gap=0.5) == pytest.approx(13.0)


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
# pad_image -- the pipeline's only padding step
# --------------------------------------------------------------------------
def test_pad_image_adds_white_border_of_the_configured_fraction():
    img = np.zeros((50, 200), dtype=np.uint8)  # all-black, so the border stands out

    padded, (pad_x, pad_y) = radon.pad_image(img, fraction=0.1)

    assert (pad_x, pad_y) == (20, 5)
    assert padded.shape == (50 + 2 * 5, 200 + 2 * 20)
    # the added border is white...
    assert (padded[:pad_y, :] == 255).all()
    assert (padded[-pad_y:, :] == 255).all()
    assert (padded[:, :pad_x] == 255).all()
    assert (padded[:, -pad_x:] == 255).all()
    # ...and the original lands exactly at (pad_y, pad_x)
    assert np.array_equal(padded[pad_y:pad_y + 50, pad_x:pad_x + 200], img)


def test_pad_image_defaults_to_the_config_fraction():
    padded, (pad_x, pad_y) = radon.pad_image(np.zeros((30, 40), dtype=np.uint8))
    assert (pad_x, pad_y) == (
        round(40 * radon.RADON_PAD_FRACTION), round(30 * radon.RADON_PAD_FRACTION),
    )
    assert padded.shape == (30 + 2 * pad_y, 40 + 2 * pad_x)


def test_pad_image_zero_area_returned_unchanged():
    empty = np.zeros((0, 5), dtype=np.uint8)
    padded, offset = radon.pad_image(empty)
    assert padded is empty
    assert offset == (0, 0)


def test_pad_image_copies_rather_than_viewing_its_input():
    img = np.zeros((20, 20), dtype=np.uint8)
    padded, (pad_x, pad_y) = radon.pad_image(img)
    img[:] = 128
    assert (padded[pad_y:pad_y + 20, pad_x:pad_x + 20] == 0).all()


# --------------------------------------------------------------------------
# gap-quality objective
# --------------------------------------------------------------------------
def test_profile_peaks_bands_two_plateaus():
    prof = np.zeros(60, dtype=np.float64)
    prof[5:15] = 10.0
    prof[40:50] = 10.0
    peaks = radon._profile_peaks(prof)
    assert len(peaks) == 2
    # smoothing widens each plateau by ~1 bin on each side
    (s0, e0), (s1, e1) = peaks
    assert s0 <= 5 and e0 >= 14
    assert s1 <= 40 and e1 >= 49


def test_gap_scores_formula():
    eps = radon.RADON_GAP_SCORE_EPS
    # a tall third peak sets the peak-detection threshold high enough that a
    # partially-filled valley (floor 40) still separates its two 100-peaks.
    prof = np.zeros(120, dtype=np.float64)
    prof[5:15] = 100.0
    prof[15:30] = 40.0     # valley floor between peak A and peak B
    prof[30:40] = 100.0
    prof[95:105] = 1000.0  # dominates -> threshold ~50
    partial, deep = radon._gap_scores(prof)
    # L = R = 100, mid = 100, M = 40 -> g = 1 - (100 - 40)/(100 + eps)
    assert partial == pytest.approx(1.0 - 60.0 / (100.0 + eps), abs=0.08)
    # B..C valley is a true zero -> near-perfect (near-zero) gap score
    assert deep < 0.05
    assert deep < partial


def test_gap_scores_needs_two_peaks():
    one = np.zeros(40, dtype=np.float64)
    one[10:20] = 10.0
    assert radon._gap_scores(one) == []


def test_skew_objective_infinite_for_single_peak():
    prof = np.zeros(40, dtype=np.float64)
    prof[10:20] = 5.0
    assert radon._skew_objective(prof) == float("inf")


def test_skew_objective_prefers_few_clean_gaps():
    clean = np.zeros(90, dtype=np.float64)
    for c in (10, 45, 78):
        clean[c:c + 6] = 10.0                      # 3 peaks, 2 deep gaps
    combed = np.zeros(90, dtype=np.float64)
    for c in range(6, 86, 8):
        combed[c:c + 3] = 10.0                     # ~10 peaks, many gaps
    assert radon._skew_objective(clean) < radon._skew_objective(combed)


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


def test_estimate_skew_single_line_via_90_degree_rule():
    """One text line has no line gaps, so the sweep locks onto the
    along-baseline letter comb; the 90-degrees-from-best check turns that
    back into the real skew."""
    w, h = 260, 60
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for word_x in (20, 100, 170):
        for i in range(3):
            x = word_x + i * 12
            d.rectangle([x, 24, x + 8, 40], fill=0)
    for true_skew in (0.0, 4.0):
        rotated = sk_rotate(np.asarray(img), -true_skew, resize=True, cval=255, preserve_range=True)
        est = radon.estimate_skew(rotated.astype(np.uint8))
        assert est == pytest.approx(true_skew, abs=2.0)


def test_estimate_skew_blank_is_zero():
    assert radon.estimate_skew(np.full((30, 30), 255, np.uint8)) == 0.0


def test_estimate_skew_project_hook_is_transparent():
    """Passing the default `_project` explicitly must be identical to
    omitting it -- the hook only exists so a notebook can swap the
    projection."""
    rotated = sk_rotate(_text_image(), -4.0, resize=True, cval=255, preserve_range=True)
    ink = radon.to_ink(rotated.astype(np.uint8))
    assert radon.estimate_skew_from_mask(ink) == radon.estimate_skew_from_mask(
        ink, project=radon._project,
    )


# --------------------------------------------------------------------------
# line bands / spacing
# --------------------------------------------------------------------------
def test_line_bands_counts_lines():
    img = _text_image(lines=4)
    prof = radon.row_profile(radon.to_ink(img))
    assert len(radon.line_bands(prof)) == 4
    assert radon.line_spacing(prof) > 0


def test_line_bands_keeps_a_shallow_valley_as_one_line():
    """Two peaks whose separating valley never drops toward zero (a bad
    gap -- overlapping ascenders/descenders) stay a single band; a deep
    white valley splits."""
    peak = 100.0
    good = np.zeros(90, dtype=np.float64)
    good[10:25] = peak
    good[65:80] = peak                       # deep zero valley between them
    assert len(radon.line_bands(good)) == 2

    bad = np.full(90, 0.55 * peak, dtype=np.float64)  # valley floor ~55% of peak
    bad[10:25] = peak
    bad[65:80] = peak
    assert len(radon.line_bands(bad)) == 1


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


def test_segment_clusters_captures_each_words_own_crop_image():
    """Each Segment's `image` is captured directly from the deskewed
    render during segmentation, so OCR never has to re-render from
    vectors -- non-empty, 2-D, and distinct per word. It is also already
    white-padded (`pad_image`), since the OCR backend does no crop
    normalization of its own."""
    cluster = _two_word_cluster()

    segments = radon.segment_clusters([cluster])

    assert len(segments) == 2
    for seg in segments:
        assert seg.image is not None
        assert seg.image.ndim == 2
        assert seg.image.size > 0
        # the per-word pad ran: every outer edge is blank white
        assert (seg.image[0, :] == 255).all()
        assert (seg.image[-1, :] == 255).all()
        assert (seg.image[:, 0] == 255).all()
        assert (seg.image[:, -1] == 255).all()
    # the two words' crops are independently sized/positioned, not the same
    # array reused
    assert segments[0].image.shape != segments[1].image.shape or not np.array_equal(
        segments[0].image, segments[1].image,
    )


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


def test_segment_clusters_pools_gaps_across_lines():
    """Line A's five wide (20pt) gaps drag the cluster-wide pooled median
    gap up to ~12pt, so the shared threshold (1.3x that) lands near ~16pt.
    Line B's own gaps are narrow -- judged on its own, its median gap is
    1pt and its threshold a mere 1.3pt, well under the 5pt gap separating
    its two 3-run "words" (each already clearing the 3-character minimum on
    its own, so `_enforce_min_run_count` doesn't confound this), which
    would therefore stay separate at that local threshold. Against the
    cluster-wide pool, that same 5pt gap falls under the shared threshold,
    so line B's two words merge into one 6-run segment -- the cross-line
    pooling behavioral difference a per-line-only threshold would not
    produce."""
    vectors: list[Vector] = []
    seq = 0
    # line A: 6 runs separated by 20pt gaps -- pulls the pooled median up.
    for x0 in (10, 34, 58, 82, 106, 130):
        vectors.append(_word_vector((x0, 20, x0 + 4, 35), seq))
        seq += 1
    # line B, word 1: 3 runs, 1pt gaps.
    for x0 in (10, 15, 20):
        vectors.append(_word_vector((x0, 50, x0 + 4, 65), seq))
        seq += 1
    # line B, word 2: same pattern, starting 5pt after word 1 ends (x=24).
    for x0 in (29, 34, 39):
        vectors.append(_word_vector((x0, 50, x0 + 4, 65), seq))
        seq += 1
    line_b_seqnos = {v.seqno for v in vectors[6:]}

    segments = radon.segment_clusters([vectors])

    line_b_segments = [seg for seg in segments if {v.seqno for v in seg.vectors} & line_b_seqnos]
    assert len(line_b_segments) == 1
    assert {v.seqno for v in line_b_segments[0].vectors} == line_b_seqnos


def test_rotation_inverse_round_trips():
    out_shape, forward, inverse = radon._rotation((50, 80), 12.0)
    pts = np.array([(0.0, 0.0), (79.0, 0.0), (40.0, 25.0)])
    back = inverse(forward(pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_group_by_aspect_gives_each_segment_its_own_tight_y_extent():
    """Two words on one baseline with different glyph heights (a short
    word, then a tall one) must get different y-extents -- not the whole
    line's shared ink bbox repeated for both. Each "word" is two small ink
    runs (like _text_image's dashes) with a small intra-word gap and a wide
    inter-word gap. `max_aspect` is set low so the two stay separate
    segments rather than grouping into one."""
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
    boxes = radon.group_by_aspect(
        np.asarray(img), gap_threshold=10.0, min_chars=1, max_aspect=3.0,
    )
    assert len(boxes) == 2
    (_, short_y0, _, short_y1), (_, tall_y0, _, tall_y1) = boxes
    assert (short_y0, short_y1) != (tall_y0, tall_y1)
    assert (short_y1 - short_y0) < (tall_y1 - tall_y0)


def test_group_by_aspect_caps_segment_aspect_ratio():
    """A long single baseline of many equal words is split into several
    segments, each roughly under the aspect-ratio cap -- and a single word
    is never split, even when it alone is wider than the cap."""
    w, h = 600, 30
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    x = 10
    for _ in range(8):
        for _ in range(4):  # 4 ticks per word -> clears min_chars
            d.rectangle([x, 8, x + 4, 22], fill=0)
            x += 8
        x += 24  # wide inter-word gap

    boxes = radon.group_by_aspect(np.asarray(img), gap_threshold=12.0)

    assert len(boxes) >= 2  # 8 words, line ink height ~15 -> total aspect ~35
    line_h = 15
    for x0, _y0, x1, _y1 in boxes:
        assert (x1 - x0) / line_h < radon.RADON_MAX_SEGMENT_ASPECT + 2

    # one solid wide word (no internal gaps) is one segment, not split
    solid = Image.new("L", (400, 30), 255)
    ImageDraw.Draw(solid).rectangle([10, 8, 380, 22], fill=0)
    assert len(radon.group_by_aspect(np.asarray(solid), gap_threshold=12.0)) == 1
