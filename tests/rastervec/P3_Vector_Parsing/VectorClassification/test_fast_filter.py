from __future__ import annotations

import numpy as np

from rastervec.P3_Vector_Parsing.VectorClassification import fast_filter


# --- group_quads_anchored -------------------------------------------------

def test_overlapping_quads_with_near_centers_share_a_group():
    a = (0.0, 0.0, 10.0, 4.0)
    b = (3.0, 0.0, 13.0, 4.0)  # overlaps a, centers 3pt apart
    assert fast_filter.group_quads_anchored([a, b], 10.0) == [[0, 1]]


def test_overlapping_quads_with_distant_centers_stay_apart():
    a = (0.0, 0.0, 30.0, 4.0)
    b = (25.0, 0.0, 55.0, 4.0)  # overlaps a, but centers 25pt apart
    assert fast_filter.group_quads_anchored([a, b], 10.0) == [[0], [1]]


def test_near_but_disjoint_quads_stay_apart():
    a = (0.0, 0.0, 4.0, 4.0)
    b = (5.0, 0.0, 9.0, 4.0)  # centers 5pt apart, but no overlap
    assert fast_filter.group_quads_anchored([a, b], 10.0) == [[0], [1]]


def test_grouping_is_anchored_not_chained():
    # a~b and b~c, but c is too far from the seed a -> c seeds its own group.
    a = (0.0, 0.0, 10.0, 4.0)
    b = (6.0, 0.0, 16.0, 4.0)
    c = (12.0, 0.0, 22.0, 4.0)
    assert fast_filter.group_quads_anchored([a, b, c], 10.0) == [[0, 1], [2]]


# --- crop geometry ----------------------------------------------------------

def test_group_crop_rect_pads_rounds_and_clamps():
    rects = [(2.4, 3.6, 10.2, 8.1), (9.0, 1.0, 20.5, 6.0)]
    assert fast_filter.group_crop_rect(rects, 2, 100, 100) == (0, 0, 23, 11)
    assert fast_filter.group_crop_rect(rects, 2, 21, 9) == (0, 0, 21, 9)


def test_pad_to_aspect_pads_thin_crop_and_reports_offset():
    rgb = np.zeros((10, 100, 3), dtype=np.uint8)
    padded, (ox, oy) = fast_filter.pad_to_aspect(rgb, 4.0)
    assert padded.shape == (25, 100, 3)
    assert (ox, oy) == (0, 7)
    assert (padded[oy:oy + 10, ox:ox + 100] == 0).all()
    assert (padded[:oy] == 255).all() and (padded[oy + 10:] == 255).all()


def test_pad_to_aspect_leaves_squarish_crop_alone():
    rgb = np.zeros((30, 60, 3), dtype=np.uint8)
    padded, offset = fast_filter.pad_to_aspect(rgb, 4.0)
    assert padded is rgb and offset == (0, 0)


def test_crop_heat_undoes_aspect_padding():
    g = _group(crop_rect=(5, 5, 9, 7), input_offset=(1, 3))
    mask = np.zeros((10, 8), dtype=np.float32)
    mask[3:5, 1:5] = 0.9
    heat = fast_filter.crop_heat(g, mask)
    assert heat.shape == (2, 4)
    assert np.allclose(heat, 0.9)


# --- vector_ink_fraction ----------------------------------------------------

def test_ink_fully_inside_hot_crop_scores_one():
    mask = np.ones((3, 3), dtype=bool)
    heat = np.ones((10, 10), dtype=np.float32)
    assert fast_filter.vector_ink_fraction(mask, (12.0, 12.0), (10, 10, 20, 20), heat, 0.5) == 1.0


def test_ink_outside_the_crop_counts_as_cold():
    # A 1x20 "line" starting inside a 10px-wide hot crop: half its ink is out.
    mask = np.ones((1, 20), dtype=bool)
    heat = np.ones((10, 10), dtype=np.float32)
    frac = fast_filter.vector_ink_fraction(mask, (10.0, 15.0), (10, 10, 20, 20), heat, 0.5)
    assert frac == 0.5


def test_ink_on_cold_heat_scores_zero_and_no_ink_scores_zero():
    heat = np.zeros((10, 10), dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)
    assert fast_filter.vector_ink_fraction(mask, (0.0, 0.0), (0, 0, 10, 10), heat, 0.5) == 0.0
    assert fast_filter.vector_ink_fraction(np.zeros((2, 2), bool), (0.0, 0.0), (0, 0, 10, 10), heat + 1, 0.5) == 0.0


# --- split_text_vectors -----------------------------------------------------

def _group(**overrides) -> fast_filter.FastGroup:
    fields = dict(
        cluster_vectors=[], dpi=72, padding=0.0, quad_idxs=[0], crop_rect=(0, 0, 10, 10),
        page_bbox=(0.0, 0.0, 10.0, 10.0), crop_page_bbox=(0.0, 0.0, 10.0, 10.0),
        fast_input=None, input_offset=(0, 0),
    )
    fields.update(overrides)
    return fast_filter.FastGroup(**fields)


def test_split_keeps_only_highlighted_vectors_of_accepted_groups(vector):
    # Cluster frame at 72 dpi with no padding: 1 px == 1 pt, origin at the
    # cluster's union-bbox corner (0, 0).
    glyph = vector(kind="re", bbox=(2.0, 2.0, 6.0, 6.0), seqno=1)
    line = vector(kind="l", bbox=(6.0, 4.0, 40.0, 4.0), seqno=2)
    far = vector(kind="l", bbox=(0.0, 30.0, 40.0, 30.0), seqno=3)  # not under the group
    cluster = [glyph, line, far]

    def ink(v, dpi, _gray):  # 1 px per pt, ink over the vector's whole bbox
        x0, y0, x1, y1 = v.bbox
        return np.ones((int(y1 - y0) + 1, int(x1 - x0) + 1), dtype=bool), (x0, y0)

    accepted = _group(cluster_vectors=cluster, heat=np.ones((10, 10), np.float32), accepted=True)
    text_ids = fast_filter.split_text_vectors(
        [accepted], heat_threshold=0.5, ink_fraction=0.5, gray_threshold=250, ink_mask_fn=ink,
    )
    assert text_ids == {id(glyph)}
    scored = {id(v): f for v, f in accepted.scores}
    assert scored[id(glyph)] == 1.0 and scored[id(line)] < 0.5
    assert id(far) not in scored


def test_split_ignores_blank_only_groups(vector):
    glyph = vector(kind="re", bbox=(2.0, 2.0, 6.0, 6.0), seqno=1)
    rejected = _group(cluster_vectors=[glyph], heat=np.ones((10, 10), np.float32), accepted=False)
    text_ids = fast_filter.split_text_vectors(
        [rejected], heat_threshold=0.5, ink_fraction=0.5, gray_threshold=250,
        ink_mask_fn=lambda *_a: (_ for _ in ()).throw(AssertionError("must not score")),
    )
    assert text_ids == set() and rejected.scores == []


def test_vector_ink_mask_renders_real_ink(vector):
    v = vector(kind="l", bbox=(0.0, 5.0, 20.0, 5.0), width=1.0, seqno=1)
    mask, origin = fast_filter.vector_ink_mask(v, 72, 250)
    assert mask.any()
    assert origin == (-1.5, 3.5)  # bbox origin minus (width/2 + 1)


def test_downscale_heat_caps_longest_side():
    small = fast_filter.downscale_heat(np.full((50, 1000), 0.5, np.float32), 256)
    assert max(small.shape) <= 256 and small.dtype == np.uint8
    assert int(small[0, 0]) == 127
