from __future__ import annotations

import numpy as np
import pytest

from rastervec.commons.models import Vector
from rastervec.OCR import radon


def _word_vector(bbox: tuple[float, float, float, float], seqno: int) -> Vector:
    """A single filled rect Vector -- stands in for one "dash" of ink in a
    synthetic text-like cluster."""
    return Vector(
        type="f", items=[("re", bbox, 1)], color=None, fill=(0, 0, 0), width=None,
        dashes=None, closePath=True, lineCap=0, lineJoin=0, even_odd=False,
        stroke_opacity=None, fill_opacity=None, layer=None, rect=bbox,
        scissor=None, seqno=seqno, blendmode=None, isolated=False, knockout=False,
        opacity=None, page_index=0,
    )


def _line_cluster(y0: float = 20.0, y1: float = 36.0, n: int = 4, spacing: float = 30.0) -> list[Vector]:
    """`n` short dashes on one horizontal baseline -- a fake single text
    line for the sweep to lock onto."""
    return [
        _word_vector((i * spacing, y0, i * spacing + 8, y1), i)
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# rotate_pts / item_subsegments / cluster_centre / rotated_segments
# --------------------------------------------------------------------------
def test_rotate_pts_identity_at_zero_theta():
    pts = np.array([(1.0, 2.0), (3.0, 4.0)])
    out = radon.rotate_pts(pts, 0.0, centre=np.array([0.0, 0.0]))
    assert np.allclose(out, pts)


def test_rotate_pts_rotates_about_centre():
    # A point directly "east" of centre, rotated by theta=90 (this module
    # applies -theta_deg internally), lands on the -y axis.
    pts = np.array([(10.0, 0.0)])
    out = radon.rotate_pts(pts, 90.0, centre=np.array([0.0, 0.0]))
    assert out[0] == pytest.approx((0.0, -10.0), abs=1e-6)


def test_item_subsegments_line():
    segs = radon.item_subsegments(("l", (0.0, 0.0), (10.0, 0.0)))
    assert segs == [((0.0, 0.0), (10.0, 0.0))]


def test_item_subsegments_rect_has_four_edges():
    segs = radon.item_subsegments(("re", (0.0, 0.0, 10.0, 5.0), 1))
    assert len(segs) == 4


def test_item_subsegments_curve_approximates_with_chords():
    segs = radon.item_subsegments(
        ("c", (0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)),
    )
    assert len(segs) == radon._CURVE_SAMPLES


def test_item_subsegments_unknown_kind_is_empty():
    assert radon.item_subsegments(("m", (0.0, 0.0))) == []


def test_cluster_centre_is_bbox_midpoint(vector):
    v1 = vector(bbox=(0.0, 0.0, 10.0, 10.0))
    v2 = vector(bbox=(10.0, 10.0, 20.0, 20.0))
    centre = radon.cluster_centre([v1, v2])
    assert tuple(centre) == pytest.approx((10.0, 10.0))


def test_rotated_segments_zero_theta_matches_original_bbox():
    cluster = _line_cluster()
    per_vector, (X0, X1, Y0, Y1) = radon.rotated_segments(cluster, 0.0)
    assert len(per_vector) == len(cluster)
    assert Y0 == pytest.approx(20.0)
    assert Y1 == pytest.approx(36.0)
    assert X0 == pytest.approx(0.0)


def test_rotated_segments_vector_with_no_items_yields_empty_subsegments():
    v = _word_vector((0.0, 0.0, 10.0, 10.0), 0)
    v.items = []
    per_vector, _bbox = radon.rotated_segments([v], 0.0)
    assert per_vector[0].shape == (0, 2, 2)


# --------------------------------------------------------------------------
# n_lines_for / project
# --------------------------------------------------------------------------
def test_n_lines_for_scales_with_height_and_clamps():
    assert radon.n_lines_for(1.0, line_spacing=1.0, max_lines=600) == 2  # floor
    assert radon.n_lines_for(50.0, line_spacing=1.0, max_lines=600) == 50
    assert radon.n_lines_for(10000.0, line_spacing=1.0, max_lines=100) == 100  # cap


def test_project_zero_height_cluster_returns_zero_profile():
    v = _word_vector((0.0, 0.0, 10.0, 0.0), 0)  # zero-height rect
    line_y, profile = radon.project([v], 0.0)
    assert np.all(profile == 0.0)


def test_project_counts_intersections_per_ray():
    cluster = _line_cluster(y0=20.0, y1=36.0, n=3)
    line_y, profile = radon.project(cluster, 0.0)
    assert len(line_y) == len(profile)
    # every ray strictly inside the shared y-extent hits all 3 vectors
    inside = (line_y > 20.0) & (line_y < 36.0)
    assert np.all(profile[inside] == 3.0)


def test_project_profile_sharper_at_true_orientation():
    """A single horizontal line of dashes: the profile at theta=0 (aligned)
    should be strictly sharper (higher sum-of-squares) than at a
    meaningfully rotated theta, since rotating spreads the ink over more
    distinct y-rays."""
    cluster = _line_cluster()
    _, profile_aligned = radon.project(cluster, 0.0)
    _, profile_rotated = radon.project(cluster, 20.0)
    score_aligned, _ = radon.val_postl_sq(profile_aligned, profile_aligned)
    score_rotated, _ = radon.val_postl_sq(profile_rotated, profile_rotated)
    assert score_aligned > score_rotated


# --------------------------------------------------------------------------
# value functions
# --------------------------------------------------------------------------
def test_val_postl_sq_sum_of_squares():
    a = np.array([1.0, 2.0, 3.0])
    score, a_is_correct = radon.val_postl_sq(a, a)
    assert score == pytest.approx(14.0)
    assert a_is_correct is True


def test_val_variance_matches_numpy_var():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    score, _ = radon.val_variance(a, a)
    assert score == pytest.approx(float(np.var(a)))


# --------------------------------------------------------------------------
# sweep_rotation -- currently a no-op passthrough (rotation correction
# disabled at the user's request, pending a future revisit).
# --------------------------------------------------------------------------
def test_sweep_rotation_empty_vectors_returns_center_unchanged():
    result = radon.sweep_rotation([], center_theta_deg=12.0)
    assert result.theta_deg == 12.0


def test_sweep_rotation_never_changes_the_input_angle():
    """No-op passthrough: whatever `center_theta_deg` comes in from
    PaddleOCR's own coarse estimate comes back unchanged, regardless of
    `vectors`/`range_deg`/`step_deg`/`value_fn`."""
    cluster = _line_cluster()
    result = radon.sweep_rotation(cluster, center_theta_deg=3.0, range_deg=10.0, step_deg=0.5)
    assert result.theta_deg == 3.0


def test_sweep_rotation_ignores_value_fn():
    cluster = _line_cluster()
    calls = []

    def spy_value_fn(a, b):
        calls.append(1)
        return radon.val_variance(a, b)

    result = radon.sweep_rotation(
        cluster, center_theta_deg=7.5, range_deg=1.0, step_deg=1.0, value_fn=spy_value_fn,
    )
    assert result.theta_deg == 7.5
    assert not calls  # the sweep loop never runs, so value_fn is never invoked
