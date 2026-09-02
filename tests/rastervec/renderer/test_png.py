from __future__ import annotations

import pytest

from rastervec.models import VectorPath
from rastervec.renderer import pixel_to_page_bbox, render_vector_cluster
from rastervec.renderer.png import _cluster_frame


def _make_path(
    *, kind="re", bbox=(0, 0, 10, 10), stroke_color=None, fill_color=None,
    stroke_width=None, dashes=None, closed=None, seq=0, item_index=0,
    even_odd=False,
) -> VectorPath:
    if kind == "l":
        points = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
    elif kind == "c":
        points = [
            (bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[1]), (bbox[2], bbox[3]),
        ]
    else:
        points = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
    return VectorPath(
        seq=seq, item_index=item_index, kind=kind, fill_rule="", points=points, bbox=bbox,
        stroke_color=stroke_color, fill_color=fill_color,
        stroke_opacity=None, fill_opacity=None, stroke_width=stroke_width,
        dashes=dashes, closed=closed, layer=None, page_index=0, even_odd=even_odd,
    )


def test_render_vector_cluster_requires_at_least_one_path():
    with pytest.raises(ValueError):
        render_vector_cluster([], 150)


def test_render_vector_cluster_draws_something():
    # A filled black rectangle should show up as non-white pixels in the render.
    path = _make_path(kind="re", bbox=(0, 0, 20, 10), fill_color=(0, 0, 0))
    image = render_vector_cluster([path], dpi=150)

    assert image.width > 0 and image.height > 0
    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255  # something was actually drawn, not a blank white page


def test_render_vector_cluster_blank_stroke_and_fill_stays_blank():
    # No stroke_color and no fill_color -> nothing should render.
    path = _make_path(kind="re", bbox=(0, 0, 20, 10))
    image = render_vector_cluster([path], dpi=150)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest == 255


def test_render_vector_cluster_line_kind():
    path = _make_path(kind="l", bbox=(0, 0, 20, 20), stroke_color=(0, 0, 0), stroke_width=2)
    image = render_vector_cluster([path], dpi=150)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_vector_cluster_even_odd_keeps_counter_open():
    # One drawing (shared seq), fill_rule "f", even_odd True: an outer rect
    # and a nested inner rect. Replayed as one composite even-odd path, the
    # inner area must stay WHITE (the glyph-counter case). Without the
    # per-drawing even_odd replay, the inner rect fills solid black.
    outer = _make_path(
        kind="re", bbox=(0, 0, 60, 60), fill_color=(0, 0, 0),
        seq=3, item_index=0, even_odd=True,
    )
    inner = _make_path(
        kind="re", bbox=(20, 20, 40, 40), fill_color=(0, 0, 0),
        seq=3, item_index=1, even_odd=True,
    )
    dpi = 300
    image = render_vector_cluster([outer, inner], dpi=dpi).convert("L")

    # Padding is asymmetric (see _cluster_frame) so the outer rect isn't
    # centered in the canvas -- derive the ring point from the real frame
    # instead of assuming a fixed fraction of the image width lands inside
    # the outer rect's left edge.
    _x0, _y0, pad_x, _pad_y = _cluster_frame([outer, inner])
    zoom = dpi / 72.0
    cx, cy = image.width // 2, image.height // 2
    ring_x = int((pad_x + 5) * zoom)  # just inside the outer rect's left edge, outside the inner
    assert image.getpixel((cx, cy)) > 200        # counter preserved (white hole)
    assert image.getpixel((ring_x, cy)) < 60     # outer ring still filled


def test_render_vector_cluster_no_even_odd_fills_counter_solid():
    # Control: same nested rects but even_odd False -> non-zero winding
    # fills the whole outer rect, centre goes dark.
    outer = _make_path(
        kind="re", bbox=(0, 0, 60, 60), fill_color=(0, 0, 0),
        seq=3, item_index=0, even_odd=False,
    )
    inner = _make_path(
        kind="re", bbox=(20, 20, 40, 40), fill_color=(0, 0, 0),
        seq=3, item_index=1, even_odd=False,
    )
    image = render_vector_cluster([outer, inner], dpi=300).convert("L")

    cx, cy = image.width // 2, image.height // 2
    assert image.getpixel((cx, cy)) < 60


def test_pixel_to_page_bbox_round_trips_cluster_frame():
    path = _make_path(kind="re", bbox=(0, 0, 20, 10), fill_color=(0, 0, 0))
    dpi = 150
    zoom = dpi / 72.0
    x0, y0, pad_x, pad_y = _cluster_frame([path])

    # A pixel-space point at the padded top-left corner should map back to
    # the cluster's own bbox origin in page space.
    page_bbox = pixel_to_page_bbox(
        [path], dpi, [(pad_x * zoom, pad_y * zoom), ((pad_x + 20) * zoom, (pad_y + 10) * zoom)],
    )
    assert page_bbox == pytest.approx((x0, y0, x0 + 20, y0 + 10))


def test_cluster_frame_horizontal_padding_more_generous_than_vertical():
    # A tall bbox (height 200) pushes both fraction-based margins well past
    # the stroke-safety floor, so the asymmetry actually engages: pad_x
    # (30% of height) should end up well past pad_y (5% of height).
    path = _make_path(kind="re", bbox=(0, 0, 20, 200), fill_color=(0, 0, 0))
    _x0, _y0, pad_x, pad_y = _cluster_frame([path])
    assert pad_x == pytest.approx(60.0)  # 200 * 0.30
    assert pad_y == pytest.approx(10.0)  # 200 * 0.05
    assert pad_x > pad_y
