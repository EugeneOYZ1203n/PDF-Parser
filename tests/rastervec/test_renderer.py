from __future__ import annotations

import pytest

from rastervec.models import VectorPath
from rastervec.renderer import Renderer


class _Meta:
    index = 0


class _Page:
    meta = _Meta()


def _make_path(
    *, kind="re", bbox=(0, 0, 10, 10), stroke_color=None, fill_color=None,
    stroke_width=None, dashes=None, closed=None,
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
        seq=0, item_index=0, kind=kind, fill_rule="", points=points, bbox=bbox,
        stroke_color=stroke_color, fill_color=fill_color, stroke_width=stroke_width,
        dashes=dashes, closed=closed, layer=None, page_index=0,
    )


def test_render_vector_cluster_requires_at_least_one_path():
    with pytest.raises(ValueError):
        Renderer().render_vector_cluster([], _Page(), 150)


def test_render_vector_cluster_draws_something():
    # A filled black rectangle should show up as non-white pixels in the render.
    path = _make_path(kind="re", bbox=(0, 0, 20, 10), fill_color=(0, 0, 0))
    image = Renderer().render_vector_cluster([path], _Page(), dpi=150)

    assert image.width > 0 and image.height > 0
    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255  # something was actually drawn, not a blank white page


def test_render_vector_cluster_blank_stroke_and_fill_stays_blank():
    # No stroke_color and no fill_color -> nothing should render.
    path = _make_path(kind="re", bbox=(0, 0, 20, 10))
    image = Renderer().render_vector_cluster([path], _Page(), dpi=150)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest == 255


def test_render_vector_cluster_line_kind():
    path = _make_path(kind="l", bbox=(0, 0, 20, 20), stroke_color=(0, 0, 0), stroke_width=2)
    image = Renderer().render_vector_cluster([path], _Page(), dpi=150)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255
