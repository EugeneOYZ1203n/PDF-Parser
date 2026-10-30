"""`commons.raster_text.debug_tools`: the debug / sample-run plumbing every raster backend shares
(layer PDFs for the viewer, clip rendering, diff canvases), not copied per backend."""

from __future__ import annotations

import numpy as np
import pytest

from raster_parser.commons.raster_text import debug_tools as dt
from raster_parser.models.raster_text import RasterItem


@pytest.mark.parametrize("backend", ["junction"])
def test_backends_use_the_shared_helpers(backend):
    import importlib

    pkg = f"raster_parser.phase_raster_vectorize.{backend}"
    predict = importlib.import_module(f"{pkg}.predict")
    diff = importlib.import_module(f"{pkg}.diff")
    adapter = importlib.import_module(f"{pkg}.adapter")
    assert predict.LayerSet is dt.LayerSet and predict.parse_clip is dt.parse_clip
    assert (
        predict.render_region is dt.render_region and predict.save_histograms is dt.save_histograms
    )
    assert diff.paste_codes is dt.paste_codes and diff.rendered_to_rgba is dt.rendered_to_rgba
    assert adapter.RasterItem is RasterItem and adapter._finish_doc is dt.finish_doc


def test_parse_clip():
    assert dt.parse_clip(None) is None
    assert dt.parse_clip("1, 2,3,4") == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(SystemExit):
        dt.parse_clip("3,2,1,4")


def test_paste_codes_keeps_existing_ink():
    canvas = np.zeros((3, 3), np.uint8)
    canvas[0, 0] = 2
    dt.paste_codes(canvas, np.array([[0, 1], [1, 1]], np.uint8), 0, 0)
    assert canvas.tolist() == [[2, 1, 0], [1, 1, 0], [0, 0, 0]]


def test_traced_vectors_share_one_paint_default():
    from raster_parser.commons.raster_text.debug_tools import traced_vector_kwargs
    from raster_parser.phase_raster_vectorize.junction import adapter as jn

    assert jn._base_vector_kwargs is traced_vector_kwargs
    kw = traced_vector_kwargs(2, 7, (1.0, 0.0, 0.0))
    assert (kw["page_index"], kw["seqno"], kw["color"], kw["fill"]) == (2, 7, (1.0, 0.0, 0.0), None)
