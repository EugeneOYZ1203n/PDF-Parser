"""Sanity checks that the not-yet-implemented Vector/Raster/Renderer/
helpers modules import cleanly and raise NotImplementedError on use --
catches import-time typos before those stages are actually built."""
from __future__ import annotations

import pytest

from rastervec.evaluation import Evaluation
from rastervec.helpers.clustering import Clustering
from rastervec.helpers.junction import JunctionDetector
from rastervec.helpers.masking import Masking
from rastervec.helpers.render_ocr import RenderOCR
from rastervec.raster import Raster
from rastervec.renderer import Renderer


@pytest.mark.parametrize(
    "obj, method, args",
    [
        (Raster(), "extract_images", (None,)),
        (Renderer(), "render_vector_cluster", (None, None, 300)),
        (Evaluation(), "build_pdf", ([], "out.pdf")),
        (Clustering(), "cluster_hsv", (None,)),
        (Masking(), "dilate_mask", (None, 1.0)),
        (JunctionDetector(), "generate_synthetic_data", (1,)),
        (RenderOCR(), "render_rotations", (None,)),
    ],
)
def test_stub_raises_not_implemented(obj, method, args):
    with pytest.raises(NotImplementedError):
        getattr(obj, method)(*args)
