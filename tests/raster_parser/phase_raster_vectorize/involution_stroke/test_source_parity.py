"""The ported InvolutionStroke backend reproduces the original implementation on a rasterised
fixture page (`tests/references/p2_golden/`, recorded with that implementation's own code and the
authors' weights). Needs torch (the `ml` extra) and the weights (`download_weights`)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch")

from raster_parser.models.settings import RunSettings  # noqa: E402
from raster_parser.evaluation.conversion import rasterise_page  # noqa: E402
from raster_parser.phase_native_read.native_read import read_and_extract  # noqa: E402
from raster_parser.phase_raster_vectorize.involution_stroke import config  # noqa: E402
from raster_parser.phase_raster_vectorize.involution_stroke.adapter import extract  # noqa: E402
from tests.raster_parser.phase_raster_vectorize.junction.test_source_parity import snapshot  # noqa: E402

REFS = Path(__file__).resolve().parents[3] / "references"

pytestmark = [
    pytest.mark.slow,
    # ~2 h on CPU for this one page (the source run took 128 min): opt in explicitly.
    pytest.mark.skipif(
        not os.environ.get("RUN_VERY_SLOW"), reason="set RUN_VERY_SLOW=1 (about 2 h on CPU)"
    ),
]


def test_involution_stroke_matches_the_source_implementation():
    from raster_parser.phase_raster_vectorize.involution_stroke.inference import (
        default_centerline_path,
        default_intersection_path,
    )

    if not (default_centerline_path().is_file() and default_intersection_path().is_file()):
        pytest.skip("InvolutionStroke weights not downloaded")
    want = json.loads(
        (REFS / "p2_golden" / "involution_stroke_test_pdfs_1_p0_150dpi.json").read_text(
            encoding="utf-8"
        )
    )
    pdf = Path(tempfile.mkdtemp()) / "r.pdf"
    rasterise_page(str(REFS / "test_pdfs_1.pdf"), 0, str(pdf), dpi=150)
    native = read_and_extract(str(pdf), 0)
    got = json.loads(json.dumps(snapshot(*extract(native.images, native.page, RunSettings()))))
    assert got["vectors"] == want["vectors"]
    assert [t[0] for t in got["texts"]] == [t[0] for t in want["texts"]]
    assert config.CENTERLINE_WEIGHTS_FILENAME
