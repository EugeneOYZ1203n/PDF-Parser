"""The ported Junction backend reproduces the source (rastervec) implementation: the snapshots in
`tests/references/p2_golden/` were recorded by running rastervec's Junction on the same
rasterised fixture pages (`p2_snapshot.py`, 150 dpi)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from raster_parser.models.settings import RunSettings
from raster_parser.evaluation.conversion import rasterise_page
from raster_parser.phase_native_read.native_read import read_and_extract
from raster_parser.phase_raster_vectorize.junction.adapter import extract

REFS = Path(__file__).resolve().parents[3] / "references"

pytestmark = pytest.mark.slow


def _r(x):
    return round(float(x), 2)


def snapshot(vectors, texts) -> dict:
    """The same compact form `p2_snapshot.py` writes."""
    return {
        "vectors": sorted(
            [
                v.type,
                [_r(c) for c in v.rect],
                [round(c, 3) for c in (v.color or ())],
                [round(c, 3) for c in (v.fill or ())],
                _r(v.width or 0),
                [
                    [it[0]]
                    + [
                        [_r(p[0]), _r(p[1])]
                        for p in it[1:]
                        if isinstance(p, (tuple, list)) and len(p) == 2
                    ]
                    for it in v.items
                ],
            ]
            for v in vectors
        ),
        "texts": sorted([t.text, [_r(c) for c in t.bbox]] for t in texts),
    }


@pytest.mark.parametrize("n", [1, 3])
def test_junction_matches_the_source_implementation(n):
    want = json.loads(
        (REFS / "p2_golden" / f"junction_test_pdfs_{n}_p0_150dpi.json").read_text(encoding="utf-8")
    )
    pdf = Path(tempfile.mkdtemp()) / "r.pdf"
    rasterise_page(str(REFS / f"test_pdfs_{n}.pdf"), 0, str(pdf), dpi=150)
    native = read_and_extract(str(pdf), 0)
    got = json.loads(json.dumps(snapshot(*extract(native.images, native.page, RunSettings()))))
    assert len(got["vectors"]) == len(want["vectors"])
    assert got["vectors"] == want["vectors"]
    assert [t[0] for t in got["texts"]] == [t[0] for t in want["texts"]]
