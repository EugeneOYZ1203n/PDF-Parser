"""Unit tests for `rastervec/notebooks/_liu_dori_showcase_helpers.py` -- the
synthetic cases, value extraction and pictures behind
`liu_dori_showcase.ipynb`."""
from __future__ import annotations

import math

import pytest

from rastervec.notebooks import _liu_dori_showcase_helpers as h

BUILDERS = [
    h.bars_cases, h.matching_cases, h.factor_cases, h.line_cases, h.page_cases, h.split_cases,
    h.pixel_cases, h.ssim_cases, h.sanity_cases, h.bias_cases, h.diagnostic_cases,
]

DOCUMENTED = [
    "D_v", "F_v", "VRI", "Q_pt", "Q_od", "Q_w", "Q_v(c)", "Q_b(g)", "Q_fr(g)", "Q_v(g)", "Q_b(k)",
    "Q_fr(k)", "Q_v(k)", "|D(g)|", "|G(k)|", "mean |D(g)|", "mean |G(k)|", "D_p", "F_p", "PRI",
    "SSIM", "PSNR", "length ratio", "direction", "D_v text", "D_v non-text", "F_v text", "D_v solid",
    "D_v dashed", "D_v vector_to_raster", "D_v original_raster",
]


@pytest.fixture
def book(tmp_path):
    return h.Book(root=tmp_path)


def test_paper_cases_reproduce_known_values():
    good = h.values(h.run(h.Scenario("good", [h.G()], [h.G()])))
    assert good["D_v"] == pytest.approx(1.0) and good["F_v"] == pytest.approx(0.0)
    frag = h.values(h.run(h.Scenario("frag", [h.G()], [h.line((10, 20), (35, 20)), h.line((50, 20), (90, 20))],
                                     cfg=h.PAPER)))
    assert frag["D_v"] == pytest.approx(0.59, abs=0.01)
    miss = h.values(h.run(h.Scenario("miss", [h.G()], [h.line((10, 35), (90, 35))])))
    assert miss["D_v"] == 0.0 and miss["F_v"] == 1.0 and miss["overlaps"] == 0


def test_values_has_every_documented_number_and_na_without_preds():
    v = h.values(h.run(h.Scenario("none", [h.G()], [])))
    assert set(DOCUMENTED) <= set(v)
    assert math.isnan(v["F_v"]) and math.isnan(v["VRI"]) and v["D_v"] == 0.0


def test_fragments_follow_inverse_sqrt():
    for n in (2, 3, 4):
        v = h.values(h.run(h.fragment_case(n)))
        assert v["Q_fr(g)"] == pytest.approx(1 / math.sqrt(n), abs=1e-6)


def test_overlap_segment_matches_scored_length():
    for sc in h.matching_cases() + h.factor_cases():
        o = h.run(sc)
        for i in range(len(o.overlaps)):
            c0, c1 = h.overlap_segment(o, i)
            assert math.dist(c0, c1) == pytest.approx(o.overlaps[i].length, abs=1e-6)


@pytest.mark.parametrize("builder", BUILDERS, ids=[b.__name__ for b in BUILDERS])
def test_every_case_runs(builder):
    for o in h.run_all(builder()):
        h.values(o)


def test_pictures_are_written(book):
    outs = h.run_all(h.factor_cases()[:2])
    paths = [
        book.cases("cases", outs, ["D_v", "F_v"]),
        book.sweep("sweep", h.offset_case, [0, 1, 2], ["Q_od"], xlabel="offset", refs={"one": lambda x: 1.0}),
        book.pixels("pixels", outs, ["D_p"]),
        book.histograms("hist", h.run(h.drawing_case())),
        book.direction("dir", outs),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
    assert book.saved == paths


def test_ink_overlay_colours():
    o = h.run(h.page_cases()[1])  # one line missed
    img = h.ink_overlay(o)
    assert img.shape[2] == 3
    assert ((img == (220, 40, 40)).all(axis=2)).any()   # GT-only ink
    assert ((img == (0, 0, 0)).all(axis=2)).any()       # both


def test_summary_table_shape():
    t = h.summary_table(h.run_all(h.page_cases()[:2]), ["D_v", "F_v"])
    assert t.count("\n") == 3 and "n/a" not in t
