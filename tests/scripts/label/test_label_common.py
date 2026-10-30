"""`scripts/label/_common.py` helpers that don't need a Tk display: the
(layer, color, width) separation behind `manual_vector_text_label.py`'s bucket filter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from raster_parser.phase_native_read.reader import Reader
from raster_parser.phase_native_read.vector_extract import extract_vectors

pytest.importorskip("tkinter")  # _common.py is Tk plumbing; Ubuntu ships Tk as python3-tk

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PDFS = sorted((REPO_ROOT / "tests" / "references").glob("test_pdfs_*.pdf"))

_spec = importlib.util.spec_from_file_location(
    "label_common", REPO_ROOT / "scripts" / "label" / "_common.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

separate_by_layer = _mod.separate_by_layer
separate_by_color = _mod.separate_by_color
separate_by_width = _mod.separate_by_width
separate_by_layer_color_width = _mod.separate_by_layer_color_width


def test_separate_by_layer_groups_by_layer_field(vector):
    vectors = [
        vector(layer="A"),
        vector(layer="A"),
        vector(layer="B"),
        vector(layer=None),
    ]

    groups = separate_by_layer(vectors)

    assert set(groups.keys()) == {"A", "B", ""}
    assert len(groups["A"]) == 2
    assert len(groups["B"]) == 1
    assert len(groups[""]) == 1
    for key, members in groups.items():
        assert all((v.layer or "") == key for v in members)


def test_separate_by_color_groups_by_color_fill_and_opacity(vector):
    v0 = vector(color=(1, 0, 0), fill=(0, 1, 0))
    v1 = vector(color=None, fill=(0, 1, 0))
    v2 = vector(color=None, fill=None)
    # same color/fill as v0 but different opacity -> its own group.
    v3 = vector(color=(1, 0, 0), fill=(0, 1, 0), stroke_opacity=0.5)
    vectors = [v0, v1, v2, v3]

    groups = separate_by_color(vectors)

    assert groups[(1, 0, 0), (0, 1, 0), None, None] == [v0]
    assert groups[None, (0, 1, 0), None, None] == [v1]
    assert groups[None, None, None, None] == [v2]
    assert groups[(1, 0, 0), (0, 1, 0), 0.5, None] == [v3]
    for key, members in groups.items():
        for v in members:
            assert (v.color, v.fill, v.stroke_opacity, v.fill_opacity) == key


@pytest.mark.skipif(not REFERENCE_PDFS, reason="tests/references/test_pdfs_*.pdf not generated")
def test_separation_invariants_hold_for_reference_pdf_vectors():
    with Reader(str(REFERENCE_PDFS[0])) as reader:
        page = reader.get_page(0)
        vectors = extract_vectors(page)

    by_layer = separate_by_layer(vectors)
    for layer, members in by_layer.items():
        assert all((v.layer or "") == layer for v in members)

    by_color = separate_by_color(vectors)
    for key, members in by_color.items():
        for v in members:
            assert (v.color, v.fill, v.stroke_opacity, v.fill_opacity) == key


def test_separate_by_layer_color_width_splits_on_all_three_axes(vector):
    a = vector(layer="A", color=(0, 0, 0), width=1.0)
    b = vector(layer="A", color=(0, 0, 0), width=1.0)
    c = vector(layer="A", color=(0, 0, 0), width=2.0)
    d = vector(layer="A", color=(1, 0, 0), width=1.0)
    e = vector(layer="B", color=(0, 0, 0), width=1.0)

    buckets = separate_by_layer_color_width([a, b, c, d, e])

    assert sorted(len(bk) for bk in buckets) == [1, 1, 1, 2]
    assert [a, b] in buckets
