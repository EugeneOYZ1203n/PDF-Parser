from __future__ import annotations

import fitz
import numpy as np
import pytest

from rastervec.Evaluation.Labelling import visible_geometry as vg
from rastervec.Evaluation.Labelling.raster_label import _all_geometry_for_page, raster_geometry_for_page


def _pdf(tmp_path, ops: bytes, name="p.pdf", size=(300, 300)):
    """One page whose content stream is `ops` (PDF y-up coordinates)."""
    doc = fitz.open()
    page = doc.new_page(width=size[0], height=size[1])
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, ops)
    doc.xref_set_key(page.xref, "Contents", f"[{xref} 0 R]")
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return str(path)


def _spans(annotations, axis=0):
    return sorted((min(p[axis] for p in a.points), max(p[axis] for p in a.points)) for a in annotations)


def test_plain_rect_keeps_its_four_exact_edges(tmp_path):
    path = _pdf(tmp_path, b"0 0 0 RG 1 w 50 50 100 80 re S")
    geom = vg.visible_geometry_for_page(path, 0, dpi=144)
    assert len(geom.annotations) == 4
    assert {tuple(map(tuple, a.points)) for a in geom.annotations} == {
        tuple(map(tuple, a.points)) for a in _all_geometry_for_page(path, 0)}
    assert all(lay == (0,) for lay in geom.layers_of)
    assert geom.labels.dtype == np.uint8 and (geom.labels == vg.PAPER_LABEL).mean() > 0.9


def test_white_fill_next_to_grid_line_is_dropped(tmp_path):
    # white cell fill 0.2 pt inside a black grid rect (fill drawn first)
    path = _pdf(tmp_path, b"1 1 1 rg 100.2 100.2 59.6 49.6 re f 0 0 0 RG 0.5 w 100 100 60 50 re S")
    geom = vg.visible_geometry_for_page(path, 0, dpi=300)
    assert len(geom.annotations) == 4
    assert all(a.color == (0.0, 0.0, 0.0) for a in geom.annotations)
    assert geom.stats["paint_dropped_vectors"] == 1


def test_light_gray_fill_gives_no_layer_and_no_gt(tmp_path):
    lines = b"".join(b"0 0 0 RG 1 w 20 %d m 280 %d l S " % (y, y) for y in range(160, 290, 10))
    path = _pdf(tmp_path, b"0.9 0.9 0.9 rg 40 40 80 60 re f 0 0 0 RG 1 w 30 70 m 200 70 l S " + lines)
    geom = vg.visible_geometry_for_page(path, 0, dpi=300)
    assert geom.n_layers == 1
    assert len(geom.annotations) == 14 and all(a.color == (0.0, 0.0, 0.0) for a in geom.annotations)


def test_clipped_line_keeps_only_its_visible_span(tmp_path):
    # line x 60..290, clip x 60..100 -> ~17 % visible
    path = _pdf(tmp_path, b"q 60 100 40 40 re W n 0 0 1 RG 2 w 60 120 m 290 120 l S Q")
    geom = vg.visible_geometry_for_page(path, 0, dpi=300)
    assert len(geom.annotations) == 1
    (x0, x1), = _spans(geom.annotations)
    assert x0 == pytest.approx(60, abs=0.5) and x1 == pytest.approx(100, abs=0.5)
    assert geom.stats["cut_pieces"] == 1


def test_dashed_line_becomes_one_piece_per_dash(tmp_path):
    # 3 on / 9 off over 120 pt -> 10 dashes, each 3 pt long
    path = _pdf(tmp_path, b"0 0 0 RG 1 w [3 9] 0 d 40 150 m 160 150 l S")
    geom = vg.visible_geometry_for_page(path, 0, dpi=300)
    spans = _spans(geom.annotations)
    assert len(spans) == 10
    for k, (x0, x1) in enumerate(spans):
        assert x0 == pytest.approx(40 + 12 * k, abs=0.6)
        assert x1 == pytest.approx(43 + 12 * k, abs=0.6)


def test_stroke_partly_under_opaque_fill_is_cut(tmp_path):
    path = _pdf(tmp_path, b"0 0 0 RG 1 w 20 150 m 280 150 l S 1 1 1 rg 100 100 100 100 re f")
    geom = vg.visible_geometry_for_page(path, 0, dpi=300)
    spans = _spans(geom.annotations)
    assert len(spans) == 2
    assert spans[0][1] == pytest.approx(100, abs=0.6) and spans[1][0] == pytest.approx(200, abs=0.6)


def test_cut_curve_stays_on_the_original_curve(tmp_path):
    path = _pdf(tmp_path, b"q 0 0 150 300 re W n 0 0 0 RG 1 w 50 150 m 100 250 200 50 250 150 c S Q")
    geom = vg.visible_geometry_for_page(path, 0, dpi=300)
    assert len(geom.annotations) == 1 and geom.annotations[0].kind == "c"
    full = np.array(_all_geometry_for_page(path, 0)[0].points)
    sub = np.array(geom.annotations[0].points)
    assert sub[0] == pytest.approx(full[0], abs=1e-6)
    assert sub[3][0] == pytest.approx(150, abs=0.6)

    def on_curve(p, c, n=4001):
        t = np.linspace(0, 1, n)[:, None]
        pts = (1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3]
        return np.min(np.linalg.norm(pts - p, axis=1))

    for t in (0.25, 0.5, 0.75):
        p = vg._sub_points("c", sub.tolist(), t, t)[0]
        assert on_curve(np.array(p), full) < 0.05


def test_thin_black_line_on_one_layer_and_colored_outline_on_two(tmp_path):
    path = _pdf(tmp_path, b"0 0 0 RG 0.3 w 20 250 m 280 250 l S "
                          b"1 1 0 rg 0 0 0 RG 0.3 w 50 50 120 80 re B")
    geom = vg.visible_geometry_for_page(path, 0, dpi=300)
    assert geom.n_layers == 2
    by_color = {}
    for a, lay in zip(geom.annotations, geom.layers_of):
        by_color.setdefault(a.fill is None, set()).add(lay)
    black = next(k for k in range(2) if tuple(geom.colors_rgb[k]) < (60, 60, 60))
    assert by_color[True] == {(black,)}                      # thin line: black layer only
    assert by_color[False] == {(0, 1)}                       # hairline outline of a yellow box: both


def test_rotated_page_annotations_land_on_ink(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=120)
    page.draw_line((20, 30), (180, 30), color=(0, 0, 0), width=1.5)
    page.set_rotation(90)
    path = tmp_path / "rot.pdf"
    doc.save(path)
    doc.close()
    geom = vg.visible_geometry_for_page(str(path), 0, dpi=144)
    assert len(geom.annotations) == 1
    assert geom.shape == (int(round(200 * 2)), int(round(120 * 2)))


def test_empty_page_and_wrapper(tmp_path):
    empty = _pdf(tmp_path, b"", name="e.pdf")
    assert raster_geometry_for_page(empty, 0) == []
    path = _pdf(tmp_path, b"0 0 0 RG 1 w 50 50 100 80 re S")
    assert len(raster_geometry_for_page(path, 0)) == 4


def test_paint_reads_as_ink():
    assert vg.paint_reads_as_ink((0, 0, 0), 120.0)
    assert not vg.paint_reads_as_ink((1, 1, 1), 120.0)
    assert not vg.paint_reads_as_ink((0.95, 0.95, 0.95), 120.0)
    assert vg.paint_reads_as_ink((1, 1, 0), 120.0)          # pale but chromatic
    assert not vg.paint_reads_as_ink(None, 120.0)


def test_subpixel_feature_rendered_only_as_antialiasing_stays_visible(tmp_path):
    # a 0.15 pt black sliver (a serif foot) renders as light-grey anti-aliasing the split calls
    # paper; its outline must still count as visible (next to a real line so the page has ink)
    path = _pdf(tmp_path, b"0 0 0 RG 1 w 20 250 m 280 250 l S 0 0 0 rg 100 100.25 20 0.15 re f")
    geom = vg.visible_geometry_for_page(path, 0, dpi=150)
    sliver = [a for a in geom.annotations if a.fill is not None]
    assert len(sliver) == 4 and geom.stats["invisible_pieces"] == 0
    # without the rendered-ink rule the whole sliver read as hidden
    assert vg.visible_runs(vg._pieces_px(sliver, geom.matrix), geom.labels, geom.n_layers,
                           vg.PAPER_LABEL)[1]["invisible_pieces"] == 4
