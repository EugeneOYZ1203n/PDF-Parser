"""`commons/renderer/draw.py` -- the one spec -> draw layer.

Text is checked against the rendered result, not the spec: ink is
rasterized and projected onto the text's own direction / normal, so a
sheared, mis-rotated, mis-sized or mis-placed string fails no matter how
the spec was computed.
"""
from __future__ import annotations

import math

import numpy as np
import pymupdf as fitz
import pytest

from rastervec.commons.models import PageMeta
from rastervec.commons.renderer.draw import (
    DotSpec,
    ImageSpec,
    PathSpec,
    TextSpec,
    arrow_spec,
    bbox_spec,
    dot_spec,
    image_spec,
    oriented_box,
    polyline_spec,
    quad_spec,
    render_specs_pdf,
    text_spec,
    vector_spec,
)

ANGLES = [0, 15, 30, 60, 90, 135, 180, 225, 270, 330]
_FONT = fitz.Font("helv")
_SPAN = _FONT.ascender - _FONT.descender


def _meta(width=400.0, height=400.0, rotation=0) -> PageMeta:
    return PageMeta(
        index=0, number=1, mediabox=(0, 0, width, height), rotation=rotation,
        width=width, height=height,
    )


def _rect_quad(cx, cy, w, h, angle):
    r = math.radians(angle)
    d = (math.cos(r), math.sin(r))
    n = (-d[1], d[0])
    return tuple(
        (cx + a * d[0] + b * n[0], cy + a * d[1] + b * n[1])
        for a, b in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))
    )


def _envelope(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def _ink_points(pdf_bytes: bytes, zoom: float = 4.0) -> np.ndarray:
    """Unrotated-page-space centres of every dark pixel (pages here are
    unrotated, so display space == page space)."""
    doc = fitz.open("pdf", pdf_bytes)
    try:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    finally:
        doc.close()
    ys, xs = np.nonzero(a < 128)
    return np.stack([(xs + 0.5) / zoom, (ys + 0.5) / zoom], axis=1)


def _extents(points: np.ndarray, angle: float):
    """(along length, normal height, centre) of `points` in the text frame."""
    r = math.radians(angle)
    d = np.array([math.cos(r), math.sin(r)])
    n = np.array([-d[1], d[0]])
    al, nl = points @ d, points @ n
    centre = d * (al.min() + al.max()) / 2 + n * (nl.min() + nl.max()) / 2
    return al.max() - al.min(), nl.max() - nl.min(), tuple(centre)


class _T:
    """Minimal duck-typed Text: what `text_spec` reads."""

    def __init__(self, text, bbox, angle, quad=None, color=None):
        self.text, self.bbox, self._angle, self.quad_points, self.color = text, bbox, angle, quad, color

    def angle(self):
        return self._angle


# ---------------------------------------------------------------------------
# text_spec
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("angle", ANGLES)
def test_text_spec_matrix_is_rotation_times_scale_never_shear(angle):
    spec = text_spec(("ABC", _envelope(_rect_quad(200, 200, 120, 20, angle)), angle))
    a, b, c, d, _e, _f = spec.morph_matrix
    # rows of a scale-then-rotate matrix are orthogonal (no shear);
    # fitz.Matrix holds float32, hence the tolerance
    assert a * c + b * d == pytest.approx(0.0, abs=1e-6 * math.hypot(a, b))


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("text", ["ABC", "AB CD EFG"])
def test_rendered_text_is_rotated_and_scaled_only(angle, text):
    # Same text, same oriented box, every angle: the ink, measured in the
    # text's own frame, must be identical -- any shear, mis-rotation, or
    # axis-aligned sizing would change it with the angle.
    def measure(a):
        t = _T(text, (0, 0, 0, 0), a, quad=_rect_quad(200, 200, 120, 20, a))
        return _extents(_ink_points(render_specs_pdf(_meta(), [text_spec(t)])), a)

    along0, normal0, _ = measure(0)
    along, normal, centre = measure(angle)
    assert along == pytest.approx(along0, abs=1.0)
    assert normal == pytest.approx(normal0, abs=1.0)
    assert along == pytest.approx(120, abs=5.0)  # fills the box length
    assert normal <= 20 + 0.5  # inside the box height
    # centred on the box (ink of caps sits above the em-centre, so allow
    # half the gap between cap-height and the full span along the normal)
    assert math.dist(centre, (200, 200)) < 3.0


@pytest.mark.parametrize("angle", ANGLES)
def test_rendered_direction_matches_angle(angle):
    spec = text_spec(("Xy", _envelope(_rect_quad(200, 200, 60, 20, angle)), angle))
    doc = fitz.open("pdf", render_specs_pdf(_meta(), [spec]))
    try:
        [trace] = doc[0].get_texttrace()
        got = math.degrees(math.atan2(trace["dir"][1], trace["dir"][0]))
    finally:
        doc.close()
    assert (got - angle + 180) % 360 - 180 == pytest.approx(0.0, abs=0.5)


def test_font_size_is_box_height_over_glyph_span():
    spec = text_spec(_T("HELLO", (0, 0, 0, 0), 30.0, quad=_rect_quad(100, 100, 80, 24, 30.0)))
    assert spec.fontsize == pytest.approx(24 / _SPAN)


def test_text_spec_point_sits_on_box_centre_after_morph():
    # insert_text maps `point` through FLIP*M*FLIP about the fixpoint; the
    # unmorphed glyph box is laid out centred on the fixpoint.
    spec = text_spec(("ABC", (100, 100, 220, 120), 0.0))
    assert spec.morph_fixpoint == pytest.approx((160, 110))
    natural = _FONT.text_length("ABC", fontsize=spec.fontsize)
    assert spec.point[0] == pytest.approx(160 - natural / 2)


@pytest.mark.parametrize("angle", [0, 10, 30, 60, 80, 90, 120, 170, 200, 300])
def test_oriented_box_inverts_the_bbox_envelope(angle):
    bbox = _envelope(_rect_quad(50, 60, 100, 12, angle))
    cx, cy, w, h = oriented_box(bbox, angle)
    assert (cx, cy) == pytest.approx((50, 60))
    assert (w, h) == pytest.approx((100, 12), abs=1e-6)


@pytest.mark.parametrize("angle", [45, 135, 44, 226])
def test_oriented_box_near_45_degrees_falls_back_to_font_aspect(angle):
    # The envelope is ill-conditioned here: assume the text's natural
    # Helvetica aspect, still positive and inside the envelope.
    bbox = _envelope(_rect_quad(50, 60, 100, 12, angle))
    _cx, _cy, w, h = oriented_box(bbox, angle, text="ABCDEFGH")
    k = _FONT.text_length("ABCDEFGH", fontsize=1.0) / _SPAN
    assert w > 0 and h > 0
    assert w / h == pytest.approx(k)


def test_quad_takes_precedence_over_bbox():
    quad = _rect_quad(100, 100, 80, 10, 45.0)  # unrecoverable from its envelope
    cx, cy, w, h = oriented_box((0, 0, 1, 1), 45.0, quad=quad)
    assert (cx, cy, w, h) == pytest.approx((100, 100, 80, 10))


def test_text_spec_skips_blank_and_degenerate():
    assert text_spec(("   ", (0, 0, 10, 10), 0.0)) is None
    assert text_spec(("A", (5, 5, 5, 5), 0.0)) is None


def test_text_spec_colour_from_text_or_override():
    t = _T("A", (0, 0, 10, 10), 0.0, color=0xFF0000)
    assert text_spec(t).color == (1.0, 0.0, 0.0)
    assert text_spec(t, color=(0.0, 1.0, 0.0)).color == (0.0, 1.0, 0.0)
    assert text_spec(("A", (0, 0, 10, 10), 0.0, (0.0, 0.0, 1.0))).color == (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# paths / dots
# ---------------------------------------------------------------------------
def _drawing_points(pdf_bytes: bytes) -> set:
    doc = fitz.open("pdf", pdf_bytes)
    try:
        pts = set()
        for drawing in doc[0].get_drawings():
            for item in drawing["items"]:
                if item[0] == "qu":
                    corners = [item[1].ul, item[1].ur, item[1].lr, item[1].ll]
                elif item[0] == "re":
                    r = item[1]
                    corners = [r.tl, r.tr, r.br, r.bl]
                else:
                    corners = item[1:]
                pts |= {(round(p.x, 2), round(p.y, 2)) for p in corners}
        return pts
    finally:
        doc.close()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_paths_land_at_their_unrotated_page_points(rotation):
    # get_drawings reports unrotated page space, whatever the /Rotate.
    meta = _meta(300, 200, rotation)
    quad = ((50.0, 20.0), (90.0, 40.0), (80.0, 60.0), (40.0, 40.0))
    assert _drawing_points(render_specs_pdf(meta, [quad_spec(quad, (0, 0, 1))])) == set(quad)
    assert _drawing_points(render_specs_pdf(meta, [bbox_spec((10, 20, 30, 50), (1, 0, 0))])) == {
        (10, 20), (30, 20), (30, 50), (10, 50),
    }
    line = ((5.0, 5.0), (25.0, 15.0), (45.0, 5.0))
    assert _drawing_points(render_specs_pdf(meta, [polyline_spec(line, (0, 0, 0))])) == set(line)


@pytest.mark.parametrize("angle", [0, 37, 90, 200])
def test_arrow_spec_points_along_angle(angle):
    shaft, head = arrow_spec((100, 100), angle, 40, (1, 0, 0))
    (tx, ty), (px, py) = shaft.points
    got = math.degrees(math.atan2(py - ty, px - tx))
    assert (got - angle + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-6)
    assert ((tx + px) / 2, (ty + py) / 2) == pytest.approx((100, 100))
    assert head.points[1] == pytest.approx((px, py))  # head meets the tip
    assert not shaft.closed and not head.closed


def test_dot_spec_draws_a_filled_circle():
    doc = fitz.open("pdf", render_specs_pdf(_meta(), [dot_spec((50, 60), (1, 0, 0), radius=2)]))
    try:
        [drawing] = doc[0].get_drawings()
        assert drawing["fill"] == pytest.approx((1, 0, 0))
        assert fitz.Rect(drawing["rect"]) == fitz.Rect(48, 58, 52, 62)
    finally:
        doc.close()


def test_spec_types():
    assert isinstance(bbox_spec((0, 0, 1, 1), (0, 0, 0)), PathSpec)
    assert isinstance(dot_spec((0, 0), (0, 0, 0)), DotSpec)
    assert isinstance(text_spec(("A", (0, 0, 10, 10), 0.0)), TextSpec)
    assert isinstance(image_spec(np.zeros((2, 2), np.uint8), (0, 0, 1, 1)), ImageSpec)


# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------
def test_vector_spec_real_paint_replays_unchanged(vector):
    v = vector(kind="re", bbox=(10, 10, 50, 30), fill=(0, 1, 0), color=None)
    doc = fitz.open("pdf", render_specs_pdf(_meta(), [vector_spec(v)]))
    try:
        [drawing] = doc[0].get_drawings()
        assert drawing["fill"] == pytest.approx((0, 1, 0))
        assert fitz.Rect(drawing["rect"]) == fitz.Rect(10, 10, 50, 30)
    finally:
        doc.close()


def test_vector_spec_recolor_strokes_in_that_colour(vector):
    v = vector(kind="re", bbox=(10, 10, 50, 30), fill=(0, 1, 0), color=None)
    spec = vector_spec(v, recolor=(1, 0, 0))
    assert spec.vector.fill is None and spec.vector.color == (1, 0, 0)
    assert v.fill == (0, 1, 0)  # the original is untouched


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rot", [0, 90, 180, 270])
def test_image_spec_orients_embedded_placements(rot):
    # A red top-left block in an embedded image placed at `rot`: drawn from
    # its own pixel frame + get_image_info transform, it must land where
    # PyMuPDF itself renders the original placement.
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 10), 0)
    pix.clear_with(255)
    for x in range(4):
        for y in range(4):
            pix.set_pixel(x, y, (255, 0, 0))
    src = fitz.open()
    page = src.new_page(width=300, height=300)
    page.insert_image(fitz.Rect(100, 100, 200, 150), pixmap=pix, rotate=rot, keep_proportion=False)
    info = page.get_image_info()[0]

    arr = np.frombuffer(pix.samples, np.uint8).reshape(10, 20, 3)
    out = render_specs_pdf(_meta(300, 300), [image_spec(arr, info["bbox"], transform=info["transform"])])

    def red_centroid(p):
        pm = p.get_pixmap(alpha=False)
        a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, 3)
        ys, xs = np.nonzero((a[:, :, 0] > 200) & (a[:, :, 1] < 80))
        return xs.mean(), ys.mean()

    doc = fitz.open("pdf", out)
    try:
        assert red_centroid(doc[0]) == pytest.approx(red_centroid(page), abs=1.0)
    finally:
        doc.close()
        src.close()


# ---------------------------------------------------------------------------
# render_specs_pdf
# ---------------------------------------------------------------------------
def test_render_specs_pdf_sets_rotation_and_size():
    doc = fitz.open("pdf", render_specs_pdf(_meta(300, 200, 90), []))
    try:
        assert doc[0].rotation == 90
        assert (doc[0].mediabox.width, doc[0].mediabox.height) == (300, 200)
    finally:
        doc.close()


def test_render_specs_pdf_one_stream_for_marks_one_for_text():
    specs = [bbox_spec((i, i, i + 5, i + 5), (1, 0, 0)) for i in range(20)]
    specs += [dot_spec((50, 50), (0, 0, 1))]
    specs += [text_spec((f"W{i}", (10, 100 + 8 * i, 60, 107 + 8 * i), 0.0)) for i in range(5)]
    doc = fitz.open("pdf", render_specs_pdf(_meta(), specs))
    try:
        streams = [x for x in doc[0].get_contents() if (doc.xref_stream(x) or b"").strip()]
        assert len(streams) == 2
    finally:
        doc.close()


def test_render_specs_pdf_ignores_none_specs():
    assert render_specs_pdf(_meta(), [None, text_spec(("  ", (0, 0, 1, 1), 0.0))])[:4] == b"%PDF"
