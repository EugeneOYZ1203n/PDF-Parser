"""v1_raster_to_vec step 5: colour-fill clustering / exclusive masks, no autotrace binary (copied from
tests/raster_parser/test_colour_fill_autotrace.py)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from raster_parser.models.vector import Vector
from raster_parser.models.raster_vectorize import ColourGroup
from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s5_colour_autotrace import (
    _covers_most_of_page,
    _is_achromatic,
    _is_near_black,
    _is_near_white,
    _is_page_wash,
    _is_white_fill,
    _label_map,
    _stamp_rgb,
    dump_colour_vectors_json,
    group_colours,
    isoperimetric_quotient,
    mask_to_bw_png,
    hsv_embedding,
    unique_colours,
)


def _bgr_two_blobs() -> np.ndarray:
    """White page with a red blob and a blue blob (BGR)."""
    img = np.full((48, 64, 3), 255, dtype=np.uint8)
    img[4:20, 4:28] = (0, 0, 255)  # red
    img[24:44, 32:60] = (255, 0, 0)  # blue
    return img


def _vector(
    *,
    rect: tuple[float, float, float, float],
    items: list | None = None,
    fill: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    seqno: int = 0,
) -> Vector:
    return Vector(
        type="f",
        items=items if items is not None else [],
        color=None,
        fill=fill,
        width=None,
        dashes=None,
        closePath=True,
        lineCap=0,
        lineJoin=0.0,
        even_odd=False,
        stroke_opacity=None,
        fill_opacity=1.0,
        layer=None,
        rect=rect,
        scissor=None,
        seqno=seqno,
        blendmode=None,
        isolated=False,
        knockout=False,
        opacity=None,
        page_index=0,
    )


def test_hsv_embedding_shapes() -> None:
    single = hsv_embedding(np.array([255, 0, 0], dtype=np.uint8))
    assert single.shape == (3,)
    many = hsv_embedding(np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8))
    assert many.shape == (2, 3)
    slab = hsv_embedding(np.zeros((4, 5, 3), dtype=np.uint8))
    assert slab.shape == (4, 5, 3)


def test_achromatic_black_white_helpers() -> None:
    assert _is_achromatic((10, 10, 12))
    assert not _is_achromatic((255, 0, 0))
    assert _is_near_black((5, 5, 5))
    assert not _is_near_black((200, 200, 200))
    assert _is_near_white((240, 240, 240))
    assert not _is_near_white((100, 100, 100))


def test_unique_colours_skips_background() -> None:
    bgr = _bgr_two_blobs()
    rgb_u, counts = unique_colours(bgr, bg_thresh=250)
    assert len(rgb_u) == 2
    assert counts.sum() == (16 * 24) + (20 * 28)
    packed = {(int(r), int(g), int(b)) for r, g, b in rgb_u}
    assert (255, 0, 0) in packed
    assert (0, 0, 255) in packed


def test_group_colours_clusters_near_reds() -> None:
    unique_rgb = np.array(
        [
            [255, 0, 0],
            [250, 5, 5],
            [0, 0, 255],
            [0, 0, 0],
            [245, 245, 245],
        ],
        dtype=np.uint8,
    )
    counts = np.array([8000, 2000, 6000, 3000, 1000], dtype=np.int64)
    groups = group_colours(unique_rgb, counts, dist_thresh=150.0, min_pixels=1)
    kinds = {g.kind for g in groups}
    assert "black" in kinds
    assert "white" in kinds
    chromatic = [g for g in groups if g.kind is None]
    assert len(chromatic) == 2  # red cluster + blue


def test_stamp_rgb_reserved_kinds() -> None:
    assert _stamp_rgb(ColourGroup(0, (12, 12, 12), 10, "black")) == (0, 0, 0)
    assert _stamp_rgb(ColourGroup(1, (250, 250, 250), 10, "white")) == (255, 255, 255)
    assert _stamp_rgb(ColourGroup(2, (10, 20, 30), 10, None)) == (10, 20, 30)


def test_label_map_and_exclusive_masks_partition_ink() -> None:
    bgr = _bgr_two_blobs()
    rgb_u, counts = unique_colours(bgr, bg_thresh=250)
    groups = group_colours(rgb_u, counts, dist_thresh=150.0, min_pixels=1)
    assert len(groups) == 2

    labels = _label_map(bgr, groups, dist_thresh=150.0, bg_thresh=250, chunk_rows=8)
    assert labels.shape == bgr.shape[:2]
    assert (labels == -1).sum() > 0  # background
    for i, g in enumerate(groups):
        assert (labels == i).sum() > 0

    # One mask per group, as colour_fill_autotrace builds them from the label map.
    masks = {g.label: labels == i for i, g in enumerate(groups)}
    # Exclusive: no pixel in two masks.
    stacked = np.stack([masks[g.label].astype(np.int32) for g in groups], axis=0)
    assert int(stacked.sum(axis=0).max()) <= 1
    # Every ink pixel assigned.
    ink = ~((bgr[:, :, 0] >= 250) & (bgr[:, :, 1] >= 250) & (bgr[:, :, 2] >= 250))
    assert bool(np.all(stacked.sum(axis=0)[ink] == 1))


def test_mask_to_bw_png(tmp_path: Path) -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:8] = True
    out = mask_to_bw_png(mask, tmp_path / "m.png")
    assert out.is_file()
    loaded = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert loaded is not None
    assert loaded[3, 4] == 0  # ink black
    assert loaded[0, 0] == 255


def test_dump_colour_vectors_json(tmp_path: Path) -> None:
    v = _vector(rect=(1, 2, 3, 4), fill=(0.1, 0.2, 0.3), seqno=7)
    path = dump_colour_vectors_json(
        [v], tmp_path / "vecs.json", page_width=100.0, page_height=200.0, source="x.png"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["page_width"] == 100.0
    assert payload["page_height"] == 200.0
    assert payload["source"] == "x.png"
    assert len(payload["vectors"]) == 1
    assert payload["vectors"][0]["seqno"] == 7


def test_page_wash_helpers() -> None:
    page = _vector(rect=(0, 0, 100, 100), fill=(0.0, 0.0, 0.0), items=[("re", (0, 0, 100, 100))])
    assert _covers_most_of_page(page, 100.0, 100.0, cover=0.95)
    assert _is_white_fill(_vector(rect=(0, 0, 10, 10), fill=(1.0, 1.0, 1.0)))
    assert not _is_white_fill(_vector(rect=(0, 0, 10, 10), fill=(0.0, 0.0, 0.0)))

    white_wash = _vector(
        rect=(0, 0, 100, 100),
        fill=(1.0, 1.0, 1.0),
        items=[("re", (0, 0, 100, 100))],
    )
    assert _is_page_wash(white_wash, 100.0, 100.0, stamp_rgb=(0, 0, 0))

    small = _vector(rect=(0, 0, 10, 10), fill=(0.0, 0.0, 0.0), items=[("re", (0, 0, 10, 10))])
    assert not _is_page_wash(small, 100.0, 100.0, stamp_rgb=(0, 0, 0))


def test_isoperimetric_quotient_of_a_quad_is_its_rectangles() -> None:
    """A "qu" item is a closed box: Q = 4*pi*A/P^2 = pi/4 for a square. Walking the corners in
    pymupdf.Quad order (ul, ur, ll, lr) instead draws a self-crossing bow-tie with zero area."""
    quad = (
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0, 10.0),
    )  # ul ur lr ll, as Vector.items store it
    v = _vector(rect=(0, 0, 10, 10), items=[("qu", quad)])
    assert isoperimetric_quotient(v) == pytest.approx(math.pi / 4)


# --- artefacts are written only when asked for ---


def _rectangle_png(path: Path) -> Path:
    img = np.full((120, 160, 3), 255, np.uint8)
    cv2.rectangle(img, (30, 30), (110, 90), (0, 0, 0), thickness=4)
    cv2.imwrite(str(path), img)
    return path


def test_no_files_are_left_without_keep_intermediates(tmp_path: Path) -> None:
    from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s5_colour_autotrace import (
        colour_fill_autotrace,
    )

    out = tmp_path / "out"
    result = colour_fill_autotrace(
        _rectangle_png(tmp_path / "d.png"), out_dir=out, stem="d", keep_intermediates=False
    )
    assert result.combined_vectors
    assert sorted(p.name for p in out.rglob("*") if p.is_file()) == []


def test_keep_intermediates_still_writes_the_qa_files(tmp_path: Path) -> None:
    from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s5_colour_autotrace import (
        colour_fill_autotrace,
    )

    out = tmp_path / "out"
    colour_fill_autotrace(
        _rectangle_png(tmp_path / "d.png"), out_dir=out, stem="d", keep_intermediates=True
    )
    names = {p.name for p in out.rglob("*") if p.is_file()}
    assert {
        "d_colour_palette.png",
        "d_colour_centroid_map.png",
        "d_colour_combined.pdf",
        "d_colour_summary.txt",
        "d_colour_vectors.json",
        "d_colour_vectors.txt",
    } <= names
