"""v1_raster_to_vec debug layers (`RasterDebugLayers`): off without callbacks, rasters and boxes mapped
to page space through the render's `to_points`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest

from raster_parser.models.page import PageMeta
from raster_parser.models.settings import RunSettings
from raster_parser.phase_raster_vectorize.v1_raster_to_vec.debug_layers import (
    BOX_FOLDER,
    RasterDebugLayers,
)
from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s2_render_page import page_to_bgr


def _meta(width: float, height: float, rotation: int) -> PageMeta:
    return PageMeta(
        index=0,
        number=1,
        mediabox=(0, 0, width, height),
        rotation=rotation,
        width=width,
        height=height,
    )


def _pdf(tmp_path: Path, rotation: int) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    page.draw_rect(pymupdf.Rect(20, 20, 60, 40), color=(0, 0, 0), fill=(0, 0, 0))
    page.set_rotation(rotation)
    path = tmp_path / f"r{rotation}.pdf"
    doc.save(path)
    doc.close()
    return path


def test_no_callbacks_means_no_debug() -> None:
    assert RasterDebugLayers.for_settings(RunSettings(), _meta(10, 10, 0)) is None


@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_layers_land_on_the_unrotated_page(tmp_path: Path, rotation: int) -> None:
    pdf = _pdf(tmp_path, rotation)
    bgr, to_points = page_to_bgr(pdf, 0, 72)
    layers: list[tuple] = []
    images: list[tuple] = []
    debug = RasterDebugLayers.for_settings(
        RunSettings(
            on_debug_layer=lambda *a: layers.append(a),
            on_debug_image=lambda f, n, make: images.append((f, n, make())),
        ),
        _meta(200, 100, rotation),
    )
    gray = np.full(bgr.shape[:2], 255, np.uint8)
    debug.rendered(bgr, to_points)
    # One box around the black rectangle, in render pixels.
    ys, xs = np.nonzero(bgr[:, :, 0] < 128)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    debug.split(gray, bgr, [box], [box])

    labels = [(stage, label) for stage, label, _h, _b in layers]
    assert labels == [
        ("render", "residual page"),
        ("split", "text layer"),
        ("split", "drawing layer"),
        ("detect", "initial boxes"),
        ("detect", "refined boxes"),
    ]
    with pymupdf.open("pdf", layers[-1][3]) as doc:
        page = doc[0]
        assert page.rotation == rotation
        drawn = page.get_drawings()[0]["rect"]
    # The box maps back onto the rectangle's unrotated page coordinates.
    assert drawn.x0 == pytest.approx(20, abs=1.5) and drawn.y0 == pytest.approx(20, abs=1.5)
    assert drawn.x1 == pytest.approx(60, abs=1.5) and drawn.y1 == pytest.approx(40, abs=1.5)
    with pymupdf.open("pdf", layers[0][3]) as doc:
        info = doc[0].get_image_info()[0]
        assert info["bbox"][2] == pytest.approx(200, abs=0.5)
        assert info["bbox"][3] == pytest.approx(100, abs=0.5)
    assert [(f, n) for f, n, _i in images] == [(BOX_FOLDER, "box0000.png")]


def test_output_layers(tmp_path: Path, text, vector) -> None:
    layers: list[tuple] = []
    debug = RasterDebugLayers.for_settings(
        RunSettings(on_debug_layer=lambda *a: layers.append(a)), _meta(200, 100, 0)
    )
    debug.output([text(text="hi", bbox=(10, 10, 30, 20))], [vector()])
    assert [(s, l) for s, l, _h, _b in layers] == [("ocr", "text"), ("trace", "vectors")]


def _image_page_pdf(tmp_path: Path) -> Path:
    """A page whose only content is an embedded image of a black frame + a bar (LibreOffice keeps
    images, so the raster pass has something to trace)."""
    img = np.full((200, 400, 3), 255, np.uint8)
    img[20:180, 20:30] = 0
    img[20:180, 370:380] = 0
    img[90:110, 60:340] = 0
    pix = pymupdf.Pixmap(pymupdf.csRGB, 400, 200, img.tobytes(), 0)
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_image(page.rect, pixmap=pix)
    path = tmp_path / "image_page.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.mark.slow
def test_adapter_output_is_the_same_with_callbacks(tmp_path: Path) -> None:
    from raster_parser.core.pipeline import run_pipeline

    pdf = _image_page_pdf(tmp_path)
    layers: list[tuple] = []
    plain = run_pipeline(
        pdf,
        0,
        p2="v1_raster_to_vec",
        p3="stub",
        settings=RunSettings(ocr_dpi=150, dump_dir=tmp_path / "a"),
    )
    debugged = run_pipeline(
        pdf,
        0,
        p2="v1_raster_to_vec",
        p3="stub",
        settings=RunSettings(
            ocr_dpi=150,
            dump_dir=tmp_path / "b",
            on_debug_layer=lambda *a: layers.append(a),
            on_debug_image=lambda f, n, make: make(),
        ),
    )
    assert (plain.texts, plain.vectors) == (debugged.texts, debugged.vectors)
    assert plain.vectors  # the bars were traced
    stages = {stage for stage, _l, _h, _b in layers}
    assert stages == {"render", "split", "detect", "ocr", "trace"}
