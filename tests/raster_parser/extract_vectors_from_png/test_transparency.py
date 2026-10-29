"""Fill opacity of traced image vectors, from the image's alpha channel
(`extract_vectors_from_png._extract_image_vectors_worker`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from raster_parser.extract_vectors_from_png import _extract_image_vectors_worker


def _half_transparent_square(path: Path) -> None:
    """White, fully opaque 100 x 100 px image with a black 50 x 50 square at alpha 128."""
    arr = np.zeros((100, 100, 4), np.uint8)
    arr[..., :3] = 255
    arr[..., 3] = 255
    arr[25:75, 25:75] = (0, 0, 0, 128)
    PILImage.fromarray(arr, mode="RGBA").save(path)


def _dark_fills(vectors) -> list:
    return [v for v in vectors if v.fill is not None and max(v.fill) < 0.5]


@pytest.mark.parametrize("dpi", [300, 150, 96])
def test_opacity_is_measured_at_the_dpi_the_vectors_were_traced_at(
    tmp_path: Path, dpi: int
) -> None:
    png = tmp_path / "img.png"
    _half_transparent_square(png)
    _, vectors, _ = _extract_image_vectors_worker((1, png, tmp_path / "work", dpi))
    dark = _dark_fills(vectors)
    assert dark
    assert all(v.fill_opacity == pytest.approx(128 / 255, abs=0.02) for v in dark)


def test_worker_without_a_work_dir_still_computes_opacity(tmp_path: Path) -> None:
    png = tmp_path / "img.png"
    _half_transparent_square(png)
    _, vectors, _ = _extract_image_vectors_worker((1, png, None, 300))
    dark = _dark_fills(vectors)
    assert dark and all(v.fill_opacity == pytest.approx(128 / 255, abs=0.02) for v in dark)
