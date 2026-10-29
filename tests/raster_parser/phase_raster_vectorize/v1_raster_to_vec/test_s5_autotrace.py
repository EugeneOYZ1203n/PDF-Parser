"""v1_raster_to_vec step 5: autotrace end to end -- the real pyautotrace binary, no mocks (copied from
tests/raster_parser/autotrace/test_drawing_autotrace.py).

`drawing_autotrace` once imported the `png_to_svg` / `svg_to_pdf` /
`pdf_to_vectors` *modules* instead of the functions inside them, so every
traced colour layer raised `TypeError: 'module' object is not callable` while
the mocked unit tests stayed green.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from raster_parser.models.vector import Vector
from raster_parser.commons.v1_commons.autotrace import autotrace_drawing
from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s5_colour_autotrace import (
    colour_fill_autotrace,
)

pytest.importorskip("autotrace")


def _write(path: Path, img: np.ndarray) -> Path:
    cv2.imwrite(str(path), img)
    return path


def _black_square_png(tmp_path: Path) -> Path:
    img = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (110, 90), (0, 0, 0), thickness=4)
    return _write(tmp_path / "square.png", img)


def _two_colour_png(tmp_path: Path) -> Path:
    img = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (70, 60), (0, 0, 255), thickness=-1)  # red (BGR)
    cv2.circle(img, (115, 80), 25, (255, 0, 0), thickness=-1)  # blue
    return _write(tmp_path / "two_colour.png", img)


def _has_rect(vectors: list[Vector], expected, tol: float) -> bool:
    """Any traced vector whose bbox is `expected` (autotrace also traces the
    white background as a page-sized shape, so the overall bbox is the page)."""
    return any(all(abs(a - b) <= tol for a, b in zip(v.rect, expected)) for v in vectors)


def test_autotrace_drawing_traces_a_square(tmp_path):
    vectors = autotrace_drawing(_black_square_png(tmp_path), scratch_dir=tmp_path / "scratch")
    assert vectors and all(isinstance(v, Vector) for v in vectors)
    assert _has_rect(vectors, (28, 28, 112, 92), tol=4)  # the drawn square, in pixels


def test_autotrace_drawing_scales_to_points(tmp_path):
    vectors = autotrace_drawing(
        _black_square_png(tmp_path), dpi=144, scratch_dir=tmp_path / "scratch"
    )
    assert _has_rect(vectors, (14, 14, 56, 46), tol=2)  # pixels * 72/144


@pytest.mark.parametrize("workers", [1, 2])
def test_colour_fill_autotrace_traces_each_colour(tmp_path, workers):
    result = colour_fill_autotrace(
        _two_colour_png(tmp_path),
        out_dir=tmp_path / "out",
        dpi=72,
        workers=workers,
        min_group_pixels=200,
    )
    assert len(result.groups) == 2
    assert result.combined_vectors
    fills = {tuple(round(c, 1) for c in v.fill) for v in result.combined_vectors if v.fill}
    assert fills == {(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)}
