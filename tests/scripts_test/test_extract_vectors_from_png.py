"""Tests for recursive PDF discovery in the batch pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from pprint import pformat
import random
import pymupdf
import pytest

from raster_parser.extract_vectors_from_png import extract_vectors_from_png, get_image_polys
from raster_parser.models import VectorDTO
from extract_vectors_from_png_expected import expected1, expected2, expected3, expected4

def assert_approx(
    actual,
    expected,
    abs_tol: float,
) -> None:
    """Recursively compare values, allowing numeric differences."""

    if isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=abs_tol)
        return

    if isinstance(expected, int):
        assert actual == expected
        return

    if isinstance(expected, str) or expected is None:
        assert actual == expected
        return

    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)

        for actual_item, expected_item in zip(actual, expected):
            assert_approx(
                actual_item,
                expected_item,
                abs_tol,
            )

        return

    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()

        for key in expected:
            assert_approx(
                actual[key],
                expected[key],
                abs_tol,
            )

        return

    assert actual == expected


def assert_vector_approx(
    actual: VectorDTO,
    expected: VectorDTO,
    abs_tol: float = 0.1,
) -> None:
    assert_approx(
        actual.model_dump(),
        expected.model_dump(),
        abs_tol,
    )


def assert_vectors_approx(
    actual: list[VectorDTO],
    expected: list[VectorDTO],
    abs_tol: float = 0.1,
) -> None:
    assert len(actual) == len(expected)

    for actual_vector, expected_vector in zip(actual, expected):
        assert_vector_approx(
            actual_vector,
            expected_vector,
            abs_tol,
        )


def print_expected_vectors(
    vectors: list[VectorDTO],
    keep: int = 3,
) -> None:
    """Print a Python literal containing the first and last `keep` vectors."""

    if len(vectors) <= keep * 2:
        selected = vectors
    else:
        selected = [
            *vectors[:keep],
            *vectors[-keep:],
        ]

    output = "[\n"

    for vector in selected:
        data = vector.model_dump()

        output += "    VectorDTO(\n"

        for field, value in data.items():
            output += f"        {field}={pformat(value, width=100)},\n"

        output += "    ),\n"

    output += "]"

    print(output)



TEST_DIR = Path(__file__).resolve().parent
img_path = TEST_DIR.parent / "scripts_test" / "references" / "10.png"

def test_extract_vectors_from_png(tmp_path: Path) -> None:
    vectors = extract_vectors_from_png(
        img_path,
        tmp_dir=tmp_path,
        dpi=300,
    )

    assert_vectors_approx(vectors, expected1)


def _insert_images(
    page: pymupdf.Page,
    image_path: Path,
    rects: list[pymupdf.Rect],
    rotations: list[float] | None = None,
) -> None:
    if rotations is None:
        rotations = [0.0] * len(rects)

    assert len(rects) == len(rotations)

    for rect, rotation in zip(rects, rotations):
        page.insert_image(
            rect,
            filename=str(image_path),
            rotate=rotation,
        )


def _random_rects(
    page: pymupdf.Page,
    image_path: Path,
    count: int,
    seed: int,
) -> list[pymupdf.Rect]:
    random.seed(seed)

    pix = pymupdf.Pixmap(str(image_path))
    source_width = pix.width
    source_height = pix.height

    rects = []

    for _ in range(count):
        scale = random.uniform(0.25, 2.0)

        width = min(
            source_width * scale,
            page.rect.width,
        )
        height = min(
            source_height * scale,
            page.rect.height,
        )

        x0 = random.uniform(
            0,
            page.rect.width - width,
        )
        y0 = random.uniform(
            0,
            page.rect.height - height,
        )

        rects.append(
            pymupdf.Rect(
                x0,
                y0,
                x0 + width,
                y0 + height,
            )
        )

    return rects


def _assert_rects_match(
    actual: list[pymupdf.Rect],
    expected: list[pymupdf.Rect],
) -> None:
    assert len(actual) == len(expected)

    for actual_rect, expected_rect in zip(actual, expected):
        assert tuple(actual_rect) == pytest.approx(
            tuple(expected_rect),
        )


def _run_get_image_polys(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    tmp_path: Path,
) -> dict:
    return get_image_polys(
        doc,
        page,
        tmp_dir=tmp_path,
        dpi=300,
    )

def test_get_image_polys_scale_and_translation(
    tmp_path: Path,
) -> None:
    doc = pymupdf.open()
    page = doc.new_page(
        width=1000,
        height=1000,
    )

    expected_rects = _random_rects(
        page,
        img_path,
        count=1,
        seed=200,
    )

    _insert_images(
        page,
        img_path,
        expected_rects,
    )

    imgxref = page.get_images(full=True)[0][0]

    actual_rects = page.get_image_rects(imgxref)

    _assert_rects_match(
        actual_rects,
        expected_rects,
    )

    vectors_by_xref = _run_get_image_polys(
        doc,
        page,
        tmp_path,
    )

    assert imgxref in vectors_by_xref

    vectors_by_rect = vectors_by_xref[imgxref]

    assert len(vectors_by_rect) == 1

    rect_key = next(iter(vectors_by_rect))
    vectors = vectors_by_rect[rect_key]

    assert_vectors_approx(vectors, expected2)

    doc.close()

def test_get_image_polys_multiple_images(
    tmp_path: Path,
) -> None:
    doc = pymupdf.open()
    page = doc.new_page(
        width=1000,
        height=1000,
    )

    expected_rects = _random_rects(
        page,
        img_path,
        count=3,
        seed=200,
    )

    _insert_images(
        page,
        img_path,
        expected_rects,
    )

    imgxref = page.get_images(full=True)[0][0]

    actual_rects = page.get_image_rects(imgxref)

    _assert_rects_match(
        actual_rects,
        expected_rects,
    )

    vectors_by_xref = _run_get_image_polys(
        doc,
        page,
        tmp_path,
    )

    assert imgxref in vectors_by_xref

    vectors_by_rect = vectors_by_xref[imgxref]

    assert len(vectors_by_rect) == 3

    ind = 0
    for rect in actual_rects:
        rect_key = tuple(rect)

        assert rect_key in vectors_by_rect
        assert_vectors_approx(vectors_by_rect[rect_key], expected3[ind])
        ind += 1

    doc.close()


def test_get_image_polys_rotation(
    tmp_path: Path,
) -> None:
    """Test vector extraction from a rotated image placement."""

    doc = pymupdf.open()

    page = doc.new_page(
        width=1000,
        height=1000,
    )

    rect = pymupdf.Rect(
        200,
        200,
        700,
        600,
    )

    _insert_images(
        page,
        img_path,
        [rect],
        rotations=[90],
    )

    image_xrefs = page.get_images()
    assert len(image_xrefs) == 1

    imgxref = image_xrefs[0][0]

    placements = page.get_image_rects(
        imgxref,
        transform=True,
    )

    assert len(placements) == 1

    extracted_rect, matrix = placements[0]

    # Rotation should be represented by the placement matrix.
    assert matrix != pymupdf.Matrix()

    vectors_by_xref = get_image_polys(
        doc,
        page,
        tmp_dir=tmp_path,
    )

    assert imgxref in vectors_by_xref

    vectors_by_rect = vectors_by_xref[imgxref]

    assert len(vectors_by_rect) == 1

    vectors = next(iter(vectors_by_rect.values()))

    assert vectors

    assert_vectors_approx(vectors, expected4)

    doc.close()