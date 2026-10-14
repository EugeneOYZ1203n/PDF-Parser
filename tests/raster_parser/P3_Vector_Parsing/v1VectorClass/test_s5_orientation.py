"""v1VectorClass step 5: OCR word `rotate` from group orientation and page rotation."""

from __future__ import annotations

import pytest

from raster_parser.P3_Vector_Parsing.v1VectorClass.s5_orientation import ocr_rotate_for_target


@pytest.mark.parametrize(
    ("page_rotation", "target", "expected"),
    [
        (0, 0, 0), (0, 90, 90),
        (90, 0, 270), (90, 90, 0),
        (180, 0, 0), (180, 90, 90),
        (270, 0, 90), (270, 90, 180),
    ],
)
def test_ocr_rotate_for_target(page_rotation: int, target: int, expected: int) -> None:
    assert ocr_rotate_for_target(target, page_rotation) == expected
