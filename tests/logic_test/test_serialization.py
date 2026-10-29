"""Serialization tests for extract output DTOs."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf as fitz

from raster_parser.models import Text
from raster_parser.models import TextDTO, NativePDFElements, VectorDTO
from raster_parser.serialisation.serialization import drawing_item_to_dict


def test_drawing_item_to_dict_line() -> None:
    item = ("l", fitz.Point(1, 2), fitz.Point(3, 4))
    d = drawing_item_to_dict(item)
    assert d == {"op": "l", "args": [[1.0, 2.0], [3.0, 4.0]]}
    json.dumps(d)


def test_drawing_item_to_dict_rect() -> None:
    item = ("re", fitz.Rect(0, 0, 10, 20))
    d = drawing_item_to_dict(item)
    assert d == {"op": "re", "args": [[0.0, 0.0, 10.0, 20.0]]}
    json.dumps(d)


def test_drawing_item_to_dict_quad() -> None:
    quad = fitz.Quad(
        fitz.Point(1, 2),
        fitz.Point(3, 4),
        fitz.Point(5, 6),
        fitz.Point(7, 8),
    )
    item = ("qu", quad)
    d = drawing_item_to_dict(item)
    assert d == {
        "op": "qu",
        "args": [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
    }
    json.dumps(d)


def test_text_dto_from_text() -> None:
    from raster_parser.models import GlyphOrientation

    dto = TextDTO.from_text(
        Text(1, 2, 3, 4, "hi", 0, 0, 0, rotate=90, glyph_orientation=GlyphOrientation.VERTICAL)
    )
    payload = json.loads(dto.model_dump_json())
    assert payload["word"] == "hi"
    assert payload["glyph_orientation"] == "vertical"
    assert payload["rotate"] == 90


def test_vector_dto_serializes_items(synthetic_pdf: Path) -> None:
    from raster_parser.parsing.parser import PdfRasterParser

    v = PdfRasterParser(str(synthetic_pdf), page=0).extract_vectors()[0]
    dto = VectorDTO.from_vector(v)
    payload = json.loads(dto.model_dump_json())
    assert payload["type"] == "s"
    assert payload["items"]
    assert payload["items"][0]["op"] == "re"


def test_native_pdf_elements_round_trip() -> None:
    out = NativePDFElements(
      words=[TextDTO(x0=0, y0=0, x1=1, y1=1, word="a", block_no=0, line_no=0, word_no=0)],
      vectors=[
        VectorDTO(
          type="s",
          x0=0,
          y0=0,
          x1=10,
          y1=10,
          items=[{"op": "re", "args": [[0, 0, 10, 10]]}],
        )
      ],
    )
    restored = NativePDFElements.model_validate_json(out.model_dump_json())
    assert restored.words[0].word == "a"
    assert restored.vectors[0].items[0]["op"] == "re"
