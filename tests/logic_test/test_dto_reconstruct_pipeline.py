"""Tests for DTO reconstruct pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
from raster_parser.rendering.pdf_render.dto_reconstruct_pipeline import (
    draw_vectors,
    output_as_dto_json,
    reconstruct_from_dtos,
    reconstruct_from_output,
)
from raster_parser.main_pipeline_extract import extract
from raster_parser.serialisation.deserialization import (
    deserialize_output,
    deserialize_vector,
    drawing_item_from_dict,
)
from raster_parser.models import TextDTO, NativePDFElements, VectorDTO
from raster_parser.serialisation.serialization import drawing_item_to_dict


def test_drawing_item_round_trip_point_and_rect() -> None:
    line = ("l", pymupdf.Point(1, 2), pymupdf.Point(3, 4))
    restored = drawing_item_from_dict(drawing_item_to_dict(line))
    assert restored[0] == "l"
    assert tuple(restored[1]) == (1.0, 2.0)
    assert tuple(restored[2]) == (3.0, 4.0)

    rect_item = ("re", pymupdf.Rect(0, 0, 10, 20))
    restored_rect = drawing_item_from_dict(drawing_item_to_dict(rect_item))
    assert restored_rect[0] == "re"
    assert tuple(restored_rect[1]) == (0.0, 0.0, 10.0, 20.0)


def test_drawing_item_round_trip_quad() -> None:
    quad = pymupdf.Quad(
        pymupdf.Point(1, 2),
        pymupdf.Point(3, 4),
        pymupdf.Point(5, 6),
        pymupdf.Point(7, 8),
    )
    restored = drawing_item_from_dict(drawing_item_to_dict(("qu", quad)))
    assert restored[0] == "qu"
    assert tuple(restored[1].ul) == (1.0, 2.0)
    assert tuple(restored[1].lr) == (7.0, 8.0)


def test_drawing_item_from_legacy_quad_string() -> None:
    quad_str = (
        "Quad(Point(1152.8399658203125, 825.56005859375), "
        "Point(1152.8399658203125, 825.6800537109375), "
        "Point(1152.9599609375, 825.56005859375), "
        "Point(1152.9599609375, 825.6800537109375))"
    )
    restored = drawing_item_from_dict({"op": "qu", "args": [quad_str]})
    assert restored[0] == "qu"
    assert tuple(restored[1].ul) == (1152.8399658203125, 825.56005859375)


def test_vector_from_dto_round_trip(synthetic_pdf: Path) -> None:
    from raster_parser.parsing.parser import PdfRasterParser

    source = PdfRasterParser(str(synthetic_pdf), page=0).extract_vectors()[0]
    dto = VectorDTO.from_vector(source)
    restored = deserialize_vector(dto)
    assert restored.type == source.type
    assert restored.rect == source.rect
    assert len(restored.items) == len(source.items)


def test_deserialize_output(tmp_path: Path, synthetic_pdf: Path) -> None:
    output = extract(
        synthetic_pdf, page_index=0, ocr_dpi=150, use_angle_cls=False, enable_raster_pass=False
    )
    words, vectors = deserialize_output(output)
    assert len(words) == len(output.words)
    assert len(vectors) == len(output.vectors)
    assert words[0].word == output.words[0].word


def test_output_as_dto_json_round_trip(tmp_path: Path, synthetic_pdf: Path) -> None:
    output = extract(
        synthetic_pdf, page_index=0, ocr_dpi=150, use_angle_cls=False, enable_raster_pass=False
    )
    out_json = output_as_dto_json(output, tmp_path / "output.json")
    assert out_json.is_file()
    restored = NativePDFElements.model_validate_json(out_json.read_text(encoding="utf-8"))
    assert len(restored.words) == len(output.words)
    assert len(restored.vectors) == len(output.vectors)
    assert restored.words[0].word == output.words[0].word


def test_reconstruct_from_extract_output(tmp_path: Path, synthetic_pdf: Path) -> None:
    output = extract(
        synthetic_pdf, page_index=0, ocr_dpi=150, use_angle_cls=False, enable_raster_pass=False
    )
    result = reconstruct_from_output(
        output,
        out_txt_words=tmp_path / "words.txt",
        out_txt_vectors=tmp_path / "vectors.txt",
    )

    assert result.out_txt_words.is_file()
    assert result.out_txt_vectors.is_file()
    assert "hello" in result.out_txt_words.read_text(encoding="utf-8")
    assert "Vector 0" in result.out_txt_vectors.read_text(encoding="utf-8")
    assert len(result.words) == len(output.words)
    assert len(result.vectors) == len(output.vectors)


def test_reconstruct_from_json_dtos(tmp_path: Path, synthetic_pdf: Path) -> None:
    output = extract(
        synthetic_pdf, page_index=0, ocr_dpi=150, use_angle_cls=False, enable_raster_pass=False
    )
    payload = json.loads(output.model_dump_json())
    words = [TextDTO.model_validate(w) for w in payload["words"]]
    vectors = [VectorDTO.model_validate(v) for v in payload["vectors"]]

    result = reconstruct_from_dtos(
        words,
        vectors,
        out_txt_words=tmp_path / "from_json_words.txt",
        out_txt_vectors=tmp_path / "from_json_vectors.txt",
    )
    assert result.out_txt_words.is_file()
    assert result.out_txt_vectors.is_file()
    assert len(result.words) == len(words)


def test_draw_vectors_writes_pdf(tmp_path: Path, synthetic_pdf: Path) -> None:
    output = extract(
        synthetic_pdf, page_index=0, ocr_dpi=150, use_angle_cls=False, enable_raster_pass=False
    )
    result = reconstruct_from_output(
        output,
        out_txt_words=tmp_path / "words.txt",
        out_txt_vectors=tmp_path / "vectors.txt",
    )
    out_pdf = draw_vectors(result.vectors, tmp_path / "vectors.pdf")
    assert out_pdf.is_file()
    doc = pymupdf.open(out_pdf)
    assert len(doc) == 1
    assert len(doc[0].get_drawings()) >= 1
    doc.close()
