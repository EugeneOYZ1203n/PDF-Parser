"""Deserialization tests for DTO ? native types."""

from __future__ import annotations

import json
from pathlib import Path

from raster_parser.main_pipeline_extract import extract
from raster_parser.serialisation.deserialization import (
    deserialize,
    deserialize_text,
    deserialize_vector,
)
from raster_parser.models import TextDTO, NativePDFElements, VectorDTO


def test_deserialize_text_from_json(synthetic_pdf: Path) -> None:
    output = extract(
        synthetic_pdf, page_index=0, ocr_dpi=150, use_angle_cls=False, enable_raster_pass=False
    )
    payload = json.loads(output.words[0].model_dump_json())
    text = deserialize_text(TextDTO.model_validate(payload))
    assert text.word == output.words[0].word
    assert text.x0 == output.words[0].x0


def test_deserialize_vector_from_json(synthetic_pdf: Path) -> None:
    from raster_parser.parsing.parser import PdfRasterParser

    source = PdfRasterParser(str(synthetic_pdf), page=0).extract_vectors()[0]
    dto = VectorDTO.from_vector(source)
    restored = deserialize_vector(VectorDTO.model_validate_json(dto.model_dump_json()))
    assert restored.type == source.type
    assert restored.rect == source.rect


def test_deserialize_lists(synthetic_pdf: Path) -> None:
    output = extract(
        synthetic_pdf, page_index=0, ocr_dpi=150, use_angle_cls=False, enable_raster_pass=False
    )
    payload = NativePDFElements.model_validate_json(output.model_dump_json())
    words, vectors = deserialize(payload.words, payload.vectors)
    assert len(words) == len(payload.words)
    assert len(vectors) == len(payload.vectors)
