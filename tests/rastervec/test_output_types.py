from __future__ import annotations

import json

from rastervec.models import DrawingVector, TextWord, VectorPath
from rastervec.output_types import NativePDFElements, TextDTO, VectorDTO


def _make_word(*, text="Hi", angle=0.0, seq=0) -> TextWord:
    return TextWord(
        text=text, bbox=(0, 0, 10, 5), quad=((0, 5), (10, 5), (10, 0), (0, 0)),
        angle=angle, direction=(1, 0), font="helv", font_size=10.0, color=None,
        flags=0, origin=(0, 5), ascender=None, descender=None,
        orientation_source="text-span", page_index=0, seq=seq,
    )


def _make_path(*, seq=0, item_index=0) -> VectorPath:
    return VectorPath(
        seq=seq, item_index=item_index, kind="l", fill_rule="s",
        points=[(0, 0), (1, 1)], bbox=(0, 0, 1, 1), stroke_color=(0, 0, 0),
        fill_color=None, stroke_opacity=1.0, fill_opacity=1.0, stroke_width=1.0,
        dashes="[] 0", closed=False, layer="L1", page_index=0,
    )


def _make_drawing_vector() -> DrawingVector:
    path = _make_path()
    return DrawingVector(
        paths=[path], bbox=path.bbox, stroke_color=(0, 0, 0), fill_color=None,
        stroke_width=1.0, dashed=False, page_index=0,
    )


def test_text_dto_from_text_word_maps_bbox_and_extra_fields():
    word = _make_word(text="Hi", angle=15.0, seq=3)

    dto = TextDTO.from_text_word(word)

    assert (dto.x0, dto.y0, dto.x1, dto.y1) == word.bbox
    assert dto.word == "Hi"
    assert dto.angle == 15.0
    assert dto.seq == 3
    assert dto.font == "helv"


def test_text_dto_rotate_rounds_to_nearest_quarter():
    dto = TextDTO.from_text_word(_make_word(angle=95.0))
    assert dto.rotate == 90


def test_vector_dto_from_drawing_vector_maps_items():
    dv = _make_drawing_vector()

    dto = VectorDTO.from_drawing_vector(dv)

    assert (dto.x0, dto.y0, dto.x1, dto.y1) == dv.bbox
    assert dto.color == (0, 0, 0)
    assert dto.dashes == "[] 0"
    assert dto.layer == "L1"
    assert len(dto.items) == 1
    assert dto.items[0]["kind"] == "l"


def test_native_pdf_elements_from_extract_round_trips_json():
    word = _make_word()
    dv = _make_drawing_vector()

    elements = NativePDFElements.from_extract([word], [dv])

    assert len(elements.words) == 1
    assert len(elements.vectors) == 1
    payload = elements.model_dump_json()
    restored = NativePDFElements.model_validate_json(payload)
    # Compare via re-serialized JSON, not model_dump(): the `items` field is
    # List[dict[str, Any]], so pydantic doesn't coerce its nested
    # tuple-vs-list points back to tuples on validation the way it does for
    # typed tuple fields -- JSON has no tuple type either way, so comparing
    # what both sides serialize to is the meaningful round-trip check.
    assert json.loads(restored.model_dump_json()) == json.loads(payload)


def test_native_pdf_elements_from_extract_handles_empty_input():
    elements = NativePDFElements.from_extract([], [])
    assert elements.words == []
    assert elements.vectors == []
