from __future__ import annotations

from typing import List
from pydantic import BaseModel

from raster_parser.models import TextDTO, VectorDTO
from raster_parser.models import Text
from raster_parser.models import Vector


class NativePDFElements(BaseModel):
    """Serializable OCR + native text and non-fill drawings from one PDF page."""

    words: List[TextDTO]
    vectors: List[VectorDTO]

    @classmethod
    def from_extract(
        cls,
        words: list[Text],
        vectors: list[Vector],
    ) -> NativePDFElements:
        return cls(
            words=[TextDTO.from_text(t) for t in words],
            vectors=[VectorDTO.from_vector(v) for v in vectors],
        )
