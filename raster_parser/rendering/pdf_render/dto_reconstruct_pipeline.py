"""Convert serialized :class:`TextDTO` / :class:`VectorDTO` lists back to native objects.

Usage:
    uv run scripts/dto_reconstruct_pipeline.py drawing.pdf [--page N]
    uv run scripts/dto_reconstruct_pipeline.py INPUT.json

Writes ``--out-json`` (DTO), ``--out-words``, ``--out-vectors``, and ``--out-pdf``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as pymupdf

from raster_parser.models import Text
from raster_parser.models import Vector
from raster_parser.serialisation.deserialization import (
    deserialize,
    deserialize_text,
    deserialize_vector,
)
from raster_parser.models import TextDTO, NativePDFElements, VectorDTO
from raster_parser.rendering.pdf_render.reconstruct import (
    _finish_kwargs_reconstruct,
    _replay_items,
    drawing_from_vector,
)

# Backward-compatible aliases
text_from_dto = deserialize_text
vector_from_dto = deserialize_vector


@dataclass(frozen=True)
class DtoReconstructResult:
    out_txt_words: Path
    out_txt_vectors: Path
    words: list[Text]
    vectors: list[Vector]


def write_words_txt(path: Path | str, words: list[Text]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["=== WORDS ===", *(str(t) for t in words)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path.resolve()


def write_vectors_txt(path: Path | str, vectors: list[Vector]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["=== VECTORS ==="]
    lines.extend(f"Vector {i}: {vector}" for i, vector in enumerate(vectors))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path.resolve()


def reconstruct_from_dtos(
    words: list[TextDTO],
    vectors: list[VectorDTO],
    *,
    out_txt_words: Path | str,
    out_txt_vectors: Path | str,
) -> DtoReconstructResult:
    """Convert DTOs to native objects and write txt dumps."""
    texts, native_vectors = deserialize(words, vectors)

    return DtoReconstructResult(
        out_txt_words=write_words_txt(out_txt_words, texts),
        out_txt_vectors=write_vectors_txt(out_txt_vectors, native_vectors),
        words=texts,
        vectors=native_vectors,
    )


def output_as_dto_json(
    output: NativePDFElements,
    out_path: Path | str,
) -> Path:
    """Write :class:`NativePDFElements` as indented JSON for debugging."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return out_path.resolve()


def reconstruct_from_output(
    output: NativePDFElements,
    *,
    out_txt_words: Path | str,
    out_txt_vectors: Path | str,
) -> DtoReconstructResult:
    """Convenience wrapper for a full :class:`NativePDFElements`."""
    return reconstruct_from_dtos(
        output.words,
        output.vectors,
        out_txt_words=out_txt_words,
        out_txt_vectors=out_txt_vectors,
    )

def _page_size_from_vectors(
    vectors: list[Vector],
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    margin: float = 20.0,
) -> tuple[float, float]:
    if page_width is not None and page_height is not None:
        return page_width, page_height

    max_x = 0.0
    max_y = 0.0
    for vector in vectors:
        max_x = max(max_x, float(vector.rect[2]))
        max_y = max(max_y, float(vector.rect[3]))

    width = page_width if page_width is not None else max(max_x + margin, 400.0)
    height = page_height if page_height is not None else max(max_y + margin, 400.0)
    return width, height


def draw_vectors(
    vectors: list[Vector],
    out_pdf: Path | str,
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    page_rotation: int = 0,
) -> Path:
    """Replay all vectors onto a new single-page PDF."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    width, height = _page_size_from_vectors(
        vectors,
        page_width=page_width,
        page_height=page_height,
    )
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    if page_rotation:
        page.set_rotation(page_rotation)

    shape = page.new_shape()
    for vector in vectors:
        path = drawing_from_vector(vector)
        _replay_items(shape, path["items"])
        shape.finish(**_finish_kwargs_reconstruct(path))
    shape.commit()

    doc.save(out_pdf)
    doc.close()
    return out_pdf.resolve()
