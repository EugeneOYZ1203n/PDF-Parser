"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pymupdf as fitz
import pymupdf as pymupdf
import pytest

from raster_parser.commons.models import PageMeta, Text, Vector


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    """1-page PDF with native text and a vector stroke."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((50, 50), "hello world", fontsize=14)
    page.draw_rect(
        pymupdf.Rect(60, 100, 200, 180),
        color=(1, 0, 0),
        width=1.5,
    )
    out = tmp_path / "synthetic.pdf"
    doc.save(out)
    doc.close()
    return out


@pytest.fixture
def synthetic_pdf_with_fills(tmp_path: Path) -> Path:
    """PDF with one stroke rect and three adjacent black fill glyph stubs.

    Glyph stubs are closed 4-line paths (not MuPDF ``re``), mid aspect ratio so
    ``is_box_v1`` does not treat them as layout boxes.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.draw_rect(pymupdf.Rect(60, 100, 200, 180), color=(1, 0, 0), width=1.5)
    shape = page.new_shape()
    for x0 in (50, 58, 66):
        r = pymupdf.Rect(x0, 50, x0 + 8, 82)
        shape.draw_line(r.tl, r.tr)
        shape.draw_line(r.tr, r.br)
        shape.draw_line(r.br, r.bl)
        shape.draw_line(r.bl, r.tl)
        shape.finish(fill=(0, 0, 0), color=None, width=0, closePath=True)
    shape.commit()
    out = tmp_path / "synthetic_fills.pdf"
    doc.save(out)
    doc.close()
    return out


# --- Fixtures ported from rastervec (factories over raster_parser.commons models) ---

@pytest.fixture
def synthetic_pdf_factory() -> Callable[[list[dict]], "fitz.Document"]:
    """Returns a builder: build(pages) -> fitz.Document.

    `pages` is a list of dicts, each optionally specifying:
        width, height   -- page size in points (default 200x100)
        rotation        -- page.set_rotation() value (default 0)
        texts           -- list of {"point": (x, y), "text": str,
                            "rotate": int, "fontsize": float}
        drawings        -- list of {"rects": [(x0,y0,x1,y1), ...],
                            "lines": [((x0,y0),(x1,y1)), ...],
                            "color": (r,g,b)|None, "fill": (r,g,b)|None,
                            "width": float}
    """

    def _factory(pages: list[dict]) -> "fitz.Document":
        doc = fitz.open()
        for spec in pages:
            page = doc.new_page(
                width=spec.get("width", 200),
                height=spec.get("height", 100),
            )
            rotation = spec.get("rotation", 0)
            if rotation:
                page.set_rotation(rotation)
            for text_spec in spec.get("texts", []):
                page.insert_text(
                    text_spec["point"],
                    text_spec["text"],
                    rotate=text_spec.get("rotate", 0),
                    fontsize=text_spec.get("fontsize", 11),
                )
            for draw_spec in spec.get("drawings", []):
                shape = page.new_shape()
                for rect in draw_spec.get("rects", []):
                    shape.draw_rect(fitz.Rect(*rect))
                for (p1, p2) in draw_spec.get("lines", []):
                    shape.draw_line(fitz.Point(*p1), fitz.Point(*p2))
                shape.finish(
                    color=draw_spec.get("color", (0, 0, 0)),
                    fill=draw_spec.get("fill"),
                    width=draw_spec.get("width", 1.0),
                )
                shape.commit()
        return doc

    return _factory


@pytest.fixture
def tmp_pdf_path(tmp_path) -> Callable[["fitz.Document"], str]:
    """Returns a saver: save(doc) -> path, for tests that need a real
    file path (e.g. Reader tests) rather than an in-memory Document."""

    def _save(doc: "fitz.Document") -> str:
        path = tmp_path / "synthetic.pdf"
        doc.save(str(path))
        doc.close()
        return str(path)

    return _save


def _items_for(kind: str, bbox: tuple[float, float, float, float]) -> list[tuple]:
    x0, y0, x1, y1 = bbox
    if kind == "c":
        return [("c", (x0, y0), (x0, y1), (x1, y0), (x1, y1))]
    if kind == "qu":
        return [("qu", ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))]
    if kind == "re":
        return [("re", (x0, y0, x1, y1))]
    # "l"
    return [("l", (x0, y0), (x1, y1))]


@pytest.fixture
def vector() -> Callable[..., Vector]:
    """Returns a builder: vector(**overrides) -> Vector, with every field
    defaulted. `items` is derived from `kind`/`bbox` unless passed
    explicitly. Replaces the per-file `_make_vector`/`_make_path` helpers."""

    def _build(
        *,
        kind: str = "l",
        type: str = "s",
        bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        items: list[tuple] | None = None,
        color: tuple[float, ...] | None = (0.0, 0.0, 0.0),
        fill: tuple[float, ...] | None = None,
        width: float | None = None,
        dashes: str | None = None,
        closePath: bool | None = None,
        lineCap: int = 0,
        lineJoin: float = 0,
        even_odd: bool = False,
        stroke_opacity: float | None = None,
        fill_opacity: float | None = None,
        layer: str | None = None,
        scissor: tuple[float, float, float, float] | None = None,
        seqno: int = 0,
        blendmode: str | None = None,
        isolated: bool = False,
        knockout: bool = False,
        opacity: float | None = None,
        page_index: int = 0,
    ) -> Vector:
        return Vector(
            type=type,
            items=items if items is not None else _items_for(kind, bbox),
            color=color,
            fill=fill,
            width=width,
            dashes=dashes,
            closePath=closePath,
            lineCap=lineCap,
            lineJoin=lineJoin,
            even_odd=even_odd,
            stroke_opacity=stroke_opacity,
            fill_opacity=fill_opacity,
            layer=layer,
            rect=bbox,
            scissor=scissor,
            seqno=seqno,
            blendmode=blendmode,
            isolated=isolated,
            knockout=knockout,
            opacity=opacity,
            page_index=page_index,
        )

    return _build


@pytest.fixture
def text() -> Callable[..., Text]:
    """Returns a builder: text(**overrides) -> Text, every field
    defaulted. `origin` is derived from `bbox`/`direction` unless passed
    explicitly."""

    def _build(
        *,
        text: str = "Hi",
        bbox: tuple[float, float, float, float] = (10.0, 10.0, 20.0, 20.0),
        direction: tuple[float, float] = (1.0, 0.0),
        origin: tuple[float, float] | None = None,
        font: str = "helv",
        font_size: float = 10.0,
        color: int | None = None,
        flags: int = 0,
        ascender: float | None = None,
        descender: float | None = None,
        wmode: int = 0,
        block_no: int = 0,
        line_no: int = 0,
        word_no: int = 0,
        page_index: int = 0,
        seqno: int = 0,
        confidence: float = 0.0,
        source: str = "native",
        raw_word: tuple | None = None,
        raw_span: dict | None = None,
        quad_points: tuple | None = None,
    ) -> Text:
        from raster_parser.commons.helpers.geometry import compute_origin

        return Text(
            text=text,
            bbox=bbox,
            direction=direction,
            origin=origin if origin is not None else compute_origin(bbox, direction),
            font=font,
            font_size=font_size,
            color=color,
            flags=flags,
            ascender=ascender,
            descender=descender,
            wmode=wmode,
            block_no=block_no,
            line_no=line_no,
            word_no=word_no,
            page_index=page_index,
            seqno=seqno,
            confidence=confidence,
            source=source,
            raw_word=raw_word,
            raw_span=raw_span,
            quad_points=quad_points,
        )

    return _build


@pytest.fixture
def page_meta() -> Callable[..., PageMeta]:
    """Returns a builder: page_meta(**overrides) -> PageMeta."""

    def _build(
        *,
        index: int = 0,
        number: int | None = None,
        mediabox: tuple[float, float, float, float] | None = None,
        rotation: int = 0,
        width: float = 200.0,
        height: float = 200.0,
    ) -> PageMeta:
        return PageMeta(
            index=index,
            number=number if number is not None else index + 1,
            mediabox=mediabox if mediabox is not None else (0.0, 0.0, width, height),
            rotation=rotation,
            width=width,
            height=height,
        )

    return _build
