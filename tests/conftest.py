"""Shared pytest fixtures for the rastervec test suite.

Synthetic pymupdf-built PDFs are preferred over `references/*.pdf` because
those reference files are gitignored/proprietary (not guaranteed present
in a fresh clone or CI) and give no exact expected values to assert
against. Synthetic PDFs are deterministic and cheap to reason about. A
couple of tests may additionally run against `references/*.pdf`, guarded
by `pytest.mark.skipif(not Path(...).exists())`, purely as an
opportunistic smoke check -- never for correctness assertions.

Dataclass factory fixtures (`vector_path`, `text_word`, `page_meta`) build
one canonical instance with every field defaulted, overridable by kwarg --
so a test that needs a `VectorPath` at a bbox doesn't have to spell out
its 15+ fields. Prefer these over per-file `_make_path` helpers.
"""
from __future__ import annotations

from typing import Callable

import pymupdf as fitz
import pytest

from rastervec.models import PageMeta, TextWord, VectorPath


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


def _points_for(kind: str, bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = bbox
    if kind == "c":
        return [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
    if kind == "qu":
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    # "l" and "re" both store two corner points
    return [(x0, y0), (x1, y1)]


@pytest.fixture
def vector_path() -> Callable[..., VectorPath]:
    """Returns a builder: vector_path(**overrides) -> VectorPath, with
    every field defaulted. `points` is derived from `kind`/`bbox` unless
    passed explicitly. Replaces the per-file `_make_path` helpers."""

    def _build(
        *,
        seq: int = 0,
        item_index: int = 0,
        kind: str = "l",
        fill_rule: str = "s",
        bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        points: list[tuple[float, float]] | None = None,
        stroke_color: tuple[float, ...] | None = (0.0, 0.0, 0.0),
        fill_color: tuple[float, ...] | None = None,
        stroke_opacity: float | None = None,
        fill_opacity: float | None = None,
        stroke_width: float | None = None,
        dashes: str | None = None,
        closed: bool | None = None,
        layer: str | None = None,
        page_index: int = 0,
        even_odd: bool = False,
        line_cap: int = 0,
        line_join: int = 0,
    ) -> VectorPath:
        return VectorPath(
            seq=seq,
            item_index=item_index,
            kind=kind,
            fill_rule=fill_rule,
            points=points if points is not None else _points_for(kind, bbox),
            bbox=bbox,
            stroke_color=stroke_color,
            fill_color=fill_color,
            stroke_opacity=stroke_opacity,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            dashes=dashes,
            closed=closed,
            layer=layer,
            page_index=page_index,
            even_odd=even_odd,
            line_cap=line_cap,
            line_join=line_join,
        )

    return _build


@pytest.fixture
def text_word() -> Callable[..., TextWord]:
    """Returns a builder: text_word(**overrides) -> TextWord, every field
    defaulted. `bbox`/`quad` are derived from `origin`/`font_size` unless
    passed explicitly."""

    def _build(
        *,
        text: str = "Hi",
        origin: tuple[float, float] | None = (10.0, 20.0),
        font_size: float = 10.0,
        bbox: tuple[float, float, float, float] | None = None,
        quad: tuple | None = None,
        angle: float = 0.0,
        direction: tuple[float, float] = (1.0, 0.0),
        font: str = "helv",
        color: int | None = None,
        flags: int = 0,
        ascender: float | None = None,
        descender: float | None = None,
        orientation_source: str = "text-span",
        page_index: int = 0,
        seq: int = 0,
    ) -> TextWord:
        if bbox is None:
            ox, oy = origin if origin is not None else (0.0, 0.0)
            bbox = (ox, oy - font_size, ox + font_size, oy)
        if quad is None:
            x0, y0, x1, y1 = bbox
            quad = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        return TextWord(
            text=text,
            bbox=bbox,
            quad=quad,
            angle=angle,
            direction=direction,
            font=font,
            font_size=font_size,
            color=color,
            flags=flags,
            origin=origin,
            ascender=ascender,
            descender=descender,
            orientation_source=orientation_source,
            page_index=page_index,
            seq=seq,
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
