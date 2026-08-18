"""Shared pytest fixtures for the rastervec test suite.

Synthetic pymupdf-built PDFs are preferred over `references/*.pdf` because
those reference files are gitignored/proprietary (not guaranteed present
in a fresh clone or CI) and give no exact expected values to assert
against. Synthetic PDFs are deterministic and cheap to reason about. A
couple of tests may additionally run against `references/*.pdf`, guarded
by `pytest.mark.skipif(not Path(...).exists())`, purely as an
opportunistic smoke check -- never for correctness assertions.
"""
from __future__ import annotations

from typing import Callable

import pymupdf as fitz
import pytest


@pytest.fixture
def synthetic_pdf_factory() -> Callable[[list[dict]], "fitz.Document"]:
    """Returns a builder: build(pages) -> fitz.Document.

    `pages` is a list of dicts, each optionally specifying:
        width, height   -- page size in points (default 200x100)
        rotation        -- page.set_rotation() value (default 0)
        texts           -- list of {"point": (x, y), "text": str,
                            "rotate": int, "fontsize": float}
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
