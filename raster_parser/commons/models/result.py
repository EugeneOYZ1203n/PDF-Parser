"""`ExtractResult` -- what `extract()` returns: one page's text and vectors, plus the page they
belong to.

Coordinates are in the page's unrotated, CropBox-relative space; `page.rotation` says how the
page is displayed (see `docs/model_migration.md`).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

from raster_parser.commons.models.page import PageMeta
from raster_parser.commons.models.text import Text
from raster_parser.commons.models.vector import Vector


@dataclass
class ExtractResult:
    page: PageMeta
    texts: list[Text]
    vectors: list[Vector]

    def to_dict(self) -> dict:
        return {
            "page": dataclasses.asdict(self.page),
            "texts": [t.to_dict() for t in self.texts],
            "vectors": [v.to_dict() for v in self.vectors],
        }

    @staticmethod
    def from_dict(d: dict) -> "ExtractResult":
        page = dict(d["page"])
        for key in ("mediabox", "cropbox"):
            if page.get(key) is not None:
                page[key] = tuple(page[key])
        return ExtractResult(
            page=PageMeta(**page),
            texts=[Text.from_dict(t) for t in d["texts"]],
            vectors=[Vector.from_dict(v) for v in d["vectors"]],
        )

    def to_json(self, **kwargs) -> str:
        """`json.dumps(self.to_dict(), **kwargs)`."""
        return json.dumps(self.to_dict(), **kwargs)

    @staticmethod
    def from_json(text: str) -> "ExtractResult":
        return ExtractResult.from_dict(json.loads(text))
