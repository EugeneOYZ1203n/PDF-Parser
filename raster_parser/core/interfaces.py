"""What every Phase 2 / Phase 3 backend implements. Structural (`Protocol`), not enforced
at runtime -- the contract `core.registry` relies on.

- Phase 2 (raster -> vector) gets Phase 1's embedded images and the page. `page.doc_path`
  / `page.meta.index` identify the source, for a backend that needs the whole page.
- Phase 3 (vector parsing) gets Phase 1's vectors only -- never Phase 2's output.

Both return `(vectors, texts)` in unrotated page space. Neither imports anything from a
sibling backend; `raster_parser.commons` is the only shared code.
"""
from __future__ import annotations

from typing import Protocol

from raster_parser.commons.models import Image, Page, Text, Vector
from raster_parser.core.settings import RunSettings


class Phase2Backend(Protocol):
    def __call__(
        self, images: list[Image], page: Page, settings: RunSettings,
    ) -> tuple[list[Vector], list[Text]]: ...


class Phase3Backend(Protocol):
    def __call__(
        self, vectors: list[Vector], page: Page, settings: RunSettings,
    ) -> tuple[list[Vector], list[Text]]: ...
