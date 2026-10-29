"""What every raster_vectorize / vector_classify backend implements. Structural (`Protocol`), not enforced
at runtime -- the contract `core.registry` relies on.

- raster_vectorize (raster -> vector) gets native_read's embedded images and the page. `page.doc_path`
  / `page.meta.index` identify the source, for a backend that needs the whole page.
- vector_classify (vector parsing) gets native_read's vectors only -- never raster_vectorize's output.

Both return `(vectors, texts)` in unrotated page space. Neither imports anything from a
sibling backend; `raster_parser.commons` is the only shared code.
"""

from __future__ import annotations

from typing import Protocol

from raster_parser.models.image import Image
from raster_parser.models.page import Page
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector
from raster_parser.models.settings import RunSettings


class RasterVectorizer(Protocol):
    def __call__(
        self,
        images: list[Image],
        page: Page,
        settings: RunSettings,
    ) -> tuple[list[Vector], list[Text]]: ...


class VectorClassifier(Protocol):
    def __call__(
        self,
        vectors: list[Vector],
        page: Page,
        settings: RunSettings,
    ) -> tuple[list[Vector], list[Text]]: ...
