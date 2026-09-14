"""Protocols every P2/P3 backend module implements. Not enforced at
runtime (Python structural typing) -- documentation + type-checker contract
for `core/registry.py`'s registries."""
from __future__ import annotations

from typing import Protocol

from rastervec.commons.models import Image, Page, Text, Vector


class Phase2Backend(Protocol):
    def __call__(self, images: list[Image], page: Page) -> tuple[list[Vector], list[Text]]: ...


class Phase3Backend(Protocol):
    def __call__(
        self, vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    ) -> tuple[list[Vector], list[Text]]: ...
