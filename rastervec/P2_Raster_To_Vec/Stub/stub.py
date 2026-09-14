"""Do-nothing Phase2Backend -- documents the interface and is the default
P2 choice for any run that doesn't need raster-to-vector extraction. Fully
self-contained (no dependency on P2_Raster_To_Vec/Junction)."""
from __future__ import annotations

from rastervec.commons.models import Image, Page, Text, Vector


def extract(images: list[Image], page: Page) -> tuple[list[Vector], list[Text]]:
    return [], []
