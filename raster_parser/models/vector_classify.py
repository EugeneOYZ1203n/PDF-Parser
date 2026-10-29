"""Intermediate results of the v1_vector_class backend: word groups and their orientation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

from raster_parser.models.vector import Vector


class Orientation(Enum):
    """Deduced text-line orientation for a word group (see ``orientation_of_word_group``)."""

    HORIZONTAL = 1
    VERTICAL = 2
    OTHER = 3


@dataclass
class WordGroup:
    vectors: List[Vector]  # the group's glyphs, in seqno order
    bbox: tuple[float, float, float, float]
    orientation: Orientation
