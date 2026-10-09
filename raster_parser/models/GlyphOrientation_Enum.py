from enum import Enum

class GlyphOrientation(Enum):
    """Deduced text-line orientation for a word group (see ``orientation_of_word_group``)."""

    HORIZONTAL = 1
    VERTICAL = 2
    OTHER = 3