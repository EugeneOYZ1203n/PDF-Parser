from dataclasses import dataclass

from raster_parser.models import GlyphOrientation


@dataclass
class Text:
    """A single word extracted from a page, with bounding box and structural indices."""

    x0: float
    y0: float
    x1: float
    y1: float
    word: str
    block_no: int
    line_no: int
    word_no: int
    rotate: int = 0  # PDF draw rotation (degrees) after ``ocr_rotate_for_target``
    glyph_orientation: GlyphOrientation = GlyphOrientation.HORIZONTAL  # deduced line orientation
