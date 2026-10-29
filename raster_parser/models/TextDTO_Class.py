from pydantic import BaseModel

from raster_parser.models import Text


class TextDTO(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    word: str
    block_no: int
    line_no: int
    word_no: int
    rotate: int = 0
    glyph_orientation: str = "horizontal"

    @classmethod
    def from_text(cls, text: Text) -> "TextDTO":
        return cls(
            x0=text.x0,
            y0=text.y0,
            x1=text.x1,
            y1=text.y1,
            word=text.word,
            block_no=text.block_no,
            line_no=text.line_no,
            word_no=text.word_no,
            rotate=text.rotate,
            glyph_orientation=text.glyph_orientation.name.lower(),
        )
