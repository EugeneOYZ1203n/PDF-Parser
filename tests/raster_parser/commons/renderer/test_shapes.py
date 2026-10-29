"""`_shapes.replay_drawing_paths` (ported from rastervec's renderer tests; the
rest of that file covers renderer modules not in raster_parser yet)."""

from __future__ import annotations

import pymupdf
from raster_parser.models.vector import Vector
from raster_parser.commons.renderer._shapes import replay_drawing_paths


def _coloured_vectors(vector) -> list[Vector]:
    magenta_line = vector(kind="l", bbox=(0, 5, 40, 5), color=(1, 0, 1), width=0.0)
    cyan_tri = vector(
        type="f",
        color=None,
        fill=(0, 1, 1),
        items=[("l", (0, 10), (30, 10)), ("l", (30, 10), (0, 40)), ("l", (0, 40), (0, 10))],
        bbox=(0, 10, 30, 40),
    )
    return [magenta_line, cyan_tri]


def test_monochrome_replay_keeps_hairlines(vector):
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=50, height=50)
        replay_drawing_paths(page, _coloured_vectors(vector), monochrome=True)
        ops = b"".join(doc.xref_stream(x) for x in page.get_contents()).decode().split()
        assert ops[ops.index("w") - 1] == "0"
        assert ops[ops.index("RG") - 3 : ops.index("RG")] == ["0", "0", "0"]
    finally:
        doc.close()
