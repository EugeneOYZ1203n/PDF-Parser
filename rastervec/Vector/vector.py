"""Vector stage: extracts vector drawings from a page.

extract_vectors (plus layer/color separation, in layer_color_separation.py)
is this module's concern. One `Vector` per `get_drawings()` entry -- never
decomposed into its own items; no per-kind item parsing happens here, just
`Vector.from_pymupdf` per drawing (drawing-level field copying +
`helpers.fitz_geometry.plain_item` to strip fitz objects out of `items`).
Classification of vectors into text candidates vs. drawing content lives in
rastervec/Vector_Classification/ instead.

`get_drawings(extended=True)` is used so a path's enclosing transparency-group
blend mode / constant opacity (which plain `get_drawings()` silently drops --
`/BM` never reaches the path dict at all) can be folded onto each `Vector`,
letting the renderer reproduce e.g. a Multiply-blended line instead of
painting it fully opaque. The extra `group`/`clip` wrapper entries
`extended=True` interleaves are consumed by `_iter_drawing_paths` and never
become `Vector`s; the surviving path entries are identical (count, order,
`seqno`, geometry) to plain `get_drawings()`. Clip-path geometry itself is
not reproduced -- only blend mode + opacity are folded in.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterator

from rastervec.logging_setup import get_logger
from rastervec.models import Page, Vector
from rastervec.renderer.stages import render_vectors  # noqa: F401 -- re-exported for callers
from rastervec.Vector.layer_color_separation import (  # noqa: F401 -- re-exported for callers
    separate_by_color,
    separate_by_layer,
)

_LOG = get_logger("vector")

_PATH_TYPES = {"s", "f", "fs"}


def _iter_drawing_paths(
    fitz_page: "object",
) -> Iterator[tuple[dict, str | None, float | None]]:
    """Yield `(drawing_dict, blendmode, opacity)` for each real path entry of
    `fitz_page.get_drawings(extended=True)`, with the blend mode / constant
    opacity of its enclosing transparency group(s) resolved from the
    interleaved `group`/`clip` wrapper entries.

    `blendmode` is the innermost enclosing non-"Normal" blend mode (else
    `None`); `opacity` is the product of every enclosing group's constant
    opacity (else `None` when that product is 1.0 or nothing set it).
    `group`/`clip` wrapper entries are consumed here and never yielded.
    """
    # stack of (level, blendmode, opacity) for currently-open group/clip
    stack: list[tuple[int, str | None, float | None]] = []
    for entry in fitz_page.get_drawings(extended=True):
        level = entry.get("level", 0)
        while stack and stack[-1][0] >= level:
            stack.pop()

        etype = entry.get("type")
        if etype in ("group", "clip"):
            stack.append((level, entry.get("blendmode"), entry.get("opacity")))
            continue
        if etype not in _PATH_TYPES:
            continue

        blendmode: str | None = None
        opacity = 1.0
        for _lvl, bm, op in stack:
            if bm and bm != "Normal":
                blendmode = bm
            if op is not None:
                opacity *= op
        yield entry, blendmode, (opacity if opacity < 1.0 else None)


def extract_vectors(page: Page) -> list[Vector]:
    """One `Vector` per raw `get_drawings()` drawing, in page order, with each
    Vector's enclosing-group `blendmode`/`opacity` folded in (see the module
    docstring)."""
    fitz_page = page.fitz_page
    page_index = page.meta.index

    vectors: list[Vector] = []
    for seq, (drawing, blendmode, opacity) in enumerate(_iter_drawing_paths(fitz_page)):
        vector = Vector.from_pymupdf(drawing, page_index=page_index, seq=seq)
        if blendmode is not None or opacity is not None:
            vector = replace(
                vector,
                blendmode=blendmode if blendmode is not None else vector.blendmode,
                opacity=opacity if opacity is not None else vector.opacity,
            )
        vectors.append(vector)

    _LOG.debug("page %d: extracted %d vector(s)", page_index, len(vectors))
    return vectors
