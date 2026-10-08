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
`seqno`, geometry) to plain `get_drawings()`. Clips are folded in twice:
each path's `scissor` is the intersection of its enclosing `clip` entries' own
`scissor` rects (PyMuPDF puts the clip on the wrapper entry, never on the
clipped path itself), and `clips` keeps the exact shape (items + even-odd
rule) of every enclosing clip that is *not* an axis-aligned rectangle -- a
rectangular clip is already exact as its `scissor`. The renderer re-applies
both. A path's own `rect` stays unclipped.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterator

from rastervec.commons.helpers.fitz_geometry import plain_item
from rastervec.commons.helpers.geometry import item_points
from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Page, Vector
from rastervec.commons.renderer.stages import render_vectors  # noqa: F401 -- re-exported for callers
from rastervec.Vector.layer_color_separation import (  # noqa: F401 -- re-exported for callers
    separate_by_color,
    separate_by_layer,
    separate_by_width,
)

_LOG = get_logger("vector")

_PATH_TYPES = {"s", "f", "fs"}


Rect = tuple[float, float, float, float]


def _intersect(a: Rect | None, b: Rect | None) -> Rect | None:
    if a is None:
        return b
    if b is None:
        return a
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    # an empty intersection stays a (zero-area) rect -- it clips everything
    return (x0, y0, max(x0, min(a[2], b[2])), max(y0, min(a[3], b[3])))


Clip = tuple[tuple, bool]  # (plain items, even_odd)

_CORNER_EPS = 1e-3


def _is_axis_rect(items: tuple, scissor: Rect | None) -> bool:
    """True when a clip path is exactly its own (axis-aligned) `scissor`
    rect, so `scissor` alone reproduces it: a single `re`, a single `qu`, or
    only `l` items, every point of which sits on a corner of `scissor`."""
    if scissor is None or not items:
        return False
    if len(items) == 1 and items[0][0] == "re":
        return True
    if not all(it[0] in ("l", "qu") for it in items):
        return False
    if any(it[0] == "qu" for it in items) and len(items) != 1:
        return False
    x0, y0, x1, y1 = scissor
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    for pt in (p for it in items for p in item_points(it)):
        if not any(abs(pt[0] - cx) <= _CORNER_EPS and abs(pt[1] - cy) <= _CORNER_EPS
                   for cx, cy in corners):
            return False
    return True


def _iter_drawing_paths(
    fitz_page: "object",
) -> Iterator[tuple[dict, str | None, float | None, Rect | None, tuple[Clip, ...]]]:
    """Yield `(drawing_dict, blendmode, opacity, scissor, clips)` for each
    real path entry of `fitz_page.get_drawings(extended=True)`, with the
    blend mode / constant opacity of its enclosing transparency group(s) and
    its enclosing clip(s) resolved from the interleaved `group`/`clip`
    wrapper entries.

    `blendmode` is the innermost enclosing non-"Normal" blend mode (else
    `None`); `opacity` is the product of every enclosing group's constant
    opacity (else `None` when that product is 1.0 or nothing set it);
    `scissor` is the intersection of every enclosing clip's `scissor` rect
    (else `None`); `clips` is the exact `(items, even_odd)` shape of every
    enclosing non-rectangular clip, outermost first (rectangular ones are
    exact in `scissor` already). `group`/`clip` wrapper entries are consumed
    here and never yielded.
    """
    # stack of (level, blendmode, opacity, scissor, clip-shape-or-None)
    stack: list[tuple[int, str | None, float | None, Rect | None, Clip | None]] = []
    for entry in fitz_page.get_drawings(extended=True):
        level = entry.get("level", 0)
        while stack and stack[-1][0] >= level:
            stack.pop()

        etype = entry.get("type")
        if etype in ("group", "clip"):
            sc = entry.get("scissor") if etype == "clip" else None
            rect = (sc.x0, sc.y0, sc.x1, sc.y1) if sc is not None else None
            shape: Clip | None = None
            if etype == "clip":
                items = tuple(plain_item(it) for it in entry.get("items") or [])
                if items and not _is_axis_rect(items, rect):
                    shape = (items, bool(entry.get("even_odd", False)))
            stack.append((level, entry.get("blendmode"), entry.get("opacity"), rect, shape))
            continue
        if etype not in _PATH_TYPES:
            continue

        blendmode: str | None = None
        opacity = 1.0
        scissor: Rect | None = None
        clips: list[Clip] = []
        for _lvl, bm, op, sc, shape in stack:
            if bm and bm != "Normal":
                blendmode = bm
            if op is not None:
                opacity *= op
            scissor = _intersect(scissor, sc)
            if shape is not None:
                clips.append(shape)
        yield entry, blendmode, (opacity if opacity < 1.0 else None), scissor, tuple(clips)


def extract_vectors(page: Page) -> list[Vector]:
    """One `Vector` per raw `get_drawings()` drawing, in page order, with each
    Vector's enclosing-group `blendmode`/`opacity`, enclosing-clip bbox
    (`scissor`) and non-rectangular clip shapes (`clips`) folded in (see the
    module docstring)."""
    fitz_page = page.fitz_page
    page_index = page.meta.index

    vectors: list[Vector] = []
    for seq, (drawing, blendmode, opacity, scissor, clips) in enumerate(
        _iter_drawing_paths(fitz_page)
    ):
        vector = Vector.from_pymupdf(drawing, page_index=page_index, seq=seq)
        if blendmode is not None or opacity is not None or scissor is not None or clips:
            vector = replace(
                vector,
                blendmode=blendmode if blendmode is not None else vector.blendmode,
                opacity=opacity if opacity is not None else vector.opacity,
                scissor=_intersect(vector.scissor, scissor),
                clips=clips,
            )
        vectors.append(vector)

    _LOG.debug("page %d: extracted %d vector(s)", page_index, len(vectors))
    return vectors
