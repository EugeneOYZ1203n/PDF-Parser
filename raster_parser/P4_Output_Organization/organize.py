"""Phase 4: combine every phase's output into the final `(texts, vectors)` pair.

Not pluggable -- the same whichever Phase 2 / Phase 3 backends ran.

Ordering (one rule for every backend combination):

- texts: Phase 1 native words, then Phase 3 text, then Phase 2 text, each in the order its
  phase produced it. `block_no` is renumbered into a disjoint range per phase (Phase 3's
  start after the largest native `block_no`, Phase 2's after Phase 3's), so a block number
  identifies one block of one source.
- vectors: Phase 3's, then Phase 2's, with Phase 2's `seqno` moved past the largest
  Phase 3 `seqno`: raster-traced vectors paint on top and every output seqno is unique.

It is also a coordinate-space backstop: every `Text` / `Vector` is meant to stay in unrotated,
CropBox-relative page space. Anything whose bbox falls outside the page is logged (never
dropped or reprojected) so a backend leaking rotated-display coordinates is caught here.
"""
from __future__ import annotations

from dataclasses import replace

from raster_parser.commons.logging_setup import get_logger
from raster_parser.commons.models import Page, Text, Vector

_LOG = get_logger("P4.organize")

# Float round-trip slack, not a tolerance for real leaks (those are off by a whole axis).
_BBOX_TOLERANCE = 0.5


def _rebase_blocks(texts: list[Text], first: int) -> list[Text]:
    if not texts:
        return []
    low = min(t.block_no for t in texts)
    return [replace(t, block_no=t.block_no - low + first) for t in texts]


def _next_after(values: list[int]) -> int:
    return max(values) + 1 if values else 0


def organize_outputs(
    texts_p1: list[Text],
    texts_p2: list[Text],
    texts_p3: list[Text],
    vectors_p2: list[Vector],
    vectors_p3: list[Vector],
    page: Page,
) -> tuple[list[Text], list[Vector]]:
    texts = list(texts_p1)
    rebased_p3 = _rebase_blocks(texts_p3, _next_after([t.block_no for t in texts]))
    texts += rebased_p3
    texts += _rebase_blocks(texts_p2, _next_after([t.block_no for t in texts]))

    first_p2_seq = _next_after([v.seqno for v in vectors_p3])
    low_p2 = min((v.seqno for v in vectors_p2), default=0)
    vectors = list(vectors_p3) + [replace(v, seqno=v.seqno - low_p2 + first_p2_seq) for v in vectors_p2]

    width, height = page.meta.width, page.meta.height
    for t in texts:
        _warn_if_out_of_bounds("Text", t.bbox, width, height, page.meta.index)
    for v in vectors:
        _warn_if_out_of_bounds("Vector", v.bbox, width, height, page.meta.index)
    return texts, vectors


def _warn_if_out_of_bounds(
    kind: str, bbox: tuple[float, float, float, float], width: float, height: float, page_index: int,
) -> None:
    x0, y0, x1, y1 = bbox
    if x0 < -_BBOX_TOLERANCE or y0 < -_BBOX_TOLERANCE or x1 > width + _BBOX_TOLERANCE or y1 > height + _BBOX_TOLERANCE:
        _LOG.warning(
            "page %d: %s bbox %s falls outside the unrotated page (%.1fx%.1f) -- "
            "possible rotated-space coordinate leak from a Phase 2/3 backend",
            page_index, kind, bbox, width, height,
        )
