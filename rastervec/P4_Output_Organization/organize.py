"""Phase 4: output organization, shared by every pipeline run regardless of
which Phase 2 / Phase 3 backend produced the input -- the counterpart to
Phase 1 on the other end of the pipeline (always the same, not pluggable;
see `core/registry.py`'s docstring for why P2/P3 are pluggable and this
isn't).

Two jobs:
  1. Combine Phase 1's native text with Phase 2's/Phase 3's own text/vector
     output into the one final `(texts, vectors)` pair `core.pipeline.
     run_pipeline` returns -- previously done inline in `run_pipeline`
     itself.
  2. A coordinate-space consistency backstop: every `Text`/`Vector` in the
     pipeline is supposed to stay in unrotated MediaBox space end-to-end
     (`commons/models/__init__.py`'s documented contract). A P2/P3 backend
     that leaks rotated-display-space geometry (the shape of bug fixed in
     `P1_Reading_Native/image_extract.py::_render_whole_page` -- see that
     module) produces a bbox that doesn't fit the page's own unrotated
     `width`/`height`, most obviously on a 90/270-degree page where a
     rotated-space bbox has its axes swapped. This phase logs (never
     silently drops or tries to guess-and-reproject) anything that doesn't
     fit, so a future regression like that one is caught at the seam
     instead of silently reaching final output.
"""
from __future__ import annotations

from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Page, Text, Vector

_LOG = get_logger("P4.organize")

# Small slack for float round-trip error through PDF/pixel-space
# conversions -- not a tolerance for genuine rotated-space leaks, which are
# typically off by tens/hundreds of points (a whole axis swapped).
_BBOX_TOLERANCE = 0.5


def organize_outputs(
    texts_p1: list[Text],
    texts_p2: list[Text],
    texts_p3: list[Text],
    vectors_p3: list[Vector],
    page: Page,
) -> tuple[list[Text], list[Vector]]:
    """Combine every phase's text output with Phase 3's final vectors, and
    warn about any item whose bbox doesn't fit the page's own unrotated
    MediaBox dims. `vectors_p3` is already Phase 3's complete final vector
    output (it supersedes `vectors_p2`, which fed into Phase 3 as an
    input) -- unchanged from what `core.pipeline.run_pipeline` used to
    return directly."""
    texts = list(texts_p1) + list(texts_p2) + list(texts_p3)
    vectors = list(vectors_p3)

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
    if (
        x0 < -_BBOX_TOLERANCE or y0 < -_BBOX_TOLERANCE
        or x1 > width + _BBOX_TOLERANCE or y1 > height + _BBOX_TOLERANCE
    ):
        _LOG.warning(
            "page %d: %s bbox %s falls outside the unrotated MediaBox (%.1fx%.1f) -- "
            "possible rotated-space coordinate leak from an upstream P2/P3 backend",
            page_index, kind, bbox, width, height,
        )
