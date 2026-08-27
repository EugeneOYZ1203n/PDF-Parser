"""Rotation Correction: a layer between ocr_compare and drawing_vectors
that fixes an OCR reading's `rotation_used` when the resolved text reads
better rotated 90 degrees from what the OCR backend itself detected.

For every ocr_compare cluster result with real resolved text, compares the
text's own natural aspect ratio (`_text_aspect_ratio`) against its
`resolved.bbox`'s aspect ratio as-is, and again against that same bbox
rotated 90 deg (width/height swapped, i.e. `1 / bbox_ratio`). If the
rotated comparison is a meaningfully closer match (`error_unrotated -
error_rotated > ROTATION_VERIFY_IMPROVEMENT_MARGIN` -- avoids flipping on
a near-tie), `RotationCheck.resolved` is a *new* `TextVectorResult` (via
`dataclasses.replace`, not a mutation of ocr_compare's own object) with
`rotation_used` corrected by +90 deg (`applied=True`). ocr_compare's own
`resolved` reading is left untouched -- consumers wanting the corrected
orientation (this stage's own reconstruction view, drawing_vectors) read
`RotationCheck.resolved` instead.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pymupdf as fitz

from rastervec.models import TextVectorResult, VectorPath

if TYPE_CHECKING:
    from rastervec.pipeline import PipelineContext

# How much closer (relative-error improvement) the 90-deg-rotated
# bbox/text-aspect match must be than the as-is match before rotation_used
# gets corrected -- avoids flipping on a near-tie.
ROTATION_VERIFY_IMPROVEMENT_MARGIN = 0.15


@dataclass
class RotationCheck:
    """One rotation_verify stage result, for one ocr_compare comparison's
    `resolved` reading (see `run`): compares the OCR'd text's own natural
    width/height aspect ratio (font-metric based, any fontsize -- see
    `_text_aspect_ratio`) against `bbox`'s aspect ratio as-is, and again
    against that same bbox rotated 90 deg (width/height swapped). If the
    rotated comparison is a meaningfully closer match, `resolved` is a
    *new* `TextVectorResult` (via `dataclasses.replace`, not a mutation of
    ocr_compare's own object) with `rotation_used` corrected by +90 deg
    (`applied=True`). ocr_compare's own `resolved` reading is left
    untouched -- consumers wanting the corrected orientation (this stage's
    own reconstruction view, drawing_vectors) read `RotationCheck.
    resolved` instead."""

    cluster: list[VectorPath]
    text: str
    bbox: tuple[float, float, float, float]
    before_rotation: int
    after_rotation: int
    applied: bool
    error_unrotated: float
    error_rotated: float
    resolved: TextVectorResult


def _text_aspect_ratio(text: str) -> float | None:
    """Natural width/height ratio of `text` laid out on one line, via the
    same base14 "helv" font metrics Renderer.render_reconstructed_page
    uses for placement -- fontsize-invariant (both width and height scale
    with fontsize together), so this needs no bbox/fontsize input at all.
    `None` for blank text."""
    if not text.strip():
        return None
    font = fitz.Font("helv")
    span = font.ascender - font.descender
    if span <= 0:
        return None
    return font.text_length(text, fontsize=1.0) / span


def _relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-6)


def _run_rotation_verify(ctx: "PipelineContext") -> list[RotationCheck]:
    """New layer between ocr_compare and drawing_vectors: for every
    ocr_compare cluster result with real resolved text, compares the
    text's own natural aspect ratio (_text_aspect_ratio) against its
    `resolved.bbox`'s aspect ratio as-is, and again against that same bbox
    rotated 90 deg (width/height swapped, i.e. 1/bbox_ratio) -- if the
    rotated comparison is a meaningfully closer match (see ROTATION_VERIFY_
    IMPROVEMENT_MARGIN), `RotationCheck.resolved` is a *new* TextVectorResult
    (`dataclasses.replace`) with `rotation_used` corrected by +90 deg.
    ocr_compare's own `resolved` object is never mutated. Clusters with
    blank/failed resolved text, or a degenerate (zero-area) bbox, are
    skipped -- nothing to check, and no RotationCheck is recorded for
    them."""
    cluster_results = ctx.cluster_ocr_results or []
    checks: list[RotationCheck] = []

    for cluster_result in cluster_results:
        resolved = cluster_result.resolved
        text_ratio = _text_aspect_ratio(resolved.text)
        if text_ratio is None:
            continue

        x0, y0, x1, y1 = resolved.bbox
        width, height = x1 - x0, y1 - y0
        if width <= 0 or height <= 0:
            continue
        bbox_ratio = width / height

        error_unrotated = _relative_error(text_ratio, bbox_ratio)
        error_rotated = _relative_error(text_ratio, 1.0 / bbox_ratio)

        before = resolved.rotation_used
        applied = (error_unrotated - error_rotated) > ROTATION_VERIFY_IMPROVEMENT_MARGIN
        after = (before + 90) % 360 if applied else before
        fixed = replace(resolved, rotation_used=after) if applied else resolved

        checks.append(
            RotationCheck(
                cluster=resolved.paths, text=resolved.text, bbox=resolved.bbox,
                before_rotation=before, after_rotation=after,
                applied=applied, error_unrotated=error_unrotated, error_rotated=error_rotated,
                resolved=fixed,
            )
        )

    ctx.rotation_checks = checks
    return checks
