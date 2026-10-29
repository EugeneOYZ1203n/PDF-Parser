"""Separate stroke-like vs fill-like Vectors via the isoperimetric quotient.

``Q = 4*pi*A / P^2`` (circle -> 1, thin ribbon -> ~0). Paths with ``Q`` below a
threshold are treated as strokes; the rest as fills.

Stroke-like paths from outline autotrace stay as closed ``type="f"`` ribbons;
this module only buckets them, it does not convert them to ``type="s"``.

Main entry: ``stroke_fill_parser``.


Note: Not completed, doesn't really work well, purely experimental.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raster_parser.rendering.pdf_render.dto_reconstruct_pipeline import draw_vectors
from raster_parser.helpers.geometry import isoperimetric_quotient
from raster_parser.models import Vector

# Q below this -> stroke (elongated); at/above -> fill (compact).
_DEFAULT_Q_THRESH = 0.25
# A fill with every RGB channel (0-1) at or above this is white.
_WHITE_FILL_MIN = 0.98
_JOIN_EPS = 1e-3


@dataclass
class StrokeFillResult:
    """Split of a Vector list into stroke-like and fill-like paths."""

    stroke_vectors: list[Vector]
    fill_vectors: list[Vector]
    stroke_pdf_path: Path | None = None
    fill_pdf_path: Path | None = None
    q_thresh: float = _DEFAULT_Q_THRESH


def is_stroke_like(v: Vector, *, q_thresh: float = _DEFAULT_Q_THRESH) -> bool:
    """True if the path is elongated (stroke-like) by isoperimetric Q."""
    # White hole covers belong with fills (topology), not stroke ink.
    if v.fill is not None and all(c >= _WHITE_FILL_MIN for c in v.fill):
        return False
    return isoperimetric_quotient(v) < q_thresh


def stroke_fill_parser(
    vectors: list[Vector],
    *,
    out_dir: str | Path | None = None,
    stem: str = "drawing",
    page_width: float | None = None,
    page_height: float | None = None,
    q_thresh: float = _DEFAULT_Q_THRESH,
    write_pdfs: bool = True,
) -> StrokeFillResult:
    """Split ``vectors`` into stroke-like and fill-like lists via ``Q = 4*pi*A/P^2``.

    Paths are left unchanged (still typically ``type="f"`` outline ribbons).

    When ``write_pdfs`` is True, writes::

        {stem}_colour_strokes.pdf
        {stem}_colour_fills.pdf

    under ``out_dir`` (default: cwd).
    """
    strokes: list[Vector] = []
    fills: list[Vector] = []
    for v in vectors:
        if is_stroke_like(v, q_thresh=q_thresh):
            strokes.append(v)
        else:
            fills.append(v)

    stroke_pdf: Path | None = None
    fill_pdf: Path | None = None
    if write_pdfs and out_dir is not None:
        parent = Path(out_dir)
        parent.mkdir(parents=True, exist_ok=True)
        if strokes:
            stroke_pdf = parent / f"{stem}_colour_strokes.pdf"
            draw_vectors(
                strokes,
                stroke_pdf,
                page_width=page_width,
                page_height=page_height,
            )
        if fills:
            fill_pdf = parent / f"{stem}_colour_fills.pdf"
            draw_vectors(
                fills,
                fill_pdf,
                page_width=page_width,
                page_height=page_height,
            )

    return StrokeFillResult(
        stroke_vectors=strokes,
        fill_vectors=fills,
        stroke_pdf_path=stroke_pdf,
        fill_pdf_path=fill_pdf,
        q_thresh=q_thresh,
    )
