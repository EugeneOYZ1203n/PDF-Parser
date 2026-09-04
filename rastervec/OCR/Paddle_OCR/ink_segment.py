"""Ink-projection line/word segmentation for the light OCR backend.

Ported / adapted from `archive/raster_parser/ocr/region_ocr.py`'s
`split_words` (projection-profile word segmentation): a rendered text crop
is split into line- and word-level sub-boxes purely from where ink columns
/ rows sit and how wide the gaps between them are -- no OpenCV, no model.
Used by `rastervec/OCR/Paddle_OCR/light_backend.py` to turn one cluster
render into individually recognizable word crops before PaddleOCR
recognition-only.

`split_lines_by_ink` runs first on the whole cluster render (a row gap
wider than a typical line's own height is a line break); `split_words_by_ink`
then runs per line crop (a column gap wider than the median inter-stroke
gap is a word break, the legacy rule). All coordinates are pixel indices
in the passed 2-D `gray` array (0 = black, 255 = white).
"""
from __future__ import annotations

import numpy as np

# A pixel darker than this counts as glyph ink (matches region_ocr.py).
INK_LEVEL = 250


def _ink_runs(has_ink: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous inclusive ``[start, end]`` index runs of True in a 1-D
    bool array -- each run is one stroke / glyph fragment along the axis."""
    idx = np.flatnonzero(has_ink)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def _group_runs(runs: list[tuple[int, int]], gap_thresh: float) -> list[tuple[int, int]]:
    """Merge `runs` into spans, starting a new span whenever the gap
    between two consecutive runs exceeds `gap_thresh`."""
    if not runs:
        return []
    spans: list[tuple[int, int]] = []
    grp_start, grp_end = runs[0]
    for start, end in runs[1:]:
        if start - grp_end - 1 > gap_thresh:
            spans.append((grp_start, grp_end))
            grp_start = start
        grp_end = max(grp_end, end)
    spans.append((grp_start, grp_end))
    return spans


def _split_on_gaps(
    has_ink: np.ndarray, *, gap_factor: float, min_gap: float,
) -> list[tuple[int, int]]:
    """Legacy word rule: group ``_ink_runs(has_ink)`` on any gap wider than
    ``max(min_gap, gap_factor * median(inter-run gaps))``. Fewer than two
    runs -> a single span over the whole ink extent (``[]`` if no ink)."""
    runs = _ink_runs(has_ink)
    if len(runs) < 2:
        return [(runs[0][0], runs[-1][1])] if runs else []
    gaps = [runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]
    thresh = max(min_gap, gap_factor * float(np.median(gaps)))
    return _group_runs(runs, thresh)


def split_lines_by_ink(
    gray: np.ndarray, *, gap_factor: float = 0.8, min_gap: float = 3.0, pad: int = 1,
) -> list[tuple[int, int, int, int]]:
    """``(x0, y0, x1, y1)`` pixel boxes, one per text line, within `gray`.
    x-extent is the crop's full ink width; the row profile is split on any
    gap wider than ``max(min_gap, gap_factor * median(line height))`` -- a
    vertical gap taller than most of a line's own height is a line break.
    Empty / ink-free crop -> ``[]``."""
    if gray.ndim != 2 or gray.size == 0:
        return []
    h, w = gray.shape
    ink = gray < INK_LEVEL
    col_has_ink = ink.any(axis=0)
    if not col_has_ink.any():
        return []
    xs = np.flatnonzero(col_has_ink)
    x0, x1 = int(xs[0]), int(xs[-1]) + 1

    runs = _ink_runs(ink.any(axis=1))
    if len(runs) < 2:
        spans = [(runs[0][0], runs[-1][1])] if runs else []
    else:
        heights = [end - start + 1 for start, end in runs]
        thresh = max(min_gap, gap_factor * float(np.median(heights)))
        spans = _group_runs(runs, thresh)

    return [
        (max(0, x0 - pad), max(0, sy0 - pad), min(w, x1 + pad), min(h, sy1 + 1 + pad))
        for sy0, sy1 in spans
    ]


def split_words_by_ink(
    gray: np.ndarray, *, gap_factor: float = 1.9, min_gap: float = 2.0, pad: int = 1,
) -> list[tuple[int, int, int, int]]:
    """``(x0, y0, x1, y1)`` pixel boxes, one per word, within `gray` (one
    line's crop). y-extent is that crop's full ink height; the column
    profile is split on the legacy word rule (see ``_split_on_gaps``).
    Empty / ink-free crop -> ``[]``; a single ink blob -> one box."""
    if gray.ndim != 2 or gray.size == 0:
        return []
    h, w = gray.shape
    ink = gray < INK_LEVEL
    row_has_ink = ink.any(axis=1)
    if not row_has_ink.any():
        return []
    ys = np.flatnonzero(row_has_ink)
    y0, y1 = int(ys[0]), int(ys[-1]) + 1

    return [
        (max(0, sx0 - pad), max(0, y0 - pad), min(w, sx1 + 1 + pad), min(h, y1 + pad))
        for sx0, sx1 in _split_on_gaps(ink.any(axis=0), gap_factor=gap_factor, min_gap=min_gap)
    ]
