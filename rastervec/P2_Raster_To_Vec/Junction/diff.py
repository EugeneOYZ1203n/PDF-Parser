"""Step 10 (debug): compare the traced vectors back against the ink they
were traced from, pixel by pixel.

`diff_codes` rasterizes a component's polylines at 1 px
(`render_geometry`) and compares them with that component's binarized, text-removed ink, each
side dilated by `tol_px` so a 1-2 px width/position slop isn't flagged:

    0  background        (neither)
    1  matched           (ink covered by a vector)
    2  missed            (ink with no vector within tol -- a line the tracer lost)
    3  spurious          (vector with no ink within tol -- an invented line)

Fill outlines (`junction_test.fills`) count as covering everything they
enclose: they are drawn filled on the comparison canvas (stroked on the
returned `rendered` canvas).

Codes 2 are the raw material for a future missed-line recovery pass (e.g.
re-running LSD/Hough on just that mask). `codes_to_rgba` colours them for
the `vector_diff` debug layers; `rendered_to_rgba` does the plain
`vector_render` layer."""
from __future__ import annotations

import cv2
import numpy as np

BACKGROUND, MATCHED, MISSED, SPURIOUS = 0, 1, 2, 3

_PALETTE = np.array([
    (0, 0, 0, 0),          # background: transparent
    (160, 160, 160, 110),  # matched: faint gray
    (220, 38, 38, 255),    # missed: red
    (37, 99, 235, 255),    # spurious: blue
], dtype=np.uint8)


def _int_pts(points) -> np.ndarray:
    return np.round(np.asarray(points, float)).astype(np.int32).reshape(-1, 1, 2)


def render_geometry(shape: tuple[int, int], polylines, outlines=()) -> np.ndarray:
    """Polylines drawn at their own (rounded) width -- always 1.0 now -- plus
    fill outline rings at 1 px, onto a bool canvas."""
    canvas = np.zeros(shape[:2], np.uint8)
    for pl in polylines:
        cv2.polylines(canvas, [_int_pts(pl.points)], False, 255, max(1, int(round(pl.width))))
    for o in outlines:
        cv2.polylines(canvas, [_int_pts(r) for r in o.rings], True, 255, 1)
    return canvas > 0


def diff_codes(ink: np.ndarray, polylines, tol_px: int, outlines=()) -> "tuple[np.ndarray, np.ndarray]":
    """`(codes uint8, rendered bool)` in `ink`'s own pixel frame."""
    ink = np.asarray(ink, dtype=bool)
    rendered = render_geometry(ink.shape, polylines, outlines)
    covered = rendered
    if outlines:
        solid = np.zeros(ink.shape, np.uint8)
        for o in outlines:
            cv2.fillPoly(solid, [_int_pts(r) for r in o.rings], 255)  # holes stay open
        covered = rendered | (solid > 0)
    if tol_px > 0:
        k = np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8)
        rendered_near = cv2.dilate(covered.astype(np.uint8), k) > 0
        ink_near = cv2.dilate(ink.astype(np.uint8), k) > 0
    else:
        rendered_near, ink_near = covered, ink
    codes = np.zeros(ink.shape, dtype=np.uint8)
    codes[ink & rendered_near] = MATCHED
    codes[ink & ~rendered_near] = MISSED
    codes[covered & ~ink_near] = SPURIOUS
    return codes, rendered


def paste_codes(canvas: np.ndarray, codes: np.ndarray, x0: int, y0: int) -> None:
    """Paste `codes` into `canvas` at `(x0, y0)`, never overwriting a
    non-background canvas pixel with background."""
    region = canvas[y0:y0 + codes.shape[0], x0:x0 + codes.shape[1]]
    np.copyto(region, codes[:region.shape[0], :region.shape[1]], where=codes[:region.shape[0], :region.shape[1]] > 0)


def codes_to_rgba(codes: np.ndarray) -> np.ndarray:
    return _PALETTE[codes]


def rendered_to_rgba(rendered: np.ndarray) -> np.ndarray:
    """Vector render as black-on-transparent RGBA."""
    out = np.zeros((*rendered.shape, 4), dtype=np.uint8)
    out[rendered, 3] = 255
    return out
