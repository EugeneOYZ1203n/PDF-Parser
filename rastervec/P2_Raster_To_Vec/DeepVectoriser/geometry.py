"""Pure numpy stroke geometry shared by inference (`inference.py`, `adapter.py`)
and training (`train_data.py`).

A *stroke* is a chain of cubic Bezier pieces, `np.ndarray (K, 4, 2)` float --
piece k is `(p0, c1, c2, p3)` and piece k's `p3` is piece k+1's `p0` (the
paper's "partial format": each curve starts where the previous one ended).
A straight `"l"` is the cubic with its controls at 1/3 and 2/3 of the chord.
Coordinates are pixels of whatever frame the caller works in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Stroke = np.ndarray  # (K, 4, 2)


def line_to_cubic(a, b) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.stack([a, a + (b - a) / 3.0, a + (b - a) * 2.0 / 3.0, b])


def bezier_eval(piece: np.ndarray, t) -> np.ndarray:
    """Points `(len(t), 2)` of one cubic at parameters `t`."""
    t = np.asarray(t, float)[:, None]
    mt = 1.0 - t
    p0, c1, c2, p3 = piece
    return mt ** 3 * p0 + 3 * mt ** 2 * t * c1 + 3 * mt * t ** 2 * c2 + t ** 3 * p3


def sample_stroke(stroke: Stroke, per_piece: int = 16) -> np.ndarray:
    """Polyline `(N, 2)` through every piece (shared joints not repeated)."""
    if len(stroke) == 0:
        return np.zeros((0, 2))
    t = np.linspace(0.0, 1.0, per_piece + 1)
    pts = [bezier_eval(stroke[0], t)]
    for piece in stroke[1:]:
        pts.append(bezier_eval(piece, t)[1:])
    return np.concatenate(pts, axis=0)


def stroke_length(stroke: Stroke) -> float:
    pts = sample_stroke(stroke, 8)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def split_bezier(piece: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """The sub-cubic of `piece` over `[t0, t1]` (de Casteljau)."""
    def split_at(p: np.ndarray, t: float):
        a = p[0] + (p[1] - p[0]) * t
        b = p[1] + (p[2] - p[1]) * t
        c = p[2] + (p[3] - p[2]) * t
        ab = a + (b - a) * t
        bc = b + (c - b) * t
        m = ab + (bc - ab) * t
        return np.stack([p[0], a, ab, m]), np.stack([m, bc, c, p[3]])

    piece = np.asarray(piece, float)
    if t1 >= 1.0:
        right = piece
    else:
        right = split_at(piece, t1)[0]
    if t0 <= 0.0:
        return right
    # rescale t0 into the left part's parameter range
    return split_at(right, t0 / t1 if t1 > 0 else 0.0)[1]


def reverse_stroke(stroke: Stroke) -> Stroke:
    return np.ascontiguousarray(stroke[::-1, ::-1, :])


def _lex_key(p) -> tuple[float, float]:
    # rounded so sub-pixel noise doesn't flip the order of a vertical line
    return (round(float(p[0]), 1), round(float(p[1]), 1))


def orient_stroke(stroke: Stroke) -> Stroke:
    """Start the stroke at its lexicographically smaller (x, y) endpoint --
    the paper's "sort the endpoints in each ground-truth stroke"."""
    if _lex_key(stroke[-1, 3]) < _lex_key(stroke[0, 0]):
        return reverse_stroke(stroke)
    return stroke


def sort_strokes(strokes: list[Stroke]) -> list[Stroke]:
    """Orient every stroke, then order the list lexicographically by
    (start, end) -- the paper's GT feature-list order."""
    oriented = [orient_stroke(s) for s in strokes]
    return sorted(oriented, key=lambda s: (_lex_key(s[0, 0]), _lex_key(s[-1, 3])))


def split_long(stroke: Stroke, max_prims: int) -> list[Stroke]:
    return [stroke[i:i + max_prims] for i in range(0, len(stroke), max_prims)]


def piece_is_flat(piece: np.ndarray, tol: float) -> bool:
    """Both controls within `tol` of the chord (or a degenerate chord)."""
    p0, c1, c2, p3 = np.asarray(piece, float)
    d = p3 - p0
    n = float(np.hypot(*d))
    if n < 1e-9:
        return max(np.hypot(*(c1 - p0)), np.hypot(*(c2 - p0))) <= tol
    for c in (c1, c2):
        v = c - p0
        if abs(d[0] * v[1] - d[1] * v[0]) / n > tol:
            return False
        # a control far outside the chord's span also bends the curve
        s = float(np.dot(v, d)) / (n * n)
        if s < -tol / n or s > 1 + tol / n:
            return False
    return True


# ---------------------------------------------------------------------------
# Clipping to a rectangle
# ---------------------------------------------------------------------------
def rect_predicate(rect):
    """`inside(points (N, 2)) -> bool (N,)` for `rect = (x0, y0, x1, y1)`."""
    x0, y0, x1, y1 = rect

    def inside(pts: np.ndarray) -> np.ndarray:
        return (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    return inside


def _refine(piece, inside, t_in: float, t_out: float, iters: int = 12) -> float:
    """Bisect for the boundary between an inside and an outside parameter."""
    for _ in range(iters):
        mid = 0.5 * (t_in + t_out)
        if inside(bezier_eval(piece, [mid]))[0]:
            t_in = mid
        else:
            t_out = mid
    return t_in


def clip_stroke_by(stroke: Stroke, inside, samples: int = 32, min_len: float = 1.5) -> list[Stroke]:
    """The parts of `stroke` where `inside(points) -> bool mask` holds, each
    a stroke of its own (leaving and re-entering splits it). Pieces are cut
    at the boundary (sampled, then bisected); parts shorter than `min_len`
    are dropped."""
    out: list[Stroke] = []
    cur: list[np.ndarray] = []
    t = np.linspace(0.0, 1.0, samples + 1)

    def flush():
        if cur:
            s = np.stack(cur)
            if stroke_length(s) >= min_len:
                out.append(s)
            cur.clear()

    for piece in stroke:
        mask = inside(bezier_eval(piece, t))
        if mask.all():
            cur.append(np.asarray(piece, float))
            continue
        if not mask.any():
            flush()
            continue
        i = 0
        while i < len(t):
            if not mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(t) and mask[j + 1]:
                j += 1
            ta = 0.0 if i == 0 else _refine(piece, inside, t[i], t[i - 1])
            tb = 1.0 if j == len(t) - 1 else _refine(piece, inside, t[j], t[j + 1])
            if i != 0:
                flush()  # entered from outside: a new stroke starts here
            if tb > ta:
                cur.append(split_bezier(piece, ta, tb))
            if j != len(t) - 1:
                flush()  # leaves before the piece ends
            i = j + 1
    flush()
    return out


def clip_stroke(stroke: Stroke, rect, samples: int = 32, min_len: float = 1.5) -> list[Stroke]:
    """`clip_stroke_by` against an axis-aligned rect `(x0, y0, x1, y1)`."""
    return clip_stroke_by(stroke, rect_predicate(rect), samples, min_len)


# ---------------------------------------------------------------------------
# Tile grid + merge
# ---------------------------------------------------------------------------
def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tile_grid(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Top-left `(x0, y0)` of every tile covering an `h`x`w` image."""
    return [(x, y) for y in tile_starts(h, tile, overlap) for x in tile_starts(w, tile, overlap)]


def core_rect(x0: int, y0: int, size: int, h: int, w: int, overlap: int) -> tuple[float, float, float, float]:
    """The part of a tile it "owns": its rect shrunk by half the overlap on
    every side that has a neighbour (image borders keep their full extent),
    so neighbouring cores tile the image without gaps or double ownership."""
    half = overlap / 2.0
    cx0 = x0 + half if x0 > 0 else float("-inf")
    cy0 = y0 + half if y0 > 0 else float("-inf")
    cx1 = x0 + size - half if x0 + size < w else float("inf")
    cy1 = y0 + size - half if y0 + size < h else float("inf")
    return (cx0, cy0, cx1, cy1)


@dataclass
class TileStrokes:
    """One tile's predicted strokes, already in the layer's pixel frame."""
    tile_id: int
    core: tuple[float, float, float, float]
    strokes: list[Stroke]


def merge_tiles(tiles: list[TileStrokes], snap_px: float) -> list[Stroke]:
    """Overlapping tiles -> one stroke set:

    1. ownership -- every tile's strokes are clipped to the tile's core, so
       the overlap's duplicate copy is cut away and a stroke crossing a seam
       ends exactly on it, from both sides;
    2. seam snapping -- stroke ends from *different* tiles within `snap_px`
       of each other are moved to their mean (each end's adjacent control
       point moves with it, so the end tangent is kept);
    3. joining -- where exactly two stroke ends from different tiles met in
       step 2, the two strokes are concatenated into one.
    """
    kept: list[tuple[int, Stroke]] = []
    for t in tiles:
        for s in t.strokes:
            for part in clip_stroke(s, t.core, min_len=1.0):
                kept.append((t.tile_id, np.array(part, float, copy=True)))
    if not kept or snap_px <= 0:
        return [s for _, s in kept]

    # endpoint records: (stroke index, 0=start / 1=end)
    ends = [(i, e) for i in range(len(kept)) for e in (0, 1)]
    pts = np.array([kept[i][1][0, 0] if e == 0 else kept[i][1][-1, 3] for i, e in ends])
    parent = list(range(len(ends)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cell = max(snap_px, 1e-6)
    grid: dict[tuple[int, int], list[int]] = {}
    for k, p in enumerate(pts):
        grid.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(k)
    for k, p in enumerate(pts):
        gx, gy = int(p[0] // cell), int(p[1] // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if j <= k:
                        continue
                    if kept[ends[j][0]][0] == kept[ends[k][0]][0]:
                        continue  # same tile: the model's own topology, leave it
                    if np.hypot(*(pts[j] - p)) <= snap_px:
                        parent[find(j)] = find(k)
    groups: dict[int, list[int]] = {}
    for k in range(len(ends)):
        groups.setdefault(find(k), []).append(k)

    pairs: list[tuple[int, int]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        mean = pts[members].mean(axis=0)
        for k in members:
            i, e = ends[k]
            s = kept[i][1]
            if e == 0:
                d = mean - s[0, 0]
                s[0, 0] += d
                s[0, 1] += d
            else:
                d = mean - s[-1, 3]
                s[-1, 3] += d
                s[-1, 2] += d
        if len(members) == 2 and ends[members[0]][0] != ends[members[1]][0]:
            pairs.append((members[0], members[1]))

    # join strokes pairwise along the snapped seams (chains of any length)
    strokes = [s for _, s in kept]
    link: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b in pairs:
        link[ends[a]] = ends[b]
        link[ends[b]] = ends[a]
    used = [False] * len(strokes)
    out: list[Stroke] = []

    def walk(i: int, entry_end: int) -> list[Stroke]:
        """Follow the chain from stroke i entered at `entry_end`."""
        chain: list[Stroke] = []
        while True:
            used[i] = True
            s = strokes[i] if entry_end == 0 else reverse_stroke(strokes[i])
            chain.append(s)
            exit_end = 1 - entry_end
            nxt = link.get((i, exit_end))
            if nxt is None or used[nxt[0]]:
                return chain
            i, entry_end = nxt

    for i in range(len(strokes)):
        if used[i]:
            continue
        # start from a chain end (an end with no link), unless it's a loop
        start, entry = i, 0
        cur, cur_entry = i, 0
        seen = {i}
        while True:
            back = link.get((cur, cur_entry))
            if back is None or back[0] in seen or used[back[0]]:
                start, entry = cur, cur_entry
                break
            cur, cur_entry = back[0], 1 - back[1]
            seen.add(cur)
        out.append(np.concatenate(walk(start, entry), axis=0))
    return out
