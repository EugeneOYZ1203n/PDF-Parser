"""Skeleton -> node/edge graph -> strokes.

Originally ported from archive/scripts/testing_paper/chaining.py, now
array-based and linear in the number of skeleton pixels:

  1. junction pixels (8-neighbour degree >= 3) are grouped into 8-connected
     clusters, and each cluster becomes ONE node at its centroid. Before
     this, every junction pixel was its own node, and the 1-3 px chains
     between adjacent junction pixels were most of the output.
  2. The remaining (degree <= 2) pixels form simple paths and loops. Each is
     walked once via precomputed neighbour lists, and its end pixels are
     attached to the junction clusters they touch. Every edge extends to its
     nodes' centroids.
  3. Graph clean-up, all on the in-memory graph (no re-tracing):
       * barbs -- a leaf edge off a junction shorter than `barb_min_px`
         (Dosch 2000 Sec 2.3) -- are dropped, never taking a node below
         degree 2;
       * junction-zone collapse: two nodes joined by an edge shorter than
         the local stroke width (2 * distance transform) merge. This turns
         a thick X crossing, which the skeleton splits into two Ys, back
         into one node;
       * degree-2 nodes are dissolved (their two edges joined), and
         degree-1 nodes become free ends.
  4. `continue_through_junctions` -- good continuation: at every node, the
     most collinear pairs of incident edges (tangents measured just
     outside the junction zone) are joined, so a line crossing N others is
     one stroke rather than N+1.

All points are (x, y) pixel coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types_ import Graph, Point

# 8-neighbour offsets as (dy, dx)
_OFF8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_FREE = -1


def _length(pts) -> float:
    if len(pts) < 2:
        return 0.0
    p = np.asarray(pts, float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


# ---------------------------------------------------------------------------
# Mutable graph
# ---------------------------------------------------------------------------
@dataclass
class _Edge:
    pts: list[Point]          # interior points (node positions are added on demand)
    a: int = _FREE            # node at the start, or _FREE
    b: int = _FREE            # node at the end, or _FREE
    closed: bool = False      # node-less loop: pts[0] == pts[-1]

    def reversed(self) -> "_Edge":
        return _Edge(self.pts[::-1], self.b, self.a, self.closed)


@dataclass
class _G:
    pos: dict[int, Point] = field(default_factory=dict)
    width: dict[int, float] = field(default_factory=dict)
    edges: dict[int, _Edge] = field(default_factory=dict)
    inc: dict[int, list[int]] = field(default_factory=dict)  # node -> edge ids (twice for a self-loop)
    _next_edge: int = 0

    def add_edge(self, e: _Edge) -> int:
        eid = self._next_edge
        self._next_edge += 1
        self.edges[eid] = e
        for n in (e.a, e.b):
            if n != _FREE:
                self.inc[n].append(eid)
        return eid

    def remove_edge(self, eid: int) -> _Edge:
        e = self.edges.pop(eid)
        for n in (e.a, e.b):
            if n != _FREE:
                self.inc[n].remove(eid)
        return e

    def points(self, e: _Edge) -> list[Point]:
        out = list(e.pts)
        if e.a != _FREE:
            out.insert(0, self.pos[e.a])
        if e.b != _FREE:
            out.append(self.pos[e.b])
        return out

    def degree(self, n: int) -> int:
        return len(self.inc[n])

    def remove_node(self, n: int) -> None:
        self.pos.pop(n, None)
        self.width.pop(n, None)
        self.inc.pop(n, None)


# ---------------------------------------------------------------------------
# 1-2. extraction
# ---------------------------------------------------------------------------
def _neighbours(flat: np.ndarray, h: int, w: int) -> np.ndarray:
    """(n, 8) index into the sorted flat pixel indices `flat` of each
    pixel's 8 neighbours, -1 where there is none. Sparse (`searchsorted`),
    so cost scales with skeleton pixels, not crop area."""
    ys, xs = np.divmod(flat, w)
    out = np.full((flat.size, 8), -1, np.int64)
    last = flat.size - 1
    for k, (dy, dx) in enumerate(_OFF8):
        ny, nx = ys + dy, xs + dx
        q = ny * w + nx
        idx = np.minimum(np.searchsorted(flat, q), last)
        hit = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w) & (flat[idx] == q)
        out[hit, k] = idx[hit]
    return out


def _extract(sk: np.ndarray, dt: "np.ndarray | None") -> _G:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    h, w = sk.shape
    g = _G()
    flat = np.flatnonzero(sk)
    if flat.size == 0:
        return g
    ys, xs = np.divmod(flat, w)
    nb = _neighbours(flat, h, w)
    is_j = (nb >= 0).sum(axis=1) >= 3

    # junction pixels -> 8-connected clusters -> one node each (1..k)
    jlab = np.zeros(flat.size, np.int64)
    jidx = np.flatnonzero(is_j)
    if jidx.size:
        local = np.full(flat.size, -1, np.int64)
        local[jidx] = np.arange(jidx.size)
        jn = nb[jidx]
        r, c = np.nonzero(jn >= 0)
        other = local[jn[r, c]]
        keep = other >= 0
        adj = coo_matrix((np.ones(int(keep.sum())), (r[keep], other[keep])), shape=(jidx.size, jidx.size))
        n_j, comp = connected_components(adj, directed=False)
        jlab[jidx] = comp + 1
        cnt = np.bincount(comp, minlength=n_j)
        cx = np.bincount(comp, xs[jidx], minlength=n_j) / cnt
        cy = np.bincount(comp, ys[jidx], minlength=n_j) / cnt
        for k in range(n_j):
            g.pos[k + 1] = (float(cx[k]), float(cy[k]))
            g.inc[k + 1] = []
            g.width[k + 1] = _width_at(dt, g.pos[k + 1])

    # edge pixels: compact ids + up to two edge-pixel neighbours each
    eidx = np.flatnonzero(~is_j)
    n_e = eidx.size
    if n_e == 0:
        return g
    elocal = np.full(flat.size, -1, np.int64)
    elocal[eidx] = np.arange(n_e)
    en = nb[eidx]
    nb_arr = np.where(en >= 0, elocal[en], -1)
    jn_arr = np.where(en >= 0, jlab[en], 0)
    nbrs = _rows(nb_arr)
    edeg = (nb_arr >= 0).sum(axis=1)
    xs_l, ys_l = xs[eidx].tolist(), ys[eidx].tolist()

    def junctions_of(i: int) -> list[int]:
        row = jn_arr[i]
        return sorted(set(row[row > 0].tolist()))

    visited = bytearray(n_e)

    def walk(start: int) -> list[int]:
        order = [start]
        visited[start] = 1
        prev, cur = -1, start
        while True:
            nxt = -1
            for q in nbrs[cur]:
                if q != prev and not visited[q]:
                    nxt = q
                    break
            if nxt < 0:
                return order
            visited[nxt] = 1
            order.append(nxt)
            prev, cur = cur, nxt

    def add_path(order: list[int]) -> None:
        pts = [(float(xs_l[i]), float(ys_l[i])) for i in order]
        ja, jb = junctions_of(order[0]), junctions_of(order[-1])
        a = ja[0] if ja else _FREE
        if len(order) == 1:
            b = next((j for j in ja if j != a), _FREE)
        else:
            b = next((j for j in jb if j != a), jb[0] if jb else _FREE)
        if a == _FREE and b == _FREE and len(order) < 2:
            return  # isolated single pixel
        g.add_edge(_Edge(pts, a, b))

    # paths start at their ends (edge-degree <= 1) ...
    for i in np.flatnonzero(edeg <= 1).tolist():
        if not visited[i]:
            add_path(walk(i))
    # ... what is left are loops with no junction and no end on them
    for i in range(n_e):
        if not visited[i]:
            order = walk(i)
            if len(order) >= 3:
                pts = [(float(xs_l[j]), float(ys_l[j])) for j in order]
                g.add_edge(_Edge(pts + [pts[0]], closed=True))
    return g


def _rows(nb_arr: np.ndarray) -> list[list[int]]:
    """Per-row list of the non-negative entries, vectorised (rows hold at
    most 2 of them -- every edge pixel has degree <= 2)."""
    srt = np.sort(nb_arr, axis=1)[:, ::-1]  # non-negatives first, descending
    first, second = srt[:, 0].tolist(), srt[:, 1].tolist()
    return [[] if a < 0 else [a] if b < 0 else [a, b] for a, b in zip(first, second)]


def _width_at(dt: "np.ndarray | None", p: Point) -> float:
    """Local stroke width at `p`: 2x the ink distance transform, >= 1."""
    if dt is None:
        return 1.0
    h, w = dt.shape
    x = min(max(int(round(p[0])), 0), w - 1)
    y = min(max(int(round(p[1])), 0), h - 1)
    return max(1.0, 2.0 * float(dt[y, x]))


# ---------------------------------------------------------------------------
# 3. clean-up
# ---------------------------------------------------------------------------
def _normalize(g: _G) -> None:
    """Dissolve degree-0/1/2 nodes until every node has degree >= 3."""
    stack = list(g.pos)
    while stack:
        n = stack.pop()
        if n not in g.pos:
            continue
        d = g.degree(n)
        if d >= 3:
            continue
        if d == 0:
            g.remove_node(n)
        elif d == 1:
            eid = g.inc[n][0]
            e = g.remove_edge(eid)
            if e.b != n:
                e = e.reversed()
            g.add_edge(_Edge(e.pts + [g.pos[n]], e.a, _FREE))
            g.remove_node(n)
        else:  # d == 2
            e1id, e2id = g.inc[n]
            if e1id == e2id:  # a self-loop and nothing else: a closed ring through n
                e = g.remove_edge(e1id)
                p = g.pos[n]
                g.add_edge(_Edge([p] + e.pts + [p], closed=True))
                g.remove_node(n)
                continue
            e1, e2 = g.remove_edge(e1id), g.remove_edge(e2id)
            if e1.b != n:
                e1 = e1.reversed()
            if e2.a != n:
                e2 = e2.reversed()
            g.add_edge(_Edge(e1.pts + [g.pos[n]] + e2.pts, e1.a, e2.b))
            g.remove_node(n)
            stack.extend(x for x in (e1.a, e2.b) if x != _FREE)


def _prune_barbs(g: _G, min_len: float) -> None:
    """Drop leaf edges (one free end, the other on a junction) shorter than
    `min_len`, shortest first, never taking a node below degree 3 within
    one pass -- a short *isolated* stroke (both ends free) is a real
    primitive and is kept."""
    for _ in range(10):
        barbs = []
        for eid, e in g.edges.items():
            if (e.a == _FREE) != (e.b == _FREE):
                ln = _length(g.points(e))
                if ln < min_len:
                    barbs.append((ln, eid))
        removed = 0
        for _ln, eid in sorted(barbs):
            e = g.edges[eid]
            n = e.a if e.a != _FREE else e.b
            if g.degree(n) >= 3:
                g.remove_edge(eid)
                removed += 1
        if not removed:
            return
        _normalize(g)


def _collapse_junction_zones(g: _G, factor: float) -> None:
    """Merge two nodes joined by an edge shorter than `factor` x the local
    stroke width. Self-loops left behind that are smaller than the stroke
    itself (a pinhole bubble) are dropped."""
    if factor <= 0:
        return
    cand = sorted(
        (_length(g.points(e)), eid) for eid, e in g.edges.items()
        if e.a != _FREE and e.b != _FREE and e.a != e.b
    )
    for _ln, eid in cand:
        e = g.edges.get(eid)
        if e is None or e.a == e.b:
            continue
        a, b = e.a, e.b
        if _length(g.points(e)) >= factor * max(g.width[a], g.width[b]):
            continue
        g.remove_edge(eid)
        pa, pb = g.pos[a], g.pos[b]
        g.pos[a] = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
        g.width[a] = max(g.width[a], g.width[b])
        for oid in list(dict.fromkeys(g.inc[b])):
            o = g.remove_edge(oid)
            g.add_edge(_Edge(o.pts, a if o.a == b else o.a, a if o.b == b else o.b, o.closed))
        g.remove_node(b)
    for eid, e in list(g.edges.items()):
        if e.a != _FREE and e.a == e.b and _length(g.points(e)) < np.pi * g.width[e.a]:
            g.remove_edge(eid)
    _normalize(g)


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------
def build_graph(
    skeleton: np.ndarray, barb_min_px: float = 0.0, *,
    dt: "np.ndarray | None" = None, junction_merge_factor: float = 0.0,
) -> Graph:
    """Skeleton -> cleaned `Graph` (see module docstring). `dt` is the ink's
    distance transform, used for the local stroke width at each node
    (junction-zone collapse, continuation tangents)."""
    g = _extract(skeleton.astype(bool), dt)
    _normalize(g)
    if barb_min_px > 0:
        _prune_barbs(g, barb_min_px)
    _collapse_junction_zones(g, junction_merge_factor)
    return _to_graph(g)


def _to_graph(g: _G) -> Graph:
    ids = sorted(g.pos)
    index = {n: i for i, n in enumerate(ids)}
    chains, ends = [], []
    for e in g.edges.values():
        pts = g.points(e)
        if len(pts) < 2:
            continue
        chains.append(pts)
        ends.append((index.get(e.a, _FREE), index.get(e.b, _FREE)))
    return Graph(
        nodes=[g.pos[n] for n in ids], chains=chains, ends=ends,
        node_widths=[g.width[n] for n in ids],
    )


def _end_direction(pts_out: list[Point], skip: float, span: float) -> "np.ndarray | None":
    """Unit direction of a chain leaving its node (`pts_out[0]`), from the
    points at arc length [skip, skip + span] -- outside the junction zone,
    where the skeleton is distorted. A chain shorter than that uses its far
    end."""
    p = np.asarray(pts_out, float)
    if len(p) < 2:
        return None
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    sel = (s >= skip) & (s <= skip + span)
    target = p[sel].mean(axis=0) if sel.any() else p[-1]
    d = target - p[0]
    n = float(np.hypot(*d))
    return d / n if n > 1e-9 else None


def continue_through_junctions(graph: Graph, max_deg: float = 20.0, span_px: float = 10.0) -> list[list[Point]]:
    """Join chains that continue straight through a junction into one
    stroke. At each node every pair of incident chain ends is scored by how
    far `d_i` is from `-d_j`; pairs within `max_deg` are taken greedily,
    best first, each end at most once. Returns the strokes (closed ones
    repeat their first point)."""
    chains, ends = graph.chains, graph.ends
    at_node: dict[int, list[tuple[int, int]]] = {}
    for ci, (a, b) in enumerate(ends):
        if a != _FREE:
            at_node.setdefault(a, []).append((ci, 0))
        if b != _FREE:
            at_node.setdefault(b, []).append((ci, 1))

    cos_max = np.cos(np.radians(max_deg))
    partner: dict[tuple[int, int], tuple[int, int]] = {}
    for n, incident in at_node.items():
        skip = graph.node_widths[n] if graph.node_widths else 1.0
        dirs = []
        for ci, side in incident:
            pts = chains[ci] if side == 0 else chains[ci][::-1]
            dirs.append(_end_direction(pts, skip, span_px))
        pairs = []
        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                if dirs[i] is None or dirs[j] is None:
                    continue
                straight = -float(np.dot(dirs[i], dirs[j]))  # 1 = perfectly straight through
                if straight >= cos_max:
                    pairs.append((-straight, i, j))
        used: set[int] = set()
        for _s, i, j in sorted(pairs):
            if i in used or j in used:
                continue
            used.update((i, j))
            partner[incident[i]] = incident[j]
            partner[incident[j]] = incident[i]

    visited = [False] * len(chains)

    def oriented(ci: int, side: int) -> list[Point]:
        return chains[ci] if side == 0 else chains[ci][::-1]

    def walk(ci: int, side: int) -> list[Point]:
        visited[ci] = True
        pts = list(oriented(ci, side))
        cur = (ci, 1 - side)
        while cur in partner:
            nci, nside = partner[cur]
            if visited[nci]:
                break
            visited[nci] = True
            pts.extend(oriented(nci, nside)[1:])
            cur = (nci, 1 - nside)
        return pts

    strokes = []
    for ci in range(len(chains)):
        if visited[ci]:
            continue
        if (ci, 0) not in partner:
            strokes.append(walk(ci, 0))
        elif (ci, 1) not in partner:
            strokes.append(walk(ci, 1))
    for ci in range(len(chains)):  # every end paired: closed through junctions
        if not visited[ci]:
            strokes.append(walk(ci, 0))
    return [s for s in strokes if len(s) >= 2]
