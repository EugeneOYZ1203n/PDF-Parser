"""Post-processing in the dual-contouring domain (paper Sec. 7), on a sparse
graph so a whole page never needs dense cell arrays:

  refine_flags      edge-flag refinement (Fig. 5): the occupancy map (cells
                    with an X or Y flag) is skeletonized (Zhang-Suen) and
                    flags outside the skeleton are dropped -- removes the
                    parallel, redundant flag lines. Applied only where the
                    occupancy is 2 cells thick (ours: a thin staircase is
                    already a line, and thinning would break it). Dense, per
                    tile.
  build_graph       dual contouring: one vertex per cell at its predicted
                    position, joined across every flagged edge.
  repair_breaks     broken lines (Fig. 6): skeletonizing a staircase drops
                    its corner cells' flags; the paper finds the four broken
                    patterns with 3x3 kernels and restores the missing flag.
                    We connect the two open ends of such a break directly --
                    degree-1 vertices in adjacent cells, 4-neighbours first,
                    then diagonal (ours, same effect).
  refine_topology   topology surgery with the under-sampling map and the
                    keypoints (Fig. 7): per connected under-sampled region,
                    its vertices and inner edges are removed and the
                    truncated edges reconnected -- to a keypoint inside the
                    region if there is one; else, with valence 4, in pairs
                    by closest tangent direction (parallel lines); else to
                    the mean of the removed vertices.
  downsample        DC downsampling (Fig. 10): k x k blocks of cells become
                    one coarse vertex -- the block's single open endpoint,
                    else the mean of its vertices -- unless the block is
                    under-sampled or holds more than one separate piece of
                    line, which keep their fine geometry.
  split_crossings   (ours) a junction -- a vertex of degree >= 3 -- is split
                    into straight-through pairs of branches (directions
                    measured a few vertices out), the rest ending there, so
                    crossing / T-ing lines come out as straight strokes
                    rather than whichever arms the longest-path grouping
                    happens to join;
  group_lines       line grouping: repeatedly take the longest shortest path
                    through the graph and remove it (the paper computes
                    all-pairs shortest paths; we use the double-Dijkstra
                    farthest pair per component -- exact on trees -- for
                    speed).
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np

from .config import GROUP_EXACT_MAX_NODES, SR

Node = object  # a cell (a, b) tuple, or ("new", i) for vertices added by surgery


@dataclass
class DCGraph:
    pos: dict = field(default_factory=dict)
    adj: dict = field(default_factory=dict)
    _new: int = 0

    def add_node(self, node, p) -> None:
        self.pos[node] = np.asarray(p, float)
        self.adj.setdefault(node, set())

    def new_node(self, p):
        node = ("new", self._new)
        self._new += 1
        self.add_node(node, p)
        return node

    def add_edge(self, u, v) -> None:
        if u == v:
            return
        self.adj.setdefault(u, set()).add(v)
        self.adj.setdefault(v, set()).add(u)

    def remove_node(self, u) -> None:
        for v in self.adj.pop(u, ()):
            self.adj[v].discard(u)
        self.pos.pop(u, None)

    def degree(self, u) -> int:
        return len(self.adj.get(u, ()))

    def edges(self):
        for u, vs in self.adj.items():
            for v in vs:
                if _key(u) < _key(v):
                    yield u, v

    def segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.pos[u], self.pos[v]) for u, v in self.edges()]


def _key(n) -> tuple:
    """A total order over the node kinds: cells (a, b), surgery vertices
    ("new", i), downsampled blocks ("blk", (a, b))."""
    if isinstance(n[0], (int, np.integer)):
        return (0, int(n[0]), int(n[1]))
    if n[0] == "new":
        return (1, int(n[1]), 0)
    return (2, int(n[1][0]), int(n[1][1]))


def cell_centre(cell) -> np.ndarray:
    return (np.asarray(cell, float) + 0.5) / SR


# ---------------------------------------------------------------------------
# Flags -> graph
# ---------------------------------------------------------------------------
def refine_flags(er: np.ndarray, eb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fig. 5: drop flags of cells off the skeleton of the occupancy map --
    only where the occupancy is thick (cells in a fully occupied 2 x 2
    block, i.e. parallel redundant flag lines). A thin staircase is already
    a valid line, and Zhang-Suen would delete its corner cells (ours)."""
    from skimage.morphology import skeletonize

    occ = er | eb
    if not occ.any():
        return er, eb
    full = occ[:-1, :-1] & occ[1:, :-1] & occ[:-1, 1:] & occ[1:, 1:]
    thick = np.zeros_like(occ)
    thick[:-1, :-1] |= full
    thick[1:, :-1] |= full
    thick[:-1, 1:] |= full
    thick[1:, 1:] |= full
    if not thick.any():
        return er, eb
    keep = ~thick | skeletonize(occ)
    return er & keep, eb & keep


def build_graph(right_cells, bottom_cells, vertex_pos: dict) -> DCGraph:
    """`right_cells` / `bottom_cells`: `(N, 2)` cells (a, b) whose right /
    bottom edge is flagged; `vertex_pos`: cell -> px position."""
    g = DCGraph()

    def node(cell):
        if cell not in g.pos:
            g.add_node(cell, vertex_pos.get(cell, cell_centre(cell)))
        return cell

    for a, b in np.asarray(right_cells, np.int64).reshape(-1, 2):
        g.add_edge(node((int(a), int(b))), node((int(a) + 1, int(b))))
    for a, b in np.asarray(bottom_cells, np.int64).reshape(-1, 2):
        g.add_edge(node((int(a), int(b))), node((int(a), int(b) + 1)))
    return g


def repair_breaks(g: DCGraph) -> int:
    """Join open ends in adjacent cells (a staircase whose step flag was
    lost to skeletonizing). Returns the number of edges added."""
    added = 0
    for offsets in (((1, 0), (-1, 0), (0, 1), (0, -1)), ((1, 1), (1, -1), (-1, 1), (-1, -1))):
        ends = sorted(n for n in g.adj if g.degree(n) == 1 and isinstance(n[0], (int, np.integer)))
        end_set = set(ends)
        for a, b in ends:
            if g.degree((a, b)) != 1:
                continue
            for da, db in offsets:
                other = (a + da, b + db)
                if other in end_set and g.degree(other) == 1 and other not in g.adj[(a, b)]:
                    g.add_edge((a, b), other)
                    added += 1
                    break
    return added


# ---------------------------------------------------------------------------
# Topology refinement (Fig. 7)
# ---------------------------------------------------------------------------
def _components(cells: set) -> list[set]:
    """8-connected components of a cell set."""
    left = set(cells)
    out = []
    while left:
        seed = left.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            a, b = stack.pop()
            for da, db in itertools.product((-1, 0, 1), repeat=2):
                c = (a + da, b + db)
                if c in left:
                    left.remove(c)
                    comp.add(c)
                    stack.append(c)
        out.append(comp)
    return out


def _dilate(cells: set) -> set:
    return {(a + da, b + db) for a, b in cells for da, db in itertools.product((-1, 0, 1), repeat=2)}


def _unit(v) -> np.ndarray:
    n = float(np.hypot(*v))
    return np.asarray(v, float) / n if n > 1e-12 else np.zeros(2)


def refine_topology(g: DCGraph, usm_cells: set, keypoints: np.ndarray) -> list[set]:
    """In place. `keypoints (K, 2)` px. Returns the regions handled."""
    kp = np.asarray(keypoints, float).reshape(-1, 2)
    kp_cells = [tuple(c) for c in np.floor(kp * SR + 1e-6).astype(np.int64)]
    regions = []
    for region in _components(set(usm_cells)):
        inside = [n for n in region if n in g.pos]
        if not inside:
            continue
        removed_pos = [g.pos[n] for n in inside]
        truncated = []  # (outside vertex, direction pointing into the region)
        for n in inside:
            for v in g.adj[n]:
                if v not in region:
                    prev = [w for w in g.adj[v] if w not in region]
                    t = _unit(g.pos[v] - g.pos[prev[0]]) if prev else _unit(g.pos[n] - g.pos[v])
                    truncated.append((v, t))
        # an open end right beside the region is a branch the network lost
        # into it -- reconnect it too (ours)
        for v in _dilate(region) - region:
            if v in g.adj and g.degree(v) == 1:
                (w,) = g.adj[v]
                if w not in region:
                    truncated.append((v, _unit(g.pos[v] - g.pos[w])))
        if not truncated:
            # nothing leaves the region (e.g. two whole lines closer than a
            # cell): no topology to reconnect -- keep its geometry (ours)
            continue
        for n in inside:
            g.remove_node(n)
        regions.append(region)
        outs = list(dict.fromkeys(v for v, _ in truncated))
        tangent = {v: t for v, t in truncated}
        near = _dilate(region)
        hits = [i for i, c in enumerate(kp_cells) if c in near]
        if hits:
            centre = np.mean(removed_pos, axis=0)
            best = min(hits, key=lambda i: float(np.hypot(*(kp[i] - centre))))
            hub = g.new_node(kp[best])
            for v in outs:
                g.add_edge(v, hub)
        elif len(outs) == 4:
            pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]

            def cost(pairing) -> float:
                # good pair: tangents opposite (both point into the region)
                return sum(1.0 + float(np.dot(tangent[outs[i]], tangent[outs[j]])) for i, j in pairing)

            for i, j in min(pairings, key=cost):
                g.add_edge(outs[i], outs[j])
        else:
            hub = g.new_node(np.mean(removed_pos, axis=0))
            for v in outs:
                g.add_edge(v, hub)
    return regions


# ---------------------------------------------------------------------------
# DC downsampling (Fig. 10)
# ---------------------------------------------------------------------------
def downsample(g: DCGraph, k: int, protected_cells: set) -> DCGraph:
    if k <= 1:
        return g
    blocks: dict[tuple, list] = {}
    for n in g.pos:
        if isinstance(n[0], (int, np.integer)):
            blocks.setdefault((n[0] // k, n[1] // k), []).append(n)
    protected = {(a // k, b // k) for a, b in protected_cells}
    rep: dict = {}
    out = DCGraph(_new=g._new)
    for blk, members in blocks.items():
        if blk in protected or _pieces(g, set(members)) > 1:
            for n in members:
                rep[n] = n
                out.add_node(n, g.pos[n])
            continue
        ends = [n for n in members if g.degree(n) == 1]
        p = g.pos[ends[0]] if len(ends) == 1 else np.mean([g.pos[n] for n in members], axis=0)
        coarse = ("blk", blk)
        out.add_node(coarse, p)
        for n in members:
            rep[n] = coarse
    for n in g.pos:
        if n not in rep:  # surgery vertices stay as they are
            rep[n] = n
            out.add_node(n, g.pos[n])
    for u, v in g.edges():
        out.add_edge(rep[u], rep[v])
    return out


def _pieces(g: DCGraph, members: set) -> int:
    """Connected pieces of the block's vertices using only in-block edges."""
    left = set(members)
    count = 0
    while left:
        count += 1
        stack = [left.pop()]
        while stack:
            u = stack.pop()
            for v in g.adj[u]:
                if v in left:
                    left.remove(v)
                    stack.append(v)
    return count


# ---------------------------------------------------------------------------
# Crossings (ours) + line grouping
# ---------------------------------------------------------------------------
def _branch_dir(g: DCGraph, u, v, steps: int = 4) -> np.ndarray:
    """Direction from junction `u` along the branch through `v`, measured a
    few vertices out (single cell steps are too jittery to pair on)."""
    prev, cur = u, v
    for _ in range(steps - 1):
        nxt = [w for w in g.adj[cur] if w != prev]
        if len(nxt) != 1:
            break
        prev, cur = cur, nxt[0]
    return _unit(g.pos[cur] - g.pos[u])


def split_crossings(g: DCGraph, straight_cos: float = -0.8) -> int:
    """In place: every vertex of degree >= 3 is split into copies at the same
    position -- one per pair of branches that continue nearly straight
    through it (direction dot product below `straight_cos`, best pairs
    first), plus one per remaining branch, which then simply ends there.
    Returns the number of vertices split."""
    count = 0
    for u in [n for n in list(g.adj) if g.degree(n) >= 3]:
        if u not in g.adj:
            continue
        nbrs = list(g.adj[u])
        dirs = {v: _branch_dir(g, u, v) for v in nbrs}
        pairs = sorted(((float(np.dot(dirs[a], dirs[b])), a, b) for a, b in itertools.combinations(nbrs, 2)),
                       key=lambda x: x[0])
        used: set = set()
        chosen = []
        for d, a, b in pairs:
            if d < straight_cos and a not in used and b not in used:
                chosen.append((a, b))
                used.update((a, b))
        if not chosen:
            continue
        p = g.pos[u]
        g.remove_node(u)
        for a, b in chosen:
            c = g.new_node(p)
            g.add_edge(a, c)
            g.add_edge(c, b)
        for v in nbrs:
            if v not in used:
                g.add_edge(v, g.new_node(p))
        count += 1
    return count


def _dijkstra(adj: dict, pos: dict, src):
    dist = {src: 0.0}
    prev = {}
    heap = [(0.0, 0, src)]
    tie = itertools.count(1)
    while heap:
        d, _, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v in adj[u]:
            nd = d + float(np.hypot(*(pos[u] - pos[v])))
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, next(tie), v))
    far = max(dist, key=dist.get)
    return far, dist, prev


def group_lines(g: DCGraph, min_len: float = 0.0, exact_max_nodes: int = GROUP_EXACT_MAX_NODES) -> list[np.ndarray]:
    """Polylines `(N, 2)` covering every edge once. Per connected component:
    longest shortest path first, repeatedly (the paper's grouping) -- or, for
    a component above `exact_max_nodes` vertices, where that is quadratic,
    its maximal chains through degree-2 vertices (ours; junctions are
    already split into straight-through pairs by `split_crossings`). Paths
    that meet end to end at an ordinary (degree-2) vertex are then chained,
    so a closed outline comes out as one closed stroke (ours)."""
    adj = {u: set(vs) for u, vs in g.adj.items() if vs}
    pos = g.pos
    paths: list[list] = []
    for comp in _component_nodes(adj):
        sub = {u: adj[u] for u in comp}
        if len(comp) > exact_max_nodes:
            paths.extend(_chains(sub))
            continue
        while sub:
            start = next(iter(sub))
            a, _, _ = _dijkstra(sub, pos, start)
            b, dist, prev = _dijkstra(sub, pos, a)
            path = [b]
            while path[-1] != a:
                path.append(prev[path[-1]])
            for u, v in zip(path, path[1:]):
                sub[u].discard(v)
                sub[v].discard(u)
            for u in set(path):
                if not sub.get(u):
                    sub.pop(u, None)
            if len(path) >= 2:
                paths.append(path)
    paths = _chain(paths, g)
    out = []
    for path in paths:
        pts = np.array([pos[u] for u in path], float)
        if geo_len(pts) >= min_len:
            out.append(pts)
    return out


def _component_nodes(adj: dict) -> list[list]:
    seen: set = set()
    out = []
    for n in adj:
        if n in seen:
            continue
        comp = [n]
        seen.add(n)
        stack = [n]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    comp.append(v)
                    stack.append(v)
        out.append(comp)
    return out


def _chains(adj: dict) -> list[list]:
    """Maximal paths through degree-2 vertices (cycles included), linear time."""
    used: set = set()

    def ekey(u, v):
        return (u, v) if _key(u) < _key(v) else (v, u)

    out = []
    starts = [u for u in adj if len(adj[u]) != 2] + list(adj)
    for s in starts:
        for v in adj[s]:
            if ekey(s, v) in used:
                continue
            path = [s, v]
            used.add(ekey(s, v))
            while len(adj[path[-1]]) == 2:
                nxt = [w for w in adj[path[-1]] if ekey(path[-1], w) not in used]
                if not nxt:
                    break
                used.add(ekey(path[-1], nxt[0]))
                path.append(nxt[0])
            out.append(path)
    return out


def geo_len(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def _chain(paths: list[list], g: DCGraph) -> list[list]:
    """Join paths whose ends meet at a vertex of degree 2 in `g`."""
    changed = True
    while changed:
        changed = False
        ends: dict = {}
        for i, p in enumerate(paths):
            for e in (p[0], p[-1]):
                ends.setdefault(e, []).append(i)
        for node, idx in ends.items():
            if g.degree(node) != 2 or len(idx) != 2:
                continue
            i, j = idx
            if i == j:
                continue  # already a closed loop
            a, b = paths[i], paths[j]
            if a[-1] != node:
                a = a[::-1]
            if b[0] != node:
                b = b[::-1]
            paths[i] = a + b[1:]
            del paths[j]
            changed = True
            break
    return paths
