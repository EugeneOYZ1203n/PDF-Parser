"""Skeleton -> node/chain graph.

Ported from archive/scripts/testing_paper/chaining.py (find_nodes / trace_chain /
chain_skeleton), with two changes:
  * output is (x, y) not (row, col), matching the rest of the spike;
  * short leaf branches (barbs) are pruned before chaining (Dosch 2000 Sec 2.3:
    "a single threshold on the significance of a branch enables correct removal
    of the smallest barbs").
"""
from __future__ import annotations

import numpy as np

from .types_ import Graph, Point

_OFF8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _neighbors(sk: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    h, w = sk.shape
    out = []
    for dy, dx in _OFF8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and sk[ny, nx]:
            out.append((ny, nx))
    return out


def _degree_map(sk: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve

    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    deg = convolve(sk.astype(np.uint8), k, mode="constant")
    deg[~sk] = 0
    return deg


def _find_nodes(sk: np.ndarray) -> set[tuple[int, int]]:
    deg = _degree_map(sk)
    ys, xs = np.nonzero(sk & ((deg == 1) | (deg >= 3)))
    return set(zip(ys.tolist(), xs.tolist()))


def _trace(sk, start, first, nodes, visited_edges):
    chain = [start]
    prev, cur = start, first
    while True:
        chain.append(cur)
        visited_edges.add(tuple(sorted((prev, cur))))
        if cur in nodes and cur != start:
            break
        nbrs = [n for n in _neighbors(sk, *cur) if n != prev]
        if not nbrs:
            break
        nxt = nbrs[0]
        if tuple(sorted((cur, nxt))) in visited_edges:
            break
        prev, cur = cur, nxt
    return chain


def _chain_skeleton(sk: np.ndarray):
    nodes = _find_nodes(sk)
    visited_edges: set = set()
    chains: list[list[tuple[int, int]]] = []
    for node in nodes:
        for nb in _neighbors(sk, *node):
            if tuple(sorted((node, nb))) in visited_edges:
                continue
            ch = _trace(sk, node, nb, nodes, visited_edges)
            if len(ch) >= 2:
                chains.append(ch)
    # closed loops with no node on them
    px = set(zip(*[a.tolist() for a in np.nonzero(sk)]))
    seen = {p for e in visited_edges for p in e}
    remaining = px - seen
    while remaining:
        start = next(iter(remaining))
        loop = [start]
        prev, cur = None, start
        while True:
            cand = [n for n in _neighbors(sk, *cur) if n != prev]
            if not cand:
                break
            nxt = cand[0]
            if nxt == start:
                loop.append(start)
                break
            if nxt in loop:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            chains.append(loop)
        remaining -= set(loop)
    return nodes, chains


def _chain_len(chain) -> float:
    p = np.array(chain, float)
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def prune_barbs(sk: np.ndarray, min_branch_px: float, iterations: int = 3) -> np.ndarray:
    """Iteratively drop leaf chains (one endpoint is degree-1) shorter than
    min_branch_px. Keeps junction topology intact."""
    sk = sk.copy()
    for _ in range(iterations):
        nodes, chains = _chain_skeleton(sk)
        dm = _degree_map(sk)
        removed = 0
        for ch in chains:
            deg0 = int(dm[ch[0]])
            deg1 = int(dm[ch[-1]])
            has_junction_end = deg0 >= 3 or deg1 >= 3
            has_free_end = deg0 == 1 or deg1 == 1
            # a true barb: sticks off a junction with a free tip, and is short.
            # a short *isolated* stroke (both ends free) is a real primitive -- keep it.
            if has_junction_end and has_free_end and _chain_len(ch) < min_branch_px:
                for (y, x) in (ch[:-1] if deg0 == 1 else ch[1:]):
                    sk[y, x] = False
                removed += 1
        if removed == 0:
            break
    # re-thin any 2px blobs left behind
    return sk


def build_graph(skeleton: np.ndarray, barb_min_px: float = 0.0) -> Graph:
    sk = skeleton.astype(bool)
    if barb_min_px > 0:
        sk = prune_barbs(sk, barb_min_px)
    nodes_yx, chains_yx = _chain_skeleton(sk)
    nodes: list[Point] = [(float(x), float(y)) for (y, x) in nodes_yx]
    chains: list[list[Point]] = [[(float(x), float(y)) for (y, x) in ch] for ch in chains_yx]
    return Graph(nodes=nodes, chains=chains)
