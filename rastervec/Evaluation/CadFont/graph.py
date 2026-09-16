"""`CharGraph` construction, complexity scoring, and anchor-point selection
-- steps 2-3 of `docs/cad_font_vector_recognition.md`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rastervec.Evaluation.CadFont.geometry import (
    Point,
    Segment,
    merge_close_points,
    merge_colinear_segments,
    split_at_intersections,
)


@dataclass
class CharGraph:
    """A planar topological graph built from one labelled character's
    vectors, in baseline-relative frame (see
    `character_bank.to_baseline_relative_vectors`): `nodes[i]` is an
    `(x, y)` point, `edges` is a list of `(node_i, node_j)` index pairs
    (`node_i < node_j`, deduped)."""

    nodes: list[Point]
    edges: list[tuple[int, int]]

    def degrees(self) -> list[int]:
        """Degree of every node, index-aligned with `self.nodes` -- one
        pass over edges rather than one `degree()` call per node."""
        deg = [0] * len(self.nodes)
        for a, b in self.edges:
            deg[a] += 1
            deg[b] += 1
        return deg

    def degree(self, node_idx: int) -> int:
        return sum(1 for a, b in self.edges if a == node_idx or b == node_idx)

    def max_degree(self) -> int:
        deg = self.degrees()
        return max(deg) if deg else 0

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)


def build_char_graph(
    segments: list[Segment],
    *, angle_tol_deg: float = 2.0, perp_tol: float = 0.75, gap_tol: float = 1.0,
    point_merge_tol: float = 0.5,
) -> CharGraph:
    """Orchestrates `geometry.py`'s pipeline: `merge_colinear_segments` ->
    `split_at_intersections` -> `merge_close_points` -> `CharGraph(nodes,
    edges)`. The single entry point `character_bank.py` and the notebook
    both call."""
    merged = merge_colinear_segments(
        segments, angle_tol_deg=angle_tol_deg, perp_tol=perp_tol, gap_tol=gap_tol,
    )
    split = split_at_intersections(merged)
    nodes, edges = merge_close_points(split, point_merge_tol=point_merge_tol)
    return CharGraph(nodes=nodes, edges=edges)


def complexity(graph: CharGraph) -> float:
    """`num_nodes * num_edges * max_degree`, per the algorithm spec.

    Documented interpretation: the spec's wording says "Num endpoints *
    Num edges * max degree", but the worked anchor-selection example in
    the same step treats every graph node as a candidate "endpoint" for
    that purpose (leaves AND internal junction/intersection nodes alike),
    not only degree-1 leaves. This function therefore uses `num_nodes()`
    for the "Num endpoints" factor -- the reading consistent with that
    worked example."""
    return graph.num_nodes() * graph.num_edges() * graph.max_degree()


def _collinear(p: Point, q: Point, r: Point, tol: float = 1e-6) -> bool:
    """True if `p`, `q`, `r` lie on (approximately) one line -- the cross
    product of `(q - p)` and `(r - p)`, normalized by the pair's own scale
    (a relative, not absolute, tolerance) is near zero."""
    ux, uy = q[0] - p[0], q[1] - p[1]
    vx, vy = r[0] - p[0], r[1] - p[1]
    cross = ux * vy - uy * vx
    scale = max(math.hypot(ux, uy) * math.hypot(vx, vy), 1e-9)
    return abs(cross) / scale <= tol


def select_anchor_points(graph: CharGraph, max_anchors: int = 3) -> tuple[int, ...]:
    """Greedy anchor selection: sort node indices by degree descending
    (stable tie-break by original index). The first 2 picked nodes are
    always anchors (any 2 points are trivially non-collinear); a 3rd
    candidate is added only if it is NOT collinear with the 2
    already-chosen anchors. Stops at `max_anchors` or when nodes are
    exhausted -- 2 anchors when the graph has fewer than 3 non-collinear
    nodes total (including a plain 2-node graph)."""
    degrees = graph.degrees()
    order = sorted(range(len(graph.nodes)), key=lambda i: (-degrees[i], i))

    anchors: list[int] = []
    for idx in order:
        if len(anchors) < 2:
            anchors.append(idx)
        elif not _collinear(graph.nodes[anchors[0]], graph.nodes[anchors[1]], graph.nodes[idx]):
            anchors.append(idx)
        if len(anchors) >= max_anchors:
            break
    return tuple(anchors)
