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
    split_at_intersections,
)


@dataclass
class CharGraph:
    """A planar topological graph built from one labelled character's
    vectors, in baseline-relative frame (see
    `character_bank.to_baseline_relative_vectors`): `nodes[i]` is an
    `(x, y)` point, `edges` is a list of `(node_i, node_j)` index pairs
    (`node_i < node_j`, deduped).

    `original_vertex_indices` and `synthetic_anchor_indices` both default
    to empty so every hand-built graph (e.g. in tests) that doesn't set
    them keeps `select_anchor_points`'s original, unrestricted behavior --
    only a graph produced by `build_char_graph` populates
    `original_vertex_indices`, and only `select_anchor_points` itself ever
    populates `synthetic_anchor_indices`."""

    nodes: list[Point]
    edges: list[tuple[int, int]]
    original_vertex_indices: frozenset[int] = frozenset()
    synthetic_anchor_indices: frozenset[int] = frozenset()

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
    vertex_groups: list[frozenset[Point]],
    *, point_merge_tol: float = 1e-3, area_tol: float,
) -> CharGraph:
    """Orchestrates `geometry.py`'s pipeline -- `split_at_intersections` ->
    `merge_close_points` (coincidence dedup, protecting each `"l"`/`"c"`
    item's own 2 vertices from merging with each other) -> protected-node
    computation -> degree-aware area simplification -> `CharGraph(nodes,
    edges)`. The single entry point `character_bank.py` and the notebook
    both call.

    `segments` is already-flattened, adaptively-sampled geometry (see
    `geometry.flatten_item_to_segments`); `vertex_groups` is that same
    flatten pass's per-item groups of true data-defined points (never
    synthetic curve-interior samples) -- a node built from one of these is
    always protected from simplification, alongside any node with graph
    degree > 2 (a real junction/crossing), and the resulting `CharGraph`'s
    `original_vertex_indices` records exactly which final nodes they are
    (consumed by `select_anchor_points`). `area_tol` has no default:
    callers must derive it from their own scale (see
    `character_bank.build_character_bank` for the per-character dynamic
    derivation this was designed for)."""
    original_vertices: set[Point] = set()
    for group in vertex_groups:
        original_vertices.update(group)

    split = split_at_intersections(segments)
    nodes, edges, point_to_node = merge_close_points(
        split, point_merge_tol=point_merge_tol, forbidden_groups=vertex_groups,
    )
    original_vertex_node_indices = {
        point_to_node[p] for p in original_vertices if p in point_to_node
    }
    protected = _compute_protected_nodes(edges, len(nodes), original_vertices, point_to_node)
    simplified_nodes, simplified_edges, final_original_vertex_indices = _simplify_graph(
        nodes, edges, protected, area_tol, original_vertex_node_indices,
    )
    return CharGraph(
        nodes=simplified_nodes,
        edges=simplified_edges,
        original_vertex_indices=final_original_vertex_indices,
    )


def _compute_protected_nodes(
    edges: list[tuple[int, int]],
    num_nodes: int,
    original_vertices: set[Point],
    point_to_node: dict[Point, int],
) -> set[int]:
    """A node is protected from the simplification pass if it's a real
    graph junction (degree > 2) or if it's the canonical node of at least
    one original (non-synthetic) data vertex. Every degree-1 node is
    necessarily an original vertex too (nothing else could have produced a
    dead end), so this rule alone also naturally protects every chain's
    true endpoints -- no separate degree-1 special case needed."""
    deg = [0] * num_nodes
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    protected = {i for i, d in enumerate(deg) if d > 2}
    for p in original_vertices:
        node_idx = point_to_node.get(p)
        if node_idx is not None:
            protected.add(node_idx)
    return protected


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _build_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _find_chains(
    num_nodes: int, edges: list[tuple[int, int]], protected: set[int],
) -> list[list[int]]:
    """Walks the graph's adjacency structure, yielding one ordered
    node-index chain per maximal run of non-protected (degree-2, synthetic)
    nodes between two protected nodes, inclusive of both protected ends.
    Every non-protected node has exactly one unvisited neighbor to advance
    through by construction (see `_compute_protected_nodes`), so each walk
    is a straight-line follow with no branching decision to make.

    A chain that loops back to its own single protected anchor (a closed
    curve/loop with no other junction, e.g. a single-item "O") is handled
    by the same walk -- it simply ends when the walk returns to a
    protected node, which may be its own start. The trailing defensive
    pass over any still-unvisited edges only matters if a chain had zero
    protected nodes at all, which the protection rule above should never
    actually produce."""
    adj = _build_adjacency(num_nodes, edges)
    visited: set[tuple[int, int]] = set()

    def walk(start: int, first: int) -> list[int]:
        chain = [start]
        prev, cur = start, first
        while True:
            visited.add(_edge_key(prev, cur))
            chain.append(cur)
            if cur in protected:
                break
            nxts = [n for n in adj[cur] if n != prev]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
        return chain

    chains: list[list[int]] = []
    for start in sorted(protected):
        for nxt in adj[start]:
            if _edge_key(start, nxt) in visited:
                continue
            chains.append(walk(start, nxt))

    for a, b in edges:
        if _edge_key(a, b) in visited:
            continue
        chains.append(walk(a, b))

    return chains


def _triangle_area(a: Point, b: Point, c: Point) -> float:
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0


def _simplify_chain(nodes: list[Point], chain: list[int], area_tol: float) -> list[int]:
    """Cascading-anchor area sweep (Visvalingam-Whyte-style) over one
    chain of node indices. Both ends of the chain are always kept
    (protected by construction -- see `_find_chains`); an interior point
    `b` between the current anchor `a` and its successor `c` is dropped
    (anchor unchanged) when `triangle(a, b, c)`'s area is below
    `area_tol`, otherwise kept and promoted to the new anchor -- so several
    consecutive points can collapse against one anchor in a single pass."""
    if len(chain) <= 2:
        return list(chain)
    anchor = chain[0]
    kept = [anchor]
    for i in range(1, len(chain) - 1):
        b, c = chain[i], chain[i + 1]
        if _triangle_area(nodes[anchor], nodes[b], nodes[c]) < area_tol:
            continue
        kept.append(b)
        anchor = b
    kept.append(chain[-1])
    return kept


def _simplify_graph(
    nodes: list[Point],
    edges: list[tuple[int, int]],
    protected: set[int],
    area_tol: float,
    original_vertex_node_indices: set[int],
) -> tuple[list[Point], list[tuple[int, int]], frozenset[int]]:
    """Runs `_simplify_chain` over every chain from `_find_chains`, then
    re-indexes the surviving node indices into a compact final node/edge
    list. `original_vertex_node_indices` is remapped through that same
    re-indexing and returned alongside -- every original vertex is
    protected (see `_compute_protected_nodes`), so it always survives;
    this only ever renumbers it."""
    chains = _find_chains(len(nodes), edges, protected)

    kept_old_indices: set[int] = set()
    raw_new_edges: list[tuple[int, int]] = []
    for chain in chains:
        simplified = _simplify_chain(nodes, chain, area_tol)
        kept_old_indices.update(simplified)
        raw_new_edges.extend(zip(simplified, simplified[1:]))

    old_to_new: dict[int, int] = {}
    new_nodes: list[Point] = []
    for old_idx in sorted(kept_old_indices):
        old_to_new[old_idx] = len(new_nodes)
        new_nodes.append(nodes[old_idx])

    seen: set[tuple[int, int]] = set()
    new_edges: list[tuple[int, int]] = []
    for a, b in raw_new_edges:
        na, nb = old_to_new[a], old_to_new[b]
        if na == nb:
            continue
        key = (na, nb) if na < nb else (nb, na)
        if key in seen:
            continue
        seen.add(key)
        new_edges.append(key)

    new_original_vertex_indices = frozenset(
        old_to_new[i] for i in original_vertex_node_indices if i in old_to_new
    )
    return new_nodes, new_edges, new_original_vertex_indices


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


def _signed_area2(a: Point, b: Point, c: Point) -> float:
    """Twice the signed area of triangle `a, b, c` -- positive when the
    ordered triple turns counter-clockwise. Same cross-product core as
    `_triangle_area`, just without the `abs()`/halving, since only the
    sign matters here."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def _synthesize_baseline_anchor(anchor0: Point, anchor1: Point) -> Point | None:
    """One deterministic point on the baseline (`y = 0`) to serve as a 3rd
    anchor when only 2 real (eligible) anchors were found.

    Candidates are every real on-baseline point whose distance to *at
    least one* of `anchor0`/`anchor1` equals `dist(anchor0, anchor1)`
    (solving `(x - ref_x)**2 + ref_y**2 = d**2` for each reference anchor,
    up to 2 real roots per reference). Among all such candidates, the one
    minimizing total distance to *both* anchors wins; a tie (rare, exact
    equality) is broken deterministically by picking whichever candidate
    makes the ordered triple `(anchor0, anchor1, candidate)` turn
    counter-clockwise (positive signed area). Candidates that coincide with
    `anchor0`/`anchor1` themselves are excluded first -- minimizing total
    distance to both anchors would otherwise degenerate to picking a point
    exactly on top of one of them whenever both anchors already lie on the
    baseline (e.g. a straight horizontal stroke), since that trivially
    achieves the smallest possible total distance. Returns `None` if
    neither anchor admits any real, non-coincident solution at all (an
    anchor's own distance from the baseline already exceeds `d`) --
    callers should then just keep the 2 real anchors rather than force an
    approximate point."""
    d = math.hypot(anchor1[0] - anchor0[0], anchor1[1] - anchor0[1])

    def _is_anchor(p: Point) -> bool:
        return math.hypot(p[0] - anchor0[0], p[1] - anchor0[1]) < 1e-9 or (
            math.hypot(p[0] - anchor1[0], p[1] - anchor1[1]) < 1e-9
        )

    candidates: set[Point] = set()
    for ref in (anchor0, anchor1):
        rx, ry = ref
        under_sqrt = d * d - ry * ry
        if under_sqrt < 0:
            continue
        s = math.sqrt(under_sqrt)
        for p in ((rx + s, 0.0), (rx - s, 0.0)):
            if not _is_anchor(p):
                candidates.add(p)
    if not candidates:
        return None

    def total_dist(p: Point) -> float:
        return (
            math.hypot(p[0] - anchor0[0], p[1] - anchor0[1])
            + math.hypot(p[0] - anchor1[0], p[1] - anchor1[1])
        )

    best = min(total_dist(p) for p in candidates)
    tied = [p for p in candidates if abs(total_dist(p) - best) < 1e-9]
    if len(tied) == 1:
        return tied[0]
    ccw = [p for p in tied if _signed_area2(anchor0, anchor1, p) > 0]
    return ccw[0] if ccw else sorted(tied)[0]


def select_anchor_points(graph: CharGraph, max_anchors: int = 3) -> tuple[CharGraph, tuple[int, ...]]:
    """Greedy anchor selection: sort *eligible* node indices by degree
    descending (stable tie-break by original index). The first 2 picked
    nodes are always anchors (any 2 points are trivially non-collinear); a
    3rd candidate is added only if it is NOT collinear with the 2
    already-chosen anchors. Stops at `max_anchors` or when eligible nodes
    are exhausted.

    Eligible = `graph.original_vertex_indices | {degree > 2 nodes}` -- a
    real original vertex or a genuine junction/crossing, never a synthetic
    curve-interior point that merely survived simplification. When
    `graph.original_vertex_indices` is empty (a hand-built graph with no
    such info, e.g. in tests), every node is eligible -- today's original,
    unrestricted behavior.

    In restricted mode (`graph.original_vertex_indices` non-empty), when
    exactly 2 eligible anchors are found and a 3rd is wanted, one synthetic
    point is added via `_synthesize_baseline_anchor` and returned as part
    of a new `CharGraph` (same nodes/edges plus the new point, flagged in
    `synthetic_anchor_indices`) -- callers must use the returned graph,
    not their original one, since the 3rd anchor index may not exist in
    it. Fewer than 2 eligible anchors is out of scope for synthesis
    (returns however many are found). In unrestricted mode, no synthesis
    ever happens and the input graph is returned unchanged -- an exact
    match for this function's pre-restriction behavior."""
    if graph.original_vertex_indices:
        eligible = set(graph.original_vertex_indices)
        eligible.update(i for i, d in enumerate(graph.degrees()) if d > 2)
    else:
        eligible = set(range(len(graph.nodes)))

    degrees = graph.degrees()
    order = sorted(eligible, key=lambda i: (-degrees[i], i))

    anchors: list[int] = []
    for idx in order:
        if len(anchors) < 2:
            anchors.append(idx)
        elif not _collinear(graph.nodes[anchors[0]], graph.nodes[anchors[1]], graph.nodes[idx]):
            anchors.append(idx)
        if len(anchors) >= max_anchors:
            break

    if graph.original_vertex_indices and len(anchors) == 2 and max_anchors > 2:
        new_point = _synthesize_baseline_anchor(graph.nodes[anchors[0]], graph.nodes[anchors[1]])
        if new_point is not None:
            new_idx = len(graph.nodes)
            graph = CharGraph(
                nodes=[*graph.nodes, new_point],
                edges=list(graph.edges),
                original_vertex_indices=graph.original_vertex_indices,
                synthetic_anchor_indices=graph.synthetic_anchor_indices | {new_idx},
            )
            anchors.append(new_idx)

    return graph, tuple(anchors)
