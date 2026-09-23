"""`CharGraph` construction, complexity scoring, and anchor-point selection
-- steps 2-3 of `docs/cad_font_vector_recognition.md`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rastervec.Evaluation.CadFont.geometry import (
    Point,
    Segment,
    connect_nearby_points,
    dedupe_exact_points,
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


@dataclass
class GraphBuildStats:
    """Per-character diagnostics from one `build_char_graph` call --
    surfaced so a caller (the matching-lab notebook) can print/tune
    without recomputing the pipeline itself."""

    epsilon: float
    points_before_rdp: int
    points_after_rdp: int
    points_removed: int


# Floor so a degenerate segment set (every segment zero-length, or a
# single-point character) never produces a zero/negative epsilon.
_MIN_EPSILON = 1e-9


def _compute_epsilon(segments: list[Segment]) -> float:
    """Half the length of the shortest segment in `segments` -- the fully
    data-derived tolerance used for both `connect_nearby_points` (how close
    counts as "the same point, topologically") and the RDP simplification
    pass. Measured from the POST-`split_at_intersections` segment set
    (splitting can only shorten some segments, and the connect/RDP steps
    should use the geometry they actually operate on)."""
    lengths = [
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segments
    ]
    lengths = [length for length in lengths if length > _MIN_EPSILON]
    if not lengths:
        return _MIN_EPSILON
    return max(min(lengths) / 2.0, _MIN_EPSILON)


def build_char_graph(
    segments: list[Segment], vertex_groups: list[frozenset[Point]],
) -> tuple[CharGraph, GraphBuildStats]:
    """Orchestrates `geometry.py`'s pipeline -- `split_at_intersections` ->
    epsilon derivation -> exact-coincidence dedup (`dedupe_exact_points`) ->
    `connect_nearby_points` -> protected-node computation -> Douglas-Peucker
    simplification -> `CharGraph(nodes, edges)`. The single entry point
    `character_bank.py` and `matching.py::build_candidate_graph` both call.

    `segments` is already-flattened geometry (see
    `geometry.flatten_item_to_segments`); `vertex_groups` is that same
    flatten pass's per-item groups of true data-defined points (never
    synthetic curve-interior samples) -- a node built from one of these is
    always protected from simplification, alongside any node with graph
    degree > 2 (a real junction/crossing, whether from `split_at_intersections`
    or from a new `connect_nearby_points` edge), and the resulting
    `CharGraph`'s `original_vertex_indices` records exactly which final
    nodes they are (consumed by `select_anchor_points`).

    No tolerance parameters: unlike the old `point_merge_tol`/`area_tol`
    (externally supplied, dynamically derived by the caller from its own
    notion of "scale"), `epsilon` is now computed here, directly from this
    character's own geometry (`_compute_epsilon`) -- see `GraphBuildStats`
    for what's returned alongside the graph."""
    original_vertices: set[Point] = set()
    for group in vertex_groups:
        original_vertices.update(group)

    split = split_at_intersections(segments)
    epsilon = _compute_epsilon(split)
    nodes, edges, point_to_node = dedupe_exact_points(split)
    edges = connect_nearby_points(nodes, edges, epsilon)

    original_vertex_node_indices = {
        point_to_node[p] for p in original_vertices if p in point_to_node
    }
    protected = _compute_protected_nodes(edges, len(nodes), original_vertices, point_to_node)
    simplified_nodes, simplified_edges, final_original_vertex_indices = _simplify_graph(
        nodes, edges, protected, epsilon, original_vertex_node_indices,
    )
    stats = GraphBuildStats(
        epsilon=epsilon,
        points_before_rdp=len(nodes),
        points_after_rdp=len(simplified_nodes),
        points_removed=len(nodes) - len(simplified_nodes),
    )
    return CharGraph(
        nodes=simplified_nodes,
        edges=simplified_edges,
        original_vertex_indices=final_original_vertex_indices,
    ), stats


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


def _perpendicular_distance(p: Point, a: Point, b: Point) -> float:
    """Distance from `p` to the infinite line through `a`/`b` (not clamped
    to the segment -- classic Douglas-Peucker measures against the line,
    not the segment). `a == b` degenerates to plain point distance."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy)
    if norm < 1e-12:
        return math.hypot(p[0] - ax, p[1] - ay)
    return abs(dx * (ay - p[1]) - (ax - p[0]) * dy) / norm


def _rdp_chain(nodes: list[Point], chain: list[int], epsilon: float) -> list[int]:
    """Classic recursive Douglas-Peucker over one chain of node indices.
    Both ends of the chain are always kept (protected by construction --
    see `_find_chains`): find the interior point with max perpendicular
    distance from the line `(chain[0], chain[-1])`; if that max distance is
    `<= epsilon`, every interior point is dropped, otherwise the
    max-distance point is kept and the chain recurses on both halves
    around it."""
    if len(chain) <= 2:
        return list(chain)
    a, b = nodes[chain[0]], nodes[chain[-1]]
    max_dist, max_idx = -1.0, -1
    for i in range(1, len(chain) - 1):
        d = _perpendicular_distance(nodes[chain[i]], a, b)
        if d > max_dist:
            max_dist, max_idx = d, i
    if max_dist <= epsilon:
        return [chain[0], chain[-1]]
    left = _rdp_chain(nodes, chain[: max_idx + 1], epsilon)
    right = _rdp_chain(nodes, chain[max_idx:], epsilon)
    return left[:-1] + right


def _simplify_graph(
    nodes: list[Point],
    edges: list[tuple[int, int]],
    protected: set[int],
    epsilon: float,
    original_vertex_node_indices: set[int],
) -> tuple[list[Point], list[tuple[int, int]], frozenset[int]]:
    """Runs `_rdp_chain` over every chain from `_find_chains`, then
    re-indexes the surviving node indices into a compact final node/edge
    list. `original_vertex_node_indices` is remapped through that same
    re-indexing and returned alongside -- every original vertex is
    protected (see `_compute_protected_nodes`), so it always survives;
    this only ever renumbers it."""
    chains = _find_chains(len(nodes), edges, protected)

    kept_old_indices: set[int] = set()
    raw_new_edges: list[tuple[int, int]] = []
    for chain in chains:
        simplified = _rdp_chain(nodes, chain, epsilon)
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


def critical_point_indices(graph: CharGraph) -> frozenset[int]:
    """A graph's "critical points" -- `original_vertex_indices | {degree >
    2 nodes}` (a real original vertex or a genuine junction/crossing, never
    a synthetic curve-interior point that merely survived simplification).
    When `graph.original_vertex_indices` is empty (a hand-built graph with
    no such info, e.g. many unit tests), every node counts as critical --
    the same fallback `select_anchor_points` has always used.

    Shared by `select_anchor_points` (the template-side anchor search pool,
    below) and `Evaluation/CadFont/matching.py` (the candidate-side
    critical-point set -- a candidate graph never goes through anchor
    selection at all, only this). A graph's own synthesized anchor, if any
    (`graph.synthetic_anchor_indices`), is deliberately NOT included here
    -- it isn't an "eligible" node by the rule above, it's a point
    `select_anchor_points` added itself; a caller wanting the full
    critical-point set of a template graph should union this with
    `graph.synthetic_anchor_indices`."""
    if graph.original_vertex_indices:
        eligible = set(graph.original_vertex_indices)
        eligible.update(i for i, d in enumerate(graph.degrees()) if d > 2)
        return frozenset(eligible)
    return frozenset(range(len(graph.nodes)))


def _centroid(nodes: list[Point], indices: set[int]) -> Point:
    cx = sum(nodes[i][0] for i in indices) / len(indices)
    cy = sum(nodes[i][1] for i in indices) / len(indices)
    return (cx, cy)


def select_anchor_points(graph: CharGraph, max_anchors: int = 3) -> tuple[CharGraph, tuple[int, ...]]:
    """Template-only anchor selection -- candidate graphs never call this
    (see `matching.py`'s module docstring: a candidate graph has no
    anchors at all, only critical points, via `critical_point_indices`
    directly). Greedy, 3-stage, over `critical_point_indices(graph)`:

    Anchor 1: the eligible node with max degree; ties broken by distance
    from the eligible/critical point set's own centroid (farther wins --
    a point far from the crowd is a more useful geometric reference);
    remaining ties broken deterministically by node index.

    Anchor 2: from the remaining eligible nodes, whichever is farthest
    from anchor 1 -- degree is no longer a factor once anchor 1 is picked.

    Anchor 3: from the remaining eligible nodes, whichever maximizes the
    triangle area of (anchor1, anchor2, candidate). This replaces the old
    separate collinearity rejection check: the argmax-area candidate is
    only ever degenerate (collinear with anchor1/anchor2) when *every*
    remaining eligible point is collinear with them, in which case no 3rd
    anchor is added here and control falls through to the existing
    synthetic-baseline-anchor fallback below, exactly as when only 2
    eligible anchors exist.

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
    eligible = set(critical_point_indices(graph))
    if not eligible or max_anchors < 1:
        return graph, ()

    degrees = graph.degrees()
    cx, cy = _centroid(graph.nodes, eligible)

    def dist_from_centroid(i: int) -> float:
        return math.hypot(graph.nodes[i][0] - cx, graph.nodes[i][1] - cy)

    anchor1 = min(eligible, key=lambda i: (-degrees[i], -dist_from_centroid(i), i))
    anchors = [anchor1]
    remaining = eligible - {anchor1}

    if remaining and max_anchors > 1:
        p1 = graph.nodes[anchor1]

        def dist_to_p1(i: int) -> float:
            return math.hypot(graph.nodes[i][0] - p1[0], graph.nodes[i][1] - p1[1])

        anchor2 = min(remaining, key=lambda i: (-dist_to_p1(i), i))
        anchors.append(anchor2)
        remaining = remaining - {anchor2}

    if len(anchors) == 2 and remaining and max_anchors > 2:
        p1, p2 = graph.nodes[anchors[0]], graph.nodes[anchors[1]]
        anchor3 = min(remaining, key=lambda i: (-_triangle_area(p1, p2, graph.nodes[i]), i))
        if not _collinear(p1, p2, graph.nodes[anchor3]):
            anchors.append(anchor3)

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
