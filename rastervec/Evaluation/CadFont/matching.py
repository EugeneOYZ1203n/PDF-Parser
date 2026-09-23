"""Candidate graph construction (step 5) and single-template-vs-single-
candidate affine matching + scoring (step 6, 6.5) of
`docs/cad_font_vector_recognition.md`.

Scoped to one character template vs one candidate graph -- iterating over
many templates (6.1-6.2, 8) and cross-candidate baseline state (7, 9-12)
are out of scope here; that's a future caller's job, not this module's.

Commons-only import convention (same as geometry.py's own docstring): no
pymupdf import, only rastervec.commons.helpers.geometry primitives.
Intra-package reuse of character_bank.py IS allowed here (this is not a
cross-P2/P3-backend import) -- `CharacterTemplate` is imported directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from rastervec.commons.helpers.geometry import transform_point
from rastervec.Evaluation.CadFont.character_bank import CharacterTemplate
from rastervec.Evaluation.CadFont.geometry import BEZIER_SAMPLE_COUNT, Point, vectors_to_segments
from rastervec.Evaluation.CadFont.graph import (
    CharGraph,
    GraphBuildStats,
    build_char_graph,
    critical_point_indices,
)

if TYPE_CHECKING:
    from rastervec.commons.models import Vector


def build_candidate_graph(
    vectors: "list[Vector]",
    *,
    bezier_sample_count: int = BEZIER_SAMPLE_COUNT,
) -> tuple[CharGraph, GraphBuildStats]:
    """Step 5: build a `CharGraph` for a candidate vector group -- the same
    flatten/split/connect/RDP pipeline a labelled character gets
    (`character_bank.build_character_bank`), but WITHOUT
    `to_baseline_relative_vectors`: `vectors` stay in raw page space, since
    no baseline is known yet for an unlabelled candidate group.
    `epsilon` is derived internally by `build_char_graph`, directly from
    this candidate's own page-space geometry -- no external scale/fraction
    needed (see `graph.py::_compute_epsilon`).

    The returned graph has no anchors -- candidate graphs never go through
    `graph.select_anchor_points` (that's template-only); its own critical
    points are `graph.critical_point_indices(result)` directly."""
    segments, vertex_groups = vectors_to_segments(vectors, bezier_sample_count=bezier_sample_count)
    return build_char_graph(segments, vertex_groups)


@dataclass
class SimilarityTransform:
    """Uniform-scale + rotation + translation, no shear -- by construction
    (a 4-DOF fit, never a general 6-DOF affine), which alone satisfies
    step 6.3.2's "no shearing transforms" requirement; no separate
    rejection check is needed. `y = scale * R(rotation_deg) @ x +
    translation`, using the same rotation convention as
    `commons.helpers.geometry.transform_point`."""

    scale: float
    rotation_deg: float
    translation: Point
    reflected: bool = False

    def apply(self, p: Point) -> Point:
        # When reflected, `rotation_deg` parameterizes the improper fit
        # matrix as `Rot(rotation_deg) @ diag(1, -1)` (see
        # `fit_similarity_transform`'s docstring) -- flipping y BEFORE the
        # scale+rotate+translate below reproduces exactly that composite,
        # not just a proper rotation.
        x, y = p
        if self.reflected:
            y = -y
        scaled = (x * self.scale, y * self.scale)
        return transform_point(scaled, offset=self.translation, rotation_deg=self.rotation_deg)

    def apply_many(self, points: list[Point]) -> list[Point]:
        return [self.apply(p) for p in points]


def fit_similarity_transform(
    src_points: list[Point], dst_points: list[Point], *, allow_reflection: bool = False,
) -> SimilarityTransform | None:
    """Closed-form least-squares similarity fit mapping `src_points[i] ->
    dst_points[i]`, via Umeyama's method (S. Umeyama, "Least-Squares
    Estimation of Transformation Parameters Between Two Point Patterns",
    IEEE TPAMI 13(4), 1991) -- well-defined for N >= 2 point pairs (exact
    for N == 2; a true least-squares fit, generally inexact, for N == 3,
    this module's real caller via `enumerate_anchor_correspondences`).
    Returns `None` when `src_points` has ~zero variance (every source
    point coincides) -- no rotation/scale is determinable from a
    degenerate source configuration.

    `would_reflect` (`det(u) * det(vt) < 0`) is true exactly when the
    unconstrained least-squares fit is a mirror image, not a proper
    rotation. By default (`allow_reflection=False`, every current caller)
    Umeyama's own sign-flip correction is still applied, forcing a proper
    rotation regardless -- `reflected` on the result is then always
    `False`, current behavior byte-identical. Passing `allow_reflection=
    True` skips that correction, so the returned transform can genuinely
    be a mirror when that's the strictly better fit, and `reflected`
    reports whether it is -- infrastructure for future per-character
    reflection support; no caller opts into this yet. When `reflected` is
    true, the improper fit matrix `r` decomposes as `Rot(rotation_deg) @
    diag(1, -1)` (flip y, then rotate) -- `SimilarityTransform.apply`
    flips `y` first in that case so applying the transform still
    reproduces the actual fit, not just its rotation component."""
    n = len(src_points)
    src = np.asarray(src_points, dtype=float)
    dst = np.asarray(dst_points, dtype=float)
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    var_src = float((src_c ** 2).sum() / n)
    if var_src < 1e-12:
        return None
    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    would_reflect = np.linalg.det(u) * np.linalg.det(vt) < 0
    s = np.eye(2)
    reflected = False
    if would_reflect and not allow_reflection:
        s[1, 1] = -1.0
    elif would_reflect and allow_reflection:
        reflected = True
    r = u @ s @ vt
    scale = float(np.trace(np.diag(d) @ s) / var_src)
    t = mu_dst - scale * (r @ mu_src)
    rotation_deg = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return SimilarityTransform(
        scale=scale, rotation_deg=rotation_deg, translation=(float(t[0]), float(t[1])),
        reflected=reflected,
    )


def enumerate_anchor_correspondences(
    template_graph: CharGraph,
    anchor_indices: tuple[int, ...],
    candidate_graph: CharGraph,
    candidate_critical_indices: frozenset[int] | None = None,
) -> list[tuple[int, ...]]:
    """Step 6.3.1: every ordered, injective assignment of `anchor_indices[i]
    -> a distinct candidate critical-point index`, filtered per-slot by
    "candidate node degree >= that specific anchor's own degree" (step
    6.3.1's exact-degree wording, relaxed to >= per this project's own
    design decision). The 3 template anchors are NOT interchangeable --
    assigning anchor_indices[0] to one candidate node vs another implies a
    different transform -- so this is an assignment-problem search over
    permutations, not a search over unordered candidate 3-subsets.
    `result[k][i]` is the candidate node matched to `anchor_indices[i]`
    for hypothesis `k`; slot order is preserved so a caller can zip
    template anchor positions against the matching candidate positions
    directly.

    Search is restricted to the candidate graph's own critical points
    (`critical_point_indices`), not every candidate node -- a candidate
    graph has no anchors of its own, so every one of its critical points
    is a valid correspondence target to test.

    Complexity: per-slot degree filtering bounds this to at most
    `|slot0| * |slot1| * |slot2|` with simple backtracking pruning on
    reuse -- for a realistic candidate cluster with ~10-30 critical points
    this is at most a few tens of thousands of hypotheses, fine for this
    module's single-pair, offline use."""
    if candidate_critical_indices is None:
        candidate_critical_indices = critical_point_indices(candidate_graph)
    cand_degrees = candidate_graph.degrees()
    tmpl_degrees = template_graph.degrees()

    slots = [
        [c for c in sorted(candidate_critical_indices) if cand_degrees[c] >= tmpl_degrees[a]]
        for a in anchor_indices
    ]

    results: list[tuple[int, ...]] = []

    def backtrack(i: int, chosen: list[int]) -> None:
        if i == len(slots):
            results.append(tuple(chosen))
            return
        for c in slots[i]:
            if c in chosen:
                continue
            chosen.append(c)
            backtrack(i + 1, chosen)
            chosen.pop()

    backtrack(0, [])
    return results


def build_candidate_tree(candidate_graph: CharGraph) -> "cKDTree | None":
    """A `scipy.spatial.cKDTree` over `candidate_graph.nodes`, built once so
    `map_template_nodes_to_candidate` can look up a nearest neighbor in
    `O(log m)` instead of a brute-force `O(m)` scan. `None` for an empty
    node list (nothing to index).

    Exposed as its own function (not inlined into
    `map_template_nodes_to_candidate`) so a future multi-template caller
    (steps 6.1/6.2/8, still out of scope for this module) can build one
    tree per candidate graph and reuse it across every template it tests
    against that same candidate, rather than rebuilding it per call."""
    if not candidate_graph.nodes:
        return None
    return cKDTree(np.asarray(candidate_graph.nodes, dtype=float))


def map_template_nodes_to_candidate(
    template_graph: CharGraph,
    transform: SimilarityTransform,
    candidate_graph: CharGraph,
    candidate_tree: "cKDTree | None" = None,
) -> dict[int, int]:
    """Every template node, mapped through `transform` into candidate/page
    space, matched to its nearest candidate-graph node -- searched over
    ALL of `candidate_graph.nodes` (not just its critical points: a
    template critical point may legitimately correspond to a non-critical,
    e.g. simplification-survivor, candidate node once transformed --
    critical-point status on the template side never restricts which
    candidate points are valid match targets). No distance cutoff.

    `candidate_tree`, if given, must be `build_candidate_tree(candidate_graph)`
    (or `None`) -- passed in by a caller (e.g. `match_template_against_candidate`)
    that builds it once and reuses it across many transforms tested against
    the same candidate graph, rather than rebuilding it here every call.
    When omitted, one is built on the fly (keeps every direct caller/test
    working unchanged, just without that reuse).

    Computed once and reused both for the rendered "matched subgraph"
    (every template node's match) and, restricted by the caller to just
    the template's own critical-point indices, for `score_match`'s step
    6.5 correspondence -- deliberately a single nearest-neighbor pass,
    not a second critical-point-restricted one."""
    if candidate_tree is None:
        candidate_tree = build_candidate_tree(candidate_graph)
    if candidate_tree is None:
        return {}
    transformed = transform.apply_many(template_graph.nodes)
    _, indices = candidate_tree.query(np.asarray(transformed, dtype=float))
    indices = np.atleast_1d(indices)
    return {i: int(idx) for i, idx in enumerate(indices)}


DEFAULT_MSE_WEIGHT = 1.0
DEFAULT_EDGE_WEIGHT = 1.0
DEFAULT_EDGE_DEGREE_BLEND = 0.5  # weight of edge-presence vs degree-mismatch inside edge_term
DEFAULT_ORIGINAL_POINT_WEIGHT = 1.0
DEFAULT_INTERSECTION_POINT_WEIGHT = 1.0
DEFAULT_SYNTHETIC_POINT_WEIGHT = 1.0


def _critical_point_type(graph: CharGraph, idx: int) -> str:
    """One of "synthetic"/"original"/"intersection" for a node in a
    template's evaluated critical-point set (`critical_point_indices(graph)
    | graph.synthetic_anchor_indices`):

    "synthetic" -- `select_anchor_points`'s own synthesized baseline anchor
    (`graph.synthetic_anchor_indices`); a brand-new, disconnected node with
    no real candidate counterpart, only a reference LINE it should lie
    near (see `_synthetic_point_sq_dist` below).

    "original" -- a genuine labelled data vertex (`graph.
    original_vertex_indices`) -- the strongest claim, checked first (a
    node can be both an original vertex and a degree > 2 junction, e.g. two
    strokes' endpoints coincide; "original" wins that case).

    "intersection" -- a real graph junction (`degree > 2`) that isn't
    itself an original vertex: a `split_at_intersections`-computed crossing,
    or a node that became a junction purely from a new
    `connect_nearby_points` edge.

    Falls back to "original" when neither `original_vertex_indices` nor
    `synthetic_anchor_indices` carry any provenance info at all (an
    unrestricted, hand-built graph, e.g. many unit tests) -- keeps every
    existing weight-1.0 caller's behavior byte-identical."""
    if idx in graph.synthetic_anchor_indices:
        return "synthetic"
    if idx in graph.original_vertex_indices:
        return "original"
    if graph.degrees()[idx] > 2:
        return "intersection"
    return "original"


def _point_segment_dist2(p: Point, a: Point, b: Point) -> float:
    """Squared distance from `p` to the segment `a`-`b` (clamped
    projection, unlike `graph._perpendicular_distance`'s infinite-line
    measure -- a synthetic point's "reference line" is a real, finite
    candidate edge, not an infinite line through it)."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = 0.0 if denom < 1e-12 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    cx, cy = ax + t * dx, ay + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2


def _synthetic_point_sq_dist(
    transformed_point: Point, matched_candidate_idx: int, candidate_graph: CharGraph,
) -> float:
    """A synthetic (synthesized-anchor) template point has no real
    candidate counterpart -- only a reference LINE it should lie near.
    `matched_candidate_idx` (from `map_template_nodes_to_candidate`'s
    nearest-neighbor lookup) identifies the relevant local neighborhood:
    the minimum squared point-to-segment distance over every candidate
    edge incident on that node. Falls back to plain point-to-point squared
    distance if that candidate node is isolated (no incident edges)."""
    incident = [
        (a, b) for a, b in candidate_graph.edges
        if a == matched_candidate_idx or b == matched_candidate_idx
    ]
    if not incident:
        cp = candidate_graph.nodes[matched_candidate_idx]
        return (transformed_point[0] - cp[0]) ** 2 + (transformed_point[1] - cp[1]) ** 2
    return min(
        _point_segment_dist2(transformed_point, candidate_graph.nodes[a], candidate_graph.nodes[b])
        for a, b in incident
    )


@dataclass
class MatchScore:
    mse_term: float  # normalized squared-distance term, lower is better
    edge_presence_cost: float  # 0..1: fraction of template edges NOT realized as candidate edges
    degree_cost: float  # 0..1ish: mean normalized degree mismatch over critical points
    edge_term: float  # blend of the two costs above, lower is better
    total: float  # mse_weight*mse_term + edge_weight*edge_term, lower is better


def score_match(
    template_graph: CharGraph,
    template_critical_indices: frozenset[int],
    transform: SimilarityTransform,
    candidate_graph: CharGraph,
    node_map: dict[int, int],
    *,
    mse_weight: float = DEFAULT_MSE_WEIGHT,
    edge_weight: float = DEFAULT_EDGE_WEIGHT,
    edge_degree_blend: float = DEFAULT_EDGE_DEGREE_BLEND,
    original_point_weight: float = DEFAULT_ORIGINAL_POINT_WEIGHT,
    intersection_point_weight: float = DEFAULT_INTERSECTION_POINT_WEIGHT,
    synthetic_point_weight: float = DEFAULT_SYNTHETIC_POINT_WEIGHT,
) -> MatchScore:
    """Step 6.5: scored ONLY over `template_critical_indices` (which
    TEMPLATE nodes get evaluated), each looked up in `node_map` -- which
    was itself built by searching the candidate graph's FULL node set, not
    just its critical points (see `map_template_nodes_to_candidate`). A
    template critical point matching a non-critical candidate node is a
    completely valid, scored correspondence.

    `mse_term`: a WEIGHTED mean squared distance (in candidate/page space)
    between each critical template point's *transformed* position and its
    match, normalized by `transform.scale ** 2` -- squared distances scale
    with `scale ** 2`, so dividing by it converts the residual back to the
    template's own (baseline-relative) unit scale, comparable across
    differently-sized candidates/templates. Each point is classified by
    `_critical_point_type` (original data vertex / true intersection-
    junction / synthesized anchor) and weighted by the matching
    `*_point_weight` kwarg; an "original"/"intersection" point's distance
    is plain point-to-point (to `candidate_graph.nodes[node_map[i]]`), but
    a "synthetic" point (the one baseline-synthesized anchor, if any) has
    no real candidate counterpart -- only a reference LINE it should lie
    near -- so its distance is `_synthetic_point_sq_dist` (nearest point on
    a candidate edge incident to its matched node) instead. All three
    weights default to `1.0`, making this an unweighted mean identical to
    the old flat behavior.

    `edge_term`: a blend (`edge_degree_blend`) of (a) `edge_presence_cost`
    -- the fraction of the template graph's OWN edges `(i, j)` whose
    mapped endpoints `(node_map[i], node_map[j])` are NOT a real edge of
    `candidate_graph`, and (b) `degree_cost` -- mean, over
    `template_critical_indices`, of the normalized absolute degree
    difference between a template critical point and its matched
    candidate node. Both terms range 0 (perfect) to ~1 (total mismatch).

    `total` is a cost (lower = better), not a bounded similarity score --
    a caller wanting a bounded, higher-is-better number can compute
    `1.0 / (1.0 + total)` itself."""
    scale = max(abs(transform.scale), 1e-9)
    critical = sorted(template_critical_indices) or list(range(template_graph.num_nodes()))
    point_weights = {
        "original": original_point_weight,
        "intersection": intersection_point_weight,
        "synthetic": synthetic_point_weight,
    }

    transformed = transform.apply_many(template_graph.nodes)
    weighted_sum = 0.0
    weight_total = 0.0
    for i in critical:
        tp = transformed[i]
        cand_idx = node_map[i]
        ptype = _critical_point_type(template_graph, i)
        weight = point_weights[ptype]
        if ptype == "synthetic":
            sq_dist = _synthetic_point_sq_dist(tp, cand_idx, candidate_graph)
        else:
            cp = candidate_graph.nodes[cand_idx]
            sq_dist = (tp[0] - cp[0]) ** 2 + (tp[1] - cp[1]) ** 2
        weighted_sum += weight * sq_dist
        weight_total += weight
    mse_term = (weighted_sum / weight_total if weight_total > 0 else 0.0) / (scale ** 2)

    cand_edge_set = {(a, b) if a < b else (b, a) for a, b in candidate_graph.edges}
    total_edges = template_graph.num_edges() or 1
    matched_edges = 0
    for a, b in template_graph.edges:
        ca, cb = node_map.get(a), node_map.get(b)
        if ca is None or cb is None:
            continue
        key = (ca, cb) if ca < cb else (cb, ca)
        if key in cand_edge_set:
            matched_edges += 1
    edge_presence_cost = 1.0 - (matched_edges / total_edges)

    tmpl_degrees = template_graph.degrees()
    cand_degrees = candidate_graph.degrees()
    degree_diffs = []
    for i in critical:
        dt, dc = tmpl_degrees[i], cand_degrees[node_map[i]]
        degree_diffs.append(abs(dt - dc) / max(dt, dc, 1))
    degree_cost = sum(degree_diffs) / len(degree_diffs)

    edge_term = edge_degree_blend * edge_presence_cost + (1 - edge_degree_blend) * degree_cost
    total = mse_weight * mse_term + edge_weight * edge_term
    return MatchScore(
        mse_term=mse_term, edge_presence_cost=edge_presence_cost,
        degree_cost=degree_cost, edge_term=edge_term, total=total,
    )


@dataclass
class CandidateMatch:
    correspondence: tuple[int, ...]  # candidate node idx per anchor slot, same order as anchor_indices
    anchor_indices: tuple[int, ...]  # template.anchor_node_indices, carried alongside for convenience
    transform: SimilarityTransform
    node_map: dict[int, int]  # ALL template node indices -> nearest candidate node index
    score: MatchScore


def passes_size_prefilter(template_graph: CharGraph, candidate_graph: CharGraph) -> bool:
    """Step 6.2: "only test vector group graphs whose max degree is >=
    character graph max degree and num end points >= character graph num
    endpoints and num edges >= character graph num edges (a match would
    not be possible otherwise)". "Num endpoints" uses this package's own
    established reading of that phrase -- `graph.py::complexity`'s
    `num_nodes()` (every node, not just degree-1 leaves), the same
    interpretation `docs/cad_font_vector_recognition.md`'s "Open design
    notes" section calls out for reuse here.

    A correctness-preserving prune -- it can never reject a candidate that
    could actually match this template, so callers apply it unconditionally
    rather than behind a toggle."""
    return (
        candidate_graph.max_degree() >= template_graph.max_degree()
        and candidate_graph.num_nodes() >= template_graph.num_nodes()
        and candidate_graph.num_edges() >= template_graph.num_edges()
    )


def match_template_against_candidate(
    template: CharacterTemplate,
    candidate_graph: CharGraph,
    *,
    mse_weight: float = DEFAULT_MSE_WEIGHT,
    edge_weight: float = DEFAULT_EDGE_WEIGHT,
    edge_degree_blend: float = DEFAULT_EDGE_DEGREE_BLEND,
    original_point_weight: float = DEFAULT_ORIGINAL_POINT_WEIGHT,
    intersection_point_weight: float = DEFAULT_INTERSECTION_POINT_WEIGHT,
    synthetic_point_weight: float = DEFAULT_SYNTHETIC_POINT_WEIGHT,
    candidate_tree: "cKDTree | None" = None,
) -> list[CandidateMatch]:
    """Step 6 (6.2-6.5), scoped to one template vs one candidate graph.

    Bails out immediately (`[]`, no search at all) if `passes_size_prefilter`
    (step 6.2) fails -- the cheapest possible rejection for a candidate that
    structurally cannot match this template.

    Otherwise: a similarity transform (scale + rotation + translation, no
    shear) is fully determined by exactly 2 point correspondences --
    `fit_similarity_transform` handles N == 2 exactly -- so only the
    template's *first 2* anchors (`template.anchor_node_indices[:2]`) are
    searched combinatorially via `enumerate_anchor_correspondences`
    (`O(n^2)` hypotheses, `n` = candidate critical-point count, down from
    `O(n^3)` for all 3 slots). Any 3rd anchor is never searched -- it falls
    out for free from `map_template_nodes_to_candidate`'s full node mapping,
    which every hypothesis needs anyway, and is read back into
    `CandidateMatch.correspondence` so a 3-anchor template still reports a
    full 3-element correspondence. `enumerate_anchor_correspondences` itself
    is untouched and remains usable directly with all 3 slots by a caller
    that wants the old, fully exhaustive (and up to N=3 exact, not
    least-squares) search.

    A single `build_candidate_tree(candidate_graph)` is built once (or
    reused, if `candidate_tree` is passed in by a caller iterating several
    templates against the same candidate) and shared by every hypothesis's
    `map_template_nodes_to_candidate` call -- `O(t*log m)` per hypothesis
    instead of `O(t*m)` (`t` = template node count, `m` = candidate node
    count). Overall: `O(n^2 * t * log m)`, down from `O(n^3 * t * m)`.

    Every valid correspondence is fit and scored -- one `CandidateMatch`
    per tested hypothesis, completely unfiltered beyond the 6.2 prefilter
    (no top-K, no similarity threshold; a caller decides sort order and
    what to render)."""
    template_graph = template.graph
    if not passes_size_prefilter(template_graph, candidate_graph):
        return []

    anchor_indices = template.anchor_node_indices
    search_anchor_indices = anchor_indices[:2]
    candidate_critical = critical_point_indices(candidate_graph)
    correspondences = enumerate_anchor_correspondences(
        template_graph, search_anchor_indices, candidate_graph, candidate_critical,
    )
    template_critical = critical_point_indices(template_graph) | template_graph.synthetic_anchor_indices
    if candidate_tree is None:
        candidate_tree = build_candidate_tree(candidate_graph)

    results: list[CandidateMatch] = []
    for corr in tqdm(correspondences, desc=f"matching {template.text!r}"):
        src = [template_graph.nodes[a] for a in search_anchor_indices]
        dst = [candidate_graph.nodes[c] for c in corr]
        transform = fit_similarity_transform(src, dst)
        if transform is None:
            continue
        node_map = map_template_nodes_to_candidate(
            template_graph, transform, candidate_graph, candidate_tree=candidate_tree,
        )
        score = score_match(
            template_graph, template_critical, transform, candidate_graph, node_map,
            mse_weight=mse_weight, edge_weight=edge_weight, edge_degree_blend=edge_degree_blend,
            original_point_weight=original_point_weight,
            intersection_point_weight=intersection_point_weight,
            synthetic_point_weight=synthetic_point_weight,
        )
        full_correspondence = tuple(node_map[a] for a in anchor_indices)
        results.append(CandidateMatch(
            correspondence=full_correspondence, anchor_indices=anchor_indices,
            transform=transform, node_map=node_map, score=score,
        ))
    return results
