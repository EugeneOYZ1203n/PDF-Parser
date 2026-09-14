"""Symbol recognition via constraint-propagation network (Dosch 2000 Sec 3.3,
Ah-Soon's model). A minimal but structurally faithful implementation of the
paper's 5 node kinds:
  NNSegment/NNArc  - input nodes, one per feature kind, seed the network
  NNCondition      - single father, tests+scores one constraint, propagates on success
  NNMerge          - two fathers, merges compatible feature bundles from both sides
  NNFinal          - terminal node per symbol; whatever reaches it IS the symbol instance
Feature bundles accumulate a soft error score as they flow down; a node drops
a bundle once its error exceeds max_error (this is the paper's tolerance for
noisy/approximated vectorization, not exact matching).

First cut: 2 of the paper's 7 symbol families (door, window) -- enough to
exercise NNMerge and node-chaining (merge feeding another merge) without a
real symbol-description-file parser yet.

`_Bridge` is a small implementation-only adapter (not one of the paper's node
kinds): it routes a plain son's `receive()` call into `NNMerge.receive_a`/
`receive_b`, since `NNMerge` has 2 distinguishable fathers but `NNNode.add_son`
doesn't carry a side label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .geom import angle_gap, dist, heading_deg, point_to_segment_dist
from .types_ import Arc, Segment, SymbolInstance


@dataclass
class Feature:
    kind: str                        # "segment" | "arc"
    obj: Segment | Arc
    at: tuple[float, float]          # the connection point relevant to downstream matching


@dataclass
class Bundle:
    features: list[Feature]
    error: float = 0.0


class NNNode:
    def __init__(self):
        self.sons: list[NNNode] = []

    def add_son(self, node: "NNNode") -> "NNNode":
        self.sons.append(node)
        return node

    def _propagate(self, bundle: Bundle) -> None:
        for son in self.sons:
            son.receive(bundle)

    def receive(self, bundle: Bundle) -> None:
        raise NotImplementedError


class NNSegment(NNNode):
    """Input node: feed every Segment on the page; only ones passing the
    pre-filter turn into a 1-feature Bundle propagated to sons."""

    def __init__(self, *, min_len: float = 0.0, max_len: float = 1e9,
                 thick: bool | None = None, dashed: bool = False):
        super().__init__()
        self.min_len, self.max_len, self.thick, self.dashed = min_len, max_len, thick, dashed

    def feed(self, seg: Segment) -> None:
        length = dist(seg.p0, seg.p1)
        if not (self.min_len <= length <= self.max_len):
            return
        if seg.dashed != self.dashed:
            return
        if self.thick is not None and seg.thick != self.thick:
            return
        for at in (seg.p0, seg.p1):
            self._propagate(Bundle([Feature("segment", seg, at)]))


class NNArc(NNNode):
    def __init__(self, *, min_radius: float = 0.0, max_radius: float = 1e9,
                 min_sweep_deg: float = 0.0, max_sweep_deg: float = 360.0):
        super().__init__()
        self.min_radius, self.max_radius = min_radius, max_radius
        self.min_sweep_deg, self.max_sweep_deg = min_sweep_deg, max_sweep_deg

    def feed(self, arc: Arc) -> None:
        if not (self.min_radius <= arc.radius <= self.max_radius):
            return
        sweep = abs(arc.a1 - arc.a0) % 360.0
        if not (self.min_sweep_deg <= sweep <= self.max_sweep_deg):
            return
        for at in (arc.polyline[0], arc.polyline[-1]):
            self._propagate(Bundle([Feature("arc", arc, at)]))


class NNCondition(NNNode):
    """Single father. `test(bundle) -> float | None`: None rejects, else the
    incremental error to add before propagating."""

    def __init__(self, test: Callable[[Bundle], float | None], max_error: float = 1.0):
        super().__init__()
        self.test, self.max_error = test, max_error

    def receive(self, bundle: Bundle) -> None:
        inc = self.test(bundle)
        if inc is None:
            return
        nb = Bundle(bundle.features, bundle.error + inc)
        if nb.error <= self.max_error:
            self._propagate(nb)


class NNMerge(NNNode):
    """Two fathers (routed via `receive_a`/`receive_b`, normally through a
    `_Bridge` adapter). Buffers incoming bundles per side; when a side-A and
    side-B bundle's last connection point are within `merge_tol` of each
    other, `test(bundle_a, bundle_b) -> float | None` is evaluated and, on
    success, a combined bundle propagates.

    `merge_tol` set very large effectively delegates all pairing to `test`
    itself (used when the constraint is a numeric relationship rather than
    shared-point proximity, e.g. window jamb pairs) -- a documented wrinkle,
    not a redesign; a cleaner v2 would let NNMerge take a custom bucketing key
    instead of hardcoding point-distance.
    """

    def __init__(self, test: Callable[[Bundle, Bundle], float | None],
                 merge_tol: float = 4.0, max_error: float = 1.0):
        super().__init__()
        self.test, self.merge_tol, self.max_error = test, merge_tol, max_error
        self._pending_a: list[Bundle] = []
        self._pending_b: list[Bundle] = []

    def receive_a(self, bundle: Bundle) -> None:
        self._try_merge(bundle, self._pending_b, self._pending_a, is_a=True)

    def receive_b(self, bundle: Bundle) -> None:
        self._try_merge(bundle, self._pending_a, self._pending_b, is_a=False)

    def _try_merge(self, bundle: Bundle, other_side: list[Bundle], own_side: list[Bundle], is_a: bool) -> None:
        for other in other_side:
            a, b = (bundle, other) if is_a else (other, bundle)
            if dist(a.features[-1].at, b.features[-1].at) > self.merge_tol:
                continue
            inc = self.test(a, b)
            if inc is None:
                continue
            merged_error = a.error + b.error + inc
            if merged_error <= self.max_error:
                combined = Bundle(a.features + b.features, merged_error)
                self._propagate(combined)
        own_side.append(bundle)


class NNFinal(NNNode):
    def __init__(self, family: str, max_error: float = 1.0):
        super().__init__()
        self.family, self.max_error = family, max_error
        self.instances: list[SymbolInstance] = []

    def receive(self, bundle: Bundle) -> None:
        if bundle.error > self.max_error:
            return
        pts = [f.at for f in bundle.features]
        for f in bundle.features:
            if f.kind == "segment":
                pts.append(f.obj.p0)
                pts.append(f.obj.p1)
        arr = np.array(pts, float)
        anchor = tuple(bundle.features[0].at)
        bbox = (float(arr[:, 0].min()), float(arr[:, 1].min()),
                float(arr[:, 0].max()), float(arr[:, 1].max()))
        self.instances.append(SymbolInstance(
            family=self.family, features=[f.obj for f in bundle.features],
            anchor=anchor, bbox=bbox, error=bundle.error,
        ))


class _Bridge(NNNode):
    """Implementation-only wiring adapter, see module docstring."""

    def __init__(self, merge: NNMerge, side: str):
        super().__init__()
        self._merge, self._side = merge, side

    def receive(self, bundle: Bundle) -> None:
        if self._side == "a":
            self._merge.receive_a(bundle)
        else:
            self._merge.receive_b(bundle)


# --------------------------------------------------------------- family networks


def _door_network(*, len_ratio_tol: float = 0.25, radial_ang_tol_deg: float = 20.0,
                   sweep_tol_deg: float = 30.0, max_error: float = 1.0):
    """door = swing arc (~quarter circle) + straight leaf segment sharing the
    hinge point, leaf length approx == arc radius, leaf direction approx ==
    radial direction (center->hinge) at the shared point."""
    leaf = NNSegment(min_len=8.0, max_len=140.0, dashed=False)
    swing = NNArc(min_radius=8.0, max_radius=140.0, min_sweep_deg=45.0, max_sweep_deg=135.0)

    def door_merge_test(seg_b: Bundle, arc_b: Bundle) -> float | None:
        seg: Segment = seg_b.features[0].obj
        arc: Arc = arc_b.features[0].obj
        hinge = arc_b.features[0].at
        leaf_len = dist(seg.p0, seg.p1)
        if arc.radius <= 0:
            return None
        len_err = abs(leaf_len - arc.radius) / arc.radius
        if len_err > len_ratio_tol:
            return None
        radial_heading = heading_deg(arc.center, hinge)
        far = seg.p1 if dist(seg.p0, hinge) <= dist(seg.p1, hinge) else seg.p0
        leaf_heading = heading_deg(hinge, far)
        ang_err = angle_gap(radial_heading, leaf_heading, 360.0)
        if ang_err > radial_ang_tol_deg:
            return None
        sweep = abs(arc.a1 - arc.a0) % 360.0
        sweep_err = abs(sweep - 90.0)
        if sweep_err > sweep_tol_deg:
            return None
        return (0.5 * (len_err / len_ratio_tol)
                + 0.3 * (ang_err / radial_ang_tol_deg)
                + 0.2 * (sweep_err / sweep_tol_deg))

    merge = NNMerge(door_merge_test, merge_tol=4.0, max_error=max_error)
    final = NNFinal("door", max_error=max_error)
    merge.add_son(final)
    leaf.add_son(_Bridge(merge, side="a"))
    swing.add_son(_Bridge(merge, side="b"))
    return [leaf, swing], final


def _window_network(*, jamb_len_tol: float = 0.3, gap_min: float = 18.0, gap_max: float = 140.0,
                     perp_ang_tol_deg: float = 15.0, wall_dist_tol: float = 6.0, max_error: float = 1.0):
    """window = two short jamb ticks, roughly parallel, roughly equal length,
    spaced gap_min..gap_max apart, both roughly perpendicular to (and with
    their base points roughly on) a nearby wall segment."""
    jamb_a = NNSegment(min_len=4.0, max_len=30.0, thick=False, dashed=False)
    jamb_b = NNSegment(min_len=4.0, max_len=30.0, thick=False, dashed=False)
    wall = NNSegment(min_len=30.0, max_len=1e9, thick=True, dashed=False)

    def jamb_pair_test(a: Bundle, b: Bundle) -> float | None:
        sa, sb = a.features[0].obj, b.features[0].obj
        if sa is sb:
            return None
        la, lb = dist(sa.p0, sa.p1), dist(sb.p0, sb.p1)
        len_err = abs(la - lb) / max(la, lb)
        if len_err > jamb_len_tol:
            return None
        ha = heading_deg(sa.p0, sa.p1) % 180.0
        hb = heading_deg(sb.p0, sb.p1) % 180.0
        par_err = angle_gap(ha, hb, 180.0)
        if par_err > perp_ang_tol_deg:
            return None
        gap = dist(a.features[0].at, b.features[0].at)
        if not (gap_min <= gap <= gap_max):
            return None
        return 0.5 * (len_err / jamb_len_tol) + 0.5 * (par_err / perp_ang_tol_deg)

    pair = NNMerge(jamb_pair_test, merge_tol=1e9, max_error=max_error)
    jamb_a.add_son(_Bridge(pair, side="a"))
    jamb_b.add_son(_Bridge(pair, side="b"))

    def pair_wall_test(pair_b: Bundle, wall_b: Bundle) -> float | None:
        wall_seg = wall_b.features[0].obj
        jamb1, jamb2 = pair_b.features[0].obj, pair_b.features[1].obj
        heading_j = heading_deg(jamb1.p0, jamb1.p1) % 180.0
        heading_w = heading_deg(wall_seg.p0, wall_seg.p1) % 180.0
        perp_err = abs(angle_gap(heading_j, heading_w, 180.0) - 90.0)
        if perp_err > perp_ang_tol_deg:
            return None
        # a jamb tick straddles the wall symmetrically -- its MIDPOINT is the
        # point that should sit on the wall line, not either endpoint (which
        # are each ~half the jamb length away from the wall on either side).
        mid1 = ((jamb1.p0[0] + jamb1.p1[0]) / 2, (jamb1.p0[1] + jamb1.p1[1]) / 2)
        mid2 = ((jamb2.p0[0] + jamb2.p1[0]) / 2, (jamb2.p0[1] + jamb2.p1[1]) / 2)
        d0 = point_to_segment_dist(mid1, wall_seg.p0, wall_seg.p1)
        d1 = point_to_segment_dist(mid2, wall_seg.p0, wall_seg.p1)
        base_err = max(d0, d1)
        if base_err > wall_dist_tol:
            return None
        return 0.6 * (perp_err / perp_ang_tol_deg) + 0.4 * (base_err / wall_dist_tol)

    final_merge = NNMerge(pair_wall_test, merge_tol=1e9, max_error=max_error)
    final = NNFinal("window", max_error=max_error)
    final_merge.add_son(final)
    pair.add_son(_Bridge(final_merge, side="a"))
    wall.add_son(_Bridge(final_merge, side="b"))
    return [jamb_a, jamb_b, wall], final


_NETWORKS = {"door": _door_network, "window": _window_network}


def recognize(
    segments: list[Segment], arcs: list[Arc], *,
    families: tuple[str, ...] = ("door", "window"),
    max_error: float = 1.0,
) -> list[SymbolInstance]:
    finals: list[NNFinal] = []
    seg_roots: list[NNSegment] = []
    arc_roots: list[NNArc] = []
    for fam in families:
        build = _NETWORKS.get(fam)
        if build is None:
            continue
        roots, final = build(max_error=max_error)
        for r in roots:
            (seg_roots if isinstance(r, NNSegment) else arc_roots).append(r)
        finals.append(final)

    for seg in segments:
        for root in seg_roots:
            root.feed(seg)
    for arc in arcs:
        for root in arc_roots:
            root.feed(arc)

    out: list[SymbolInstance] = []
    used_ids: set[int] = set()
    all_instances = sorted(
        (si for f in finals for si in f.instances), key=lambda si: si.error
    )
    for si in all_instances:
        ids = {id(x) for x in si.features}
        if ids & used_ids:
            continue
        used_ids |= ids
        out.append(si)
    return out
