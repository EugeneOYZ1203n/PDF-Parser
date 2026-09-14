"""Vector-provenance metrics: categories 9-12 of the metrics rework (vector
pairing, vector count accuracy, endpoint accuracy, per-property accuracy),
plus the vector half of category 1 (label description).

3 vector types (`VECTOR_TYPES`): `original_vector` (real `Vector`-level GT,
matched via `path_signature` against pipeline `Vector` predictions),
`vector_to_raster` (`GeometryAnnotation` entries with `source=="auto"`),
`original_raster` (`GeometryAnnotation` entries with `source=="manual"`).

Pure -- no pipeline/label_schema import; callers (`adapters.py`) build
`GeometryEntry` lists from `Vector`/`GeometryAnnotation` objects via
`geometry_entries_from_vector`/`geometry_entries_from_annotations`, which DO
accept those richer objects (duck-typed, not imported) since this is the
natural point to decompose them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from rastervec.Evaluation.Evaluate.metrics import Ratio

VECTOR_TYPES: tuple[str, ...] = ("original_vector", "vector_to_raster", "original_raster")

Point = tuple[float, float]

_NA = Ratio(0.0, math.nan)


@dataclass(frozen=True)
class VectorMetricConfig:
    endpoint_tolerance_pt: float = 2.0


# --------------------------------------------------------------------------
# GeometryEntry -- unifies GT GeometryAnnotation / full Vector GT / Vector
# predictions into one comparable shape.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GeometryEntry:
    kind: str  # "l" | "c"
    points: tuple[Point, ...]
    color: "tuple[float, ...] | None" = None
    fill: "tuple[float, ...] | None" = None
    width: "float | None" = None
    dashes: "str | None" = None
    opacity: "float | None" = None
    close_path: "bool | None" = None
    line_cap: "int | None" = None
    line_join: "int | None" = None
    even_odd: "bool | None" = None
    stroke_opacity: "float | None" = None
    fill_opacity: "float | None" = None
    layer: "str | None" = None
    blendmode: "str | None" = None

    @property
    def endpoints(self) -> tuple[Point, Point]:
        return (self.points[0], self.points[-1])


def geometry_entries_from_annotations(anns: list) -> list[GeometryEntry]:
    """`anns`: `list[GeometryAnnotation]` (duck-typed)."""
    return [
        GeometryEntry(
            kind=a.kind, points=tuple(a.points), color=a.color, fill=a.fill,
            width=a.width, dashes=a.dashes, opacity=a.opacity,
        )
        for a in anns
    ]


def geometry_entries_from_vector(v) -> list[GeometryEntry]:
    """Decomposes one `Vector` (duck-typed) into `GeometryEntry` "l"/"c"
    primitives -- "re"/"qu" expand to their 4 edges as 4 lines, matching
    `label_schema.geometry_annotations_for_vector`'s own decomposition, so
    an `original_vector` GT (real `Vector`) and prediction (real `Vector`)
    compare on identical primitive granularity. Every sub-entry carries the
    FULL Vector's properties."""
    from rastervec.helpers.geometry import item_points

    common = dict(
        color=v.color, fill=v.fill, width=v.width, dashes=v.dashes,
        opacity=v.stroke_opacity, close_path=bool(v.closePath) if v.closePath is not None else None,
        line_cap=v.lineCap, line_join=v.lineJoin, even_odd=bool(v.even_odd),
        stroke_opacity=v.stroke_opacity, fill_opacity=v.fill_opacity,
        layer=v.layer, blendmode=v.blendmode,
    )
    out: list[GeometryEntry] = []
    for item in v.items:
        kind = item[0]
        pts = item_points(item)
        if kind == "l":
            out.append(GeometryEntry(kind="l", points=tuple(pts), **common))
        elif kind == "c":
            out.append(GeometryEntry(kind="c", points=tuple(pts), **common))
        elif kind in ("re", "qu"):
            if kind == "re":
                (x0, y0), (x1, y1) = pts
                corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            else:
                corners = pts
            for p1, p2 in zip(corners, corners[1:] + corners[:1]):
                out.append(GeometryEntry(kind="l", points=(p1, p2), **common))
    return out


# --------------------------------------------------------------------------
# Category 1 (vector half) -- label description
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VectorLabelStats:
    vector_type: str
    count: int


def vector_label_stats(
    original_vector_count: int,
    vector_to_raster_geoms: list,
    original_raster_geoms: list,
) -> list[VectorLabelStats]:
    return [
        VectorLabelStats("original_vector", original_vector_count),
        VectorLabelStats("vector_to_raster", len(vector_to_raster_geoms)),
        VectorLabelStats("original_raster", len(original_raster_geoms)),
    ]


# --------------------------------------------------------------------------
# Category 9 -- vector pairing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VectorPairing:
    vector_type: str
    paired: "list[tuple[int, int]]"  # (gt_idx, pred_idx), same orientation as compared
    paired_reversed: "list[bool]"  # per pair, whether pred endpoints were reversed to match
    unpaired_gt: "list[int]"
    unpaired_pred: "list[int]"


def _endpoint_dist(a: tuple[Point, Point], b: tuple[Point, Point]) -> float:
    return math.dist(a[0], b[0]) + math.dist(a[1], b[1])


def pair_vectors(
    gt: list[GeometryEntry], preds: list[GeometryEntry], *,
    vector_type: str = "", tolerance: float = 2.0,
) -> VectorPairing:
    """Greedy nearest-first matching within same `kind` only. Distance is
    the smaller of the forward/reversed endpoint-sum distance (a line/curve
    has no guaranteed start/end orientation across GT vs. prediction
    extraction). Not optimal (no Hungarian assignment) -- acceptable at
    small tolerances per the approved plan."""
    candidates: list[tuple[float, int, int, bool]] = []  # (dist, gi, pj, reversed)
    for gi, g in enumerate(gt):
        for pj, p in enumerate(preds):
            if g.kind != p.kind:
                continue
            fwd = _endpoint_dist(g.endpoints, p.endpoints)
            rev = _endpoint_dist(g.endpoints, (p.endpoints[1], p.endpoints[0]))
            if fwd <= rev:
                if fwd <= tolerance:
                    candidates.append((fwd, gi, pj, False))
            else:
                if rev <= tolerance:
                    candidates.append((rev, gi, pj, True))
    candidates.sort(key=lambda c: c[0])

    paired_gt: set[int] = set()
    paired_pred: set[int] = set()
    paired: list[tuple[int, int]] = []
    reversed_flags: list[bool] = []
    for _dist, gi, pj, rev in candidates:
        if gi in paired_gt or pj in paired_pred:
            continue
        paired_gt.add(gi)
        paired_pred.add(pj)
        paired.append((gi, pj))
        reversed_flags.append(rev)

    unpaired_gt = [gi for gi in range(len(gt)) if gi not in paired_gt]
    unpaired_pred = [pj for pj in range(len(preds)) if pj not in paired_pred]
    return VectorPairing(
        vector_type=vector_type, paired=paired, paired_reversed=reversed_flags,
        unpaired_gt=unpaired_gt, unpaired_pred=unpaired_pred,
    )


# --------------------------------------------------------------------------
# Category 10 -- vector count accuracy
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VectorCountStats:
    vector_type: str
    n_paired: int
    n_missed: int
    n_spurious: int
    precision: Ratio
    recall: Ratio


def vector_count_stats(pairing: VectorPairing) -> VectorCountStats:
    n_paired = len(pairing.paired)
    n_missed = len(pairing.unpaired_gt)
    n_spurious = len(pairing.unpaired_pred)
    return VectorCountStats(
        vector_type=pairing.vector_type,
        n_paired=n_paired, n_missed=n_missed, n_spurious=n_spurious,
        precision=Ratio(float(n_paired), float(n_paired + n_spurious)),
        recall=Ratio(float(n_paired), float(n_paired + n_missed)),
    )


# --------------------------------------------------------------------------
# Category 11 -- endpoint accuracy
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EndpointAccuracyStats:
    vector_type: str
    rmse: float
    n_paired: int


def endpoint_accuracy_stats(
    pairing: VectorPairing, gt: list[GeometryEntry], preds: list[GeometryEntry],
) -> EndpointAccuracyStats:
    if not pairing.paired:
        return EndpointAccuracyStats(vector_type=pairing.vector_type, rmse=math.nan, n_paired=0)
    sq_errors: list[float] = []
    for (gi, pj), rev in zip(pairing.paired, pairing.paired_reversed):
        g0, g1 = gt[gi].endpoints
        p0, p1 = preds[pj].endpoints
        if rev:
            p0, p1 = p1, p0
        for (gx, gy), (px, py) in ((g0, p0), (g1, p1)):
            sq_errors.append((gx - px) ** 2 + (gy - py) ** 2)
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
    return EndpointAccuracyStats(vector_type=pairing.vector_type, rmse=rmse, n_paired=len(pairing.paired))


# --------------------------------------------------------------------------
# Category 12 -- per-property accuracy
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PropertyAccuracyRow:
    vector_type: str
    property_name: str
    kind: str  # "continuous" | "continuous_color" | "discrete"
    metric_value: float
    n_applicable: int
    applicable: bool
    none_vs_none_count: int = 0


# (property_name, kind, "both" | "full_vector")
_PROPERTY_SPECS: "tuple[tuple[str, str, str], ...]" = (
    ("color", "continuous_color", "both"),
    ("fill", "continuous_color", "both"),
    ("opacity", "continuous", "both"),
    ("width", "continuous", "both"),
    ("dashes", "discrete", "both"),
    ("close_path", "discrete", "full_vector"),
    ("line_cap", "discrete", "full_vector"),
    ("line_join", "discrete", "full_vector"),
    ("even_odd", "discrete", "full_vector"),
    ("stroke_opacity", "continuous", "full_vector"),
    ("fill_opacity", "continuous", "full_vector"),
    ("layer", "discrete", "full_vector"),
    ("blendmode", "discrete", "full_vector"),
)


def _color_distance(a: "tuple[float, ...] | None", b: "tuple[float, ...] | None"):
    """Euclidean distance over matching channels; `None` returned when
    either side has no color (caller decides how to handle)."""
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))


def _property_row(
    vector_type: str, name: str, kind: str,
    gt_vals: list, pred_vals: list,
) -> PropertyAccuracyRow:
    if kind == "continuous_color":
        none_vs_none = 0
        dists: list[float] = []
        for gv, pv in zip(gt_vals, pred_vals):
            if gv is None and pv is None:
                none_vs_none += 1
                continue
            d = _color_distance(gv, pv)
            if d is not None:
                dists.append(d)
        if not dists:
            return PropertyAccuracyRow(vector_type, name, kind, math.nan, 0, True, none_vs_none)
        rmse = math.sqrt(sum(d * d for d in dists) / len(dists))
        return PropertyAccuracyRow(vector_type, name, kind, rmse, len(dists), True, none_vs_none)

    if kind == "continuous":
        pairs = [(gv, pv) for gv, pv in zip(gt_vals, pred_vals) if gv is not None and pv is not None]
        if not pairs:
            return PropertyAccuracyRow(vector_type, name, kind, math.nan, 0, True)
        rmse = math.sqrt(sum((gv - pv) ** 2 for gv, pv in pairs) / len(pairs))
        return PropertyAccuracyRow(vector_type, name, kind, rmse, len(pairs), True)

    # discrete -- None == None counts as a match (a genuine comparable state).
    n = len(gt_vals)
    if n == 0:
        return PropertyAccuracyRow(vector_type, name, kind, math.nan, 0, True)
    matches = sum(1 for gv, pv in zip(gt_vals, pred_vals) if gv == pv)
    return PropertyAccuracyRow(vector_type, name, kind, matches / n, n, True)


def property_accuracy_table(
    pairing: VectorPairing, gt: list[GeometryEntry], preds: list[GeometryEntry], vector_type: str,
) -> list[PropertyAccuracyRow]:
    is_full_vector = vector_type == "original_vector"
    rows: list[PropertyAccuracyRow] = []
    for name, kind, scope in _PROPERTY_SPECS:
        if scope == "full_vector" and not is_full_vector:
            rows.append(PropertyAccuracyRow(vector_type, name, kind, math.nan, 0, False))
            continue
        gt_vals = [getattr(gt[gi], name) for gi, _pj in pairing.paired]
        pred_vals = [getattr(preds[pj], name) for _gi, pj in pairing.paired]
        rows.append(_property_row(vector_type, name, kind, gt_vals, pred_vals))
    return rows


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
@dataclass
class PerTypeVectorResult:
    label_stats: VectorLabelStats
    pairing: VectorPairing
    count_stats: VectorCountStats
    endpoint_stats: EndpointAccuracyStats
    property_rows: list[PropertyAccuracyRow]


@dataclass
class VectorMetricSuiteResult:
    by_type: dict[str, PerTypeVectorResult]


def evaluate_vector_metrics(
    gt_by_type: "dict[str, list[GeometryEntry]]",
    preds_by_type: "dict[str, list[GeometryEntry]]",
    label_counts: "dict[str, int]",
    *,
    cfg: VectorMetricConfig = VectorMetricConfig(),
) -> VectorMetricSuiteResult:
    label_rows = {
        r.vector_type: r for r in vector_label_stats(
            label_counts.get("original_vector", 0),
            [None] * label_counts.get("vector_to_raster", 0),
            [None] * label_counts.get("original_raster", 0),
        )
    }
    by_type: dict[str, PerTypeVectorResult] = {}
    for vector_type in VECTOR_TYPES:
        gt = gt_by_type.get(vector_type, [])
        preds = preds_by_type.get(vector_type, [])
        pairing = pair_vectors(gt, preds, vector_type=vector_type, tolerance=cfg.endpoint_tolerance_pt)
        by_type[vector_type] = PerTypeVectorResult(
            label_stats=label_rows[vector_type],
            pairing=pairing,
            count_stats=vector_count_stats(pairing),
            endpoint_stats=endpoint_accuracy_stats(pairing, gt, preds),
            property_rows=property_accuracy_table(pairing, gt, preds, vector_type),
        )
    return VectorMetricSuiteResult(by_type=by_type)


def aggregate_vector_metrics(results: "list[VectorMetricSuiteResult]") -> "VectorMetricSuiteResult | None":
    if not results:
        return None
    by_type: dict[str, PerTypeVectorResult] = {}
    for vector_type in VECTOR_TYPES:
        per_type = [r.by_type[vector_type] for r in results]
        label_count = sum(p.label_stats.count for p in per_type)
        n_paired = sum(p.count_stats.n_paired for p in per_type)
        n_missed = sum(p.count_stats.n_missed for p in per_type)
        n_spurious = sum(p.count_stats.n_spurious for p in per_type)
        count_stats = VectorCountStats(
            vector_type=vector_type, n_paired=n_paired, n_missed=n_missed, n_spurious=n_spurious,
            precision=Ratio(float(n_paired), float(n_paired + n_spurious)),
            recall=Ratio(float(n_paired), float(n_paired + n_missed)),
        )
        sq_total = sum(
            (p.endpoint_stats.rmse ** 2) * p.endpoint_stats.n_paired
            for p in per_type if p.endpoint_stats.n_paired
        )
        n_ep_total = sum(p.endpoint_stats.n_paired for p in per_type)
        endpoint_stats = EndpointAccuracyStats(
            vector_type=vector_type,
            rmse=math.sqrt(sq_total / n_ep_total) if n_ep_total else math.nan,
            n_paired=n_ep_total,
        )
        # Property rows: weighted re-aggregation by n_applicable / metric_value.
        property_rows: list[PropertyAccuracyRow] = []
        n_props = len(per_type[0].property_rows) if per_type else 0
        for idx in range(n_props):
            rows = [p.property_rows[idx] for p in per_type]
            name, kind = rows[0].property_name, rows[0].kind
            if not any(r.applicable for r in rows):
                property_rows.append(PropertyAccuracyRow(vector_type, name, kind, math.nan, 0, False))
                continue
            applicable_rows = [r for r in rows if r.applicable and r.n_applicable]
            none_vs_none = sum(r.none_vs_none_count for r in rows)
            if not applicable_rows:
                property_rows.append(
                    PropertyAccuracyRow(vector_type, name, kind, math.nan, 0, True, none_vs_none)
                )
                continue
            n_total = sum(r.n_applicable for r in applicable_rows)
            if kind == "discrete":
                value = sum(r.metric_value * r.n_applicable for r in applicable_rows) / n_total
            else:
                value = math.sqrt(
                    sum((r.metric_value ** 2) * r.n_applicable for r in applicable_rows) / n_total
                )
            property_rows.append(
                PropertyAccuracyRow(vector_type, name, kind, value, n_total, True, none_vs_none)
            )
        by_type[vector_type] = PerTypeVectorResult(
            label_stats=VectorLabelStats(vector_type, label_count),
            pairing=VectorPairing(vector_type, [], [], [], []),  # not meaningful post-aggregation
            count_stats=count_stats, endpoint_stats=endpoint_stats, property_rows=property_rows,
        )
    return VectorMetricSuiteResult(by_type=by_type)
