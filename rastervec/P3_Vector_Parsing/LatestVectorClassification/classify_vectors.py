"""LatestVectorClassification classification: turn raw vectors into text-candidate
clusters + drawing content.

Per `(layer, color, width)` bucket, `_classify_bucket` runs seven named steps:

0. **Oversize** -- every Vector whose bbox area is at least
   `MAX_VECTOR_PAGE_AREA_FRAC` of the page (a border, frame or background
   fill) is dropped to drawing.
1. **Collinear drawing** -- group the bucket's straight Vectors by same
   infinite line (`line_geometry.group_collinear`); a group with more than
   `COLLINEAR_DRAWING_MIN_COUNT` members whose length std is below
   `COLLINEAR_DRAWING_MAX_STD_PT` (a long dashed/repeated line) is dropped
   to drawing.
2. **Pattern lattice** (`pattern_lattice.pattern_drawing`) -- similar
   Vectors (same item kinds + per-item lengths, hashed into 3 pt buckets)
   repeated on a regular translation lattice (each member within
   `PATTERN_GRID_TOL_PT` of the global grid point); a lattice group with
   more than `PATTERN_MAX_GROUP` members (1D: `PATTERN_MAX_GROUP_1D`) *and*
   fewer than `PATTERN_LINK_FOREIGN_MEAN_LIMIT` unique foreign bboxes per
   link between adjacent members is dropped to drawing (`pattern`); a
   similarity bucket with more than `PATTERN_BUCKET_DRAWING_GROUPS` such
   groups drops its other groups of more than `PATTERN_MAX_GROUP` members
   too (`pattern_bucket`). Groups failing only the in-between test are
   recorded in an `info` category (`pattern_rejected`).
3. **Seq overlap merge** (`group_filters.combine_overlapping_seq`).
4. **Spatial cluster** (`cluster_filters.cluster_spatial_groups`). Both
   merges are capped: no group/cluster grows to a bbox of
   `MAX_CLUSTER_PAGE_AREA_FRAC` of the page or more (bounds each cluster's
   OCR render).
5. **Length outliers** -- per cluster, pool the lengths of every straight
   Vector in a parallel group (>= `MIN_PARALLEL_GROUP_SIZE` same-angle
   members); drop the ones more than `LENGTH_OUTLIER_STD` std from the mean.
6. **Crossings** -- per cluster (`crossed_grid`): flag every Vector made
   only of "l" items that at least `MIN_CROSSINGS` distinct foreign pieces
   properly cross (`line_geometry.line_crossing_counts` -- no flattening;
   segments and rect/quad edges count once each, every curve crossing
   counts); then drop to drawing only the flagged Vectors on the dominant
   parallel/perpendicular grid (`line_geometry.dominant_grid`: within
   `GRID_ANGLE_TOL_DEG` mod 90, holding more than `GRID_DOMINANCE` of the
   flagged "l" length). The rest of the flagged Vectors stay, recorded in
   an `info` category for the debug layers.

`classify_vectors` also returns the page's **global potential angles**:
the mean angle of every collinear group (step 1's grouping) with at least
`GLOBAL_ANGLE_MIN_GROUP_SIZE` members, across every bucket (drawing groups
included, singletons not), deduped within `ANGLE_TOL_DEG`. `parse.py` snaps
each detect quad's long-edge angle to the nearest one within
`QUAD_ANGLE_SNAP_TOL_DEG`.

Every step is timed through an optional `StepClock` (summed across
buckets): `classify_separate`, `classify_oversize`, `classify_collinear`, `classify_pattern`,
`classify_seqno`, `classify_spatial`, `classify_outliers`, `classify_crossings`,
`classify_collect`.

`keep_steps=False` (what `parse.py` passes when nothing will render the
per-step debug layers) keeps only what the result needs -- every dropped
category and the last step's `kept` -- instead of every intermediate step's
kept groups (the oversize/collinear/pattern steps' one-list-per-Vector
`kept` alone is ~3 small lists per Vector).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rastervec.commons.models import Page, Vector
from rastervec.commons.step_timing import StepClock
from rastervec.P3_Vector_Parsing.LatestVectorClassification import cluster_filters as clf
from rastervec.P3_Vector_Parsing.LatestVectorClassification import group_filters as grf
from rastervec.P3_Vector_Parsing.LatestVectorClassification.pattern_lattice import pattern_drawing
from rastervec.P3_Vector_Parsing.LatestVectorClassification.classification import CategoryResult, StepResult
from rastervec.P3_Vector_Parsing.LatestVectorClassification.config import (
    ANGLE_TOL_DEG,
    COLLINEAR_DRAWING_MAX_STD_PT,
    COLLINEAR_DRAWING_MIN_COUNT,
    COLLINEAR_OFFSET_TOL_PT,
    CROSS_EPS_PT,
    CROSS_PAIR_CHUNK,
    GLOBAL_ANGLE_MIN_GROUP_SIZE,
    GRID_ANGLE_TOL_DEG,
    GRID_DOMINANCE,
    LENGTH_OUTLIER_STD,
    MAX_CLUSTER_PAGE_AREA_FRAC,
    MAX_VECTOR_PAGE_AREA_FRAC,
    MIN_CROSSINGS,
    MIN_PARALLEL_GROUP_SIZE,
    SEQ_OVERLAP_TOLERANCE_PX,
    SPATIAL_CLUSTER_THRESHOLD,
    SPATIAL_SIZE_TOLERANCE,
    STRAIGHT_TOL_PT,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.layer_color_separation import (
    separate_by_color, separate_by_layer, separate_by_width,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.line_geometry import (
    dedupe_angles,
    dominant_grid,
    group_angle,
    group_collinear,
    group_length_std,
    group_parallel,
    line_crossing_counts,
    split_straight,
)

OVERSIZE_CATEGORY = "oversize"  # step 0's dropped category
PATTERN_CATEGORY = "pattern"  # step 2's dropped category (one entry per lattice group)
PATTERN_BUCKET_CATEGORY = "pattern_bucket"  # step 2's bucket-wide dropped groups
PATTERN_REJECTED_CATEGORY = "pattern_rejected"  # step 2's too-much-between groups (role "info")
CROSSED_CATEGORY = "crossed"  # step 6's dropped category (parse.py's `intersection` layer)
FLAGGED_KEPT_CATEGORY = "crossed_off_grid"  # step 6's flagged-but-kept Vectors (role "info")
STEP_LABELS = (
    "Oversize", "Collinear drawing", "Pattern lattice", "Seq overlap merge", "Spatial cluster",
    "Length outliers", "Crossings",
)
# The steps whose `kept` entries are single Vectors (no grouping yet).
SINGLE_VECTOR_STEPS = STEP_LABELS[:3]


@dataclass
class ClusteringStageResult:
    """One (layer, color, width) bucket's result: `steps` is exactly
    `_classify_bucket()`'s return value. `steps[-1].categories["kept"]` is
    the final (tiered) clusters; every `role="dropped"` category is
    drawing content."""

    steps: "list[StepResult]"


@dataclass
class ClassificationResult:
    """`text_clusters` is tiered (`list[list[list[Vector]]]` -- per cluster,
    its member groups). `drawing_vectors` is flat -- every dropped Vector.
    `global_angles` is the page's deduped global potential angles (deg,
    folded to [0, 180), sorted -- see the module docstring)."""

    text_clusters: list[list[list[Vector]]]
    drawing_vectors: list[Vector]
    clustering: dict
    global_angles: list[float] = field(default_factory=list)
    vectors_by_layer: dict | None = None
    vectors_by_layer_color: dict | None = None
    vectors_by_layer_color_width: dict | None = None


def oversize(vectors: list[Vector], max_area: float) -> tuple[list[Vector], list[Vector]]:
    """Step 0 for one bucket: `(kept, dropped)` -- `dropped` every Vector
    whose bbox area is at least `max_area`, `kept` the rest, both in input
    order."""
    kept, dropped = [], []
    for v in vectors:
        x0, y0, x1, y1 = v.bbox
        (dropped if (x1 - x0) * (y1 - y0) >= max_area else kept).append(v)
    return kept, dropped


def collinear_drawing(
    vectors: list[Vector],
) -> tuple[list[Vector], list[list[Vector]], list[float]]:
    """Step 1 for one bucket: `(kept, drawing_groups, angles)` -- `kept` in
    input order, `drawing_groups` the collinear groups judged to be drawing,
    `angles` the mean angle of every collinear group with at least
    `GLOBAL_ANGLE_MIN_GROUP_SIZE` members (drawing groups included) -- this
    bucket's share of the global potential angles."""
    straight, _ = split_straight(vectors, STRAIGHT_TOL_PT)
    groups = group_collinear(straight, ANGLE_TOL_DEG, COLLINEAR_OFFSET_TOL_PT)
    drawing_groups = [
        [v for v, _ in g] for g in groups
        if len(g) > COLLINEAR_DRAWING_MIN_COUNT and group_length_std(g) < COLLINEAR_DRAWING_MAX_STD_PT
    ]
    dropped = {id(v) for g in drawing_groups for v in g}
    kept = [v for v in vectors if id(v) not in dropped]
    angles = [group_angle(g) for g in groups if len(g) >= GLOBAL_ANGLE_MIN_GROUP_SIZE]
    return kept, drawing_groups, angles


def length_outliers(cluster_vectors: list[Vector]) -> list[Vector]:
    """Step 5 for one cluster: straight Vectors in parallel groups whose
    length is more than `LENGTH_OUTLIER_STD` pooled std from the pooled mean
    (pooled over every parallel group in the cluster). Nothing when fewer
    than two such strokes or a zero std."""
    straight, _ = split_straight(cluster_vectors, STRAIGHT_TOL_PT)
    members = [
        (v, fit) for g in group_parallel(straight, ANGLE_TOL_DEG)
        if len(g) >= MIN_PARALLEL_GROUP_SIZE for v, fit in g
    ]
    if len(members) < 2:
        return []
    lengths = np.asarray([fit.length for _, fit in members])
    mean, std = float(lengths.mean()), float(lengths.std())
    if std == 0.0:
        return []
    return [v for v, fit in members if abs(fit.length - mean) > LENGTH_OUTLIER_STD * std]


def crossed_grid(cluster_vectors: list[Vector]) -> tuple[list[Vector], list[Vector]]:
    """Step 6 for one cluster: `(dropped, flagged_kept)`. Flagged = "l"-only
    Vectors with at least `MIN_CROSSINGS` foreign crossings; dropped = the
    flagged Vectors on the dominant grid (see the module docstring)."""
    counts = line_crossing_counts(cluster_vectors, eps=CROSS_EPS_PT, chunk=CROSS_PAIR_CHUNK)
    flagged = [v for v, n in zip(cluster_vectors, counts) if n is not None and n >= MIN_CROSSINGS]
    if not flagged:
        return [], []
    dropped = dominant_grid(flagged, tol=GRID_ANGLE_TOL_DEG, dominance=GRID_DOMINANCE)
    gone = {id(v) for v in dropped}
    return dropped, [v for v in flagged if id(v) not in gone]


def _remove_from_clusters(
    clusters: list[list[list[Vector]]], removed_fn,
) -> tuple[list[list[list[Vector]]], list[list[Vector]]]:
    """Apply `removed_fn(flat cluster vectors) -> removed vectors` to every
    cluster, keeping group tiering: `(kept clusters, dropped per cluster)`.
    Emptied groups and clusters disappear."""
    kept_clusters, dropped = [], []
    for cluster in clusters:
        flat = [v for g in cluster for v in g]
        removed = removed_fn(flat)
        if not removed:
            kept_clusters.append(cluster)
            continue
        gone = {id(v) for v in removed}
        dropped.append(removed)
        new_cluster = [[v for v in g if id(v) not in gone] for g in cluster]
        new_cluster = [g for g in new_cluster if g]
        if new_cluster:
            kept_clusters.append(new_cluster)
    return kept_clusters, dropped


def _classify_bucket(
    vectors: list[Vector], clock: "StepClock | None" = None, *,
    page_area: float | None = None, keep_steps: bool = True,
) -> tuple[list[StepResult], list[float]]:
    """The seven-step chain for one bucket (see module docstring), plus the
    bucket's collinear-group angles; each step is timed on `clock` (a
    private one when `None`). `page_area` (pt^2) sets the oversize and
    cluster-area caps; `None` disables both. `keep_steps=False` stores every
    intermediate step's `kept` as `[]` (see the module docstring)."""
    clock = clock or StepClock()
    steps: list[StepResult] = []
    vector_cap = MAX_VECTOR_PAGE_AREA_FRAC * page_area if page_area else None
    cluster_cap = MAX_CLUSTER_PAGE_AREA_FRAC * page_area if page_area else None

    def _kept_singles(kept: list[Vector]) -> list:
        return [[v] for v in kept] if keep_steps else []

    with clock("classify_oversize"):
        if vector_cap is None:
            kept, too_big = list(vectors), []
        else:
            kept, too_big = oversize(vectors, vector_cap)
        steps.append(StepResult(STEP_LABELS[0], {
            "kept": CategoryResult(_kept_singles(kept), "kept"),
            OVERSIZE_CATEGORY: CategoryResult([[v] for v in too_big], "dropped"),
        }))

    with clock("classify_collinear"):
        kept, drawing_groups, angles = collinear_drawing(kept)
        steps.append(StepResult(STEP_LABELS[1], {
            "kept": CategoryResult(_kept_singles(kept), "kept"),
            "drawing": CategoryResult(drawing_groups, "dropped"),
        }))

    with clock("classify_pattern"):
        kept, pattern_groups, bucket_groups, rejected_groups = pattern_drawing(kept)
        steps.append(StepResult(STEP_LABELS[2], {
            "kept": CategoryResult(_kept_singles(kept), "kept"),
            PATTERN_CATEGORY: CategoryResult(pattern_groups, "dropped"),
            PATTERN_BUCKET_CATEGORY: CategoryResult(bucket_groups, "dropped"),
            PATTERN_REJECTED_CATEGORY: CategoryResult(rejected_groups, "info"),
        }))

    with clock("classify_seqno"):
        groups, _ = grf.combine_overlapping_seq([[v] for v in kept], SEQ_OVERLAP_TOLERANCE_PX, cluster_cap)
        steps.append(StepResult(STEP_LABELS[3], {
            "kept": CategoryResult(groups if keep_steps else [], "kept"),
        }))

    with clock("classify_spatial"):
        clusters = clf.cluster_spatial_groups(
            groups, SPATIAL_CLUSTER_THRESHOLD, SPATIAL_SIZE_TOLERANCE, cluster_cap,
        )
        steps.append(StepResult(STEP_LABELS[4], {
            "kept": CategoryResult(clusters if keep_steps else [], "kept"),
        }))

    with clock("classify_outliers"):
        clusters, outliers = _remove_from_clusters(clusters, length_outliers)
        steps.append(StepResult(STEP_LABELS[5], {
            "kept": CategoryResult(clusters if keep_steps else [], "kept"),
            "outliers": CategoryResult(outliers, "dropped"),
        }))

    flagged_kept: list[list[Vector]] = []

    def _crossed(flat: list[Vector]) -> list[Vector]:
        dropped, kept_flagged = crossed_grid(flat)
        if kept_flagged:
            flagged_kept.append(kept_flagged)
        return dropped

    with clock("classify_crossings"):
        clusters, crossed = _remove_from_clusters(clusters, _crossed)
        steps.append(StepResult(STEP_LABELS[6], {
            "kept": CategoryResult(clusters, "kept"),
            CROSSED_CATEGORY: CategoryResult(crossed, "dropped"),
            FLAGGED_KEPT_CATEGORY: CategoryResult(flagged_kept, "info"),
        }))
    return steps, angles


def _iter_buckets(vectors_by_layer_color_width: dict):
    return [
        ((layer, color, width), vectors)
        for layer, color_groups in vectors_by_layer_color_width.items()
        for color, width_groups in color_groups.items()
        for width, vectors in width_groups.items()
    ]


def _collect_dropped(clustering: dict) -> list[Vector]:
    out: list[Vector] = []
    for stage in clustering.values():
        for step in stage.steps:
            for category in step.categories.values():
                if category.role != "dropped":
                    continue
                for entry in category.groups:
                    if entry and isinstance(entry[0], list):
                        out.extend(v for g in entry for v in g)
                    else:
                        out.extend(entry)
    return out


def classify_vectors(
    vectors: list[Vector], page: Page, *, verbose: bool = False,
    clock: "StepClock | None" = None, keep_steps: bool = True,
) -> ClassificationResult:
    """Separate by (layer, color, width), run `_classify_bucket` per bucket,
    gather every bucket's final clusters, every dropped Vector (drawing)
    and the page's deduped global potential angles. Steps are timed on `clock` (see the module docstring).
    The page area for the oversize/cluster caps comes from `page.meta`.
    `keep_steps=False` drops intermediate step results and the
    `vectors_by_layer*` dicts (see the module docstring)."""
    clock = clock or StepClock()
    meta = getattr(page, "meta", None)
    page_area = (meta.width * meta.height) if meta is not None and meta.width and meta.height else None
    with clock("classify_separate"):
        vectors_by_layer = separate_by_layer(vectors)
        vectors_by_layer_color = {
            layer: separate_by_color(vs) for layer, vs in vectors_by_layer.items()
        }
        vectors_by_layer_color_width = {
            layer: {color: separate_by_width(vs) for color, vs in color_groups.items()}
            for layer, color_groups in vectors_by_layer_color.items()
        }

    clustering: dict = {}
    all_angles: list[float] = []
    for key, bucket in _iter_buckets(vectors_by_layer_color_width):
        steps, angles = _classify_bucket(bucket, clock, page_area=page_area, keep_steps=keep_steps)
        clustering[key] = ClusteringStageResult(steps=steps)
        all_angles.extend(angles)

    with clock("classify_collect"):
        text_clusters: list[list[list[Vector]]] = []
        for stage in clustering.values():
            text_clusters.extend(stage.steps[-1].categories["kept"].groups)
        drawing_vectors = _collect_dropped(clustering)
        global_angles = dedupe_angles(all_angles, ANGLE_TOL_DEG)

    return ClassificationResult(
        text_clusters=text_clusters,
        drawing_vectors=drawing_vectors,
        clustering=clustering,
        global_angles=global_angles,
        vectors_by_layer=vectors_by_layer if verbose and keep_steps else None,
        vectors_by_layer_color=vectors_by_layer_color if verbose and keep_steps else None,
        vectors_by_layer_color_width=vectors_by_layer_color_width if verbose and keep_steps else None,
    )
