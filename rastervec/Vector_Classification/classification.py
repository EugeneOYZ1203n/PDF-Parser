"""Vector Classification: a single fixed, non-configurable 12-step
pipeline that classifies extracted VectorPaths into text candidates vs.
drawing content, run in order by `cluster()`. Each step
is implemented as a plain function in this package's items/groups/
clusters submodules (see each submodule's own docstring for its steps'
descriptions):

  items/item_filters.py    -- steps 1-2 (filter_large_items, compute_vector_signatures)
  groups/group_filters.py  -- steps 3-5, 8 (seq dedupe/merge, tiny/large groups, group stats)
  clusters/cluster_filters.py -- steps 6-7, 9-12 (spatial cluster, mixed fill-rule,
                                 perimeter/density/constant-spacing/low-variety)

See Glossary.md for standardized group/cluster/global-group/similarity-group
terminology.

Step 6's spatial merge also tracks lineage: `cluster()`'s final
`StepResult.cluster_groups` (keyed by `id(cluster)`) records which of
step 3's pre-spatial-clustering "groups" each surviving cluster is
composed of.

All thresholds live in `rastervec/config.py` -- tune the pipeline by
editing them there, not at runtime. Each step's result is wrapped into a
`StepResult` holding one or more named `CategoryResult`s -- exactly one
per step has `role="kept"` and feeds the next step; every other category
is a side-channel for the debug UI (a `role="dropped"` category is folded
into `drawing_vectors` by pipeline.py, same as every other drop). Every
path that survives the whole chain (the last step's `"kept"` category) is
a text candidate handed downstream (pipeline.py's text_candidates/
fast_text_detect/ocr_compare stages) -- there's no separate
drawing-vs-text heuristic; everything any filter step drops along the way
is drawing content, and OCR success/failure itself is the signal for
whether a given cluster was actually text.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from rastervec.config import UNIQUE_CLUSTER_TOLERANCE
from rastervec.helpers.geometry import is_dashed, union_bbox
from rastervec.logging_setup import get_logger
from rastervec.models import DrawingVector, Page, VectorPath, VectorRecord
from rastervec.Vector_Classification import cluster_filters as clf
from rastervec.Vector_Classification import group_filters as grf  # noqa: F401 -- re-exported
from rastervec.Vector_Classification import item_filters as itf  # noqa: F401 -- re-exported

_LOG = get_logger("classification")

CategoryRole = Literal["kept", "dropped", "info"]


@dataclass
class CategoryResult:
    """One named category within a pipeline step's result -- a list of
    groups plus its role. Exactly one category per step is `role="kept"`
    (feeds the next step); a `role="dropped"` category is folded into
    `drawing_vectors` by pipeline.py's `_run_drawing_vectors`; `role=
    "info"` is never folded anywhere (used only by step 2's pass-through
    counter category)."""

    groups: list[list[VectorPath]]
    role: CategoryRole


@dataclass
class StepResult:
    """One pipeline step's full result: a display label plus every named
    category it produced (`"kept"` always present, plus any number of
    side categories for the debug UI). `signature_counts`, if set (only on
    step 2's result), is the per-`VectorSignature` occurrence count built
    by `compute_vector_signatures`. `group_stats`, if set, is the
    per-group `GroupStats` built by `compute_group_stats`, keyed by
    `id(group)`. `cluster_groups`, if set (only on the final "Low-variety
    clusters" step's result), maps `id(cluster)` (for every cluster in
    that step's own `"kept"` category) to the list of step 3's
    pre-spatial-clustering "groups" that cluster is composed of."""

    label: str
    categories: dict[str, CategoryResult]
    signature_counts: dict[itf.VectorSignature, int] | None = None
    group_stats: dict[int, grf.GroupStats] | None = None
    cluster_groups: dict[int, list[list[VectorPath]]] | None = None


def cluster(paths: list[VectorPath], page: Page) -> list[StepResult]:
    """The fixed classification chain for one (layer, color) bucket. The
    chain itself -- one named step call per step -- now lives in
    `rastervec/pipelines/sub_pipelines/vector_classification.py` so the
    pipeline reads top-down; this stays as a thin entrypoint for callers
    that only want one bucket's `StepResult` list."""
    from rastervec.pipelines.sub_pipelines.vector_classification import _classify_bucket

    return _classify_bucket(paths, page)


def group_similar_clusters(
    clusters: list[list[VectorPath]],
) -> list[list[list[VectorPath]]]:
    """Whole-page similarity grouping of text-candidate clusters -- see
    `clf.group_similar_clusters` and Glossary.md's "similarity group"
    entry. Uses `UNIQUE_CLUSTER_TOLERANCE`."""
    return clf.group_similar_clusters(clusters, UNIQUE_CLUSTER_TOLERANCE)


def classify(paths: list[VectorPath], page: Page) -> list[list[VectorPath]]:
    """Runs cluster() and returns just the final surviving groups -- a
    convenience wrapper for callers that don't need the per-step/
    per-category bookkeeping (pipeline.py's own stage wiring calls
    cluster() directly instead, to keep every step's categories for the
    debug app and drawing_vectors)."""
    steps = cluster(paths, page)
    return steps[-1].categories["kept"].groups if steps else []


def _bbox_and_representative(paths: list[VectorPath]) -> dict:
    """Shared by build_vector_records/build_drawing_vectors: a group's
    own union bbox, plus style fields read off its first member as a
    representative value (paths within one group/cluster share the
    same drawing-level style in practice)."""
    first = paths[0]
    return dict(
        bbox=union_bbox([p.bbox for p in paths]),
        stroke_color=first.stroke_color,
        fill_color=first.fill_color,
        stroke_width=first.stroke_width,
        dashed=is_dashed(first.dashes),
        page_index=first.page_index,
    )


def build_vector_records(steps: list[StepResult]) -> list[VectorRecord]:
    """Wires the final step's `cluster_groups` lineage into one
    `VectorRecord` per surviving (role="kept") text-candidate cluster --
    built right here, where the lineage is already known, rather than
    reconstructed after the fact from a flattened list. Drawing-level
    fields VectorPath doesn't carry (even_odd/line_cap/line_join/
    scissor/blendmode/isolated/knockout/opacity) fall back to
    false/0/None -- a cluster can merge paths from several different
    original drawings, so there's no single drawing left to read them
    from; `seqno` uses the cluster's first member's synthetic `seq` as
    a representative value instead."""
    if not steps:
        return []
    last = steps[-1]
    kept = last.categories["kept"].groups
    lineage = last.cluster_groups or {}

    records: list[VectorRecord] = []
    for cluster_ in kept:
        if not cluster_:
            continue
        common = _bbox_and_representative(cluster_)
        first = cluster_[0]
        records.append(
            VectorRecord(
                items=cluster_,
                even_odd=False,
                line_cap=0,
                line_join=0,
                seqno=first.seq,
                rect=common["bbox"],
                scissor=None,
                blendmode=None,
                isolated=False,
                knockout=False,
                opacity=None,
                groups=lineage.get(id(cluster_), [cluster_]),
                role="kept",
                **common,
            )
        )
    return records


def build_drawing_vectors(paths: list[VectorPath]) -> list[DrawingVector]:
    groups: dict[int, list[VectorPath]] = defaultdict(list)
    for path in paths:
        groups[path.seq].append(path)

    result = []
    for group in groups.values():
        result.append(DrawingVector(paths=group, **_bbox_and_representative(group)))

    _LOG.debug("build_drawing_vectors: %d path(s) -> %d drawing(s)", len(paths), len(result))
    return result
