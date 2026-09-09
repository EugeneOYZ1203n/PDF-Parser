"""Vector Classification: a single fixed, non-configurable 12-step
pipeline that classifies extracted Vectors into text candidates vs.
drawing content, run in order by `cluster()`. Each step
is implemented as a plain function in this package's items/groups/
clusters submodules (see each submodule's own docstring for its steps'
descriptions):

  items/item_filters.py    -- steps 1-2 (filter_large_items, compute_vector_signatures)
  groups/group_filters.py  -- steps 3-5, 8 (seq dedupe/merge, tiny/large groups, group stats)
  clusters/cluster_filters.py -- steps 6-7, 9-12 (spatial cluster, mixed fill-rule,
                                 perimeter/density/constant-spacing/low-variety)

See docs/Glossary.md for standardized group/cluster/global-group/similarity-group
terminology.

A `Vector` is never decomposed into standalone items anywhere in this chain
-- items stay nested inside their parent Vector and are only inspected
internally by filters that need item-level granularity (perimeter/density/
constant-spacing). Step 6's spatial merge produces real nested structure
(`list[list[Vector]]` per cluster, one entry per member group) instead of a
flattened cluster plus a side `id()`-keyed lineage dict -- every cluster-level
filter step from here on keeps that tiering, partitioning/appending the same
cluster object so `id(cluster)` stays a stable key for `group_stats` across
steps.

Whole-page similarity grouping of text-candidate clusters (formerly
`group_similar_clusters`, run right after this chain) has moved downstream
of Radon segmentation -- it now operates on post-Radon `Segment`s using each
segment's own known precise skew angle, not pre-Radon clusters via a
PCA-based rotation search. See `OCR/radon.py` / `pipelines/_steps.py`.

All thresholds live in `rastervec/config.py` -- tune the pipeline by
editing them there, not at runtime. Each step's result is wrapped into a
`StepResult` holding one or more named `CategoryResult`s -- exactly one
per step has `role="kept"` and feeds the next step; every other category
is a side-channel for the debug UI (a `role="dropped"` category is folded
into the final `vectors` output, same as every other drop). Every Vector
that survives the whole chain (the last step's `"kept"` category) is a
text candidate handed downstream (Radon segmentation, similarity grouping,
FAST, OCR) -- there's no separate drawing-vs-text heuristic; everything
any filter step drops along the way is drawing content, and OCR success/
failure itself is the signal for whether a given cluster was actually text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rastervec.logging_setup import get_logger
from rastervec.models import Page, Vector
from rastervec.renderer.stages import (  # noqa: F401 -- re-exported for callers
    render_clustering_steps,
    render_layer_color_buckets,
    render_layers,
    render_text_candidates,
)
from rastervec.Vector_Classification import cluster_filters as clf  # noqa: F401 -- re-exported
from rastervec.Vector_Classification import group_filters as grf  # noqa: F401 -- re-exported
from rastervec.Vector_Classification import item_filters as itf  # noqa: F401 -- re-exported

_LOG = get_logger("classification")

CategoryRole = Literal["kept", "dropped", "info"]


@dataclass
class CategoryResult:
    """One named category within a pipeline step's result -- a list of
    entries plus its role. Exactly one category per step is `role="kept"`
    (feeds the next step); a `role="dropped"` category is folded into the
    final drawing-vector output; `role="info"` is never folded anywhere
    (used only by step 2's pass-through counter category). Each entry is
    `list[Vector]` for steps 1-5 (a group) or `list[list[Vector]]` for
    steps 6-12 (a cluster, tiered by member group) -- see this module's
    docstring."""

    groups: list
    role: CategoryRole


@dataclass
class StepResult:
    """One pipeline step's full result: a display label plus every named
    category it produced (`"kept"` always present, plus any number of
    side categories for the debug UI). `signature_counts`, if set (only on
    step 2's result), is the per-`VectorSignature` occurrence count built
    by `compute_vector_signatures`. `group_stats`, if set, is the
    per-cluster `GroupStats` built by `compute_group_stats`, keyed by
    `id(cluster)`."""

    label: str
    categories: dict[str, CategoryResult]
    signature_counts: dict[itf.VectorSignature, int] | None = None
    group_stats: dict[int, grf.GroupStats] | None = None


def cluster(vectors: list[Vector], page: Page) -> list[StepResult]:
    """The fixed classification chain for one (layer, color) bucket. The
    chain itself -- one named step call per step -- now lives in
    `rastervec/pipelines/sub_pipelines/vector_classification.py` so the
    pipeline reads top-down; this stays as a thin entrypoint for callers
    that only want one bucket's `StepResult` list."""
    from rastervec.pipelines.sub_pipelines.vector_classification import _classify_bucket

    return _classify_bucket(vectors, page)


def classify(vectors: list[Vector], page: Page) -> list[list[list[Vector]]]:
    """Runs cluster() and returns just the final surviving clusters -- a
    convenience wrapper for callers that don't need the per-step/
    per-category bookkeeping."""
    steps = cluster(vectors, page)
    return steps[-1].categories["kept"].groups if steps else []
