"""Vector Classification: a reduced, non-configurable 2-step pipeline that
groups extracted Vectors into text-candidate clusters, run in order by
`_classify_bucket()` (see `classify_vectors.py`). Only two steps remain
of the original 12-step chain -- everything else was removed as part of
an experiment to isolate the effect of seqno-overlap merge + spatial
clustering alone:

  group_filters.py    -- seqno-overlap merge (`combine_overlapping_seq`)
  cluster_filters.py  -- constrained spatial clustering (`cluster_spatial_groups`)

See docs/Glossary.md for standardized group/cluster/global-group/similarity-group
terminology.

A `Vector` is never decomposed into standalone items anywhere in this chain
-- items stay nested inside their parent Vector. The spatial-clustering
step produces real nested structure (`list[list[Vector]]` per cluster, one
entry per member group) instead of a flattened cluster plus a side
`id()`-keyed lineage dict.

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
into the final `vectors` output, same as every other drop). Neither
remaining step ever drops anything, so every input Vector reaches the
last step's `"kept"` category and is handed downstream (FAST, OCR) as a
text candidate -- FAST and OCR success/failure are now the only signal
for whether a given cluster was actually text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Page, Vector
from rastervec.P3_Vector_Parsing.VectorClassification import cluster_filters as clf  # noqa: F401 -- re-exported
from rastervec.P3_Vector_Parsing.VectorClassification import group_filters as grf  # noqa: F401 -- re-exported
from rastervec.P3_Vector_Parsing.VectorClassification import item_filters as itf  # noqa: F401 -- re-exported

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
    side categories for the debug UI)."""

    label: str
    categories: dict[str, CategoryResult]


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
