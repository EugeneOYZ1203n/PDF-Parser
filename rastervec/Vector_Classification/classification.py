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
from typing import TYPE_CHECKING, Literal

from rastervec.helpers.geometry import union_bbox
from rastervec.logging_setup import get_logger
from rastervec.models import Page, Vector
from rastervec.Vector_Classification import cluster_filters as clf  # noqa: F401 -- re-exported
from rastervec.Vector_Classification import group_filters as grf  # noqa: F401 -- re-exported
from rastervec.Vector_Classification import item_filters as itf  # noqa: F401 -- re-exported

if TYPE_CHECKING:
    from rastervec.pipelines.result import PipelineResult
    from rastervec.renderer.notebook import RenderResult

_LOG = get_logger("classification")

# notebook-visualization display colors (see the render_* functions at the
# bottom of this file) -- not classification logic.
_CLUSTER_STEP_COLORS = ["#2563eb", "#7c3aed", "#ea580c", "#0d9488", "#6b7280", "#c026d3", "#65a30d", "#0284c7"]
_SIDE_CATEGORY_COLORS = ["#9ca3af", "#f59e0b", "#db2777", "#eab308", "#16a34a"]

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


# --------------------------------------------------------------------------
# notebook visualization (pipeline_stage_visualization.ipynb's "classify"
# sections) -- reads a PipelineResult, never called by the real pipeline.
# --------------------------------------------------------------------------
def _entry_vectors(entry: list) -> list[Vector]:
    """Flattens one category entry to a flat `list[Vector]` regardless of
    whether it's a pre-spatial group (`list[Vector]`) or a post-spatial
    cluster (`list[list[Vector]]`)."""
    if entry and isinstance(entry[0], list):
        return [v for g in entry for v in g]
    return entry


def render_layers(res: "PipelineResult") -> "RenderResult":
    """Vectors grouped by their PDF layer."""
    from rastervec.renderer.notebook import RenderResult

    vbl = res.vectors_by_layer or {}
    return RenderResult(
        categories=[
            {"name": f"layer {name or '(no layer)'} ({len(vs)})", "vectors": vs}
            for name, vs in vbl.items()
        ],
        note=f"{len(vbl)} layer(s)",
    )


def render_layer_color_buckets(res: "PipelineResult") -> "RenderResult":
    """Vectors grouped by (layer, color) bucket -- the classification
    chain's own unit of work."""
    from rastervec.renderer.notebook import RenderResult

    vblc = res.vectors_by_layer_color or {}
    cats = []
    for layer, by_color in vblc.items():
        for color, vs in by_color.items():
            cats.append({"name": f"{layer or '(no layer)'} / {color} ({len(vs)})", "vectors": vs})
    return RenderResult(categories=cats, note=f"{len(cats)} (layer, color) bucket(s)")


def _signature_color(sig) -> str:
    import colorsys
    import hashlib

    digest = hashlib.md5(repr(sig).encode()).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def render_clustering_steps(res: "PipelineResult", matrix, original) -> "RenderResult":
    """The 12-step chain's own per-step categories, one row per (step,
    named category) across every (layer, color) bucket, plus a "colour by
    vector type" overlay coloring every surviving Vector's items by its
    `VectorSignature`. `matrix`/`original` come from
    `renderer.notebook.page_setup()`."""
    import pymupdf as fitz
    from PIL import ImageDraw

    from rastervec.config import SIGNATURE_ROUND_PX
    from rastervec.helpers.geometry import item_points
    from rastervec.renderer.notebook import RenderResult, blank_like

    clustering = res.clustering or {}
    results = list(clustering.values())[:10]
    step_labels = [s.label for s in results[0].steps] if results else []
    cats = []
    for i, label in enumerate(step_labels):
        names = []
        for r in results:
            for nm in r.steps[i].categories:
                if nm not in names:
                    names.append(nm)
        side_i = 0
        for nm in names:
            entries, role = [], "kept"
            for r in results:
                cat = r.steps[i].categories.get(nm)
                if cat is None:
                    continue
                role = cat.role
                entries.extend(e for e in cat.groups if e)
            if role == "kept":
                color = _CLUSTER_STEP_COLORS[i % len(_CLUSTER_STEP_COLORS)]
            else:
                color = _SIDE_CATEGORY_COLORS[side_i % len(_SIDE_CATEGORY_COLORS)]
                side_i += 1
            cats.append({
                "name": f"step {i + 1} {label} / {nm} [{role}] ({len(entries)} grp)",
                "color": color,
                "bboxes": [union_bbox([v.bbox for v in _entry_vectors(e)]) for e in entries],
            })

    sig_vectors = [
        v for r in results for s in r.steps
        if s.signature_counts is not None
        for e in s.categories["kept"].groups for v in _entry_vectors(e)
    ]
    if sig_vectors:
        iso, ovl = blank_like(original), original.copy()
        for im in (iso, ovl):
            d = ImageDraw.Draw(im)
            for v in sig_vectors:
                color = _signature_color(itf.vector_signature(v, SIGNATURE_ROUND_PX))
                for item in v.items:
                    pts = [(pt.x, pt.y) for pt in (fitz.Point(x, y) * matrix for x, y in item_points(item))]
                    if len(pts) >= 2:
                        d.line(pts, fill=color, width=2)
        cats.append({"name": "colour by vector type", "isolated": iso, "overlay": ovl})

    return RenderResult(
        categories=cats,
        note=f"{len(results)} (layer, color) bucket(s), {len(step_labels)} steps",
    )


def render_text_candidates(res: "PipelineResult") -> "RenderResult":
    """Surviving text-candidate cluster boxes."""
    from rastervec.renderer.notebook import RenderResult

    tc = res.text_clusters or []
    flat_clusters = [_entry_vectors(c) for c in tc]
    return RenderResult(
        categories=[{
            "name": f"text candidate clusters ({len(tc)})",
            "color": _CLUSTER_STEP_COLORS[0],
            "bboxes": [union_bbox([v.bbox for v in c]) for c in flat_clusters if c],
            "vectors": [v for c in flat_clusters for v in c],
            "path_color": _CLUSTER_STEP_COLORS[0],
        }],
        note=f"{len(tc)} text candidate cluster(s)",
    )
