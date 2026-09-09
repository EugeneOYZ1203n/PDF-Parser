"""Every `render_<stage_name>` notebook-visualization function, one per
pipeline stage, centralized here instead of scattered next to each stage's
own code (the old convention -- see `CLAUDE.md`'s history). Only
`pipeline_stage_visualization.ipynb` calls these; the real pipeline never
does.

Each function still does its own `from rastervec.renderer.notebook import
...` **inside the function body**, never at module level -- same invariant
`renderer/notebook.py`'s own docstring documents (that module imports
matplotlib, so importing it eagerly here would drag matplotlib into every
real pipeline run's import graph, since stage modules do a cheap top-level
`from rastervec.renderer.stages import render_x` re-export). Everything
else this module needs (numpy, PIL, pymupdf, `rastervec.helpers.geometry`,
plain `rastervec.renderer` package symbols) is matplotlib-free and safe at
module level.

Each stage module that used to define its own `render_<stage_name>` keeps a
one-line re-export (`from rastervec.renderer.stages import render_x`) for
backward compatibility.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw

from rastervec.helpers.geometry import is_dashed, item_points, union_bbox
from rastervec.renderer import render_reconstructed_page

if TYPE_CHECKING:
    from rastervec.pipelines.result import PipelineResult
    from rastervec.renderer.notebook import RenderResult

# Shared pass/fail/dropped display colors -- previously redefined
# independently (same hex values) in OCR/fast_detect.py and
# OCR/Paddle_OCR/ocr_backend.py.
_PASSED_COLOR = "#059669"
_DROPPED_COLOR = "#dc2626"
_FAILED_COLOR = "#dc2626"

_NATIVE_WORD_COLOR = "#2563eb"
_CLUSTER_STEP_COLORS = ["#2563eb", "#7c3aed", "#ea580c", "#0d9488", "#6b7280", "#c026d3", "#65a30d", "#0284c7"]
_SIDE_CATEGORY_COLORS = ["#9ca3af", "#f59e0b", "#db2777", "#eab308", "#16a34a"]


def _entry_vectors(entry: list) -> list:
    """Flattens one category entry to a flat `list[Vector]` regardless of
    whether it's a pre-spatial group (`list[Vector]`) or a post-spatial
    cluster (`list[list[Vector]]`) -- also used for a Radon `Segment`'s own
    `vectors` list, which is already flat (single-element `isinstance`
    check on the first item is `False` for a `Vector`, so it passes
    through unchanged)."""
    if entry and isinstance(entry[0], list):
        return [v for g in entry for v in g]
    return entry


def _signature_color(sig) -> str:
    import colorsys
    import hashlib

    digest = hashlib.md5(repr(sig).encode()).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def _fast_mask_overlay(base: "Image.Image", mask: "np.ndarray") -> "Image.Image":
    heat = (np.clip(mask, 0.0, 1.0) * 255).astype("uint8")
    zeros = Image.new("L", base.size, 0)
    heat_img = Image.merge("RGB", (Image.fromarray(heat), zeros, zeros))
    return Image.blend(base.convert("RGB"), heat_img, alpha=0.5)


# --------------------------------------------------------------------------
# Vector extraction (Vector/vector.py)
# --------------------------------------------------------------------------
def render_vectors(res: "PipelineResult") -> "RenderResult":
    """One category per distinct `Vector.type`."""
    from rastervec.renderer.notebook import RenderResult

    vectors = res.vectors_raw or []
    kinds = sorted({v.type for v in vectors})
    return RenderResult(
        categories=[
            {"name": f"type {k!r} ({sum(v.type == k for v in vectors)})",
             "vectors": [v for v in vectors if v.type == k]}
            for k in kinds
        ],
        note=f"{len(vectors)} vectors, types={kinds}",
    )


# --------------------------------------------------------------------------
# Native text (native_text.py)
# --------------------------------------------------------------------------
def render_native(res: "PipelineResult", *, zoom: float = 1.0) -> "RenderResult":
    """Word quads over the page, plus a full page reconstruction built
    from `native_words` alone."""
    from rastervec.renderer.notebook import RenderResult

    words = res.native_words or []
    fallback = sum(1 for w in words if w.orientation_source == "fallback")
    return RenderResult(categories=[{
        "name": f"text words ({len(words)})",
        "color": _NATIVE_WORD_COLOR,
        "polys": [w.quad() for w in words],
        "isolated": render_reconstructed_page(res.page.meta, native_words=words, zoom=zoom),
    }], note=(
        f"{len(words)} word(s)"
        + (f"; {fallback} lost their rotation (no matching span, defaulted to horizontal)"
           if fallback else "")
    ))


# --------------------------------------------------------------------------
# Vector Classification (Vector_Classification/classification.py)
# --------------------------------------------------------------------------
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


def render_clustering_steps(res: "PipelineResult", matrix, original) -> "RenderResult":
    """The 12-step chain's own per-step categories, one row per (step,
    named category) across every (layer, color) bucket, plus a "colour by
    vector type" overlay coloring every surviving Vector's items by its
    `VectorSignature`. `matrix`/`original` come from
    `renderer.notebook.page_setup()`."""
    from rastervec.config import SIGNATURE_ROUND_PX
    from rastervec.renderer.notebook import RenderResult, blank_like
    from rastervec.Vector_Classification import item_filters as itf

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


# --------------------------------------------------------------------------
# Segment / Radon (OCR/radon.py)
# --------------------------------------------------------------------------
def render_radon(res: "PipelineResult") -> "RenderResult":
    """One row per word `Segment` (capped), from `res.word_segments` --
    Radon now runs on every FAST-surviving cluster (not just elected
    representatives -- dedup happens after this step), so there's no
    per-cluster grouping left to visualize here. Each row shows the word's
    own captured deskewed crop (`seg.image`) directly -- no re-render
    needed, it's the exact crop OCR itself will use."""
    from rastervec.renderer.notebook import RenderResult

    word_segments = res.word_segments or []

    rows = []
    for seg in word_segments[:8]:
        if seg.image is None:
            continue
        base = Image.fromarray(seg.image).convert("RGB")
        caption = f"angle={seg.angle:+.2f}deg"
        rows.append({"name": caption, "isolated": base, "overlay": base})

    return RenderResult(
        categories=rows,
        note=f"{len(word_segments)} word(s) across every FAST-surviving cluster",
    )


# --------------------------------------------------------------------------
# Similarity / FAST / drawing output (pipelines/_steps.py)
# --------------------------------------------------------------------------
def render_similarity(res: "PipelineResult") -> "RenderResult":
    """Notebook visualization for the (now post-Radon, word-level)
    similarity-grouping step: every word `Segment`'s bbox, plus a note on
    how much the grouping is expected to save OCR call count (Radon already
    ran on every occurrence by this point, so dedup here only saves OCR,
    not Radon -- each group beyond size 1 means every extra member skips
    recognition entirely, reusing the representative's instead)."""
    from rastervec.renderer.notebook import RenderResult

    segments = res.word_segments or []
    groups = res.similarity_groups or []
    dup_groups = [g for g in groups if len(g) > 1]
    saved = sum(len(g) - 1 for g in dup_groups)
    return RenderResult(
        categories=[{
            "name": f"words ({len(segments)}) in {len(groups)} similarity group(s)",
            "bboxes": [union_bbox([v.bbox for v in seg.vectors]) for seg in segments if seg.vectors],
        }],
        note=(
            f"{len(segments)} word(s) -> {len(groups)} group(s) "
            f"({len(dup_groups)} with >1 member, dedup saves {saved} OCR call(s))"
        ),
    )


def render_fast(res: "PipelineResult", *, enable_fast: bool) -> "RenderResult":
    """The whole-page render, its detection heatmap, and passed/dropped
    cluster boxes -- FAST runs directly on classification's clusters here
    (before Radon or any dedup), so "passed" is `res.fast_passed`'s plain
    clusters, not yet split into words. `enable_fast=False` and a page with
    no clusters both render as a note only (no pixels to show)."""
    from rastervec.renderer.notebook import RenderResult

    fr = res.fast_result
    passed = res.fast_passed or []
    dropped_vectors = res.fast_dropped_vectors or []
    if not enable_fast:
        return RenderResult(note=(
            f"ENABLE_FAST=False -- pass-through, all {len(passed)} cluster(s) "
            "kept, none dropped, no render/detection"
        ))
    if fr is None or fr.page_image is None:
        return RenderResult(note="(no clusters on this page)")

    render = fr.page_image.convert("RGB")
    heat = _fast_mask_overlay(render, fr.page_mask) if fr.page_mask is not None else render
    return RenderResult(
        categories=[
            {"name": "FAST render", "isolated": render, "overlay": render},
            {"name": "detection heatmap", "isolated": heat, "overlay": heat},
            {"name": f"passed clusters ({len(passed)})", "color": _PASSED_COLOR,
             "bboxes": [union_bbox([v.bbox for v in c]) for c in passed if c]},
            {"name": f"dropped vectors ({len(dropped_vectors)})", "color": _DROPPED_COLOR,
             "bboxes": [v.bbox for v in dropped_vectors]},
        ],
        note=f"detect_seconds = {fr.detect_seconds}",
    )


def render_drawing(res: "PipelineResult", *, zoom: float = 1.0) -> "RenderResult":
    """Notebook visualization for the drawing-vectors output: dashed vs
    solid bboxes, plus a full page reconstruction."""
    from rastervec.renderer.notebook import DEFAULT_PATH_COLOR, RenderResult

    dv = res.vectors or []
    dashed = [d for d in dv if is_dashed(d.dashes)]
    solid = [d for d in dv if not is_dashed(d.dashes)]
    recon = render_reconstructed_page(res.page.meta, drawing_vectors=dv, zoom=zoom)
    return RenderResult(
        categories=[
            {"name": f"dashed ({len(dashed)})", "color": DEFAULT_PATH_COLOR, "bboxes": [d.bbox for d in dashed]},
            {"name": f"solid ({len(solid)})", "color": DEFAULT_PATH_COLOR, "bboxes": [d.bbox for d in solid]},
            {"name": "full reconstruction", "isolated": recon, "overlay": recon},
        ],
        note=f"{len(dv)} drawing vector(s)",
    )


# --------------------------------------------------------------------------
# OCR (OCR/Paddle_OCR/ocr_backend.py, pipelines/sub_pipelines/ocr.py)
# --------------------------------------------------------------------------
def render_ocr_results(res: "PipelineResult", *, zoom: float = 1.0) -> "RenderResult":
    """Passed (non-blank) vs failed (blank) word-level OCR readings, one per
    elected representative, in canonical frame (not restored to real page
    position -- there can be many restored instances per reading)."""
    from rastervec.renderer.notebook import RenderResult

    unique_texts = res.unique_texts or []
    passed = [t for t in unique_texts if t.text.strip()]
    failed = [t for t in unique_texts if not t.text.strip()]
    return RenderResult(categories=[
        {"name": f"passed ({len(passed)})", "color": _PASSED_COLOR, "bboxes": [t.bbox for t in passed]},
        {"name": f"failed ({len(failed)})", "color": _FAILED_COLOR, "bboxes": [t.bbox for t in failed]},
    ], note=(
        f"{len(unique_texts)} representative word(s) OCR'd -> "
        f"{len(res.restored_texts or [])} restored Text(s) across all occurrences"
    ))


def render_restore(res: "PipelineResult") -> "RenderResult":
    """Every restored `Text`'s real-position bbox -- the dedup payoff made
    visible: one representative word's OCR reading fans back out onto every
    real occurrence it covers."""
    from rastervec.renderer.notebook import RenderResult

    restored = res.restored_texts or []
    unique_word_count = len(res.unique_texts or [])
    return RenderResult(
        categories=[{
            "name": f"restored text ({len(restored)})",
            "bboxes": [t.bbox for t in restored],
        }],
        note=(
            f"{unique_word_count} unique word OCR reading(s) -> "
            f"{len(restored)} restored Text(s) at their real page positions"
        ),
    )
