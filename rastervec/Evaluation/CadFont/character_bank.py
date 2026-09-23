"""Baseline-relative normalization + character -> graph mapping (step 3 of
`docs/cad_font_vector_recognition.md`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rastervec.commons.helpers.geometry import transform_point, transform_vector, item_points
from rastervec.Evaluation.CadFont.geometry import BEZIER_SAMPLE_COUNT, vectors_to_segments
from rastervec.Evaluation.CadFont.graph import (
    CharGraph,
    GraphBuildStats,
    build_char_graph,
    complexity,
    select_anchor_points,
)
from rastervec.Evaluation.Labelling.label_schema import LabelSet, path_signature
from rastervec.commons.logging_setup import get_logger

_LOG = get_logger("cad_font.character_bank")


def to_baseline_relative_vectors(vectors, baseline_origin, baseline_direction):
    """Two-step rigid transform of a labelled character's own `Vector`s
    into baseline-relative frame, using `commons.helpers.geometry`'s
    `transform_point`/`transform_vector` as the only rotation/translation
    primitives:

    Step 1: `theta = degrees(atan2(dy, dx))` for `baseline_direction =
    (dx, dy)`. We want every point `p -> R(-theta) @ (p - baseline_origin)`
    -- baseline_origin maps to `(0, 0)`, baseline_direction maps onto `+x`.
    `transform_point(p, offset, rot) = R(rot) @ p + offset`, so substituting
    `offset = R(-theta) @ (-baseline_origin)` gives `R(-theta) @ p +
    R(-theta) @ (-baseline_origin) = R(-theta) @ (p - baseline_origin)` --
    exactly what we want, realized with a single `transform_vector` call
    per vector.

    Step 2: a pure x-shift so the character's own leftmost point (over
    every step-1-transformed vector's items) lands at `x = 0`.

    Returns new `Vector`s (input untouched -- `transform_vector` returns
    copies)."""
    theta = math.degrees(math.atan2(baseline_direction[1], baseline_direction[0]))
    offset = transform_point(
        (-baseline_origin[0], -baseline_origin[1]), offset=(0.0, 0.0), rotation_deg=-theta,
    )
    step1 = [transform_vector(v, offset=offset, rotation_deg=-theta) for v in vectors]

    xs = [p[0] for v in step1 for item in v.items for p in item_points(item)]
    min_x = min(xs) if xs else 0.0

    step2 = [transform_vector(v, offset=(-min_x, 0.0), rotation_deg=0.0) for v in step1]
    return step2


@dataclass
class CharacterTemplate:
    label_id: str
    text: str
    baseline_id: str
    graph: CharGraph
    complexity: float
    anchor_node_indices: tuple[int, ...]
    build_stats: GraphBuildStats


def build_character_bank(
    label_set: LabelSet,
    vectors_by_page: dict[int, list],
    *,
    bezier_sample_count: int = BEZIER_SAMPLE_COUNT,
) -> list[CharacterTemplate]:
    """For every `source="cad_font"` `LabelEntry` in `label_set`: resolve
    its backing `Vector`s via `path_signature` against
    `vectors_by_page[entry.page_index]` (fresh `extract_vectors` output,
    caller-supplied so this function stays pure/easy to unit test);
    resolve `entry.baseline_id` against `label_set.baselines`; skip (log a
    warning, don't raise) any entry with an unresolved vector signature or
    a missing/unresolved baseline -- it cannot be placed in
    baseline-relative frame, so it cannot become a template. Returns
    templates sorted by `complexity` descending."""
    baselines_by_id = {b.baseline_id: b for b in label_set.baselines}

    sig_maps_by_page: dict[int, dict[str, object]] = {}

    templates: list[CharacterTemplate] = []
    for entry in label_set.entries:
        if entry.source != "cad_font":
            continue

        if entry.page_index not in sig_maps_by_page:
            page_vectors = vectors_by_page.get(entry.page_index, [])
            sig_maps_by_page[entry.page_index] = {path_signature(v): v for v in page_vectors}
        sig_map = sig_maps_by_page[entry.page_index]

        resolved = []
        missing = False
        for sig in entry.vector_signatures:
            v = sig_map.get(sig)
            if v is None:
                missing = True
                break
            resolved.append(v)
        if missing or not resolved:
            _LOG.warning(
                "cad_font label %s: unresolved vector_signatures on page %d, skipping",
                entry.label_id, entry.page_index,
            )
            continue

        baseline = baselines_by_id.get(entry.baseline_id) if entry.baseline_id else None
        if baseline is None:
            _LOG.warning(
                "cad_font label %s: no resolvable baseline (baseline_id=%r), skipping",
                entry.label_id, entry.baseline_id,
            )
            continue

        transformed = to_baseline_relative_vectors(resolved, baseline.origin, baseline.direction)
        segments, vertex_groups = vectors_to_segments(transformed, bezier_sample_count=bezier_sample_count)
        graph, build_stats = build_char_graph(segments, vertex_groups)
        char_complexity = complexity(graph)  # before any anchor-synthesis augmentation
        graph, anchor_indices = select_anchor_points(graph)
        templates.append(CharacterTemplate(
            label_id=entry.label_id,
            text=entry.text,
            baseline_id=entry.baseline_id,
            graph=graph,
            complexity=char_complexity,
            anchor_node_indices=anchor_indices,
            build_stats=build_stats,
        ))

    templates.sort(key=lambda t: t.complexity, reverse=True)
    return templates
