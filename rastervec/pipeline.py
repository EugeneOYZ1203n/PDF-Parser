"""Pipeline orchestration: shared stage-running machinery driven by both
the CLI in this file and the debug Tkinter app (rastervec/debug_app.py).

Only calls high-level class methods -- no inline extraction logic. A stage
is a (key, label, run) triple appended to Pipeline.STAGES; run(ctx) mutates
the shared PipelineContext and returns this stage's output data. Adding a
new stage once it's actually implemented means: add one StageSpec here,
and one view-renderer entry in debug_app.py's registry -- the CLI and the
debug app's cycling nav both pick it up automatically.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

if __name__ == "__main__" and __package__ is None:
    # Allow running this file directly (`python rastervec/pipeline.py`),
    # not just as a module (`python -m rastervec.pipeline`), by putting
    # the repo root -- the parent of this package -- on sys.path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rastervec.logging_setup import configure_logging, get_logger
from rastervec.models import DrawingVector, Page, TextWord, VectorPath
from rastervec.native import Native
from rastervec.reader import Reader
from rastervec.vector import Vector

_LOG = get_logger("pipeline")

# (layer, color) -- one Vector.separate_by_color() bucket.
GroupKey = tuple[str, tuple]


@dataclass
class VectorStageBuckets:
    """One (layer, color) group's paths, split into what this exact stage
    decided vs. what an earlier stage in the filter/cluster pipeline already
    decided vs. what's still waiting to be decided by a later stage.

    `this_stage`/`previous` entries are always groups (`list[VectorPath]`):
    a filter stage that decides path-by-path wraps each dropped path in a
    singleton group (`[path]`); a filter stage that decides group-by-group
    (filter_aspect_ratio) contributes each dropped group as-is; a cluster
    stage's entries are its actual cluster groupings. Every filter stage's
    drops are "classified as Drawing this round" (nothing is ever discarded
    -- see _run_drawing_vectors). `pending` holds whatever the next stage
    still needs to consume: a flat list[VectorPath] for path-level filter
    stages, or a list[list[VectorPath]] (groups) once clustering has run.
    """

    this_stage: list[list[VectorPath]]
    previous: list[list[VectorPath]]
    pending: list[VectorPath] | list[list[VectorPath]]


@dataclass
class ClusteringStageResult:
    """One (layer, color) group's result from the single configurable
    clustering stage: the `order` the 4 Vector.CLUSTER_STEPS operations ran
    in, `steps[i]` = the groups snapshot after applying `order[i]` (so
    `steps[-1]` is the final clustering result fed to the two filter stages
    after this one), and `previous` = every group the two filter stages
    before this one already dropped (classified as Drawing -- carried
    through so drawing_vectors can still fold it in)."""

    order: list[str]
    steps: list[list[list[VectorPath]]]
    previous: list[list[VectorPath]]


@dataclass
class PipelineContext:
    """Accumulates state across stages for one page run."""

    reader: Reader
    page_index: int
    page: Page | None = None
    native_words: list[TextWord] | None = None
    vector_paths: list[VectorPath] | None = None
    paths_by_layer: dict[str, list[VectorPath]] | None = None
    paths_by_layer_color: dict[str, dict[tuple, list[VectorPath]]] | None = None
    filter_layout_panels: dict[GroupKey, VectorStageBuckets] | None = None
    filter_large_bbox: dict[GroupKey, VectorStageBuckets] | None = None
    # None means "use Vector.CLUSTER_STEPS's default order" -- settable by
    # the debug app before calling Pipeline.run_page to interactively
    # reorder the clustering stage's 4 operations.
    clustering_order: list[str] | None = None
    clustering: dict[GroupKey, ClusteringStageResult] | None = None
    filter_large_group_bbox: dict[GroupKey, VectorStageBuckets] | None = None
    filter_aspect_ratio: dict[GroupKey, VectorStageBuckets] | None = None
    drawing_vectors: list[DrawingVector] | None = None
    # Future stages add fields here as they're implemented (raster_images,
    # etc.) so later stages can read earlier stages' output.


@dataclass
class StageOutput:
    key: str
    label: str
    status: str  # "ok" | "error"
    data: Any = None
    error: str | None = None


@dataclass
class StageSpec:
    key: str
    label: str
    run: Callable[[PipelineContext], Any]


def _run_reader(ctx: PipelineContext) -> Page:
    ctx.page = ctx.reader.get_page(ctx.page_index)
    return ctx.page


def _run_native(ctx: PipelineContext) -> list[TextWord]:
    ctx.native_words = Native().extract_text(ctx.page)
    return ctx.native_words


def _run_vector_extract(ctx: PipelineContext) -> list[VectorPath]:
    ctx.vector_paths = Vector().extract_paths(ctx.page)
    return ctx.vector_paths


def _run_layer_separation(ctx: PipelineContext) -> dict[str, list[VectorPath]]:
    ctx.paths_by_layer = Vector().separate_by_layer(ctx.vector_paths)
    return ctx.paths_by_layer


def _run_color_separation(ctx: PipelineContext) -> dict[str, dict[tuple, list[VectorPath]]]:
    vector = Vector()
    ctx.paths_by_layer_color = {
        layer: vector.separate_by_color(paths)
        for layer, paths in ctx.paths_by_layer.items()
    }
    return ctx.paths_by_layer_color


def _iter_groups(
    paths_by_layer_color: dict[str, dict[tuple, list[VectorPath]]]
) -> list[tuple[GroupKey, list[VectorPath]]]:
    return [
        ((layer, color), paths)
        for layer, color_groups in paths_by_layer_color.items()
        for color, paths in color_groups.items()
    ]


def _split_dropped(
    paths: list[VectorPath], kept: list[VectorPath]
) -> list[VectorPath]:
    kept_ids = {id(p) for p in kept}
    return [p for p in paths if id(p) not in kept_ids]


def _run_filter_layout_panels(ctx: PipelineContext) -> dict[GroupKey, VectorStageBuckets]:
    vector = Vector()
    result: dict[GroupKey, VectorStageBuckets] = {}

    for key, paths in _iter_groups(ctx.paths_by_layer_color):
        kept = vector.filter_layout_panels(paths)
        dropped = _split_dropped(paths, kept)
        result[key] = VectorStageBuckets(
            this_stage=[[p] for p in dropped], previous=[], pending=kept
        )

    ctx.filter_layout_panels = result
    return result


def _run_filter_large_bbox(ctx: PipelineContext) -> dict[GroupKey, VectorStageBuckets]:
    vector = Vector()
    result: dict[GroupKey, VectorStageBuckets] = {}

    for key, buckets in ctx.filter_layout_panels.items():
        kept = vector.filter_large_bbox(buckets.pending, ctx.page)
        dropped = _split_dropped(buckets.pending, kept)
        result[key] = VectorStageBuckets(
            this_stage=[[p] for p in dropped],
            previous=list(buckets.this_stage),
            pending=kept,
        )

    ctx.filter_large_bbox = result
    return result


def _run_clustering(ctx: PipelineContext) -> dict[GroupKey, ClusteringStageResult]:
    vector = Vector()
    order = ctx.clustering_order or list(vector.CLUSTER_STEPS)
    result: dict[GroupKey, ClusteringStageResult] = {}

    for key, buckets in ctx.filter_large_bbox.items():
        snapshots = vector.cluster(buckets.pending, ctx.page, order)
        result[key] = ClusteringStageResult(
            order=order,
            steps=snapshots,
            previous=list(buckets.previous) + list(buckets.this_stage),
        )

    ctx.clustering = result
    return result


def _run_filter_large_group_bbox(ctx: PipelineContext) -> dict[GroupKey, VectorStageBuckets]:
    vector = Vector()
    result: dict[GroupKey, VectorStageBuckets] = {}

    for key, cluster_result in ctx.clustering.items():
        groups = cluster_result.steps[-1]
        kept = vector.filter_large_group_bbox(groups, ctx.page)
        kept_ids = {id(g) for g in kept}
        dropped = [g for g in groups if id(g) not in kept_ids]
        result[key] = VectorStageBuckets(
            this_stage=dropped,
            previous=list(cluster_result.previous),
            pending=kept,
        )

    ctx.filter_large_group_bbox = result
    return result


def _run_filter_aspect_ratio(ctx: PipelineContext) -> dict[GroupKey, VectorStageBuckets]:
    vector = Vector()
    result: dict[GroupKey, VectorStageBuckets] = {}

    for key, buckets in ctx.filter_large_group_bbox.items():
        kept = vector.filter_aspect_ratio(buckets.pending)
        kept_ids = {id(g) for g in kept}
        dropped = [g for g in buckets.pending if id(g) not in kept_ids]
        result[key] = VectorStageBuckets(
            this_stage=dropped,
            previous=list(buckets.previous) + list(buckets.this_stage),
            pending=kept,
        )

    ctx.filter_aspect_ratio = result
    return result


def _run_drawing_vectors(ctx: PipelineContext) -> list[DrawingVector]:
    """Final aggregation: every group any filter stage dropped along the
    way was already "classified as Drawing" the moment it was dropped (see
    VectorStageBuckets's docstring), so it's folded in here unconditionally
    alongside whatever the final filter_aspect_ratio round's own drawing/
    text split decides. Text clusters are the only content that doesn't
    end up here."""
    vector = Vector()
    all_drawing_paths: list[VectorPath] = []

    for buckets in ctx.filter_aspect_ratio.values():
        for group in buckets.previous:
            all_drawing_paths.extend(group)
        for group in buckets.this_stage:
            all_drawing_paths.extend(group)
        drawing_paths, _text_clusters = vector.classify_clusters(buckets.pending)
        all_drawing_paths.extend(drawing_paths)

    ctx.drawing_vectors = vector.build_drawing_vectors(all_drawing_paths)
    return ctx.drawing_vectors


class Pipeline:
    """Runs Pipeline.STAGES in order for one page, collecting each
    stage's output. A stage that raises does not stop the run or crash a
    caller (e.g. the debug app) -- it's recorded as status="error"."""

    STAGES: list[StageSpec] = [
        StageSpec(key="reader", label="Reader", run=_run_reader),
        StageSpec(key="native", label="Native Text", run=_run_native),
        StageSpec(key="vector_extract", label="Vector Extraction", run=_run_vector_extract),
        StageSpec(key="layer_separation", label="Layer Separation", run=_run_layer_separation),
        StageSpec(key="color_separation", label="Color Separation", run=_run_color_separation),
        StageSpec(
            key="filter_layout_panels",
            label="Filter: Layout Panels",
            run=_run_filter_layout_panels,
        ),
        StageSpec(
            key="filter_large_bbox",
            label="Filter: Large Bbox",
            run=_run_filter_large_bbox,
        ),
        StageSpec(key="clustering", label="Clustering", run=_run_clustering),
        StageSpec(
            key="filter_large_group_bbox",
            label="Filter: Large Group Bbox",
            run=_run_filter_large_group_bbox,
        ),
        StageSpec(
            key="filter_aspect_ratio",
            label="Filter: Aspect Ratio",
            run=_run_filter_aspect_ratio,
        ),
        StageSpec(key="drawing_vectors", label="Drawing Vectors", run=_run_drawing_vectors),
    ]

    def run_page(
        self, reader: Reader, page_index: int, clustering_order: list[str] | None = None,
    ) -> list[StageOutput]:
        ctx = PipelineContext(
            reader=reader, page_index=page_index, clustering_order=clustering_order,
        )
        outputs: list[StageOutput] = []

        for spec in self.STAGES:
            try:
                data = spec.run(ctx)
                outputs.append(StageOutput(spec.key, spec.label, "ok", data))
            except Exception as exc:  # noqa: BLE001 -- debug tool: surface, don't crash
                _LOG.exception("stage %s failed", spec.key)
                outputs.append(
                    StageOutput(spec.key, spec.label, "error", None, str(exc))
                )

        return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the rastervec extraction pipeline on one PDF page."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF.")
    parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="0-based page index to process (default: 0).",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="Only log WARNING and above."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    configure_logging(level)

    with Reader(args.pdf) as reader:
        outputs = Pipeline().run_page(reader, args.page)

        for output in outputs:
            if output.status == "error":
                _LOG.error("stage %s (%s) failed: %s", output.key, output.label, output.error)
                continue

            if output.key == "native":
                words: list[TextWord] = output.data
                _LOG.info(
                    "extracted %d native word(s) from page %d", len(words), args.page
                )
                for word in words[:10]:
                    _LOG.info("  seq=%d %r @ %s", word.seq, word.text, word.bbox)
            else:
                _LOG.info("stage %s (%s) ok", output.key, output.label)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
