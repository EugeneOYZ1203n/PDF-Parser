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

from tqdm import tqdm

if __name__ == "__main__" and __package__ is None:
    # Allow running this file directly (`python rastervec/pipeline.py`),
    # not just as a module (`python -m rastervec.pipeline`), by putting
    # the repo root -- the parent of this package -- on sys.path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rastervec.helpers.render_ocr import RenderOCR
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.models import DrawingVector, Page, TextVectorResult, TextWord, VectorPath
from rastervec.native import Native
from rastervec.reader import Reader
from rastervec.renderer import Renderer
from rastervec.vector import Vector

_LOG = get_logger("pipeline")

# (layer, color) -- one Vector.separate_by_color() bucket.
GroupKey = tuple[str, tuple]


@dataclass
class ClusteringStageResult:
    """One (layer, color) group's result from the single configurable
    clustering stage: the `order` the (up to 8) Vector.PIPELINE_STEPS
    operations ran in, `steps[i]` = the surviving-groups snapshot after
    applying `order[i]` (so `steps[-1]` is the final result, handed to
    drawing_vectors/ocr_text_clusters), and `dropped[i]` = only what step
    `i` itself dropped (empty for a pure clustering step or "none" -- only
    the 4 filter steps ever drop anything). Every dropped group was
    "classified as Drawing" the moment it was dropped (see
    _run_drawing_vectors) -- a caller wanting the cumulative drop total up
    to step i sums `dropped[0:i+1]`."""

    order: list[str]
    steps: list[list[list[VectorPath]]]
    dropped: list[list[list[VectorPath]]]


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
    # None means "use Vector.PIPELINE_STEPS's default order" -- settable by
    # the debug app before calling Pipeline.run_page to interactively
    # reorder the clustering stage's (up to 8) operations. A step may
    # repeat at more than one ordinal position.
    clustering_order: list[str] | None = None
    # Step key -> {param_name: value} overrides for that pipeline method
    # (see Vector.cluster's step_params) -- e.g. {"cluster_spatial":
    # {"threshold": 12.0}}. None/missing entries fall back to that method's
    # own instance-attribute default. Settable by the debug app alongside
    # clustering_order to interactively tune each operation's thresholds.
    clustering_params: dict[str, dict[str, float]] | None = None
    clustering: dict[GroupKey, ClusteringStageResult] | None = None
    drawing_vectors: list[DrawingVector] | None = None
    # Populated alongside drawing_vectors -- whatever the clustering
    # stage's final step left kept (every filter/cluster step's drops go
    # to drawing_vectors instead, never here). Consumed by ocr_text_clusters.
    text_clusters: list[list[VectorPath]] | None = None
    ocr_text_clusters: list[TextVectorResult] | None = None
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


def _run_clustering(ctx: PipelineContext) -> dict[GroupKey, ClusteringStageResult]:
    vector = Vector()
    order = ctx.clustering_order or list(vector.PIPELINE_STEPS)
    result: dict[GroupKey, ClusteringStageResult] = {}

    for key, paths in _iter_groups(ctx.paths_by_layer_color):
        kept_snapshots, dropped_snapshots = vector.cluster(
            paths, ctx.page, order, step_params=ctx.clustering_params,
        )
        result[key] = ClusteringStageResult(
            order=order, steps=kept_snapshots, dropped=dropped_snapshots,
        )

    ctx.clustering = result
    return result


def _run_drawing_vectors(ctx: PipelineContext) -> list[DrawingVector]:
    """Final aggregation: every group any filter step in the clustering
    chain dropped along the way was already "classified as Drawing" the
    moment it was dropped (see ClusteringStageResult's docstring), so it's
    folded in here unconditionally. Whatever the chain's final step still
    kept is *not* drawing content -- there's no further drawing-vs-text
    heuristic applied to it (Vector no longer has one); it's stashed on
    ctx.text_clusters as-is for ocr_text_clusters to actually OCR, the only
    content that doesn't end up in drawing_vectors."""
    vector = Vector()
    all_drawing_paths: list[VectorPath] = []
    all_text_clusters: list[list[VectorPath]] = []

    for cluster_result in ctx.clustering.values():
        for dropped_this_step in cluster_result.dropped:
            for group in dropped_this_step:
                all_drawing_paths.extend(group)
        if cluster_result.steps:
            all_text_clusters.extend(cluster_result.steps[-1])

    ctx.text_clusters = all_text_clusters
    ctx.drawing_vectors = vector.build_drawing_vectors(all_drawing_paths)
    return ctx.drawing_vectors


def _run_ocr_text_clusters(ctx: PipelineContext) -> list[TextVectorResult]:
    """OCRs each text-classified vector cluster from drawing_vectors's
    split (ctx.text_clusters) via Renderer.render_vector_cluster +
    RenderOCR.ocr_cluster -- one high-res isolated render per cluster,
    OCR'd across several rotations and confidence-voted (see
    RenderOCR.combine_rotation_results). RenderOCR's underlying PaddleOCR
    engine is cached at module scope (see helpers/render_ocr.py), so
    constructing a fresh RenderOCR() here per run is cheap after the
    first call. Each cluster is a real (if engine-warm-cheap) multi-
    rotation OCR round-trip, so a page with many text clusters can take a
    visible moment -- the CLI and the debug app (which runs the pipeline
    synchronously before its window becomes interactive) both just call
    Pipeline.run_page() directly, so a tqdm bar here is the only progress
    feedback available while this stage runs."""
    renderer = Renderer()
    render_ocr = RenderOCR()
    clusters = ctx.text_clusters or []
    results = [
        render_ocr.ocr_cluster(cluster, ctx.page, renderer)
        for cluster in tqdm(clusters, desc="OCR text clusters", unit="cluster")
    ]
    ctx.ocr_text_clusters = results
    return results


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
        StageSpec(key="clustering", label="Clustering", run=_run_clustering),
        StageSpec(key="drawing_vectors", label="Drawing Vectors", run=_run_drawing_vectors),
        StageSpec(
            key="ocr_text_clusters",
            label="OCR: Text Clusters",
            run=_run_ocr_text_clusters,
        ),
    ]

    @classmethod
    def stage_keys(cls) -> list[str]:
        return [spec.key for spec in cls.STAGES]

    def run_page(
        self,
        reader: Reader,
        page_index: int,
        clustering_order: list[str] | None = None,
        clustering_params: dict[str, dict[str, float]] | None = None,
        final_stage: str | None = None,
    ) -> list[StageOutput]:
        """Runs Pipeline.STAGES in order, stopping after `final_stage`
        (inclusive) instead of running every stage -- e.g. `final_stage=
        "drawing_vectors"` skips ocr_text_clusters (and any later stage)
        entirely, never even constructing a RenderOCR/PaddleOCR engine, so
        it's a real way to skip the OCR round-trip while iterating on
        earlier stages, not just a display-time filter. `None` (default)
        runs every stage, unchanged from before this parameter existed.
        `clustering_params` is threaded onto `ctx` for `_run_clustering` to
        pass as `Vector.cluster`'s `step_params` -- per-method threshold/
        parameter overrides, keyed by step (see PipelineContext.
        clustering_params)."""
        if final_stage is not None and final_stage not in self.stage_keys():
            raise ValueError(
                f"unknown final_stage {final_stage!r}; must be one of {self.stage_keys()}"
            )

        ctx = PipelineContext(
            reader=reader, page_index=page_index, clustering_order=clustering_order,
            clustering_params=clustering_params,
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
            if spec.key == final_stage:
                break

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
    parser.add_argument(
        "--final-stage",
        choices=Pipeline.stage_keys(),
        default=None,
        help="Stop after this stage instead of running the whole pipeline -- e.g. "
        "--final-stage drawing_vectors skips ocr_text_clusters (and the PaddleOCR "
        "engine it would otherwise build) entirely. Default: run every stage.",
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
        outputs = Pipeline().run_page(reader, args.page, final_stage=args.final_stage)

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
