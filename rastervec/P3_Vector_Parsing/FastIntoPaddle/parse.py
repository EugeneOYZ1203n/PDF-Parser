"""FastIntoPaddle's Phase3Backend entrypoint -- similarity grouping -> FAST
filter -> reclassify -> (layer, color, width) separation -> spatial
clustering -> per-cluster PaddleOCR detect -> overlap reassignment ->
rotation refine -> PaddleOCR recognize. Fully self-contained -- imports
nothing from P3_Vector_Parsing/VectorClassification or P2_Raster_To_Vec.
"""
from __future__ import annotations

from rastervec.commons.models import Page, Text, Vector
from rastervec.P3_Vector_Parsing.FastIntoPaddle.config import FAST_PADDLE_SEQ_MERGE_TOLERANCE
from rastervec.P3_Vector_Parsing.FastIntoPaddle.paddle_engine import recognize_segments
from rastervec.P3_Vector_Parsing.FastIntoPaddle.steps import (
    build_drawing_output,
    cluster_buckets,
    detect_text_paddle_per_cluster,
    filter_vectors_fast,
    reassign_by_overlap,
    reclassify_by_similarity,
    rotate_paddle_detections,
    separate_by_layer_color_width,
    similarity_group,
)

STEP_NAMES = [
    "similarity", "fast", "reclassify", "separation",
    "clusters", "paddle_detect", "assignment", "rotate", "ocr", "drawing",
]


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, enable_fast: bool = True, verbose: bool = False, compute=None, progress_counter=None,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors into one flat pool, then runs the FastIntoPaddle chain."""
    vectors = list(vectors_p1) + list(vectors_p2)

    groups = similarity_group(vectors)
    fast = filter_vectors_fast(
        vectors, page, enable_fast=enable_fast, verbose=verbose,
        compute=compute, progress_counter=progress_counter,
    )
    reclass = reclassify_by_similarity(fast.passed, fast.dropped, groups)
    buckets = separate_by_layer_color_width(reclass.passed)
    clusters = cluster_buckets(buckets, FAST_PADDLE_SEQ_MERGE_TOLERANCE)
    cluster_detections = detect_text_paddle_per_cluster(clusters)
    reassignment = reassign_by_overlap(clusters, cluster_detections)
    segments = rotate_paddle_detections(clusters, cluster_detections, reassignment.text)

    recognize_fn = None
    if compute is not None:
        from rastervec.P3_Vector_Parsing.FastIntoPaddle.paddle_engine import _recognize_crops_job

        recognize_fn = lambda crops: compute.apply(_recognize_crops_job, (crops,))  # noqa: E731
    texts = recognize_segments(segments, recognize_fn=recognize_fn)

    drawing = build_drawing_output(reassignment.drawing, reclass.dropped)

    return drawing, texts
