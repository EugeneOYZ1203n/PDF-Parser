"""VectorClassification's Phase3Backend entrypoint -- the restored 12-step
Vector_Classification chain + Radon word-segmentation + PaddleOCR
recognition-only OCR, exactly as it ran before being retired in favor of
FastIntoPaddle. Fully self-contained (own fast_detect.py/paddle_engine.py/
layer_color_separation.py/radon.py/config.py) -- imports nothing from
P3_Vector_Parsing/FastIntoPaddle or P2_Raster_To_Vec.
"""
from __future__ import annotations

from rastervec.commons.models import Page, Vector, Text
from rastervec.P3_Vector_Parsing.VectorClassification.classify_vectors import classify_vectors
from rastervec.P3_Vector_Parsing.VectorClassification.fast_filter import (
    detect_text_fast,
    elect_unique_segments,
    group_similar_segments,
)
from rastervec.P3_Vector_Parsing.VectorClassification.ocr import recognize_unique_words, restore_word_texts
from rastervec.P3_Vector_Parsing.VectorClassification.radon import segment_clusters

STEP_NAMES = ["classify", "fast", "segment", "similarity", "ocr", "restore", "drawing"]


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, enable_fast: bool = True, verbose: bool = False, compute=None, progress_counter=None,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors into one flat pool, then runs the old chain: classify -> FAST
    filter -> Radon word-segmentation -> similarity dedup -> OCR recognize
    -> restore each word occurrence's text -> merge every dropped Vector as
    drawing content."""
    all_vectors = list(vectors_p1) + list(vectors_p2)

    cls = classify_vectors(all_vectors, page, verbose=verbose)

    flat_clusters = [
        [v for group in cluster for v in group] for cluster in cls.text_clusters
    ]
    fast = detect_text_fast(
        flat_clusters, page, enable_fast=enable_fast, verbose=verbose,
        compute=compute, progress_counter=progress_counter,
    )

    word_segments = segment_clusters(fast.passed)

    groups = group_similar_segments(word_segments)
    uniques, metas = elect_unique_segments(word_segments, groups)

    unique_texts = recognize_unique_words(uniques, compute=compute, progress_counter=progress_counter)
    restored = restore_word_texts(unique_texts, metas)

    drawing = list(cls.drawing_vectors) + list(fast.dropped_vectors)

    return drawing, restored
