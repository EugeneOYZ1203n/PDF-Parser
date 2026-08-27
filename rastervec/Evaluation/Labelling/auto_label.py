"""Automatic labelling: derives ground-truth text labels for vector-text
clusters without any manual work, by exploiting the fact that Conversion
(`Evaluation/Conversion/conversion.py`) turns known native text into vector
paths -- so whatever the *original* native text said, at the same page
location, is the correct label for whichever vector cluster ends up there.

Pipeline: read `pdf_path`'s native `TextWord`s (ground truth), convert that
same page to vector text via `convert_page_to_vector_text`, run the vector
extraction/clustering chain (reader through text_candidates only -- OCR
isn't needed since the ground truth already gives the text) on the
converted page, then spatially match each surviving cluster's bbox against
the native words' bboxes by IoU (`helpers.geometry.bbox_iou`). A cluster
whose bbox overlaps one or more native words above `iou_threshold` gets
those words' text (sorted left-to-right, joined with spaces) as its label,
source="auto"; a cluster with no matching word is skipped (nothing to
label it with -- typically a different clustering split than the original
per-word granularity would need a lower threshold or run to run merging
instead of raising it here).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from rastervec.Evaluation.Conversion.conversion import convert_page_to_vector_text
from rastervec.Evaluation.Labelling.label_schema import (
    LabelEntry,
    LabelSet,
    cluster_signature,
)
from rastervec.helpers.geometry import bbox_iou, union_bbox
from rastervec.logging_setup import get_logger
from rastervec.Native_Text.native import Native
from rastervec.pipeline import (
    PipelineContext,
    _run_clustering,
    _run_color_separation,
    _run_layer_separation,
    _run_native,
    _run_reader,
    _run_text_candidates,
    _run_vector_extract,
)
from rastervec.Reader.reader import Reader

_LOG = get_logger("auto_label")

DEFAULT_IOU_THRESHOLD = 0.3


def auto_label_pdf(
    pdf_path: str, page_index: int, iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> LabelSet:
    """Auto-labels one page of `pdf_path` (a native-text PDF). Returns a
    `LabelSet` with one `LabelEntry` (source="auto") per matched cluster."""
    with Reader(pdf_path) as reader:
        page = reader.get_page(page_index)
        native_words = Native().extract_text(page)

    converted_bytes = convert_page_to_vector_text(pdf_path, page_index)

    with tempfile.TemporaryDirectory() as tmp_dir:
        converted_path = str(Path(tmp_dir) / "converted.pdf")
        Path(converted_path).write_bytes(converted_bytes)

        with Reader(converted_path) as conv_reader:
            ctx = PipelineContext(reader=conv_reader, page_index=0)
            _run_reader(ctx)
            _run_native(ctx)
            _run_vector_extract(ctx)
            _run_layer_separation(ctx)
            _run_color_separation(ctx)
            _run_clustering(ctx)
            _run_text_candidates(ctx)

    clusters = ctx.text_clusters or []
    entries: list[LabelEntry] = []

    for cluster in clusters:
        if not cluster:
            continue
        cluster_bbox = union_bbox([p.bbox for p in cluster])
        matched_words = [
            w for w in native_words if bbox_iou(w.bbox, cluster_bbox) >= iou_threshold
        ]
        if not matched_words:
            continue
        matched_words.sort(key=lambda w: w.bbox[0])
        text = " ".join(w.text for w in matched_words)
        entries.append(
            LabelEntry(
                page_index=page_index,
                cluster_bbox=cluster_bbox,
                cluster_signature=cluster_signature(cluster),
                text=text,
                source="auto",
            )
        )

    _LOG.debug(
        "auto_label_pdf: page %d, %d cluster(s), %d matched",
        page_index, len(clusters), len(entries),
    )
    return LabelSet(pdf_path=pdf_path, entries=entries)
