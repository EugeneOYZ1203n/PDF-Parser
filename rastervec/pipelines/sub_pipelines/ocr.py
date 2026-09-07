"""OCR sub-pipeline: segment each text-candidate cluster into deskewed
word crops (Radon), then recognise them.

`segment_for_ocr` runs before recognition -- it is the text-detection
step. `recognize` then OCRs one representative per similarity group and
reuses that reading for the rest, exactly as the old `_run_ocr_compare`
did.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from tqdm import tqdm

from rastervec.helpers.geometry import union_bbox
from rastervec.logging_setup import get_logger
from rastervec.models import ClusterOcrResult, Page, TextVectorResult, VectorPath
from rastervec.OCR.Paddle_OCR.ocr_backend import OcrBackend
from rastervec.OCR.Paddle_OCR.render_ocr import RenderOCR, render_cluster_for_ocr
from rastervec.pipelines.sub_pipelines.radon import ClusterSegmentation, segment_cluster

_LOG = get_logger("ocr")


@dataclass
class OcrResult:
    cluster_results: list[ClusterOcrResult]
    results: list[TextVectorResult]
    failed: list[list[VectorPath]] = field(default_factory=list)


def segment_for_ocr(
    clusters: list[list[VectorPath]], *, dpi: int = 300, verbose: bool = False,
) -> list[ClusterSegmentation]:
    """Deskew + line/word split each cluster's render -- the pre-OCR
    detection step. One `ClusterSegmentation` per input cluster."""
    out: list[ClusterSegmentation] = []
    for cluster in clusters:
        image, dpi_used = render_cluster_for_ocr(cluster, dpi)
        out.append(segment_cluster(image, dpi_used, verbose=verbose))
    return out


def _reuse(rep: TextVectorResult, cluster: list[VectorPath], page: Page) -> TextVectorResult:
    return TextVectorResult(
        paths=cluster,
        text=rep.text,
        confidence=rep.confidence,
        bbox=union_bbox([p.bbox for p in cluster]),
        ocr_bbox=None,
        rotation_used=rep.rotation_used,
        page_index=page.meta.index,
        words=None,
    )


def recognize(
    segmentations: list[ClusterSegmentation],
    clusters: list[list[VectorPath]],
    page: Page,
    *,
    backend: OcrBackend | None = None,
    similarity_id: dict[int, int] | None = None,
) -> OcrResult:
    """Recognise each cluster's word crops. One real recognition per
    similarity group; the reading is reused (text/confidence/rotation only)
    for every other cluster sharing that group id. A blank reading folds
    the cluster into `failed` (drawing content)."""
    render_ocr = RenderOCR(backend=backend)
    similarity_id = similarity_id or {}

    cluster_results: list[ClusterOcrResult] = []
    results: list[TextVectorResult] = []
    failed: list[list[VectorPath]] = []
    resolved_by_group: dict[int, TextVectorResult] = {}

    for cluster, seg in tqdm(
        list(zip(clusters, segmentations)), desc="OCR compare", unit="cluster",
    ):
        group_id = similarity_id.get(id(cluster))
        rep = resolved_by_group.get(group_id) if group_id is not None else None

        if rep is not None:
            resolved = _reuse(rep, cluster, page)
            ocr_seconds = 0.0
        else:
            start = time.perf_counter()
            resolved = render_ocr.recognize_segmented(seg, cluster, page)
            ocr_seconds = time.perf_counter() - start
            if group_id is not None:
                resolved_by_group[group_id] = resolved

        cluster_results.append(
            ClusterOcrResult(cluster=cluster, resolved=resolved, ocr_seconds=ocr_seconds)
        )
        results.append(resolved)
        if not resolved.text.strip():
            failed.append(cluster)

    return OcrResult(cluster_results=cluster_results, results=results, failed=failed)
