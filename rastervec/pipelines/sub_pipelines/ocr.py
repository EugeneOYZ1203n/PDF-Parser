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
from rastervec.models import ClusterOcrResult, OcrWord, Page, TextVectorResult, VectorPath
from rastervec.OCR.Paddle_OCR.ocr_backend import OcrBackend, _recognize_crops_job
from rastervec.OCR.Paddle_OCR.render_ocr import RenderOCR, render_cluster_for_ocr
from rastervec.OCR.radon import ClusterSegmentation, segment_cluster

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
    """Reuse `rep`'s reading for `cluster` (a similarity-group match), but
    re-project its per-word boxes onto `cluster`'s own bbox instead of
    dropping them -- a similarity group's clusters are geometrically similar
    copies of the same shape at a different page position, so `rep.bbox` ->
    `bbox`'s translate+scale carries each word box along with it."""
    bbox = union_bbox([p.bbox for p in cluster])
    rx0, ry0, rx1, ry1 = rep.bbox
    nx0, ny0, nx1, ny1 = bbox
    sx = (nx1 - nx0) / (rx1 - rx0) if rx1 > rx0 else 1.0
    sy = (ny1 - ny0) / (ry1 - ry0) if ry1 > ry0 else 1.0

    def _map(b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = b
        return (
            nx0 + (x0 - rx0) * sx, ny0 + (y0 - ry0) * sy,
            nx0 + (x1 - rx0) * sx, ny0 + (y1 - ry0) * sy,
        )

    words = (
        [OcrWord(text=w.text, confidence=w.confidence, bbox=_map(w.bbox)) for w in rep.words]
        if rep.words else None
    )
    return TextVectorResult(
        paths=cluster,
        text=rep.text,
        confidence=rep.confidence,
        bbox=bbox,
        ocr_bbox=_map(rep.ocr_bbox) if rep.ocr_bbox else None,
        rotation_used=rep.rotation_used,
        page_index=page.meta.index,
        words=words,
    )


def recognize(
    segmentations: list[ClusterSegmentation],
    clusters: list[list[VectorPath]],
    page: Page,
    *,
    backend: OcrBackend | None = None,
    similarity_id: dict[int, int] | None = None,
    compute=None,
) -> OcrResult:
    """Recognise each cluster's word crops. One real recognition per
    similarity group; the reading is reused (text/confidence/rotation only)
    for every other cluster sharing that group id. A blank reading folds
    the cluster into `failed` (drawing content). `compute`, when given a
    shared compute-pool proxy (see `Reader/Parallel`), replaces the actual
    engine call (`backend.recognize_crops`) with a dispatch to that pool
    -- the memoization/orchestration here stays local either way."""
    render_ocr_kwargs = {"backend": backend}
    if compute is not None:
        render_ocr_kwargs["recognize_fn"] = lambda crops: compute.apply(_recognize_crops_job, (crops,))
    render_ocr = RenderOCR(**render_ocr_kwargs)
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
