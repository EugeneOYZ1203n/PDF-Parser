"""Pipeline-facing glue for the golden-case regression suite: browse
candidate clusters/segmentations at the three stages that keep/drop
bbox'd items (vector classification, FAST detection, word-splitting),
capture a curator's pick into a `golden_schema.GoldenCase`, and replay a
`GoldenCaseBank` against fresh pipeline runs to check nothing drifted.

The only `Evaluation/Evaluate/` module besides `adapters.py` that imports
`rastervec.pipelines` -- `golden_schema.py` itself stays pure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import ImageDraw

from rastervec.Evaluation.Evaluate.golden_schema import CaseLabel, GoldenCase, Stage
from rastervec.Evaluation.Labelling.label_schema import cluster_signature
from rastervec.helpers.geometry import bbox_iou, union_bbox
from rastervec.pipelines.current import run_pipeline
from rastervec.renderer import pixel_to_page_bbox, render_vector_cluster

if TYPE_CHECKING:
    from PIL import Image

    from rastervec.models import VectorPath
    from rastervec.OCR.radon import ClusterSegmentation
    from rastervec.pipelines.result import PipelineResult

# Notebook-browsing thumbnail resolution -- a display concern, not a
# pipeline-tuning knob, so it stays local to this module rather than
# config.py.
_THUMBNAIL_DPI = 150

DEFAULT_IOU_THRESHOLD = 0.85
# A test-harness tolerance, not a pipeline-tuning knob -- deliberately not
# in config.py, and deliberately not metrics.MetricConfig.iou_edge_min
# (0.10, tuned loose for benchmark-style detection matching): this wants a
# tight "did this basically not change" regression match instead.


@dataclass
class Candidate:
    bbox: "tuple[float, float, float, float]"
    cluster: "list[VectorPath]"
    note: str
    role: str | None = None  # "kept" | "dropped" | "passed"; None for word_split
    segmentation: "ClusterSegmentation | None" = None
    image: "Image.Image | None" = None


# --------------------------------------------------------------------------
# stage pools -- shared by candidate listing (browse, thumbnails included)
# and replay matching (golden_regression.compare_case, no thumbnails)
# --------------------------------------------------------------------------
def _classification_pool(res: "PipelineResult") -> list[Candidate]:
    pool = [
        Candidate(bbox=union_bbox([p.bbox for p in c]), cluster=c, note="kept", role="kept")
        for c in (res.text_clusters or []) if c
    ]
    for bucket in (res.clustering or {}).values():
        for step in bucket.steps:
            for name, cat in step.categories.items():
                if cat.role != "dropped":
                    continue
                for group in cat.groups:
                    if not group:
                        continue
                    pool.append(Candidate(
                        bbox=union_bbox([p.bbox for p in group]), cluster=group,
                        note=f"{step.label} / {name}", role="dropped",
                    ))
    return pool


def _fast_pool(res: "PipelineResult") -> list[Candidate]:
    passed = [
        Candidate(bbox=union_bbox([p.bbox for p in c]), cluster=c, note="FAST passed", role="passed")
        for c in (res.fast_passed or []) if c
    ]
    dropped = [
        Candidate(bbox=union_bbox([p.bbox for p in c]), cluster=c, note="FAST dropped", role="dropped")
        for c in (res.fast_dropped or []) if c
    ]
    return passed + dropped


def _word_split_pool(res: "PipelineResult") -> list[Candidate]:
    pool = []
    for cluster, seg in zip(res.regrouped_clusters or [], res.segmentations or []):
        if not cluster or not seg.word_crops:
            continue
        pool.append(Candidate(
            bbox=union_bbox([p.bbox for p in cluster]), cluster=cluster,
            note=f"{len(seg.word_crops)} word(s), skew={seg.skew_deg:+.1f}",
            role=None, segmentation=seg,
        ))
    return pool


def _pool_for_stage(res: "PipelineResult", stage: Stage) -> list[Candidate]:
    if stage == "classification":
        return _classification_pool(res)
    if stage == "fast":
        return _fast_pool(res)
    if stage == "word_split":
        return _word_split_pool(res)
    raise ValueError(f"unknown stage: {stage!r}")


def _word_split_thumbnail(cand: Candidate) -> "Image.Image":
    seg = cand.segmentation
    base = render_vector_cluster(cand.cluster, seg.render_dpi).convert("RGB")
    d = ImageDraw.Draw(base)
    for corners in seg.word_corners:
        d.polygon(corners, outline="#dc2626", width=1)
    return base


def _with_thumbnails(cands: list[Candidate]) -> list[Candidate]:
    for c in cands:
        c.image = render_vector_cluster(c.cluster, _THUMBNAIL_DPI)
    return cands


# --------------------------------------------------------------------------
# browsing -- one call per stage, positives/negatives already thumbnailed
# --------------------------------------------------------------------------
def list_classification_candidates(res: "PipelineResult", n: int = 3) -> tuple[list[Candidate], list[Candidate]]:
    pool = _classification_pool(res)
    positives = _with_thumbnails([c for c in pool if c.role == "kept"][:n])
    negatives = _with_thumbnails([c for c in pool if c.role == "dropped"][:n])
    return positives, negatives


def list_fast_candidates(res: "PipelineResult", n: int = 3) -> tuple[list[Candidate], list[Candidate]]:
    pool = _fast_pool(res)
    positives = _with_thumbnails([c for c in pool if c.role == "passed"][:n])
    negatives = _with_thumbnails([c for c in pool if c.role == "dropped"][:n])
    return positives, negatives


def list_word_split_candidates(res: "PipelineResult", n: int = 6) -> list[Candidate]:
    cands = _word_split_pool(res)[:n]
    for c in cands:
        c.image = _word_split_thumbnail(c)
    return cands


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------
def capture_case(
    cand: Candidate, *, stage: Stage, label: CaseLabel,
    pdf_path: str, page_index: int, note: str = "",
) -> GoldenCase:
    word_bboxes = None
    if stage == "word_split" and cand.segmentation is not None:
        word_bboxes = [
            pixel_to_page_bbox(cand.cluster, cand.segmentation.render_dpi, corners)
            for corners in cand.segmentation.word_corners
        ]
    return GoldenCase(
        stage=stage, label=label, pdf_path=pdf_path, page_index=page_index,
        note=note or cand.note, signature=cluster_signature(cand.cluster),
        bbox=cand.bbox, role=cand.role, word_bboxes=word_bboxes,
    )


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------
@dataclass
class CaseCheckResult:
    case: GoldenCase
    passed: bool
    detail: str
    best_iou: float


def _compare_word_split(case: GoldenCase, best: Candidate, *, iou_threshold: float, best_iou: float) -> CaseCheckResult:
    stored = case.word_bboxes or []
    current = [
        pixel_to_page_bbox(best.cluster, best.segmentation.render_dpi, corners)
        for corners in best.segmentation.word_corners
    ]
    if len(stored) != len(current):
        return CaseCheckResult(
            case=case, passed=False, best_iou=best_iou,
            detail=f"word count changed: expected {len(stored)}, got {len(current)}",
        )

    pairs = sorted(
        ((bbox_iou(sb, cb), i, j) for i, sb in enumerate(stored) for j, cb in enumerate(current)),
        key=lambda t: -t[0],
    )
    matched_j: set[int] = set()
    match_iou: dict[int, float] = {}
    for iou, i, j in pairs:
        if i in match_iou or j in matched_j:
            continue
        match_iou[i] = iou
        matched_j.add(j)

    drifted = [i for i in range(len(stored)) if match_iou.get(i, 0.0) < iou_threshold]
    if drifted:
        return CaseCheckResult(
            case=case, passed=False, best_iou=best_iou,
            detail=f"word(s) drifted beyond tolerance at stored index(es) {drifted}",
        )
    return CaseCheckResult(case=case, passed=True, best_iou=best_iou, detail="ok")


def compare_case(
    case: GoldenCase, res: "PipelineResult", *, iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> CaseCheckResult:
    """Find the current pool entry closest to `case.bbox` (by IoU) and check
    its decision/word-boxes still match what was stored."""
    pool = _pool_for_stage(res, case.stage)
    best, best_iou = None, 0.0
    for cand in pool:
        iou = bbox_iou(case.bbox, cand.bbox)
        if iou > best_iou:
            best, best_iou = cand, iou

    if best is None or best_iou < iou_threshold:
        return CaseCheckResult(
            case=case, passed=False, best_iou=best_iou,
            detail=f"no cluster within IoU>={iou_threshold:.2f} of stored bbox (best={best_iou:.2f})",
        )

    if case.stage == "word_split":
        return _compare_word_split(case, best, iou_threshold=iou_threshold, best_iou=best_iou)

    if best.role != case.role:
        return CaseCheckResult(
            case=case, passed=False, best_iou=best_iou,
            detail=f"role changed: expected {case.role!r}, got {best.role!r}",
        )
    return CaseCheckResult(case=case, passed=True, best_iou=best_iou, detail="ok")


def run_regression(
    bank, *, iou_threshold: float = DEFAULT_IOU_THRESHOLD, enable_fast: bool = True,
) -> list[CaseCheckResult]:
    """Runs the pipeline once per distinct (pdf_path, page_index) among
    `bank.cases` (cached), and compares every case on that page against
    that one run."""
    cache: dict[tuple[str, int], "PipelineResult"] = {}
    results: list[CaseCheckResult] = []
    for case in bank.cases:
        key = (case.pdf_path, case.page_index)
        if key not in cache:
            cache[key] = run_pipeline(case.pdf_path, case.page_index, enable_fast=enable_fast, verbose=True)
        results.append(compare_case(case, cache[key], iou_threshold=iou_threshold))
    return results


def format_regression_report(results: list[CaseCheckResult]) -> str:
    lines = [
        f"[{'PASS' if r.passed else 'FAIL'}] {r.case.stage}/{r.case.label} "
        f"{r.case.pdf_path}#{r.case.page_index} iou={r.best_iou:.2f} {r.detail}"
        for r in results
    ]
    n_pass = sum(1 for r in results if r.passed)
    lines.append(f"{n_pass}/{len(results)} passed")
    return "\n".join(lines)
