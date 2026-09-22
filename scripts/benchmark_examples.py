"""Illustrated error-example collection for `pipeline_report_benchmark.py`'s
HTML report: crop a bbox out of the run's own converted PDF (`_crop_to_png`)
and gather up to `_EXAMPLE_CAP` examples per (text_type, category) across a
run's pages (`_collect_examples`)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf as fitz

from rastervec.Evaluation.Evaluate.confusion_metrics import closest_pred_word
from rastervec.Evaluation.Evaluate.html_report import ExampleCard
from rastervec.Evaluation.Evaluate.metrics import TEXT_TYPES, OverlapGraph, TextMetricSuiteResult
from rastervec.Evaluation.Evaluate.text_metrics import word_tokens
from rastervec.commons.logging_setup import get_logger

from scripts.benchmark_run_loading import _RASTER_TEXT_TYPES, RunEntry

_LOG = get_logger("pipeline_report_benchmark")

_EXAMPLE_CAP = 5


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_") or "key"


def _short_slug(text: str, max_len: int = 16) -> str:
    """A filename-safe slug capped to `max_len` chars (+ a short hash
    suffix for uniqueness) -- unlike `_slug`, safe to compose several of
    into one path component without hitting Windows' ~260-char MAX_PATH."""
    full = _slug(text)
    if len(full) <= max_len:
        return full
    h = hashlib.sha1(text.encode()).hexdigest()[:8]
    return f"{full[:max_len]}_{h}"


def _crop_to_png(pdf_path: Path, bbox, out_path: Path, *, dpi: float = 150.0) -> bool:
    """Crops `bbox` (page-space, unrotated MediaBox) out of `pdf_path`'s
    page 0 -- every `converted_p<N>.pdf` this benchmark scores against is a
    single-page vectorised render. `get_pixmap()` always bakes the page's own
    `/Rotate` into what it renders (same gotcha `P1_Reading_Native/
    image_extract.py`'s whole-page raster counter-rotates for), and empirically
    `clip=` is interpreted in that same rotated display space regardless of
    whether an explicit `matrix=` is also passed -- so `bbox` must be mapped
    into display space via `page.rotation_matrix` before clipping, or the crop
    silently lands on the wrong region of a rotated page."""
    try:
        doc = fitz.open(str(pdf_path))
        try:
            page = doc[0]
            rect = fitz.Rect(*bbox)
            if rect.is_empty or rect.is_infinite:
                return False
            if page.rotation:
                rect = fitz.Rect(
                    fitz.Point(rect.x0, rect.y0) * page.rotation_matrix,
                    fitz.Point(rect.x1, rect.y1) * page.rotation_matrix,
                )
                rect.normalize()
            pix = page.get_pixmap(clip=rect, dpi=int(dpi))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(out_path))
            return True
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("crop failed for %s %s: %s", pdf_path, bbox, exc)
        return False


def _collect_examples(
    entry: RunEntry,
    per_page: "list[tuple[int, TextMetricSuiteResult, dict[str, OverlapGraph]]]",
    examples_dir: Path, kslug: str,
) -> "dict[tuple[str, str], list[ExampleCard]]":
    """Up to `_EXAMPLE_CAP` examples per (text_type, category) across all of
    this key/run's pages -- extra predictions (no GT overlap at all), missed
    GT (no prediction reached it), confusion misreads (GT word vs. its
    closest overlapping predicted word)."""
    out: "dict[tuple[str, str], list[ExampleCard]]" = {}
    rslug = _short_slug(entry.run_name)
    for text_type in TEXT_TYPES:
        extra: "list[ExampleCard]" = []
        missed: "list[ExampleCard]" = []
        confusion: "list[ExampleCard]" = []
        for pi, _res, graphs_by_type in per_page:
            graph = graphs_by_type[text_type]
            conv_pdf = entry.doc_dir / (
                f"rasterised_p{pi}.pdf" if text_type in _RASTER_TEXT_TYPES
                else f"converted_p{pi}.pdf"
            )

            if len(extra) < _EXAMPLE_CAP:
                for pj, p in enumerate(graph.preds):
                    if len(extra) >= _EXAMPLE_CAP:
                        break
                    if graph.edges_by_pred[pj]:
                        continue
                    n = len(extra)
                    img = examples_dir / f"{kslug}__{rslug}__{text_type}__extra_prediction__{n}.png"
                    ok = _crop_to_png(conv_pdf, p.bbox, img)
                    extra.append(ExampleCard(
                        caption=f"p{pi} pred={p.text!r}",
                        image_path=(Path("examples") / img.name) if ok else None,
                        run=entry.run_name, text_type=text_type,
                    ))

            if len(missed) < _EXAMPLE_CAP:
                for gi in graph.missed_gt_idxs:
                    if len(missed) >= _EXAMPLE_CAP:
                        break
                    g = graph.gt[gi]
                    n = len(missed)
                    img = examples_dir / f"{kslug}__{rslug}__{text_type}__missed_gt__{n}.png"
                    ok = _crop_to_png(conv_pdf, g.bbox, img)
                    missed.append(ExampleCard(
                        caption=f"p{pi} gt={g.text!r}",
                        image_path=(Path("examples") / img.name) if ok else None,
                        run=entry.run_name, text_type=text_type,
                    ))

            if len(confusion) < _EXAMPLE_CAP:
                for gi, g in enumerate(graph.gt):
                    if len(confusion) >= _EXAMPLE_CAP:
                        break
                    overlapping = graph.overlapping_preds_by_gt[gi]
                    if not overlapping:
                        continue
                    pred_words_all: "list[str]" = []
                    for pj in overlapping:
                        pred_words_all.extend(word_tokens(graph.preds[pj].text))
                    pred_word_set = set(pred_words_all)
                    for gt_word in word_tokens(g.text):
                        if gt_word in pred_word_set:
                            continue
                        closest = closest_pred_word(gt_word, pred_words_all)
                        if closest is None or closest == gt_word:
                            continue
                        n = len(confusion)
                        img = examples_dir / f"{kslug}__{rslug}__{text_type}__confusion__{n}.png"
                        ok = _crop_to_png(conv_pdf, g.bbox, img)
                        confusion.append(ExampleCard(
                            caption=f"p{pi} gt={gt_word!r} pred={closest!r}",
                            image_path=(Path("examples") / img.name) if ok else None,
                            run=entry.run_name, text_type=text_type,
                        ))
                        break  # one example per gt region

        out[(text_type, "extra_prediction")] = extra
        out[(text_type, "missed_gt")] = missed
        out[(text_type, "confusion")] = confusion
    return out
