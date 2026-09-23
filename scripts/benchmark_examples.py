"""Illustrated error-example collection for `pipeline_report_benchmark.py`'s
HTML report: crop a bbox out of the run's own converted PDF (`_crop_to_png`
/ the caching `_Cropper`), gather up to `_EXAMPLE_CAP` examples per
(text_type, category) across a run's pages (`_collect_examples`), and up to
`_CHAR_EXAMPLE_CAP` word crops per (text_type, gt char, error kind)
(`_collect_char_examples`)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf as fitz

from rastervec.Evaluation.Evaluate.confusion_metrics import CharEvent, char_events, closest_pred_word
from rastervec.Evaluation.Evaluate.html_report import ExampleCard
from rastervec.Evaluation.Evaluate.label_overlays import gt_word_bboxes
from rastervec.Evaluation.Evaluate.metrics import (
    TEXT_TYPES,
    OverlapGraph,
    TextMetricSuiteResult,
    rotation_outcome,
)
from rastervec.Evaluation.Evaluate.text_metrics import word_tokens
from rastervec.commons.logging_setup import get_logger

from scripts.benchmark_run_loading import _RASTER_TEXT_TYPES, RunEntry

_LOG = get_logger("pipeline_report_benchmark")

_EXAMPLE_CAP = 3
_CHAR_EXAMPLE_CAP = 2
# Per-char error kinds that get example crops (`detected`/`inserted` don't).
CHAR_EXAMPLE_KINDS = ("dropped", "unreached", "misclassified")
# rotation_outcome bucket -> _collect_examples category
_ROTATION_CATEGORIES = {"off_90": "rotation_off90", "off_180": "rotation_off180"}


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


def _crop_page_to_png(page: "fitz.Page", bbox, out_path: Path, *, dpi: float = 150.0) -> bool:
    """Crops `bbox` (page-space, unrotated MediaBox) out of `page`.
    `get_pixmap()` always bakes the page's own `/Rotate` into what it renders
    (same gotcha `P1_Reading_Native/image_extract.py`'s whole-page raster
    counter-rotates for), and empirically `clip=` is interpreted in that same
    rotated display space regardless of whether an explicit `matrix=` is also
    passed -- so `bbox` must be mapped into display space via
    `page.rotation_matrix` before clipping, or the crop silently lands on the
    wrong region of a rotated page."""
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


def _crop_to_png(pdf_path: Path, bbox, out_path: Path, *, dpi: float = 150.0) -> bool:
    """Crops `bbox` out of `pdf_path`'s page 0 -- every `converted_p<N>.pdf`
    this benchmark scores against is a single-page vectorised render. See
    `_crop_page_to_png` for the rotated-page handling."""
    try:
        doc = fitz.open(str(pdf_path))
        try:
            return _crop_page_to_png(doc[0], bbox, out_path, dpi=dpi)
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("crop failed for %s %s: %s", pdf_path, bbox, exc)
        return False


class _Cropper:
    """`_crop_to_png` with each PDF opened once and each distinct
    `(pdf, bbox)` crop rendered once -- the per-char galleries reuse one word
    crop for every character it illustrates. Use as a context manager so the
    open documents get closed."""

    def __init__(self, examples_dir: Path, prefix: str) -> None:
        self._examples_dir = examples_dir
        self._prefix = prefix
        self._docs: "dict[Path, fitz.Document | None]" = {}
        self._crops: "dict[tuple, Path | None]" = {}

    def __enter__(self) -> "_Cropper":
        return self

    def __exit__(self, *_args) -> None:
        for doc in self._docs.values():
            if doc is not None:
                doc.close()
        self._docs.clear()

    def _doc(self, pdf_path: Path) -> "fitz.Document | None":
        if pdf_path not in self._docs:
            try:
                self._docs[pdf_path] = fitz.open(str(pdf_path))
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("could not open %s for crops: %s", pdf_path, exc)
                self._docs[pdf_path] = None
        return self._docs[pdf_path]

    def crop(self, pdf_path: Path, bbox) -> "Path | None":
        """Report-relative `examples/<name>.png`, or `None` if the crop failed."""
        key = (pdf_path, tuple(round(v, 3) for v in bbox))
        if key in self._crops:
            return self._crops[key]
        rel: "Path | None" = None
        doc = self._doc(pdf_path)
        if doc is not None:
            img = self._examples_dir / f"{self._prefix}__{len(self._crops)}.png"
            try:
                if _crop_page_to_png(doc[0], bbox, img):
                    rel = Path("examples") / img.name
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("crop failed for %s %s: %s", pdf_path, bbox, exc)
        self._crops[key] = rel
        return rel


def _example_pdf(entry: RunEntry, text_type: str, page_index: int) -> Path:
    return entry.doc_dir / (
        f"rasterised_p{page_index}.pdf" if text_type in _RASTER_TEXT_TYPES
        else f"converted_p{page_index}.pdf"
    )


def _collect_examples(
    entry: RunEntry,
    per_page: "list[tuple[int, TextMetricSuiteResult, dict[str, OverlapGraph]]]",
    examples_dir: Path, kslug: str,
) -> "dict[tuple[str, str], list[ExampleCard]]":
    """Up to `_EXAMPLE_CAP` examples per (text_type, category) across all of
    this key/run's pages -- extra predictions (no GT overlap at all), missed
    GT (no prediction reached it), confusion misreads (GT word vs. its
    closest overlapping predicted word), and localized GT regions whose voted
    rotation is off by 90/180 deg (`rotation_off90`/`rotation_off180`)."""
    out: "dict[tuple[str, str], list[ExampleCard]]" = {}
    rslug = _short_slug(entry.run_name)
    for text_type in TEXT_TYPES:
        extra: "list[ExampleCard]" = []
        missed: "list[ExampleCard]" = []
        confusion: "list[ExampleCard]" = []
        rotation: "dict[str, list[ExampleCard]]" = {c: [] for c in _ROTATION_CATEGORIES.values()}
        for pi, _res, graphs_by_type in per_page:
            graph = graphs_by_type[text_type]
            conv_pdf = _example_pdf(entry, text_type, pi)

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

            for gi in graph.localized_gt_idxs:
                if all(len(cards) >= _EXAMPLE_CAP for cards in rotation.values()):
                    break
                predicted, _diff, bucket = rotation_outcome(graph, gi)
                category = _ROTATION_CATEGORIES.get(bucket)
                if category is None or len(rotation[category]) >= _EXAMPLE_CAP:
                    continue
                g = graph.gt[gi]
                read = " | ".join(graph.preds[pj].text for pj in graph.assigned_preds_by_gt[gi])
                n = len(rotation[category])
                img = examples_dir / f"{kslug}__{rslug}__{text_type}__{category}__{n}.png"
                ok = _crop_to_png(conv_pdf, g.bbox, img)
                rotation[category].append(ExampleCard(
                    caption=(
                        f"p{pi} gt={g.text!r} expected={g.expected_rotation}° "
                        f"pred={predicted}° read={read!r}"
                    ),
                    image_path=(Path("examples") / img.name) if ok else None,
                    run=entry.run_name, text_type=text_type,
                ))

        out[(text_type, "extra_prediction")] = extra
        out[(text_type, "missed_gt")] = missed
        out[(text_type, "confusion")] = confusion
        for category, cards in rotation.items():
            out[(text_type, category)] = cards
    return out


def _char_caption(page_index: int, ev: CharEvent) -> str:
    if ev.kind == "unreached":
        return f"p{page_index} {ev.gt_word!r} (region unreached)"
    if ev.pred_word is None:
        return f"p{page_index} {ev.gt_word!r} (no predicted words)"
    if ev.kind == "misclassified":
        return f"p{page_index} {ev.gt_word!r} → {ev.pred_word!r} ({ev.char}→{ev.replacement})"
    return f"p{page_index} {ev.gt_word!r} → {ev.pred_word!r} ({ev.char} dropped)"


def _collect_char_examples(
    entry: RunEntry,
    per_page: "list[tuple[int, TextMetricSuiteResult, dict[str, OverlapGraph]]]",
    examples_dir: Path, kslug: str,
) -> "dict[str, dict[str, dict[str, list[ExampleCard]]]]":
    """`{text_type: {gt_char: {kind: [ExampleCard]}}}` -- up to
    `_CHAR_EXAMPLE_CAP` word crops per (text_type, gt char, kind), for each
    kind in `CHAR_EXAMPLE_KINDS`, across this key/run's pages. Walks the same
    `confusion_metrics.char_events` the per-char table is counted from, so the
    examples always match the counts. Each crop is the GT word's own slice
    of its region (`label_overlays.gt_word_bboxes`). A word is used at most
    once per (char, kind)."""
    out: "dict[str, dict[str, dict[str, list[ExampleCard]]]]" = {}
    rslug = _short_slug(entry.run_name)
    with _Cropper(examples_dir, f"{kslug}__{rslug}__charword") as cropper:
        for text_type in TEXT_TYPES:
            by_char: "dict[str, dict[str, list[ExampleCard]]]" = {}
            used: "set[tuple]" = set()
            for pi, _res, graphs_by_type in per_page:
                graph = graphs_by_type[text_type]
                pdf = _example_pdf(entry, text_type, pi)
                word_boxes: "dict[int, list]" = {}
                for ev in char_events(graph):
                    if ev.kind not in CHAR_EXAMPLE_KINDS:
                        continue
                    cards = by_char.setdefault(ev.char, {}).setdefault(ev.kind, [])
                    if len(cards) >= _CHAR_EXAMPLE_CAP:
                        continue
                    use_key = (pi, ev.gt_idx, ev.word_idx, ev.char, ev.kind)
                    if use_key in used:
                        continue
                    used.add(use_key)
                    if ev.gt_idx not in word_boxes:
                        word_boxes[ev.gt_idx] = gt_word_bboxes(graph.gt[ev.gt_idx])
                    _word, bbox = word_boxes[ev.gt_idx][ev.word_idx]
                    cards.append(ExampleCard(
                        caption=_char_caption(pi, ev),
                        image_path=cropper.crop(pdf, bbox),
                        run=entry.run_name, text_type=text_type,
                    ))
            out[text_type] = by_char
    return out
