"""The OCR backend: recognises pre-segmented word crops with PaddleOCR and
returns one `Text` per word.

Text *detection* is not PaddleOCR's job in this pipeline -- FAST (Phase D,
`pipelines/_steps.py::detect_text_fast`) first narrows classification's
clusters down to the ones that look like text, then Radon segmentation
(`OCR/radon.py::segment_clusters`, Phase E, called on every FAST-surviving
cluster) splits each one into word-level `Segment`s, each already carrying
its own deskewed, white-padded crop (`Segment.image`) -- so by the time a
`Segment` reaches this module it *is* one word, already rotated to
(approximately) upright and already OCR-ready: no render and no crop
normalization happen here at all, only a BGR conversion. Similarity dedup (Phase F,
`pipelines/_steps.py::elect_unique_segments`) then elects one representative
`Segment` per repeated shape and it's only those representatives this
module actually recognizes. The only remaining ambiguity Radon's
projection-profile skew estimate cannot resolve is a 0-vs-180-degree flip (a
baseline is a line, not an arrow) -- PaddleOCR's own angle classifier
(`use_angle_cls=True`) resolves it in one extra pass, then `text_recognizer`
runs once per batch. Radon's own residual skew angle (0.0 for an elected
representative's canonicalized copy) and the classifier's 0/180 correction
are combined into one final `direction` before the `Text` is returned --
see `recognize_segments`. (A representative's own canonicalizing rotation,
on top of this, is applied afterward by `pipelines/sub_pipelines/
ocr.py::restore_word_texts` when placing its `Text` back onto each real
word occurrence -- this module never sees that.)

`PaddleRecBackend` builds one `paddleocr.PaddleOCR` engine
(`config.OCR_VERSION` = PP-OCRv4, `config.OCR_LANG`), cached at class scope
by `(ocr_version, lang)` so a spawn pool started next finds the weights on
disk. This is the same API surface `archive/`'s `raster_parser` OCR uses,
so the `legacy` benchmark variant needs no compatibility shim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from rastervec.config import OCR_BATCH_SIZE, OCR_LANG, OCR_VERSION
from rastervec.helpers.geometry import compute_origin, transform_direction, union_bbox
from rastervec.logging_setup import get_logger
from rastervec.models import Segment, Text
from rastervec.renderer.stages import render_ocr_results  # noqa: F401 -- re-exported for callers

_LOG = get_logger("ocr.backend")


@dataclass
class OcrBox:
    """One recognised crop: text, confidence, and the classifier's own
    0/180 flip decision (degrees, 0 or 180)."""

    text: str
    confidence: float
    flip_deg: int = 0


class PaddleRecBackend:
    """paddleocr 2.x recognition + angle classification. One engine per
    `(ocr_version, lang)`, cached at class scope -- every instance shares
    it."""

    _ENGINE_CACHE: dict[tuple[str, str], object] = {}

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    @classmethod
    def warmup(cls, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        """Force the (model-downloading on first ever call) engine build
        now, in the calling process. Safe to call repeatedly."""
        cls(ocr_version, lang)._engine()

    def _engine(self):
        if self.key not in PaddleRecBackend._ENGINE_CACHE:
            # torch must load before paddle on Windows: paddleocr 2.x pulls
            # paddle (and, via albumentations, torch) at import, and a
            # paddle-first process then fails torch's DLL load (shm.dll,
            # WinError 127 -- clashing OpenMP runtimes). FAST already relies on
            # this order; make the OCR path self-sufficient too.
            import torch  # noqa: F401
            from paddleocr import PaddleOCR

            ocr_version, lang = self.key
            PaddleRecBackend._ENGINE_CACHE[self.key] = PaddleOCR(
                ocr_version=ocr_version,
                lang=lang,
                use_angle_cls=True,  # PaddleOCR's own classifier resolves the 0/180 flip
                show_log=False,
                rec_batch_num=OCR_BATCH_SIZE,
                cls_batch_num=OCR_BATCH_SIZE,
            )
        return PaddleRecBackend._ENGINE_CACHE[self.key]

    def recognize_crops(self, crops: list[np.ndarray]) -> list[OcrBox]:
        """One `OcrBox` per crop, in input order: classify each crop's
        orientation (0/180), rotate the ones flagged 180, then recognise
        the whole (now-upright) batch in a single `text_recognizer` pass --
        one recognition call total, not the old upright-and-flipped double
        call. Each crop is used as handed over, only converted to BGR: it
        arrives already deskewed and already white-padded from
        `OCR/radon.py`, and `text_recognizer` does its own resize."""
        if not crops:
            return []
        bgr = [_normalize_bgr(c) for c in crops]

        engine = self._engine()
        _, cls_results, _ = engine.text_classifier(bgr)
        flips = [180 if str(label) == "180" else 0 for label, _score in cls_results]
        upright = [np.rot90(img, 2) if flip else img for img, flip in zip(bgr, flips)]

        rec = engine.text_recognizer(upright)
        rows = rec[0] if isinstance(rec, tuple) else rec
        out: list[OcrBox] = []
        for row, flip in zip(rows, flips):
            text = str(row[0] or "").strip()
            score = float(row[1] or 0.0)
            out.append(OcrBox(text=text, confidence=score if text else 0.0, flip_deg=flip))
        return out


class PaddleDetectBackend:
    """Test-branch backend (see `pipelines/_steps.py::detect_text_paddle`):
    PaddleOCR's own text-DETECTION model (`engine.text_detector`, PP-OCR's
    DB detector) run directly on a whole-page render, replacing Radon
    segmentation + similarity dedup + recognition entirely. No text is ever
    recognized on this path -- callers get page-space bboxes only.

    Cached separately from `PaddleRecBackend` (its own `_ENGINE_CACHE`)
    because it's built with a much larger `det_limit_side_len`: a whole-page
    render at `config.FAST_PAGE_RENDER_DPI` is easily 1500-5000px on a side,
    and DB's 960px default would downsize away most small text before
    detection even runs."""

    _ENGINE_CACHE: dict[tuple[str, str], object] = {}
    _DET_LIMIT_SIDE_LEN = 4000

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    def _engine(self):
        if self.key not in PaddleDetectBackend._ENGINE_CACHE:
            # torch must load before paddle on Windows -- see PaddleRecBackend._engine.
            import torch  # noqa: F401
            from paddleocr import PaddleOCR

            ocr_version, lang = self.key
            PaddleDetectBackend._ENGINE_CACHE[self.key] = PaddleOCR(
                ocr_version=ocr_version,
                lang=lang,
                show_log=False,
                det_limit_side_len=self._DET_LIMIT_SIDE_LEN,
            )
        return PaddleDetectBackend._ENGINE_CACHE[self.key]

    def detect(self, image: np.ndarray) -> list[tuple[float, float, float, float]]:
        """One axis-aligned pixel-space bbox per detected text region, no
        confidence score (paddleocr 2.x's public detector API never returns
        the one `DBPostProcess` computes internally -- it's only used for
        its own `det_db_box_thresh` filtering)."""
        engine = self._engine()
        bgr = _normalize_bgr(image)
        dt_boxes, _elapse = engine.text_detector(bgr)
        return [
            (float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max()))
            for q in dt_boxes
        ]


def _normalize_bgr(crop: np.ndarray) -> np.ndarray:
    """A `Segment.image` -> a 3-channel BGR array (paddleocr 2.x's
    TextClassifier/TextRecognizer are cv2/BGR). Nothing else happens here:
    the crop already carries its white margin from `OCR/radon.py::pad_image`,
    and `text_recognizer` resizes to its own `rec_image_shape` internally,
    so there is no separate crop-normalization pass to run."""
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:  # grayscale -- gray RGB and gray BGR are identical
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    return np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])


def _recognize_crops_job(
    crops: list[np.ndarray],
    ocr_version: str = OCR_VERSION,
    lang: str = OCR_LANG,
) -> list[OcrBox]:
    """Top-level, picklable Pool-2 job: recognise `crops` with a
    `PaddleRecBackend` cached per Pool-2 worker process by `(ocr_version,
    lang)` (`PaddleRecBackend._ENGINE_CACHE` is keyed the same way for local
    calls). Fitz-free -- plain numpy crops in, dataclasses out."""
    return PaddleRecBackend(ocr_version, lang).recognize_crops(crops)


def recognize_segments(
    segments: list[Segment],
    *,
    batch_size: int = OCR_BATCH_SIZE,
    recognize_fn: "Callable[[list[np.ndarray]], list[OcrBox]] | None" = None,
) -> list[Text]:
    """Recognises every word-level `Segment` (each already carrying its own
    deskewed crop in `.image` -- no render happens here) in batches of
    `batch_size`, and returns one `Text` per input `Segment`, `source=
    "ocr"`, in the same coordinate frame `segment.vectors` already live in
    -- real page position for a plain word `Segment`, or an elected
    representative's own canonical frame (see `models/segment.py`'s
    docstring). `recognize_fn` defaults to a fresh
    `PaddleRecBackend().recognize_crops`; pass one (e.g. dispatching to a
    shared compute pool -- see `Reader/Parallel`) to replace the actual
    engine call without changing anything else here.

    `direction` combines that word's own Radon residual skew
    (`segment.angle` -- 0.0 for an elected representative's canonicalized
    copy) with the classifier's 0/180 flip correction. A representative's
    own canonicalizing rotation is layered on top of this by `pipelines/
    sub_pipelines/ocr.py::restore_word_texts` when placing it back onto
    each real word occurrence -- not here."""
    recognize_fn = recognize_fn if recognize_fn is not None else PaddleRecBackend().recognize_crops

    bboxes = [union_bbox([v.bbox for v in seg.vectors]) for seg in segments]

    texts: list[Text] = []
    for start in range(0, len(segments), max(1, batch_size)):
        batch = list(zip(segments[start:start + batch_size], bboxes[start:start + batch_size]))
        boxes = recognize_fn([seg.image for seg, _bbox in batch])
        for (seg, bbox), box in zip(batch, boxes):
            direction = transform_direction((1.0, 0.0), seg.angle + box.flip_deg)
            texts.append(Text(
                text=box.text,
                bbox=bbox,
                direction=direction,
                origin=compute_origin(bbox, direction),
                font="", font_size=0.0, color=None, flags=0,
                ascender=None, descender=None, wmode=0,
                block_no=0, line_no=0, word_no=0,
                page_index=0, seqno=0,
                confidence=box.confidence, source="ocr",
                orientation_source="ocr",
            ))
    return texts
