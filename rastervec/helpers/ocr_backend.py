"""Pluggable OCR backends -- the strategy-pattern split behind RenderOCR
(helpers/render_ocr.py). An OcrBackend takes one rendered PIL image and
returns every detected text box (PaddleOcrBackend: line/region-level;
TesseractOcrBackend: word-level) plus the page's own detected orientation,
all already in the *caller's* original image pixel space -- callers never
need to know about a given engine's own internal preprocessing/rotation
quirks. RenderOCR stays the single public orchestration class (render once,
detect once, build a TextVectorResult); only the raw-detection step is
swappable here.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from PIL import Image

# Must be set before paddlex reads its flags on import (which happens the
# first time PaddleOcrBackend._engine() lazily imports paddleocr below) --
# on this project's dev environment, the default (mkldnn-accelerated) CPU
# inference path hits an unimplemented PIR attribute-conversion error in
# this paddlepaddle build; plain "paddle" run mode works fine and is plenty
# fast for small, pre-cropped cluster renders. Only set if the caller
# hasn't already configured this themselves.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

# Common install locations for the tesseract.exe/tesseract binary, tried
# (in order) when it isn't already on PATH -- covers both target
# environments (this project's Windows dev machine and a Linux/WSL box)
# without needing per-machine env-var setup. `apt install tesseract-ocr`
# on Linux/WSL already puts it on PATH, so the Linux entries here are just
# a safety net for a from-source/non-package install.
_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]


def _resolve_tesseract_cmd() -> str | None:
    """RASTERVEC_TESSERACT_CMD env var wins if set; otherwise PATH lookup
    (covers the common case on Linux/WSL, where `apt install
    tesseract-ocr` already puts it there); otherwise the first existing
    path from `_TESSERACT_CANDIDATES`. Returns None (pytesseract's own
    default lookup, which will raise TesseractNotFoundError when actually
    called) only if none of those find anything."""
    override = os.environ.get("RASTERVEC_TESSERACT_CMD")
    if override:
        return override
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _TESSERACT_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


_TESSERACT_CMD = _resolve_tesseract_cmd()


@dataclass
class OcrBox:
    """One detected text box, in the caller's original image pixel space.
    `is_word` is True for a single-word box (Tesseract), False for a
    line/region box (Paddle) -- callers that want word-level placement
    (Renderer.render_reconstructed_page) can rely on `is_word` boxes each
    covering exactly one word; line/region boxes may cover several."""

    text: str
    confidence: float
    corners: list[tuple[float, float]]
    is_word: bool


@dataclass
class OcrDetection:
    """One backend's full result for one rendered image."""

    boxes: list[OcrBox] = field(default_factory=list)
    rotation: int = 0  # 0/90/180/270, page-level orientation


class OcrBackend(Protocol):
    def detect(self, image: "Image.Image") -> OcrDetection: ...


def _undo_doc_rotation_point(
    x: float, y: float, angle: int, orig_w: float, orig_h: float,
) -> tuple[float, float]:
    """Inverts PaddleOCR's own internal doc-orientation rotation for one
    point -- see `_undo_doc_rotation`'s docstring for why this exists."""
    if angle == 90:
        return orig_w - y, x
    if angle == 180:
        return orig_w - x, orig_h - y
    if angle == 270:
        return y, orig_h - x
    return x, y


def _undo_doc_rotation(
    points: list[tuple[float, float]], angle: int, orig_size: tuple[float, float],
) -> list[tuple[float, float]]:
    """When `use_doc_orientation_classify=True`, PaddleOCR's doc-preprocessor
    sub-pipeline actually rotates the input image by `doc_preprocessor_res
    .angle` degrees *before* running text detection (confirmed by reading
    paddlex's own `doc_preprocessor/pipeline.py` -- `output_img =
    rotate_image(image_array, angle)` -- and `ocr/pipeline.py`, which runs
    `text_det_model` on that same rotated `doc_preprocessor_images`, not
    the original input). So every `rec_poly`/`dt_poly` PaddleOCR returns is
    in that ROTATED image's pixel space, not the space of the image we
    actually passed to `predict()` -- for a 90/270 angle the rotated
    image's width and height are even swapped. Left uncorrected, mapping
    those corners back into PDF page space (`Renderer.pixel_to_page_bbox`)
    produces a wrong-sized, wrong-position `ocr_bbox` whenever Paddle's doc
    classifier decides the upright cluster render itself needs further
    correction (i.e. whenever `angle != 0`) -- this is the actual bug
    behind observed bad OCR bboxes/angles, not a mistake in our own
    geometry math. `rotate_image`'s rotation is derived exactly (via
    `cv2.getRotationMatrix2D`) for each of the four possible angles
    (0/90/180/270, the only values PaddleOCR's doc-orientation classifier
    ever predicts) and inverted here, per-point, back into the original
    image's own `orig_size` (width, height) pixel space."""
    if angle == 0 or not points:
        return points
    orig_w, orig_h = orig_size
    return [_undo_doc_rotation_point(x, y, angle, orig_w, orig_h) for x, y in points]


class PaddleOcrBackend:
    """PaddleOCR (PP-OCRv6, text-detection + text-recognition, with doc/
    textline orientation classification enabled). Detected boxes are
    line/region-level, not word-level. Engines are expensive to construct
    (they load model weights), so one is built lazily per `lang` and
    cached at module scope -- every PaddleOcrBackend for the same lang
    shares it."""

    _ENGINE_CACHE: dict[str, "paddleocr.PaddleOCR"] = {}

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang

    def _engine(self) -> "paddleocr.PaddleOCR":
        if self.lang not in PaddleOcrBackend._ENGINE_CACHE:
            from paddleocr import PaddleOCR

            PaddleOcrBackend._ENGINE_CACHE[self.lang] = PaddleOCR(
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                lang=self.lang,
            )
        return PaddleOcrBackend._ENGINE_CACHE[self.lang]

    def _predict_page(self, image: "Image.Image") -> dict:
        """Runs PaddleOCR's `predict()` once on one rendered image and
        returns its single-page result dict (`{}` if PaddleOCR returned
        nothing for it) -- the raw dict carries `rec_texts`/`rec_scores`/
        `rec_polys` (detected text boxes), `doc_preprocessor_res.angle`
        (document-orientation classification) and
        `textline_orientation_angles` (per-line 0/180 flip correction)."""
        import numpy as np

        array = np.asarray(image.convert("RGB"))
        pages = self._engine().predict(array)
        return pages[0] if pages else {}

    def _page_rotation(self, page: dict) -> int:
        """Reads PaddleOCR's own orientation classifiers off one
        `_predict_page` result dict -- `doc_preprocessor_res.angle` (its
        0/90/180/270 document-orientation classification) combined with
        the majority vote of `textline_orientation_angles` (each entry 0
        or 1, meaning a 0/180 per-line flip correction) turned into 0 or
        180 -- added together mod 360."""
        doc_angle = int(page.get("doc_preprocessor_res", {}).get("angle", 0) or 0)
        line_angles = list(page.get("textline_orientation_angles") or [])
        flip = 0
        if line_angles:
            ones = sum(1 for a in line_angles if a == 1)
            flip = 180 if ones * 2 >= len(line_angles) else 0
        return (doc_angle + flip) % 360

    def detect(self, image: "Image.Image") -> OcrDetection:
        page = self._predict_page(image)
        angle = int(page.get("doc_preprocessor_res", {}).get("angle", 0) or 0)
        texts = list(page.get("rec_texts") or [])
        scores = list(page.get("rec_scores") or [])
        polys = list(page.get("rec_polys") or [])

        boxes: list[OcrBox] = []
        for i, text in enumerate(texts):
            score = float(scores[i]) if i < len(scores) else 0.0
            poly = polys[i] if i < len(polys) else None
            corners = [(float(x), float(y)) for x, y in poly] if poly is not None else []
            corners = _undo_doc_rotation(corners, angle, image.size)
            boxes.append(OcrBox(text=text, confidence=score, corners=corners, is_word=False))

        return OcrDetection(boxes=boxes, rotation=self._page_rotation(page))


class TesseractOcrBackend:
    """Tesseract (via pytesseract), word-level detection --
    `image_to_data` returns one row per detected word (plus aggregate
    line/block/paragraph rows, filtered out here since they carry
    `conf == -1` and no useful text of their own). Page-level orientation
    comes from `image_to_osd`, which needs enough text on the page to
    classify -- wrapped in try/except since it routinely fails on small,
    sparse cluster crops (the only input this ever actually sees), falling
    back to `rotation=0` rather than raising."""

    def __init__(self, lang: str = "eng") -> None:
        self.lang = lang
        if _TESSERACT_CMD:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

    def detect(self, image: "Image.Image") -> OcrDetection:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(image, lang=self.lang, output_type=Output.DICT)

        boxes: list[OcrBox] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            conf = float(data["conf"][i])
            if not text or conf < 0:
                continue
            x, y = float(data["left"][i]), float(data["top"][i])
            w, h = float(data["width"][i]), float(data["height"][i])
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            boxes.append(OcrBox(text=text, confidence=conf / 100.0, corners=corners, is_word=True))

        rotation = 0
        try:
            osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
            rotation = int(osd.get("rotate", 0) or 0) % 360
        except pytesseract.TesseractError:
            pass

        return OcrDetection(boxes=boxes, rotation=rotation)
