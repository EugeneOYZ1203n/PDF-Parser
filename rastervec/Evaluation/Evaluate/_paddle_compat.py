"""PaddleOCR 2.x -> 3.x compatibility shim for the archive `legacy`
benchmark variant.

`archive/raster_parser/` was written against PaddleOCR 2.x; this repo's
venv ships paddleocr 3.4.x (required by the current pipeline's PP-OCRv5).
The two disagree in three ways archive can't survive:

1. **Constructor kwargs.** Archive builds `PaddleOCR(use_angle_cls=...,
   use_gpu=..., show_log=..., det_limit_side_len=..., det_limit_type=...,
   drop_score=...)`. 3.x's `__init__` accepts none of these and raises
   `ValueError: Unknown argument: <name>` from its `**kwargs` passthrough.
2. **`.ocr()` result shape.** 2.x returned
   `[[ [box_4pts, (text, conf)], ... ]]`; 3.x `.predict()` returns
   per-page dict-likes carrying `rec_texts` / `rec_scores` / `rec_polys`.
3. **`.ocr(img, cls=/det=/rec=)` + `ocr.text_recognizer`.** Gone in 3.x.

`install()` swaps `paddleocr.PaddleOCR` for `_PaddleOCRv2Compat`, which
translates the constructor kwargs, wraps `.ocr()` to return the 2.x
nested shape, and exposes a `text_recognizer` callable backed by
`paddleocr.TextRecognition` (recognition-only, like
`OCR/Paddle_OCR/light_backend.py`). Nothing in `archive/` is modified --
only the symbol it imports is. `legacy_adapter._ensure_archive_importable`
calls `install()` once, so the shim is active only for a legacy run.
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

import numpy as np

from rastervec.config import OCR_VERSION, REC_BATCH_SIZE
from rastervec.logging_setup import get_logger

_LOG = get_logger("eval.paddle_compat")

# archive ctor kwargs -> 3.x ctor kwargs (None value = drop the kwarg)
_CTOR_KWARG_MAP: dict[str, str | None] = {
    "use_gpu": None,
    "show_log": None,
    "use_angle_cls": "use_textline_orientation",
    "drop_score": "text_rec_score_thresh",
    "det_limit_side_len": "text_det_limit_side_len",
    "det_limit_type": "text_det_limit_type",
    "det": None,
    "rec": None,
    "cls": None,
}
# archive .ocr()/.predict() call kwargs that 3.x rejects
_CALL_KWARGS_TO_DROP = ("cls", "det", "rec", "bin", "inv", "alpha_color")

_original: type | None = None


def _rec_field(result: object, key: str) -> Any:
    """Pull one field from a paddlex predictor result (dict-like or
    attribute-style) -- same tolerance as light_backend._rec_field."""
    try:
        return result[key]  # type: ignore[index]
    except (TypeError, KeyError, IndexError):
        return getattr(result, key, None)


def _reshape_page(page: object) -> list[list[Any]]:
    """One 3.x predict() page result -> 2.x
    `[[box_4pts, (text, conf)], ...]`."""
    texts = list(_rec_field(page, "rec_texts") or [])
    scores = list(_rec_field(page, "rec_scores") or [])
    polys = _rec_field(page, "rec_polys")
    if polys is None:
        polys = _rec_field(page, "dt_polys")
    polys = list(polys or [])

    lines: list[list[Any]] = []
    for i, text in enumerate(texts):
        conf = float(scores[i]) if i < len(scores) else 0.0
        if i < len(polys) and polys[i] is not None:
            box = [[float(x), float(y)] for x, y in np.asarray(polys[i]).reshape(-1, 2)]
        else:
            box = []
        lines.append([box, (text, conf)])
    return lines


class _TextRecognizerAdapter:
    """Callable matching archive's `ocr.text_recognizer(chunk)` contract:
    takes a list of image crops, returns `([(text, score), ...], None)`."""

    def __init__(self) -> None:
        from paddleocr import TextRecognition

        self._engine = TextRecognition(model_name=f"{OCR_VERSION}_mobile_rec")

    def __call__(self, crops: list[np.ndarray]) -> tuple[list[tuple[str, float]], None]:
        arrs = [np.asarray(c) for c in crops]
        results = self._engine.predict(arrs, batch_size=REC_BATCH_SIZE)
        out: list[tuple[str, float]] = []
        for r in results:
            text = str(_rec_field(r, "rec_text") or "")
            score = float(_rec_field(r, "rec_score") or 0.0)
            out.append((text, score))
        return out, None


class _PaddleOCRv2Compat:
    """Drop-in for `paddleocr.PaddleOCR` that speaks archive's 2.x API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        assert _original is not None
        translated: dict[str, Any] = {
            "ocr_version": OCR_VERSION,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": bool(
                kwargs.pop("use_angle_cls", kwargs.pop("use_textline_orientation", False))
            ),
        }
        for key, value in kwargs.items():
            if key in _CTOR_KWARG_MAP:
                new_key = _CTOR_KWARG_MAP[key]
                if new_key is not None:
                    translated.setdefault(new_key, value)
            else:
                translated.setdefault(key, value)
        self._engine = _original(**translated)

        self._text_recognizer: _TextRecognizerAdapter | None
        try:
            self._text_recognizer = _TextRecognizerAdapter()
        except Exception as exc:  # noqa: BLE001 -- optional fast path
            _LOG.warning("TextRecognition adapter unavailable (%s)", exc)
            self._text_recognizer = None

    # archive checks `getattr(ocr, "text_recognizer", None)`
    @property
    def text_recognizer(self):  # noqa: ANN201
        if self._text_recognizer is None:
            raise AttributeError("text_recognizer")
        return self._text_recognizer

    def ocr(self, img: Any, **kwargs: Any) -> list[list[list[Any]]]:
        for key in _CALL_KWARGS_TO_DROP:
            kwargs.pop(key, None)
        try:
            pages = self._engine.predict(img, **kwargs)
        except TypeError:
            pages = self._engine.predict(img)
        if not pages:
            return [[]]
        return [_reshape_page(p) for p in pages]

    # a few archive call sites use .predict directly on the raw engine name
    def predict(self, img: Any, **kwargs: Any):  # noqa: ANN201
        return self._engine.predict(img, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


def install() -> None:
    """Swap `paddleocr.PaddleOCR` for the compat shim (idempotent)."""
    global _original
    import paddleocr

    if _original is not None:
        return
    _original = paddleocr.PaddleOCR
    paddleocr.PaddleOCR = _PaddleOCRv2Compat  # type: ignore[assignment,misc]
    _LOG.info("PaddleOCR 2.x->3.x compat shim installed for the legacy variant")


def uninstall() -> None:
    """Restore the real `paddleocr.PaddleOCR` (for tests)."""
    global _original
    if _original is None:
        return
    import paddleocr

    paddleocr.PaddleOCR = _original  # type: ignore[assignment,misc]
    _original = None
