"""Step 4 (engine) -- the PaddleOCR wrapper word-group OCR runs on.

Copied from `ocr/paddle_ocr.py`.
"""

from __future__ import annotations

from typing import Any, Iterator, List

import numpy as np

from raster_parser.commons.v1_commons.ocr import paddle_lines


class PaddleOcr:
    """Run PaddleOCR on in-memory image crops (CPU only for now)."""

    def __init__(
        self,
        use_angle_cls,
        *,
        lang: str = "en",
        drop_score: float = 0.0,
        use_gpu: bool = False,
    ) -> None:
        from paddleocr import PaddleOCR

        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self._ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=self.use_gpu,
            show_log=False,
            # Detection downscales any crop whose longer side exceeds this (px); word-group
            # crops are far smaller, so they are detected at full resolution.
            det_limit_side_len=4096,
            det_limit_type="max",
            drop_score=drop_score,
        )

    @staticmethod
    def _extent(box: Any) -> tuple[float, float, float, float] | None:
        if box is None:
            return None
        arr = np.asarray(box, dtype=np.float64)
        if arr.size == 0:
            return None
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] > 0:
            x0, x1 = float(arr[:, 0].min()), float(arr[:, 0].max())
            y0, y1 = float(arr[:, 1].min()), float(arr[:, 1].max())
            if x1 > x0 and y1 > y0:
                return x0, y0, x1, y1
        return None

    def _lines(
        self, results: Any
    ) -> Iterator[tuple[tuple[float, float, float, float], str, float]]:
        """PaddleOCR 2.x: ``[[[box, (text, conf)], ...]]`` or ``[None]``."""
        for box, text, conf in paddle_lines(results):
            text = (text or "").strip()
            if not text:
                continue
            ext = self._extent(box)
            if ext is not None:
                yield ext, text, float(conf)

    def ocr_image(
        self,
        image: np.ndarray,
        *,
        min_confidence: float = 0.5,
        cls: bool | None = None,
    ) -> List[tuple[tuple[float, float, float, float], str, float]]:
        """OCR one RGB image and return [(box_xyxy, text, conf), ...]."""
        if cls is None:
            cls = self.use_angle_cls
        raw = self._ocr.ocr(image, cls=cls)
        out: List[tuple[tuple[float, float, float, float], str, float]] = []
        for box, text, conf in self._lines(raw):
            if conf >= min_confidence:
                out.append((box, text, conf))
        return out
