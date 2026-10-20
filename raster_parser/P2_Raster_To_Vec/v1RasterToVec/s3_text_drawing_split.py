"""Step 3 -- OCR-based text/drawing separation for raster (AEC) drawings.

PaddleOCR only *locates* text; it never decides pixels. The text layer is every
non-white pixel inside a detection box (gray < ink_thresh), including thin
strokes Otsu would drop and lines that cross the box. The drawing layer keeps
the full line mask (from Otsu+morphology/Hough) plus all ink outside those boxes.

Copied from `parsing/text_drawing_parser.py`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from raster_parser.commons.progress import bar

# Per-process OCR for ProcessPoolExecutor workers (spawn).
_DET_WORKER_OCR = None


def _det_worker_init(lang: str, min_conf: float) -> None:
    """Limit BLAS threads and load one PaddleOCR per process (CPU)."""
    global _DET_WORKER_OCR
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    _DET_WORKER_OCR = make_ocr(lang, min_conf, use_gpu=False)


def _det_tile_job(job: dict) -> list:
    """Process-pool tile detection job: load spilled crop → boxes in page coords."""
    crop = cv2.imread(job["crop_path"])
    if crop is None:
        return []
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    raw = _DET_WORKER_OCR.ocr(np.ascontiguousarray(crop), cls=True)
    return _boxes_from_result(raw, job["min_conf"], job["ox"], job["oy"])


def paddle_ocr_call(ocr, img, **kwargs):
    """Call ``ocr.ocr`` (main-process path; process workers call the engine directly)."""
    return ocr.ocr(img, **kwargs)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold a grayscale image to a binary mask where ink = 255, background = 0."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw


def line_mask(bw: np.ndarray, line_min: int, grow: int, threshold: int, max_gap: int) -> np.ndarray:
    """Long straight runs (any angle) -> drawing."""
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_min))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel),
    )
    segs = cv2.HoughLinesP(bw, 1, np.pi / 180, threshold, minLineLength=line_min, maxLineGap=max_gap)
    if segs is not None:
        # OpenCV may return (N, 1, 4) or (N, 4) depending on build/version.
        for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
            cv2.line(lines, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=grow)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (grow, grow)))
    return cv2.bitwise_and(lines, bw)

def remove_lines_for_ocr(rgb: np.ndarray, lines: np.ndarray, mode: str) -> np.ndarray:
    """Return a DETECTION-ONLY copy of `rgb` with the line mask cleaned off, so PaddleOCR's
    detector sees text uncrossed by walls/pipes/leaders and boxes more of it.

    mode:
      none    -> unchanged (return rgb as-is)
      inpaint -> reconstruct glyph pixels across the line via cv2.inpaint

    This output is used ONLY to find boxes; erasing/outputs still use the original arrays,
    so no real ink is lost by cleaning here.
    """
    if mode == "none":
        return rgb
    if mode == "inpaint":
        # dilate the mask a touch so the inpainter has clean borders to sample from
        m = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        return cv2.inpaint(rgb, m, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    raise ValueError(f"unknown ocr-line-removal mode: {mode}")


def _boxes_from_result(raw, min_conf, ox, oy):
    """Pull axis-aligned boxes out of a PaddleOCR result, offset into page coords."""
    out = []
    for page in raw or []:
        for line in page or []:
            if not line or len(line) < 2:
                continue
            box, rec = line[0], line[1]
            if not isinstance(rec, (list, tuple)) or len(rec) < 2 or float(rec[1]) < min_conf:
                continue
            arr = np.asarray(box, dtype=np.float64)
            out.append((int(arr[:, 0].min()) + ox, int(arr[:, 1].min()) + oy,
                        int(arr[:, 0].max()) + ox, int(arr[:, 1].max()) + oy))
    return out


def merge_boxes(boxes, gap=0):
    """Union boxes that overlap (optionally within `gap` px). Stitches the duplicate
    detections that tiling produces and labels split across a tile seam."""
    boxes = [list(b) for b in boxes]
    out = []
    while boxes:
        a = boxes.pop()
        merged = True
        while merged:
            merged, rest = False, []
            for b in boxes:
                if not (a[2] + gap < b[0] or b[2] + gap < a[0]
                        or a[3] + gap < b[1] or b[3] + gap < a[1]):
                    a = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    merged = True
                else:
                    rest.append(b)
            boxes = rest
        out.append(tuple(a))
    return out


def _iou(a, b) -> float:
    """Intersection-over-union of two (x0, y0, x1, y1) boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes, iou_thresh: float = 0.6):
    """Drop near-duplicate boxes by IoU without growing survivors (unlike ``merge_boxes``).

    Keeps larger boxes first, suppressing any later box whose IoU with a kept box exceeds
    ``iou_thresh``. Nested word/line boxes have low IoU (small vs large), so they survive;
    only genuine duplicates of roughly the same extent (e.g. an upright + rotated detection,
    or a refine sub-box that lands on a neighbour) are removed.
    """
    ordered = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: list = []
    for b in ordered:
        if all(_iou(b, k) <= iou_thresh for k in kept):
            kept.append(tuple(b))
    return kept


def filter_boxes_by_min_area(boxes, min_area: int):
    """Drop Paddle det boxes whose pixel area (w*h) is strictly below ``min_area``.

    ``min_area <= 0`` disables the filter (returns boxes unchanged).
    """
    if min_area <= 0:
        return [tuple(b) for b in boxes]
    return [
        tuple(b) for b in boxes
        if (b[2] - b[0]) * (b[3] - b[1]) >= min_area
    ]


def make_ocr(lang: str, min_conf: float, *, use_gpu: bool = True):
    """Build a single PaddleOCR instance to reuse across passes/tiles.

    Process-pool workers should pass ``use_gpu=False`` to avoid multi-process GPU fights.
    """
    # OneDNN multi-thread inference has been observed to SIGSEGV; keep it off for safety.
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, use_gpu=use_gpu,
                     det_db_box_thresh=0.6, det_limit_side_len=9000,
                     det_db_unclip_ratio=1.0, det_db_thresh=0.3,
                     det_limit_type="max", drop_score=min_conf)


def ocr_boxes(rgb: np.ndarray, min_conf: float, lang: str, tile: int, overlap: int,
              ocr=None, *, workers: int = 1, scratch_dir: str | Path | None = None):
    """Text boxes [(x0,y0,x1,y1), ...] via PaddleOCR.

    PaddleOCR downscales any side > det_limit_side_len before detection, so on a big
    sheet small/rotated labels are lost. Running overlapping full-res tiles keeps the
    detector at native scale; `overlap` must exceed the largest label so none is split.
    Pass a prebuilt `ocr` to avoid reloading the model.

    ``workers > 1`` runs tiles via ``ProcessPoolExecutor`` (spawn): spill each tile PNG,
    one PaddleOCR per process (CPU). Serial path (``workers == 1``) uses ``ocr`` on this
    process.
    """
    if ocr is None:
        ocr = make_ocr(lang, min_conf)
    H, W = rgb.shape[:2]
    if tile <= 0 or (W <= tile and H <= tile):
        return merge_boxes(
            _boxes_from_result(paddle_ocr_call(ocr, np.ascontiguousarray(rgb), cls=True),
                               min_conf, 0, 0)
        )

    step = max(1, tile - overlap)
    xs = list(range(0, max(1, W - overlap), step))
    ys = list(range(0, max(1, H - overlap), step))
    coords = [(oy, ox) for oy in ys for ox in xs]
    n_workers = max(1, int(workers))

    if n_workers == 1 or len(coords) <= 1:
        boxes: list = []
        for oy, ox in bar(coords, desc="det tiles", unit="tile"):
            crop = np.ascontiguousarray(rgb[oy:min(oy + tile, H), ox:min(ox + tile, W)])
            if crop.size == 0:
                continue
            boxes += _boxes_from_result(
                paddle_ocr_call(ocr, crop, cls=True), min_conf, ox, oy
            )
        return merge_boxes(boxes)

    own_scratch = scratch_dir is None
    root = Path(scratch_dir) if scratch_dir is not None else Path(
        tempfile.mkdtemp(prefix="det_tiles_")
    )
    root.mkdir(parents=True, exist_ok=True)
    tile_dir = root / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    try:
        for i, (oy, ox) in enumerate(coords):
            crop = rgb[oy:min(oy + tile, H), ox:min(ox + tile, W)]
            if crop.size == 0:
                continue
            path = tile_dir / f"t{i:05d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            jobs.append({
                "crop_path": str(path),
                "ox": ox,
                "oy": oy,
                "min_conf": float(min_conf),
            })

        boxes = []
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_det_worker_init,
            initargs=(lang, float(min_conf)),
        ) as pool:
            futures = [pool.submit(_det_tile_job, job) for job in jobs]
            for fut in bar(as_completed(futures), desc="det tiles", total=len(futures), unit="tile"):
                boxes.extend(fut.result())
        return merge_boxes(boxes)
    finally:
        if own_scratch:
            for p in tile_dir.glob("*.png"):
                p.unlink(missing_ok=True)
            try:
                tile_dir.rmdir()
                root.rmdir()
            except OSError:
                pass


def _rotate_image(rgb: np.ndarray, angle: float):
    """Rotate `rgb` CCW by `angle` deg about its center, expanding the canvas so nothing
    is clipped. Returns (rotated_image, M) where M maps original -> rotated coords."""
    h, w = rgb.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    rotated = cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_LINEAR,
                             borderValue=(255, 255, 255))
    return rotated, M


def _map_box_back(box, Minv) -> tuple:
    """Map an axis-aligned box from the rotated frame back to the original frame via the
    inverse affine, returning the axis-aligned bbox of the four mapped corners."""
    x0, y0, x1, y1 = box
    corners = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], dtype=np.float64)
    pts = corners @ Minv.T                       # 4x2 in original coords
    xs, ys = pts[:, 0], pts[:, 1]
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def ocr_boxes_rotated(rgb: np.ndarray, min_conf: float, lang: str, tile: int, overlap: int,
                      ocr=None, *, workers: int = 1, scratch_dir: str | Path | None = None):
    """Upright OCR plus a fixed 90° CCW pass for vertical labels.

    Rotates the page 90° CCW, runs detection, maps boxes back to the original frame,
    and merges with the upright finds. ``workers`` / ``scratch_dir`` are forwarded to
    each tiled pass (ProcessPool when workers > 1).
    """
    if ocr is None:
        ocr = make_ocr(lang, min_conf)
    boxes = list(ocr_boxes(rgb, min_conf, lang, tile, overlap, ocr=ocr,
                           workers=workers, scratch_dir=scratch_dir))
    rot, M = _rotate_image(rgb, 90.0)
    Minv = cv2.invertAffineTransform(M)
    # Separate scratch subtree so upright/rotated spills do not collide.
    rot_scratch = None
    if scratch_dir is not None:
        rot_scratch = Path(scratch_dir) / "rot90"
        rot_scratch.mkdir(parents=True, exist_ok=True)
    for b in ocr_boxes(rot, min_conf, lang, tile, overlap, ocr=ocr,
                       workers=workers, scratch_dir=rot_scratch):
        boxes.append(_map_box_back(b, Minv))
    return merge_boxes(boxes)


def ink_in_boxes(bw, boxes, *, margin: int = 0) -> np.ndarray:
    """All ink=255 mask pixels that fall inside any OCR det box (union of rectangles)."""
    H, W = bw.shape
    out = np.zeros_like(bw)
    for x0, y0, x1, y1 in boxes:
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(W, x1 + margin)
        y1 = min(H, y1 + margin)
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0:y1, x0:x1] = bw[y0:y1, x0:x1]
    return out


def nonwhite_ink(gray: np.ndarray, thresh: int = 250) -> np.ndarray:
    """Soft ink mask: every pixel darker than ``thresh`` -> 255 (keeps thin/faint strokes)."""
    return np.where(gray < thresh, np.uint8(255), np.uint8(0))


def nonwhite_ink_in_boxes(gray, boxes, *, margin: int = 0, thresh: int = 250) -> np.ndarray:
    """Non-white ink inside OCR det boxes (soft threshold, not Otsu)."""
    return ink_in_boxes(nonwhite_ink(gray, thresh), boxes, margin=margin)


@dataclass
class OcrParseResult:
    drawing: np.ndarray   # uint8 grayscale: text erased -> white, lines/drawing kept
    text: np.ndarray      # uint8 mask, text ink = 255 on 0 background
    text_gray: np.ndarray # uint8 grayscale: original ink where text, white elsewhere
    lines: np.ndarray     # uint8 mask, line ink = 255 on 0 background
    binary: np.ndarray    # uint8 mask, all ink = 255 on 0 background
    boxes: list           # OCR boxes [(x0, y0, x1, y1), ...]

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)

    @property
    def text_ink_px(self) -> int:
        return int((self.text == 255).sum())


class OcrTextDrawingParser:
    """Separate rasterized text from drawings using PaddleOCR boxes + within-box ink split.

    Build once (optionally with a shared ``ocr`` instance) and call :meth:`parse` per page.
    """

    def __init__(
        self,
        *,
        min_conf: float = 0.0,
        lang: str = "en",
        tile: int = 512,
        tile_overlap: int = 64,
        ocr_line_removal: str = "none",
        margin: int = 1,
        line_min: int = 100,
        line_grow: int = 3,
        hough_threshold: int = 50,
        hough_max_gap: int = 3,
        ink_thresh: int = 250,
        min_box_area: int = 0,
        workers: int = 1,
        ocr=None,
    ):
        self.min_conf = min_conf
        self.lang = lang
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.ocr_line_removal = ocr_line_removal
        self.margin = margin
        self.line_min = line_min
        self.line_grow = line_grow
        self.hough_threshold = hough_threshold
        self.hough_max_gap = hough_max_gap
        self.ink_thresh = ink_thresh
        self.min_box_area = max(0, int(min_box_area))
        self.workers = max(1, int(workers))
        self.ocr = ocr

    def _get_ocr(self):
        if self.ocr is None:
            self.ocr = make_ocr(self.lang, self.min_conf)
        return self.ocr

    def parse(self, image: np.ndarray, *, scratch_dir: str | Path | None = None) -> OcrParseResult:
        """Separate text from linework on one rendered page.

        ``image`` is a BGR raster (e.g. a rendered PDF page). Returns the drawing
        (text erased), text, and line masks plus the OCR boxes.
        ``scratch_dir``: spill dir for process-pool tile crops when ``workers > 1``.
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if image.ndim == 2 \
            else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bw = binarize(gray)
        lines = line_mask(bw, self.line_min, self.line_grow, self.hough_threshold, self.hough_max_gap)
        # Detection-only: strip the line mask so lines crossing labels don't break OCR detection.
        # Boxes come back in the SAME coordinates; erasing/outputs still use the original arrays.
        ocr_rgb = remove_lines_for_ocr(rgb, lines, self.ocr_line_removal)
        ocr = self._get_ocr()
        boxes = ocr_boxes_rotated(
            ocr_rgb, self.min_conf, self.lang, self.tile, self.tile_overlap,
            ocr=ocr, workers=self.workers, scratch_dir=scratch_dir,
        )
        boxes = filter_boxes_by_min_area(boxes, self.min_box_area)

        text = nonwhite_ink_in_boxes(gray, boxes, margin=self.margin, thresh=self.ink_thresh)
        # Text = non-white ink inside det boxes. Drawing keeps every line pixel (even those
        # that also sit inside a det box) plus ink outside boxes. Line crossings therefore
        # appear in BOTH layers; only non-line text ink is erased from the drawing.
        drawing = gray.copy()
        erase = (text == 255) & (lines != 255)
        drawing[erase] = 255
        text_gray = np.full_like(gray, 255)
        text_gray[text == 255] = gray[text == 255]
        return OcrParseResult(drawing=drawing, text=text, text_gray=text_gray,
                              lines=lines, binary=bw, boxes=boxes)
