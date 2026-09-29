from __future__ import annotations

import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.Junction import text_ocr
from rastervec.P2_Raster_To_Vec.Junction.paddle_engine import OcrBox
from rastervec.P2_Raster_To_Vec.Junction.text_ocr import TileBox, merge_cross_tile_boxes, tile_grid


@pytest.mark.parametrize("h,w", [(500, 700), (960, 960), (2000, 3000), (961, 2500)])
def test_tile_grid_covers_image_with_overlap(h, w):
    tiles = tile_grid(h, w, tile=960, overlap_frac=0.2)
    covered = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in tiles:
        assert x1 - x0 <= 960 and y1 - y0 <= 960
        covered[y0:y1, x0:x1] = True
    assert covered.all()
    xs = sorted({t[0] for t in tiles})
    for a, b in zip(xs, xs[1:]):
        assert a + 960 - b >= 192  # neighbouring tiles overlap by >= 20 %


def test_split_box_across_tiles_is_merged_to_extremes():
    merged = merge_cross_tile_boxes([
        TileBox((900, 100, 960, 120), 0),
        TileBox((770, 102, 1010, 121), 1),
    ])
    assert merged == [(770, 100, 1010, 121)]


def test_same_tile_neighbours_are_not_merged():
    merged = merge_cross_tile_boxes([
        TileBox((0, 0, 50, 10), 0),
        TileBox((49, 0, 90, 10), 0),
    ])
    assert len(merged) == 2


def test_merge_is_transitive_and_dedupes():
    merged = merge_cross_tile_boxes([
        TileBox((0, 0, 50, 10), 0),
        TileBox((40, 0, 100, 10), 1),
        TileBox((95, 0, 150, 10), 2),
        TileBox((1, 1, 49, 9), 3),        # same word, seen again in a 4th tile
        TileBox((500, 500, 520, 510), 0),
    ])
    assert sorted(merged) == [(0, 0, 150, 10), (500, 500, 520, 510)]


def _ink_bbox_detector(images):
    """Fake PaddleOCR detect: one quad around each image's dark pixels."""
    out = []
    for im in images:
        ys, xs = np.nonzero(np.asarray(im).min(axis=2) < 128)
        if xs.size == 0:
            out.append([])
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        out.append([np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)])
    return out


def test_run_ocr_end_to_end_with_fakes():
    bgr = np.full((1200, 1500, 3), 255, dtype=np.uint8)
    bgr[500:520, 700:900] = 0  # one "text line" straddling the x=768 tile seam
    seen = []

    def recognize(crops):
        seen.extend(crops)
        return [OcrBox("TXT", 0.9) for _ in crops]

    result = text_ocr.run_ocr(
        bgr, px_per_pt=2.0, bg_bgr=(255, 255, 255),
        detect_many=_ink_bbox_detector, recognize=recognize, recognize_raw=recognize,
    )
    assert len(result.merged) == 1
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.text == "TXT" and hit.retry_count == 0
    qx0, qy0 = hit.quad.min(axis=0)
    qx1, qy1 = hit.quad.max(axis=0)
    assert (qx0, qy0, qx1, qy1) == pytest.approx((700, 500, 900, 520), abs=1.0)
    # pad 1 = 15 pt at 2 px/pt = 30 px around the merged box
    assert hit.padded_box == pytest.approx((670, 470, 930, 550), abs=1.0)
    # pad 2 border: the recognized crop is larger than the quad itself
    assert seen[0].shape[0] > 20 and seen[0].shape[1] > 200


def test_blank_recognition_retries_then_records_rotation():
    bgr = np.full((300, 300, 3), 255, dtype=np.uint8)
    bgr[100:115, 50:200] = 0
    calls = {"raw": 0}

    def recognize(crops):
        return [OcrBox("", 0.0) for _ in crops]

    def recognize_raw(crops):
        calls["raw"] += 1
        return [OcrBox("OK" if calls["raw"] == 2 else "", 0.8) for _ in crops]

    result = text_ocr.run_ocr(
        bgr, px_per_pt=1.0, bg_bgr=(255, 255, 255),
        detect_many=_ink_bbox_detector, recognize=recognize, recognize_raw=recognize_raw,
    )
    hit = result.hits[0]
    assert hit.text == "OK"
    assert hit.retry_count == 2
    assert hit.angle_deg == pytest.approx(0.0)  # 180 deg retry, normalized mod 180
