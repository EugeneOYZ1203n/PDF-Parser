from __future__ import annotations

import fitz
import numpy as np
import pytest

from rastervec.Evaluation.Evaluate.vector_metrics import image_similarity
from rastervec.notebooks import _gt_fidelity_helpers as h


def _pdf(tmp_path, ops: bytes, *, text: str | None = None, rotation: int = 0, name="p.pdf"):
    doc = fitz.open()
    page = doc.new_page(width=200, height=150)
    if text:
        page.insert_text((20, 130), text, fontsize=12)
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, ops)
    doc.xref_set_key(page.xref, "Contents", "[" + " ".join(f"{x} 0 R" for x in page.get_contents() + [xref]) + "]")
    if rotation:
        page.set_rotation(rotation)
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return str(path)


def test_identical_images_score_perfectly():
    img = np.full((40, 60), 255, np.uint8)
    img[10:12, 5:50] = 0
    stats = image_similarity(img, img.copy())
    assert stats.ssim == pytest.approx(1.0) and stats.psnr == pytest.approx(stats.psnr_cap_db)


def test_paint_white_whitens_the_box():
    img = np.zeros((150, 200), np.uint8)
    out = h.paint_white(img, [(10, 10, 20, 20)], dpi=72, pad_px=0)
    assert (out[10:20, 10:20] == 255).all() and out[30, 30] == 0 and img[15, 15] == 0


def test_visible_gt_beats_paint_blind_gt_on_white_fills(tmp_path):
    # a white wipeout over part of a line: the paint-blind GT keeps the hidden part + the wipeout edges
    path = _pdf(tmp_path, b"0 0 0 RG 1.5 w 20 75 m 180 75 l S 1 1 1 rg 0 0 0 RG 0.5 w 30 30 60 60 re B "
                          b"1 1 1 rg 100 60 50 30 re f")
    r = h.evaluate_page(path, 0, dpi=150, cache_dir=tmp_path / "cache")
    assert r.original.shape == r.gt_visible.shape == r.gt_all.shape
    assert r.ssim_visible > r.ssim_all and r.psnr_visible > r.psnr_all
    assert r.stats["cut_pieces"] >= 1 and r.n_all > r.stats["annotations"] - 1
    # cached: a second call doesn't rebuild and gives the same numbers
    assert list((tmp_path / "cache").glob("*.npz"))
    again = h.evaluate_page(path, 0, dpi=150, cache_dir=tmp_path / "cache")
    assert again.ssim_visible == pytest.approx(r.ssim_visible) and again.stats == r.stats


def test_text_mask_and_rotated_page_frame(tmp_path):
    path = _pdf(tmp_path, b"0 0 0 RG 1 w 20 100 m 180 100 l S", text="HELLO WORLD", rotation=90)
    masked = h.evaluate_page(path, 0, dpi=100)
    plain = h.evaluate_page(path, 0, dpi=100, mask_text=False)
    hh, ww = masked.original.shape  # unrotated frame (portrait page stays landscape)
    assert abs(hh - 150 * 100 / 72) <= 1 and abs(ww - 200 * 100 / 72) <= 1
    assert masked.masked_boxes == 2 and plain.masked_boxes == 0
    assert masked.ssim_visible > plain.ssim_visible       # text ink has no GT


def test_summary_rows_figure_and_outputs(tmp_path):
    path = _pdf(tmp_path, b"0 0 0 RG 1 w 20 100 m 180 100 l S")
    results = [h.evaluate_page(path, 0, dpi=72)]
    rows = h.summary_rows(results)
    assert [r["page"] for r in rows] == [0, "mean"]
    assert "ssim_visible" in h.markdown_table(rows)
    run = h.new_run_dir(tmp_path / "run")
    assert h.save_summary(rows, run).read_text().startswith("pdf,page")
    assert h.save_page(results[0], run).is_file()
    assert h.diff_map(results[0].original, results[0].gt_visible).shape[-1] == 3
    assert h.page_indices(path, None, None) == [0] and h.page_indices(path, [0, 5], 1) == [0]
