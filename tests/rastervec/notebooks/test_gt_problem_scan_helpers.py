from __future__ import annotations

import fitz
import pytest

from rastervec.notebooks import _gt_problem_scan_helpers as h


def _pdf(tmp_path, ops: bytes, *, text: str | None = None, name="p.pdf", size=(300, 300)):
    doc = fitz.open()
    page = doc.new_page(width=size[0], height=size[1])
    if text:
        page.insert_text((20, 30), text, fontsize=14)
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, ops)
    doc.xref_set_key(page.xref, "Contents", "[" + " ".join(f"{x} 0 R" for x in page.get_contents() + [xref]) + "]")
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return str(path)


def _cats(tmp_path, ops, **kw):
    return {k: v for k, v in h.scan_page(_pdf(tmp_path, ops, **kw), 0, dpi=150).counts().items() if v}


def test_plain_drawing_has_no_findings(tmp_path):
    assert _cats(tmp_path, b"0 0 0 RG 1 w 50 50 100 80 re S 20 200 m 280 200 l S") == {}


@pytest.mark.parametrize("ops, expected", [
    (b"1 1 1 rg 100.2 100.2 59.6 49.6 re f 0 0 0 RG 0.5 w 100 100 60 50 re S", "paper_paint_near_ink"),
    (b"1 1 1 rg 200 200 40 40 re f 0 0 0 RG 1 w 20 50 m 120 50 l S", "paper_paint"),
    (b"q 60 100 40 40 re W n 0 0 0 RG 2 w 60 120 m 290 120 l S Q", "clipped"),
    (b"0 0 0 RG 1 w [3 9] 0 d 40 150 m 160 150 l S", "dashed"),
    (b"0 0 0 RG 1 w 20 150 m 280 150 l S 1 1 1 rg 100 100 100 100 re f", "occluded"),
])
def test_each_vector_category(tmp_path, ops, expected):
    cats = _cats(tmp_path, ops)
    assert expected in cats, cats


def test_light_fill(tmp_path):
    lines = b"".join(b"0 0 0 RG 1 w 20 %d m 280 %d l S " % (y, y) for y in range(160, 290, 10))
    cats = _cats(tmp_path, b"0.9 0.9 0.9 rg 40 40 80 60 re f " + lines)
    assert cats.get("light_fill") == 1 and "paper_paint" in cats


def test_native_text_is_unbacked_ink(tmp_path):
    cats = _cats(tmp_path, b"0 0 0 RG 1 w 20 150 m 280 150 l S", text="HELLO WORLD")
    assert cats.get("unbacked_text", 0) >= 1 and "unbacked_other" not in cats


def test_lost_dark_ink_when_a_big_light_fill_pulls_otsu(tmp_path):
    # at 150 dpi this proportion makes Otsu split grey from white and the black line reads as paper
    # (whether it happens depends on fill share and dpi -- exactly why the scan looks for it)
    cats = _cats(tmp_path, b"0.9 0.9 0.9 rg 20 20 200 150 re f 0 0 0 RG 1 w 30 150 m 270 150 l S")
    assert cats.get("lost_dark_ink", 0) >= 1


def test_tables_figures_and_outputs(tmp_path):
    path = _pdf(tmp_path, b"0 0 0 RG 1 w [3 9] 0 d 40 150 m 160 150 l S", text="AB")
    scans = [h.scan_page(path, 0, dpi=100)]
    rows = h.summary_rows(scans)
    assert [r["page"] for r in rows] == [0, "total"] and rows[1]["dashed"] == 1
    finds = h.findings_rows(scans)
    assert {f["category"] for f in finds} >= {"dashed", "unbacked_text"}
    assert "dashed" in h.markdown_table(rows)
    run = h.new_run_dir(tmp_path / "run")
    assert h.save_csv(finds, run / "findings.csv", h.FINDING_COLUMNS).read_text().startswith("pdf,page,category")
    assert h.save_fig(h.page_overlay(scans[0]), run / "pages" / "p.png").is_file()
    assert h.save_fig(h.gallery(scans, "dashed"), run / "gallery" / "dashed.png").is_file()
    assert h.gallery(scans, "clipped") is None
    assert "dashed 1" in h.one_line(scans[0])
