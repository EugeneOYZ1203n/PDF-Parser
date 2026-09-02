from __future__ import annotations

from pathlib import Path

from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet, save_labels
from rastervec.Reader.dataset import (
    collect_dataset,
    find_label_sets,
    find_pdfs,
    resolve_pdf_path,
)


def _label_set(pdf_path: str, entries: list[LabelEntry]) -> LabelSet:
    return LabelSet(pdf_path=pdf_path, entries=entries)


def _manual(page_index: int, text: str) -> LabelEntry:
    return LabelEntry(
        page_index=page_index, cluster_bbox=(0, 0, 10, 5),
        cluster_signature="1:0.0:0.0:10.0:5.0", text=text, source="manual",
    )


def _auto(page_index: int, text: str) -> LabelEntry:
    return LabelEntry(
        page_index=page_index, cluster_bbox=(0, 0, 10, 5),
        cluster_signature=f"line:{page_index}:0:0", text=text, source="auto",
    )


def test_find_pdfs_is_recursive(tmp_path, synthetic_pdf_factory):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()
    synthetic_pdf_factory([{}]).save(str(tmp_path / "top.pdf"))
    synthetic_pdf_factory([{}]).save(str(tmp_path / "a" / "b" / "deep.pdf"))

    found = find_pdfs(tmp_path)

    assert [p.name for p in found] == ["deep.pdf", "top.pdf"]


def test_find_label_sets_skips_non_labelset_json(tmp_path):
    (tmp_path / "notes.json").write_text('{"hello": 1}', encoding="utf-8")
    save_labels(_label_set("x.pdf", [_manual(0, "hi")]), str(tmp_path / "labels.json"))

    found = find_label_sets(tmp_path)

    assert [p.name for p, _ in found] == ["labels.json"]


def test_resolve_pdf_path_by_basename(tmp_path, synthetic_pdf_factory):
    (tmp_path / "pdfs").mkdir()
    pdf = tmp_path / "pdfs" / "drawing.pdf"
    synthetic_pdf_factory([{}]).save(str(pdf))

    resolved = resolve_pdf_path("drawing.pdf", tmp_path / "elsewhere", [pdf])

    assert resolved == pdf.resolve()


def test_collect_dataset_pairs_manual_labels_to_pages(tmp_path, synthetic_pdf_factory):
    pdf = tmp_path / "drawing.pdf"
    synthetic_pdf_factory([{}, {}, {}]).save(str(pdf))
    save_labels(
        _label_set("drawing.pdf", [_manual(0, "a"), _auto(0, "a"), _manual(2, "c")]),
        str(tmp_path / "labels.json"),
    )

    dataset = collect_dataset(tmp_path)

    by_page = {d.page_index: d for d in dataset}
    assert set(by_page) == {0, 2}
    assert [e.text for e in by_page[0].manual_entries] == ["a"]
    assert [e.text for e in by_page[2].manual_entries] == ["c"]
    assert all(d.pdf_path == str(pdf.resolve()) for d in dataset)


def test_collect_dataset_unlabelled_pdf_is_page_capped(tmp_path, synthetic_pdf_factory):
    synthetic_pdf_factory([{}, {}, {}, {}]).save(str(tmp_path / "no_labels.pdf"))

    dataset = collect_dataset(tmp_path, pages_per_pdf=2)

    assert [(Path(d.pdf_path).name, d.page_index) for d in dataset] == [
        ("no_labels.pdf", 0),
        ("no_labels.pdf", 1),
    ]
    assert all(d.manual_entries == () for d in dataset)


def test_collect_dataset_mixed_tree(tmp_path, synthetic_pdf_factory):
    (tmp_path / "labelled").mkdir()
    (tmp_path / "raw").mkdir()

    labelled_pdf = tmp_path / "labelled" / "has_labels.pdf"
    synthetic_pdf_factory([{}, {}]).save(str(labelled_pdf))
    save_labels(
        _label_set("has_labels.pdf", [_manual(1, "x")]),
        str(tmp_path / "labelled" / "has_labels.json"),
    )
    synthetic_pdf_factory([{}, {}, {}]).save(str(tmp_path / "raw" / "plain.pdf"))

    dataset = collect_dataset(tmp_path, pages_per_pdf=1)

    got = {(Path(d.pdf_path).name, d.page_index) for d in dataset}
    assert got == {("has_labels.pdf", 1), ("plain.pdf", 0)}


def test_collect_dataset_include_unlabelled_false(tmp_path, synthetic_pdf_factory):
    synthetic_pdf_factory([{}]).save(str(tmp_path / "orphan.pdf"))
    labelled = tmp_path / "kept.pdf"
    synthetic_pdf_factory([{}]).save(str(labelled))
    save_labels(_label_set("kept.pdf", [_manual(0, "y")]), str(tmp_path / "kept.json"))

    dataset = collect_dataset(tmp_path, include_unlabelled=False)

    assert [Path(d.pdf_path).name for d in dataset] == ["kept.pdf"]
