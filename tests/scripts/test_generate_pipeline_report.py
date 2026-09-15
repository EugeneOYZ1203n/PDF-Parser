"""ReportConfig validation for scripts/generate_pipeline_report.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pymupdf as fitz
import pytest

_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_pipeline_report.py"
_spec = importlib.util.spec_from_file_location("generate_pipeline_report", _MOD_PATH)
gpr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpr)


def test_defaults_and_pages():
    cfg = gpr.ReportConfig(input_files=["a.pdf"])
    assert cfg.pipeline == "current"
    assert cfg.final_stage is None
    assert cfg.pages_for("a") == [0]

    cfg2 = gpr.ReportConfig(pages={"a": [1, 2], "*": [0]})
    assert cfg2.pages_for("a") == [1, 2]
    assert cfg2.pages_for("b") == [0]

    cfg3 = gpr.ReportConfig(pages=[3, 4])
    assert cfg3.pages_for("anything") == [3, 4]


def test_filter_valid_pages_drops_out_of_range(tmp_path, caplog):
    pdf_path = tmp_path / "two_pages.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    with caplog.at_level("WARNING"):
        kept = gpr._filter_valid_pages(pdf_path, [0, 1, 2], "two_pages")

    assert kept == [0, 1]
    assert "two_pages" in caplog.text
    assert "[2]" in caplog.text


def test_filter_valid_pages_all_valid_no_warning(tmp_path, caplog):
    pdf_path = tmp_path / "one_page.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    with caplog.at_level("WARNING"):
        kept = gpr._filter_valid_pages(pdf_path, [0], "one_page")

    assert kept == [0]
    assert caplog.text == ""


def test_rejects_unknown_pipeline():
    with pytest.raises(ValueError):
        gpr.ReportConfig(pipeline="nope")


def test_rejects_unknown_final_stage():
    with pytest.raises(ValueError):
        gpr.ReportConfig(final_stage="nope")


def test_rejects_unknown_vectorise_mode():
    with pytest.raises(ValueError):
        gpr.ReportConfig(vectorise_mode="nope")


def test_benchmark_flag_and_vectorise_conflict():
    cfg = gpr.ReportConfig(benchmark=True, input_files=["a.pdf"])
    assert cfg.benchmark is True
    with pytest.raises(ValueError):
        gpr.ReportConfig(benchmark=True, vectorise=True, input_files=["a.pdf"])


def test_benchmark_inputs_keys(tmp_path):
    (tmp_path / "A.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (tmp_path / "C.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    labels = tmp_path / "C.json"
    labels.write_text(
        f'{{"pdf_path": "{(tmp_path / "C.pdf").as_posix()}", "entries": []}}',
        encoding="utf-8",
    )
    cfg = gpr.ReportConfig(
        benchmark=True,
        input_files=[str(tmp_path / "A.pdf"), str(labels)],
    )
    inputs = {b.key: b for b in cfg.benchmark_inputs()}
    assert set(inputs) == {"pdf:A", "labels:C"}
    assert inputs["pdf:A"].labels_path is None
    assert inputs["labels:C"].labels_path == labels.resolve()
    assert inputs["labels:C"].pdf_path == (tmp_path / "C.pdf").resolve()


def test_benchmark_inputs_directory_is_master_label_folder(tmp_path):
    folder = tmp_path / "D_label"
    folder.mkdir()
    (folder / "original.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (folder / "native_labels.json").write_text(
        f'{{"pdf_path": "{(folder / "original.pdf").as_posix()}", "entries": []}}',
        encoding="utf-8",
    )
    cfg = gpr.ReportConfig(benchmark=True, input_files=[str(folder)])
    inputs = {b.key: b for b in cfg.benchmark_inputs()}
    assert set(inputs) == {"labels:D_label"}
    assert inputs["labels:D_label"].labels_path == folder.resolve()
    assert inputs["labels:D_label"].pdf_path == (folder / "original.pdf").resolve()
    assert inputs["labels:D_label"].rasterised_pdf_path is None


def test_benchmark_inputs_directory_detects_rasterised_pdf(tmp_path):
    folder = tmp_path / "E_label"
    folder.mkdir()
    (folder / "original.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (folder / "rasterised.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (folder / "native_labels.json").write_text(
        f'{{"pdf_path": "{(folder / "original.pdf").as_posix()}", "entries": []}}',
        encoding="utf-8",
    )
    cfg = gpr.ReportConfig(benchmark=True, input_files=[str(folder)])
    inputs = {b.key: b for b in cfg.benchmark_inputs()}
    assert inputs["labels:E_label"].rasterised_pdf_path == (folder / "rasterised.pdf").resolve()


def test_bench_doc_name_disambiguates_master_label_originals():
    # Two different master_label.py folders both name their PDF copy
    # "original.pdf" -- `bench.pdf_path.stem` is "original" for both, which
    # is exactly the collision that used to make one input's report
    # overwrite the other's. `_bench_doc_name` must key off `bench.key`
    # instead, which `ReportConfig.benchmark_inputs()` already guarantees
    # unique per input.
    foo = gpr.BenchInput(
        key="labels:foo_label", pdf_path=Path("/x/foo_label/original.pdf"),
        labels_path=Path("/x/foo_label"),
    )
    bar = gpr.BenchInput(
        key="labels:bar_label", pdf_path=Path("/x/bar_label/original.pdf"),
        labels_path=Path("/x/bar_label"),
    )
    assert foo.pdf_path.stem == bar.pdf_path.stem == "original"
    assert gpr._bench_doc_name(foo) == "foo_label"
    assert gpr._bench_doc_name(bar) == "bar_label"
    assert gpr._bench_doc_name(foo) != gpr._bench_doc_name(bar)


def test_bench_doc_name_bare_pdf_and_sanitizes_unsafe_chars():
    bare = gpr.BenchInput(key="pdf:A", pdf_path=Path("/x/A.pdf"), labels_path=None)
    assert gpr._bench_doc_name(bare) == "A"

    unsafe = gpr.BenchInput(
        key='labels:weird?name', pdf_path=Path("/x/original.pdf"), labels_path=Path("/x"),
    )
    assert gpr._bench_doc_name(unsafe) == "weird_name"


def test_bench_ground_truth_by_type_synthesizes_native_to_raster(tmp_path):
    from rastervec.Evaluation.Labelling.label_schema import LabelEntry, save_labels

    folder = tmp_path / "F_label"
    folder.mkdir()
    (folder / "original.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    native_entry = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 10), cluster_signature="s",
        label_id="line:0:0:0", text="HELLO", source="native",
    )
    save_labels(
        gpr.LabelSet(pdf_path=str((folder / "original.pdf").resolve()), entries=[native_entry]),
        str(folder / "native_labels.json"),
    )
    bench = gpr.BenchInput(
        key="labels:F_label", pdf_path=(folder / "original.pdf").resolve(),
        labels_path=folder,
    )
    by_type = gpr._bench_ground_truth_by_type(bench, [0])
    assert [e.text for e in by_type["native_to_vector"].entries] == ["HELLO"]
    assert [e.text for e in by_type["native_to_raster"].entries] == ["HELLO"]
    assert by_type["native_to_raster"].entries[0].label_id.startswith("natsync:")
