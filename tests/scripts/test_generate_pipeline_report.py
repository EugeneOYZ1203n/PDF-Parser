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


# ---------------------------------------------------------------------------
# Per-backend pre-OCR debug image dumpers -- each reads a `p3_debug`-shaped
# dict (what `res.extra["p3_debug"]` holds for that backend), never the old
# dead `res.*` attributes.
# ---------------------------------------------------------------------------
import numpy as np
from collections import namedtuple

from rastervec.commons.models import Segment

_FakeText = namedtuple("FakeText", "text")


def test_save_segment_recog_images_writes_one_png_per_segment(tmp_path):
    segs = [Segment(vectors=[], angle=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))]
    texts = [_FakeText(text="AB/CD")]
    folder = tmp_path / "recog"
    n = gpr._save_segment_recog_images(segs, texts, folder, page_index=0)
    assert n == 1
    files = list(folder.glob("*.png"))
    assert len(files) == 1
    assert "AB_CD" in files[0].name


def test_save_segment_recog_images_empty_input_no_folder(tmp_path):
    folder = tmp_path / "recog"
    assert gpr._save_segment_recog_images([], [], folder, page_index=0) == 0
    assert not folder.exists()


def test_save_fastintopaddle_recog_images_reads_p3_debug(tmp_path):
    segs = [Segment(vectors=[], angle=0.0, image=np.zeros((2, 2, 3), dtype=np.uint8))]
    p3_debug = {"segments": segs, "texts": [_FakeText(text="X")]}
    folder = tmp_path / "recog"
    assert gpr._save_fastintopaddle_recog_images(p3_debug, folder, 0) == 1


def test_save_vectorclassification_recog_images_reads_ocr_crops(tmp_path):
    crop = np.zeros((2, 2, 3), dtype=np.uint8)
    p3_debug = {"ocr_crops": [(crop, "Y")]}
    folder = tmp_path / "recog"
    assert gpr._save_vectorclassification_recog_images(p3_debug, folder, 0) == 1


def test_save_legacyrecreation_ocr_images_reads_ocr_crops(tmp_path):
    crop = np.zeros((3, 3, 3), dtype=np.uint8)
    p3_debug = {"ocr_crops": [(crop, "HELLO")]}
    folder = tmp_path / "ocr"
    n = gpr._save_legacyrecreation_ocr_images(p3_debug, folder, page_index=2)
    assert n == 1
    files = list(folder.glob("*.png"))
    assert len(files) == 1
    assert "HELLO" in files[0].name
    assert "p2_" in files[0].name


def test_save_legacyrecreation_ocr_images_missing_key_no_crash(tmp_path):
    folder = tmp_path / "ocr"
    assert gpr._save_legacyrecreation_ocr_images({}, folder, 0) == 0
    assert not folder.exists()


def test_save_fastintopaddle_detect_images_missing_keys_no_crash(tmp_path):
    folder = tmp_path / "detect"
    assert gpr._save_fastintopaddle_detect_images({}, folder, 0) == 0
    assert not folder.exists()


def test_save_vectorclassification_detect_images_draws_quads(tmp_path):
    bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    quads = [np.array([(1, 1), (10, 1), (10, 10), (1, 10)], dtype=np.float64)]
    p3_debug = {"cluster_detections": [(bgr, quads)]}
    folder = tmp_path / "detect"
    n = gpr._save_vectorclassification_detect_images(p3_debug, folder, page_index=0)
    assert n == 1
    assert len(list(folder.glob("*.png"))) == 1


def test_save_vectorclassification_detect_images_missing_key_no_crash(tmp_path):
    folder = tmp_path / "detect"
    assert gpr._save_vectorclassification_detect_images({}, folder, 0) == 0
    assert not folder.exists()


def test_layer_writer_skips_all_blank_layers(tmp_path):
    from rastervec.commons.models import PageMeta
    from rastervec.commons.renderer import render_boxes_pdf

    pm = PageMeta(index=0, number=1, mediabox=(0, 0, 100, 100), rotation=0, width=100, height=100)
    blank = render_boxes_pdf(pm, [])
    drawn = render_boxes_pdf(pm, [((10, 10, 20, 20), (1, 0, 0))])
    writer = gpr._LayerWriter()
    for page_bytes in (blank, blank):
        writer.add("empty.pdf", {"file": "empty.pdf"}, page_bytes)
    for page_bytes in (blank, drawn):  # blank on page 0 only -> still written
        writer.add("some.pdf", {"file": "some.pdf"}, page_bytes)
    assert writer.filenames() == ["some.pdf"]
    writer.finalize(tmp_path)
    assert not (tmp_path / "empty.pdf").exists()
    with fitz.open(str(tmp_path / "some.pdf")) as doc:
        assert doc.page_count == 2  # page-aligned with the report's pages


def test_new_engine_artifacts_skip_phase1_and_final():
    variant = gpr.PipelineVariant(name="x", engine="current", p2="Stub", p3="FastIntoPaddle")
    stems = [row[0] for row in gpr._active_artifacts(gpr.ReportConfig(), variant)]
    assert stems == ["phase2", "reconstructed"]


def test_debug_images_flag_default_and_off(tmp_path):
    from types import SimpleNamespace

    import numpy as np

    assert gpr.ReportConfig().debug_images is True
    crop = np.zeros((4, 4, 3), dtype=np.uint8)
    res = SimpleNamespace(extra={"p3_debug": {"ocr_crops": [(crop, "A")]}})
    ocr_dir = gpr._image_dirs(tmp_path)[2]
    gpr._accumulate_page(res, 0, [], gpr._LayerWriter(), {}, None, p3="LegacyRecreation")
    assert not ocr_dir.exists()
    reservoirs = gpr._image_reservoirs(*gpr._image_dirs(tmp_path), seed_name="doc")
    debug: dict = {}
    gpr._accumulate_page(
        res, 0, [], gpr._LayerWriter(), {}, reservoirs, p3="LegacyRecreation",
        clock=gpr.StepClock(debug),
    )
    assert len(list(ocr_dir.glob("*.png"))) == 1
    assert set(debug) == {"stage_layers", "debug_images"}


def test_image_reservoir_caps_randomly_and_deterministically(tmp_path):
    from PIL import Image

    from scripts.debug_image_savers import _ImageReservoir

    def fill(folder, seed):
        r = _ImageReservoir(folder, cap=10, seed=seed)
        made = 0
        for page in range(5):
            for i in range(40):
                def _make():
                    nonlocal made
                    made += 1
                    return Image.new("RGB", (2, 2))
                r.offer(f"p{page}_word_{i:03d}.png", _make)
        return sorted(p.name for p in folder.glob("*.png")), made, r

    names_a, made_a, r = fill(tmp_path / "a", seed=7)
    names_b, _made, _r = fill(tmp_path / "b", seed=7)
    assert len(names_a) == 10 == len(r.kept)
    assert names_a == names_b  # fixed seed -> same picks
    assert made_a < 200  # rejected crops are never encoded
    assert len({n.split("_")[0] for n in names_a}) > 1  # not just the first page


def test_extra_prediction_layers_only_with_content(tmp_path):
    from types import SimpleNamespace

    from rastervec.commons.models import PageMeta

    pm = PageMeta(index=0, number=1, mediabox=(0, 0, 100, 100), rotation=0, width=100, height=100)
    res = SimpleNamespace(texts=[], vectors=[], extra={})
    writer = gpr._LayerWriter()
    gpr._add_extra_prediction_layers(writer, res, pm, gt_boxes=[], manual_boxes=[])
    assert writer.filenames() == []  # all three layers blank -> none written
    assert set(writer.meta) == {
        "benchmark__extra_text.pdf", "benchmark__extra_vectors.pdf", "benchmark__missed_vectors.pdf",
    }
