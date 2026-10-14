"""scripts/generate_pipeline_report.py: config validation, the artifact writers, and end-to-end
runs on synthetic PDFs (Stub P2, so no LibreOffice)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pymupdf as fitz
import pytest
from PIL import Image

from raster_parser.commons.models import PageMeta
from raster_parser.commons.renderer.pdf import render_boxes_pdf
from raster_parser.evaluation import dump_io
from raster_parser.evaluation.report import report_artifacts as ra
from raster_parser.evaluation.report.report_config import ReportConfig, layer_slug
from raster_parser.P3_Vector_Parsing.v1VectorClass import s4_word_ocr

_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_pipeline_report.py"
_spec = importlib.util.spec_from_file_location("generate_pipeline_report", _MOD_PATH)
gpr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpr)


def _pm(rotation: int = 0) -> PageMeta:
    return PageMeta(index=0, number=1, mediabox=(0, 0, 100, 100), rotation=rotation, width=100, height=100)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_defaults_and_pages():
    cfg = ReportConfig(input_files=["a.pdf"])
    assert (cfg.p2, cfg.p3) == ("v1RasterToVec", "v1VectorClass")
    assert cfg.final_stage is None
    assert cfg.pages_for("a") == [0]
    assert ReportConfig(pages={"a": [1, 2], "*": [0]}).pages_for("b") == [0]
    assert ReportConfig(pages=[3, 4]).pages_for("anything") == [3, 4]


@pytest.mark.parametrize("field, value", [
    ("p2", "nope"), ("p3", "nope"), ("final_stage", "phase4"), ("vectorise_mode", "nope"),
])
def test_rejects_unknown_values(field, value):
    with pytest.raises(ValueError):
        ReportConfig(**{field: value})


def test_benchmark_and_vectorise_conflict():
    with pytest.raises(ValueError):
        ReportConfig(benchmark=True, vectorise=True, input_files=["a.pdf"])


def test_filter_valid_pages_drops_out_of_range(tmp_path, caplog):
    pdf_path = tmp_path / "two_pages.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.new_page()
        doc.save(str(pdf_path))
    with caplog.at_level("WARNING"):
        assert gpr.filter_valid_pages(pdf_path, [0, 1, 2], "two_pages") == [0, 1]
    assert "[2]" in caplog.text


def test_benchmark_inputs_keys(tmp_path):
    (tmp_path / "A.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (tmp_path / "C.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    labels = tmp_path / "C.json"
    labels.write_text(json.dumps({"pdf_path": (tmp_path / "C.pdf").as_posix(), "entries": []}), encoding="utf-8")
    folder = tmp_path / "D_label"
    folder.mkdir()
    (folder / "original.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (folder / "native_labels.json").write_text(
        json.dumps({"pdf_path": (folder / "original.pdf").as_posix(), "entries": []}), encoding="utf-8")

    cfg = ReportConfig(benchmark=True, input_files=[tmp_path / "A.pdf", labels, folder])
    inputs = {b.key: b for b in cfg.benchmark_inputs()}
    assert set(inputs) == {"pdf:A", "labels:C", "labels:D_label"}
    assert inputs["pdf:A"].labels_path is None
    assert inputs["labels:C"].pdf_path == (tmp_path / "C.pdf").resolve()
    assert inputs["labels:D_label"].pdf_path == (folder / "original.pdf").resolve()
    assert gpr.bench_doc_name(inputs["labels:D_label"]) == "D_label"


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------
def test_layer_writer_skips_all_blank_layers(tmp_path):
    blank = render_boxes_pdf(_pm(), [])
    drawn = render_boxes_pdf(_pm(), [((10, 10, 20, 20), (1, 0, 0))])
    writer = ra.LayerWriter()
    for pos in range(2):
        writer.begin_page(pos, 100, 100)
        writer.add("empty.pdf", {"file": "empty.pdf"}, blank)
        writer.add("some.pdf", {"file": "some.pdf"}, (blank, drawn)[pos])
    assert writer.filenames() == ["some.pdf"]
    writer.finalize(tmp_path)
    assert not (tmp_path / "empty.pdf").exists()
    with fitz.open(str(tmp_path / "some.pdf")) as doc:
        assert doc.page_count == 2


def test_layer_writer_keeps_sparse_and_repeated_layers_page_aligned(tmp_path):
    def box(x):
        return render_boxes_pdf(_pm(90), [((x, 10, x + 5, 20), (1, 0, 0))])

    spill = tmp_path / "spill"
    writer = ra.LayerWriter(spill)
    for pos in range(3):
        writer.begin_page(pos, 100, 100, 90)
        writer.add("dense.pdf", {"file": "dense.pdf"}, box(10))
        if pos == 1:
            writer.add("sparse.pdf", {"file": "sparse.pdf"}, box(30))
            writer.add("sparse.pdf", {"file": "sparse.pdf"}, box(60))
    writer.finalize(tmp_path)
    assert not spill.exists()
    with fitz.open(str(tmp_path / "sparse.pdf")) as doc:
        assert doc.page_count == 3
        assert [page.rotation for page in doc] == [90, 90, 90]
        assert [len(page.get_drawings()) for page in doc] == [0, 2, 0]
        assert sorted(d["rect"].x0 for d in doc[1].get_drawings()) == pytest.approx([30, 60], abs=1.0)


def test_debug_layer_sink_merges_per_page_count_labels():
    writer = ra.LayerWriter()
    added: list[tuple[str, dict]] = []
    writer.add = lambda fname, meta, pdf_bytes: added.append((fname, meta))
    sink = ra.debug_layer_sink(writer)
    sink("word_grouping", "word groups (25)", "#000000", b"")
    sink("word_grouping", "word groups (31)", "#000000", b"")
    sink("ocr", "kept (off-grid) (4)", "#000000", b"")
    assert [f for f, _m in added] == [
        "word_grouping__word_groups.pdf", "word_grouping__word_groups.pdf",
        f"ocr__{layer_slug('kept (off-grid)')}.pdf",
    ]
    assert added[0][1]["layer"] == "word groups"


def test_image_reservoir_caps_randomly_and_deterministically(tmp_path):
    def fill(folder, seed):
        r = ra.ImageReservoir(folder, cap=10, seed=seed)
        made = 0
        for page in range(5):
            for i in range(40):
                def _make():
                    nonlocal made
                    made += 1
                    return Image.new("RGB", (2, 2))
                r.offer(f"p{page}_word_{i:03d}.png", _make)
        return sorted(p.name for p in folder.glob("*.png")), made

    names_a, made_a = fill(tmp_path / "a", seed=7)
    names_b, _ = fill(tmp_path / "b", seed=7)
    assert len(names_a) == 10 and names_a == names_b
    assert made_a < 200  # rejected images are never built
    assert len({n.split("_")[0] for n in names_a}) > 1


def test_debug_image_sink_prefixes_the_page(tmp_path):
    reservoirs = ra.ImageReservoirs(tmp_path, cap=None)
    sink = ra.debug_image_sink(reservoirs, 7)
    sink("v1_ocr_input", "group0000.png", lambda: Image.new("RGB", (4, 4), "white"))
    assert [p.name for p in (tmp_path / "v1_ocr_input").iterdir()] == ["p7_group0000.png"]


def test_extra_prediction_layers_only_with_content(vector):
    writer = ra.LayerWriter()
    ra.add_extra_prediction_layers(writer, _pm(), [], [], [], gt_boxes=[], manual_boxes=[])
    assert writer.filenames() == []
    assert set(writer.meta) == {
        "benchmark__extra_text.pdf", "benchmark__extra_vectors.pdf", "benchmark__missed_vectors.pdf",
    }


def test_text_routed_vectors_are_the_inputs_not_kept(vector):
    a, b = vector(seqno=1), vector(seqno=2, bbox=(50.0, 50.0, 60.0, 60.0))
    assert ra.text_routed_vectors([a, b], [b]) == [a]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
class _FakeOcr:
    def __init__(self, use_angle_cls, *, lang="en", drop_score=0.0):
        pass

    def ocr_image(self, image, *, min_confidence=0.5, cls=None):
        return [((0, 0, 1, 1), "ABC", 0.9)]


def _write_config(tmp_path: Path, **fields) -> Path:
    path = tmp_path / "cfg.json"
    path.write_text(ReportConfig(output_root=tmp_path / "out", **fields).model_dump_json(), encoding="utf-8")
    return path


def test_report_run_writes_layers_dump_and_manifest(tmp_path, monkeypatch, synthetic_pdf_with_fills):
    monkeypatch.setattr(s4_word_ocr, "PaddleOcr", _FakeOcr)
    config = _write_config(tmp_path, p2="Stub", p3="v1VectorClass", input_files=[synthetic_pdf_with_fills])
    assert gpr.main(["--config", str(config)]) == 0

    (run_dir,) = (tmp_path / "out").iterdir()
    doc_dir = run_dir / synthetic_pdf_with_fills.stem
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    files = {layer["file"] for layer in manifest["layers"]}
    assert {"fill_filter__glyph_candidates.pdf", "word_grouping__word_groups.pdf",
            "ocr__text.pdf", "reconstructed__page.pdf"} <= files
    assert all((doc_dir / f).is_file() for f in files)
    assert list((doc_dir / "v1_ocr_input").glob("p0_group*.png"))
    dump = dump_io.load_dump(doc_dir / "dump.json")
    assert [t.text for t in dump.pages[0].texts] == ["ABC"]
    assert (run_dir / "config_and_hyperparameters.txt").is_file()


def test_final_stage_skips_later_backends(tmp_path, monkeypatch, synthetic_pdf_with_fills):
    monkeypatch.setattr(s4_word_ocr, "PaddleOcr", _FakeOcr)
    config = _write_config(tmp_path, p2="Stub", p3="v1VectorClass", final_stage="phase2",
                           input_files=[synthetic_pdf_with_fills])
    assert gpr.main(["--config", str(config)]) == 0
    (run_dir,) = (tmp_path / "out").iterdir()
    manifest = json.loads((run_dir / synthetic_pdf_with_fills.stem / "manifest.json").read_text(encoding="utf-8"))
    assert {layer["stage"] for layer in manifest["layers"]} == {"reconstructed"}


def test_benchmark_run_writes_ground_truth_and_overlays(tmp_path, synthetic_pdf_factory):
    doc = synthetic_pdf_factory([{"texts": [{"point": (20, 50), "text": "HELLO"}],
                                  "drawings": [{"lines": [((10, 80), (150, 80))]}]}])
    pdf = tmp_path / "in" / "A.pdf"
    pdf.parent.mkdir()
    doc.save(str(pdf))
    doc.close()
    config = _write_config(tmp_path, p2="Stub", p3="Stub", benchmark=True, input_files=[pdf])
    assert gpr.main(["--config", str(config)]) == 0

    (run_dir,) = (tmp_path / "out").iterdir()
    marker = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert marker["benchmark"] is True
    assert [(e["key"], e["dir"], e["sources"]) for e in marker["entries"]] == [
        ("pdf:A", "A", ["native_to_vector"]),
    ]
    doc_dir = run_dir / "A"
    assert (doc_dir / "ground_truth_native_to_vector.json").is_file()
    assert (doc_dir / "native_to_vector_bbox.pdf").is_file()
    assert (doc_dir / "converted_p0.pdf").is_file()


def test_benchmark_without_layers_images_or_converted_pdfs_writes_no_pdfs(tmp_path, synthetic_pdf_factory):
    doc = synthetic_pdf_factory([{"texts": [{"point": (20, 50), "text": "HELLO"}]}])
    pdf = tmp_path / "in" / "A.pdf"
    pdf.parent.mkdir()
    doc.save(str(pdf))
    doc.close()
    config = _write_config(tmp_path, p2="Stub", p3="Stub", benchmark=True, input_files=[pdf],
                           debug_layers=False, debug_images=False, keep_converted_pdfs=False)
    assert gpr.main(["--config", str(config)]) == 0
    (run_dir,) = (tmp_path / "out").iterdir()
    assert (run_dir / "A" / "dump.json").is_file()
    assert list(run_dir.rglob("*.pdf")) == []
    assert list(run_dir.rglob("*.png")) == []
