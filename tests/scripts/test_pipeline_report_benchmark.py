"""pipeline_report_benchmark: folder matching, text/vector scoring, report.html."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rastervec.Evaluation import dump_io
from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet, save_labels
from rastervec.commons.models import PageMeta, Text

_MOD = Path(__file__).resolve().parents[2] / "scripts" / "pipeline_report_benchmark.py"
_spec = importlib.util.spec_from_file_location("pipeline_report_benchmark", _MOD)
prb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prb)


def _text(s: str, bbox) -> Text:
    return Text(
        text=s, bbox=bbox, direction=(1.0, 0.0), origin=(bbox[0], bbox[3]),
        font="helv", font_size=10.0, color=0, flags=0, ascender=0.9, descender=-0.2,
        wmode=0, block_no=0, line_no=0, word_no=0, page_index=0, seqno=0, source="ocr",
    )


def _meta() -> PageMeta:
    return PageMeta(index=0, number=1, mediabox=(0, 0, 200, 200), rotation=0,
                    width=200, height=200)


def _label(text, bbox, source):
    return LabelEntry(page_index=0, cluster_bbox=bbox, cluster_signature="s",
                      label_id=f"{source}:{text}", text=text, source=source)


def _write_doc(doc: Path, *, ocr_text: str, manual: bool) -> None:
    doc.mkdir(parents=True, exist_ok=True)
    dump_io.write_dump(
        doc / "dump.json", "x.pdf",
        [dump_io.PageDump(
            _meta(), [_text(ocr_text, (0, 0, 50, 12))], [], "current",
            {"phase1": 0.1, "phase2": 0.0, "phase3": 2.0, "phase4": 0.05},
            substep_durations={"fast": 0.5, "ocr": 1.2},
            raster_step_durations={"phase1": 0.2, "phase3": 1.0},
        )],
    )
    save_labels(
        LabelSet(pdf_path="x.pdf", entries=[_label("HELLO WORLD", (0, 0, 50, 12), "native")]),
        str(doc / "ground_truth_native_to_vector.json"),
    )
    if manual:
        save_labels(
            LabelSet(pdf_path="x.pdf", entries=[_label("FOO BAR", (0, 100, 50, 112), "vector")]),
            str(doc / "ground_truth_original_vector.json"),
        )


def _make_run(root: Path, name: str, entries: list[dict], ocr_text: str) -> Path:
    run = root / name
    for e in entries:
        _write_doc(run / e["dir"], ocr_text=ocr_text, manual="original_vector" in e["sources"])
    (run / "benchmark.json").write_text(
        json.dumps({"benchmark": True, "entries": entries}), encoding="utf-8")
    return run


_A = {"key": "pdf:A", "pdf_stem": "A", "dir": "A", "sources": ["native_to_vector"]}
_C = {"key": "labels:C", "pdf_stem": "C", "dir": "C", "sources": ["native_to_vector", "original_vector"]}


def test_rejects_non_benchmark_folder(tmp_path):
    bad = tmp_path / "plain"
    bad.mkdir()
    (bad / "benchmark.json").write_text('{"benchmark": false}')
    with pytest.raises(SystemExit):
        prb.main(["--run", str(bad)])


def test_rejects_three_runs(tmp_path):
    r = _make_run(tmp_path, "r", [_A], "HELLO WORLD")
    with pytest.raises(SystemExit):
        prb.main(["--run", str(r), "--run", str(r), "--run", str(r)])


def test_two_runs_shared_key_report_html_and_viewer_cmds(tmp_path):
    r1 = _make_run(tmp_path, "run1", [_A, _C, {"key": "labels:B", "pdf_stem": "B", "dir": "B", "sources": ["native_to_vector", "original_vector"]}], "HELLO WORLD")
    r2 = _make_run(tmp_path, "run2", [_A, _C, {"key": "pdf:B", "pdf_stem": "B", "dir": "B", "sources": ["native_to_vector"]}], "HELLO")
    out_root = tmp_path / "out"
    assert prb.main(["--run", str(r1), "--run", str(r2), "--out-root", str(out_root)]) == 0

    run_out = next(out_root.iterdir())
    text = (run_out / "benchmark.txt").read_text(encoding="utf-8")
    assert "pdf:A" in text
    assert "labels:C" in text
    assert "OCR confusion characters" in text
    assert "vectorised run wall-clock per page (seconds)" in text
    assert "median seconds per page by run" in text
    assert "labels:B" not in text  # labels:B vs pdf:B not shared

    html = (run_out / "report.html").read_text(encoding="utf-8")
    assert "pdf:A" in html
    assert "labels:C" in html
    assert "Label description" in html
    assert "Char overlap" in html
    assert "Rotation accuracy" in html
    assert "Per-character OCR accuracy -- native_to_vector" in html
    assert "detected 1 (100.0%)" in html  # run1 reads HELLO WORLD exactly
    assert "Per-character examples" not in html
    assert not (run_out / "examples").exists()
    assert "Vector classification funnel" in html
    assert "Font size distribution" in html
    assert "Wall-clock per page -- vectorised run" in html
    assert "Wall-clock per page -- rasterised run" in html
    assert "phase3 › ocr" in html
    assert "Δ median vs run1: run2" in html
    assert 'src="charts/aggregate__timing__vectorised.png"' in html  # grand aggregate

    charts = sorted(p.name for p in (run_out / "charts").glob("*.png"))
    assert "labels_C__aggregate__labels.png" in charts
    assert "labels_C__timing__vectorised.png" in charts
    assert "aggregate__timing__rasterised.png" in charts
    assert 'src="charts/labels_C__timing__vectorised.png"' in html
    assert "aggregate__labels.png" in charts
    # per-page charts are no longer generated
    assert not any("__p0__" in c for c in charts)

    font_size_charts = [c for c in charts if "font_size" in c]
    assert font_size_charts
    assert all(
        f'src="charts/{c}"' in html for c in font_size_charts
    ), "font-size charts must be generated AND linked into report.html"

    cmds = (run_out / "viewer_commands.txt").read_text(encoding="utf-8")
    assert "pipeline_report_viewer.py" in cmds
    assert cmds.count("\n") >= 3  # header + pdf:A + labels:C


def test_single_run_allowed(tmp_path):
    r = _make_run(tmp_path, "solo", [_C], "HELLO WORLD")
    out_root = tmp_path / "out"
    assert prb.main(["--run", str(r), "--out-root", str(out_root)]) == 0
    out = next(out_root.iterdir())
    text = (out / "benchmark.txt").read_text(encoding="utf-8")
    assert "labels:C" in text
    assert (out / "report.html").is_file()


def test_score_text_routes_vectorised_vs_rasterised_predictions(tmp_path):
    doc = tmp_path / "doc"
    doc.mkdir()
    dump_io.write_dump(
        doc / "dump.json", "x.pdf",
        [dump_io.PageDump(
            _meta(),
            texts=[_text("HELLO", (0, 0, 50, 12))],
            vectors=[], engine="current", step_durations={},
            raster_texts=[_text("WORLD", (0, 100, 50, 112))],
        )],
    )
    save_labels(
        LabelSet(pdf_path="x.pdf", entries=[_label("HELLO", (0, 0, 50, 12), "native")]),
        str(doc / "ground_truth_native_to_vector.json"),
    )
    vector_to_raster_label = LabelEntry(
        page_index=0, cluster_bbox=(0, 100, 50, 112), cluster_signature="s",
        label_id="vecsync:x", text="WORLD", source="raster",
    )
    save_labels(
        LabelSet(pdf_path="x.pdf", entries=[vector_to_raster_label]),
        str(doc / "ground_truth_vector_to_raster.json"),
    )

    entry = prb.RunEntry(
        run_name="run", run_dir=tmp_path, key="labels:doc", pdf_stem="doc",
        doc_dir=doc, dump_path=doc / "dump.json", gt=prb._merge_gt(doc),
    )
    _per_page, agg = prb._score_text(entry, prb.MetricConfig())
    assert agg.by_type["native_to_vector"].char_overlap.matched == 5  # "HELLO"
    assert agg.by_type["native_to_vector"].char_overlap.missing == 0
    assert agg.by_type["vector_to_raster"].char_overlap.matched == 5  # "WORLD"
    assert agg.by_type["vector_to_raster"].char_overlap.missing == 0


def test_load_fast_cluster_stats_sums_across_pages(tmp_path):
    doc = tmp_path / "doc"
    doc.mkdir()
    dump_io.write_dump(
        doc / "dump.json", "x.pdf",
        [
            dump_io.PageDump(
                _meta(), texts=[], vectors=[], engine="current", step_durations={},
                fast_cluster_stats={"total": 4, "passed": 3},
            ),
            dump_io.PageDump(
                _meta(), texts=[], vectors=[], engine="current", step_durations={},
                fast_cluster_stats={"total": 2, "passed": 0},
            ),
        ],
    )
    entry = prb.RunEntry(
        run_name="run", run_dir=tmp_path, key="pdf:doc", pdf_stem="doc",
        doc_dir=doc, dump_path=doc / "dump.json", gt=LabelSet(pdf_path="x.pdf"),
    )
    stats = prb._load_fast_cluster_stats(entry)
    assert stats == {"total": 6, "passed": 3}


def test_load_fast_cluster_stats_none_when_no_page_recorded_it(tmp_path):
    doc = tmp_path / "doc"
    doc.mkdir()
    dump_io.write_dump(
        doc / "dump.json", "x.pdf",
        [dump_io.PageDump(_meta(), texts=[], vectors=[], engine="current", step_durations={})],
    )
    entry = prb.RunEntry(
        run_name="run", run_dir=tmp_path, key="pdf:doc", pdf_stem="doc",
        doc_dir=doc, dump_path=doc / "dump.json", gt=LabelSet(pdf_path="x.pdf"),
    )
    assert prb._load_fast_cluster_stats(entry) is None


def test_add_fast_cluster_section_renders_counts_and_skips_when_all_none():
    from rastervec.Evaluation.Evaluate.html_report import ReportBuilder

    builder = ReportBuilder("t")
    prb._add_fast_cluster_section(builder, {"current": {"total": 10, "passed": 6}, "legacy": None})
    html = builder.render()
    assert "Clusters dropped by FAST" in html
    assert "6" in html and "10" in html

    empty_builder = ReportBuilder("t")
    prb._add_fast_cluster_section(empty_builder, {"legacy": None})
    assert "Clusters dropped by FAST" not in empty_builder.render()


def test_load_retry_stats_sums_across_pages(tmp_path):
    doc = tmp_path / "doc"
    doc.mkdir()
    dump_io.write_dump(
        doc / "dump.json", "x.pdf",
        [
            dump_io.PageDump(
                _meta(), texts=[], vectors=[], engine="current", step_durations={},
                retry_stats={"0": 3, "1": 1, "2": 0, "3": 0, "failed": 1},
            ),
            dump_io.PageDump(
                _meta(), texts=[], vectors=[], engine="current", step_durations={},
                retry_stats={"0": 1, "1": 0, "2": 1, "3": 0, "failed": 0},
            ),
        ],
    )
    entry = prb.RunEntry(
        run_name="run", run_dir=tmp_path, key="pdf:doc", pdf_stem="doc",
        doc_dir=doc, dump_path=doc / "dump.json", gt=LabelSet(pdf_path="x.pdf"),
    )
    stats = prb._load_retry_stats(entry)
    assert stats == {"0": 4, "1": 1, "2": 1, "3": 0, "failed": 1}


def test_load_retry_stats_none_when_no_page_recorded_it(tmp_path):
    doc = tmp_path / "doc"
    doc.mkdir()
    dump_io.write_dump(
        doc / "dump.json", "x.pdf",
        [dump_io.PageDump(_meta(), texts=[], vectors=[], engine="current", step_durations={})],
    )
    entry = prb.RunEntry(
        run_name="run", run_dir=tmp_path, key="pdf:doc", pdf_stem="doc",
        doc_dir=doc, dump_path=doc / "dump.json", gt=LabelSet(pdf_path="x.pdf"),
    )
    assert prb._load_retry_stats(entry) is None


def test_add_retry_stats_section_renders_counts_and_skips_when_all_none():
    from rastervec.Evaluation.Evaluate.html_report import ReportBuilder

    builder = ReportBuilder("t")
    prb._add_retry_stats_section(
        builder, {"current": {"0": 4, "1": 2, "2": 1, "3": 0, "failed": 1}, "legacy": None},
    )
    html = builder.render()
    assert "Blank-recognition retries" in html
    assert "4" in html and "2" in html

    empty_builder = ReportBuilder("t")
    prb._add_retry_stats_section(empty_builder, {"legacy": None})
    assert "Blank-recognition retries" not in empty_builder.render()


def test_merge_gt_supports_legacy_auto_manual_filenames(tmp_path):
    doc = tmp_path / "doc"
    doc.mkdir()
    save_labels(
        LabelSet(pdf_path="x.pdf", entries=[_label("HELLO", (0, 0, 10, 10), "native")]),
        str(doc / "ground_truth_auto.json"),
    )
    save_labels(
        LabelSet(pdf_path="x.pdf", entries=[_label("WORLD", (0, 20, 10, 30), "vector")]),
        str(doc / "ground_truth_manual.json"),
    )
    merged = prb._merge_gt(doc)
    assert {e.text for e in merged.entries} == {"HELLO", "WORLD"}


def test_char_order_sorts_by_error_rate():
    from rastervec.Evaluation.Evaluate.confusion_metrics import CharStats
    from scripts.benchmark_report_sections import char_order

    class _PT:
        def __init__(self, cs):
            self.char_stats = cs

    class _Res:
        def __init__(self, cs):
            self.by_type = {t: _PT({}) for t in prb.TEXT_TYPES}
            self.by_type["native_to_vector"] = _PT(cs)

    res = _Res({
        "A": CharStats(detected=9, dropped=1),
        "B": CharStats(detected=1, misclassified=1),
        "X": CharStats(inserted=4),
        "C": CharStats(detected=5),
    })
    assert char_order({"r": res, "none": None}, "native_to_vector") == ["B", "A", "C", "X"]


def test_gallery_links_run_images_in_place(tmp_path):
    r = _make_run(tmp_path, "run1", [_A], "HELLO WORLD")
    img_dir = r / "A" / "paddle_recog_images"
    img_dir.mkdir()
    (img_dir / "p0_word_000__HELLO.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    out_root = tmp_path / "out"
    assert prb.main(["--run", str(r), "--out-root", str(out_root)]) == 0
    run_out = next(out_root.iterdir())
    assert not (run_out / "gallery").exists()
    html = (run_out / "report.html").read_text(encoding="utf-8")
    assert 'src="../../run1/A/paddle_recog_images/p0_word_000__HELLO.png"' in html
    assert not any("extra_chars" in p.name for p in (run_out / "charts").glob("*.png"))
