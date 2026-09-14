"""pipeline_report_benchmark: folder matching, text/vector scoring, report.html."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rastervec.Evaluation import dump_io
from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet, save_labels
from rastervec.models import PageMeta, Text

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
        [dump_io.PageDump(_meta(), [_text(ocr_text, (0, 0, 50, 12))], [], "current", {})],
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
    assert "labels:B" not in text  # labels:B vs pdf:B not shared

    html = (run_out / "report.html").read_text(encoding="utf-8")
    assert "pdf:A" in html
    assert "labels:C" in html
    assert "Label description" in html
    assert "Char overlap" in html
    assert "Rotation accuracy" in html
    assert "Vector classification funnel" in html

    charts = sorted(p.name for p in (run_out / "charts").glob("*.png"))
    assert "labels_C__aggregate__labels.png" in charts
    assert "aggregate__labels.png" in charts
    # per-page charts are no longer generated
    assert not any("__p0__" in c for c in charts)

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
