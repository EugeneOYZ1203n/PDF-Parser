"""pipeline_report_benchmark: folder matching, multiclass scoring, charts."""
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
                      text=text, source=source)


def _write_doc(doc: Path, *, ocr_text: str, manual: bool) -> None:
    doc.mkdir(parents=True, exist_ok=True)
    dump_io.write_dump(
        doc / "dump.json", "x.pdf",
        [dump_io.PageDump(_meta(), [_text(ocr_text, (0, 0, 50, 12))], [], "current", {})],
    )
    save_labels(LabelSet(pdf_path="x.pdf", entries=[_label("HELLO WORLD", (0, 0, 50, 12), "auto")]),
                str(doc / "ground_truth_auto.json"))
    if manual:
        save_labels(LabelSet(pdf_path="x.pdf", entries=[_label("FOO BAR", (0, 100, 50, 112), "manual")]),
                    str(doc / "ground_truth_manual.json"))


def _make_run(root: Path, name: str, entries: list[dict], ocr_text: str) -> Path:
    run = root / name
    for e in entries:
        _write_doc(run / e["dir"], ocr_text=ocr_text, manual="manual" in e["sources"])
    (run / "benchmark.json").write_text(
        json.dumps({"benchmark": True, "entries": entries}), encoding="utf-8")
    return run


_A = {"key": "pdf:A", "pdf_stem": "A", "dir": "A", "sources": ["auto"]}
_C = {"key": "labels:C", "pdf_stem": "C", "dir": "C", "sources": ["auto", "manual"]}


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


def test_two_runs_shared_key_charts_and_viewer_cmds(tmp_path):
    r1 = _make_run(tmp_path, "run1", [_A, _C, {"key": "labels:B", "pdf_stem": "B", "dir": "B", "sources": ["auto", "manual"]}], "HELLO WORLD")
    r2 = _make_run(tmp_path, "run2", [_A, _C, {"key": "pdf:B", "pdf_stem": "B", "dir": "B", "sources": ["auto"]}], "HELLO")
    out_root = tmp_path / "out"
    assert prb.main(["--run", str(r1), "--run", str(r2), "--out-root", str(out_root)]) == 0

    run_out = next(out_root.iterdir())
    text = (run_out / "benchmark.txt").read_text(encoding="utf-8")
    assert "pdf:A  --  AUTO ground truth" in text
    assert "labels:C  --  MANUAL ground truth" in text
    assert "confusion matrix" in text
    assert "labels:B" not in text  # labels:B vs pdf:B not shared

    charts = sorted(p.name for p in (run_out / "charts").glob("*.png"))
    assert any("labels_C__p0__auto.png" == c for c in charts)
    assert any("labels_C__p0__confusion.png" == c for c in charts)
    assert any("labels_C__p0__manual.png" == c for c in charts)
    assert "aggregate__auto.png" in charts

    cmds = (run_out / "viewer_commands.txt").read_text(encoding="utf-8")
    assert "pipeline_report_viewer.py" in cmds
    assert cmds.count("\n") >= 3  # header + pdf:A + labels:C


def test_single_run_allowed(tmp_path):
    r = _make_run(tmp_path, "solo", [_C], "HELLO WORLD")
    out_root = tmp_path / "out"
    assert prb.main(["--run", str(r), "--out-root", str(out_root)]) == 0
    text = (next(out_root.iterdir()) / "benchmark.txt").read_text(encoding="utf-8")
    assert "labels:C  --  AUTO ground truth" in text
