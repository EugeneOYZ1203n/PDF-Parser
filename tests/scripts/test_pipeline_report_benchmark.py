"""scripts/pipeline_report_benchmark.py: folder matching, scoring and report.html, on synthetic
run folders (no pipeline run)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")  # the `report` dependency group

from raster_parser.models.page import PageMeta  # noqa: E402
from raster_parser.models.result import ExtractResult  # noqa: E402
from raster_parser.models.text import Text  # noqa: E402
from raster_parser.evaluation import dump_io  # noqa: E402
from raster_parser.evaluation.evaluate.benchmark_report_sections import char_order  # noqa: E402
from raster_parser.evaluation.evaluate.benchmark_run_loading import load_run, score_text  # noqa: E402
from raster_parser.evaluation.evaluate.metrics import MetricConfig  # noqa: E402
from raster_parser.evaluation.labelling.label_schema import LabelEntry, LabelSet, save_labels  # noqa: E402

_MOD = Path(__file__).resolve().parents[2] / "scripts" / "pipeline_report_benchmark.py"
_spec = importlib.util.spec_from_file_location("pipeline_report_benchmark", _MOD)
prb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prb)


def _text(s: str, bbox) -> Text:
    return Text(
        text=s,
        bbox=bbox,
        direction=(1.0, 0.0),
        origin=(bbox[0], bbox[3]),
        font="",
        font_size=0.0,
        color=None,
        flags=0,
        ascender=None,
        descender=None,
        wmode=0,
        block_no=0,
        line_no=0,
        word_no=0,
        page_index=0,
        seqno=0,
        source="ocr",
    )


def _meta() -> PageMeta:
    return PageMeta(index=0, number=1, mediabox=(0, 0, 200, 200), rotation=0, width=200, height=200)


def _label(text, bbox, source):
    return LabelEntry(
        page_index=0, bbox=bbox, label_id=f"{source}:{text}", text=text, source=source
    )


def _write_doc(doc: Path, *, ocr_text: str, manual: bool) -> None:
    doc.mkdir(parents=True, exist_ok=True)
    texts = [_text(ocr_text, (0, 0, 50, 12)), _text("NOISE", (150, 150, 190, 160))]
    dump_io.write_dump(
        doc / "dump.json",
        "x.pdf",
        [
            dump_io.PageDump(
                ExtractResult(
                    _meta(),
                    texts,
                    [],
                    step_durations={
                        "native_read": 0.1,
                        "raster_vectorize": 0.0,
                        "vector_classify": 2.0,
                        "output_assemble": 0.05,
                    },
                ),
                "stub",
                "v1_vector_class",
                {"conversion": 0.2},
            )
        ],
        {"layer_save": 0.3},
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


def _make_run(
    root: Path, name: str, entries: list[dict], ocr_text: str, p3: str = "v1_vector_class"
) -> Path:
    run = root / name
    for e in entries:
        _write_doc(run / e["dir"], ocr_text=ocr_text, manual="original_vector" in e["sources"])
    full = [
        {**e, "source_pdf": f"/data/{e['pdf_stem']}.pdf", "labels": None, "pages": [0]}
        for e in entries
    ]
    (run / "benchmark.json").write_text(
        json.dumps(
            {
                "benchmark": True,
                "p2": "stub",
                "p3": p3,
                "final_stage": None,
                "config_path": "/cfg/bench.json",
                "entries": full,
            }
        ),
        encoding="utf-8",
    )
    return run


_A = {"key": "pdf:A", "pdf_stem": "A", "dir": "A", "sources": ["native_to_vector"]}
_C = {
    "key": "labels:C",
    "pdf_stem": "C",
    "dir": "C",
    "sources": ["native_to_vector", "original_vector"],
}


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


def test_two_runs_report(tmp_path):
    r1 = _make_run(
        tmp_path,
        "run1",
        [_A, _C, {**_C, "key": "labels:B", "pdf_stem": "B", "dir": "B"}],
        "HELLO WORLD",
    )
    r2 = _make_run(
        tmp_path,
        "run2",
        [_A, _C, {**_A, "key": "pdf:B", "pdf_stem": "B", "dir": "B"}],
        "HELLO",
        p3="stub",
    )
    out_root = tmp_path / "out"
    assert prb.main(["--run", str(r1), "--run", str(r2), "--out-root", str(out_root)]) == 0

    (run_out,) = out_root.iterdir()
    html = (run_out / "report.html").read_text(encoding="utf-8")
    # Header first: the runs and the files scored, then the summary, then each input.
    assert (
        html.index("Files scored")
        < html.index("Summary (all inputs)")
        < html.index('<h2 id="pdf:A">')
    )
    assert "/data/A.pdf" in html and "/cfg/bench.json" in html
    assert "labels:B" not in html  # labels:B vs pdf:B are different inputs
    for heading in (
        "char F1",
        "word F1",
        "bbox mean IoU",
        "rotation correct",
        "candidate precision",
        "pipeline s/page (median)",
        "Char overlap",
        "Rotation accuracy",
        "Per-character OCR accuracy (confusion) -- native_to_vector",
        "Wall-clock per page",
        "Δ median vs run1: run2",
    ):
        assert heading in html, heading
    assert "(2/4 OCR readings touch a label)" in html  # NOISE touches nothing, on 2 inputs
    for dropped in ("Font size", "Reading order", "funnel", "FAST"):
        assert dropped not in html, dropped
    assert 'src="charts/aggregate__char.png"' in html

    text = (run_out / "benchmark.txt").read_text(encoding="utf-8")
    assert "OCR confusion characters" in text and "median s/page by run" in text
    assert json.loads((run_out / "runs.json").read_text(encoding="utf-8"))["shared_keys"] == [
        "labels:C",
        "pdf:A",
    ]
    assert (run_out / "viewer_commands.txt").read_text(encoding="utf-8").count(
        "pipeline_report_viewer.py"
    ) == 2


def test_single_run_allowed(tmp_path):
    r = _make_run(tmp_path, "solo", [_A], "HELLO WORLD")
    assert prb.main(["--run", str(r), "--out-root", str(tmp_path / "out")]) == 0


def test_score_text_counts_candidates(tmp_path):
    run = _make_run(tmp_path, "r", [_C], "HELLO WORLD")
    _meta_, entries = load_run(run, parser=None)
    score = score_text(entries["labels:C"], MetricConfig())
    assert (score.n_ocr, score.n_ocr_touching_gt) == (2, 1)
    agg = score.aggregate.by_type
    assert agg["native_to_vector"].char_overlap.matched == 10
    assert agg["original_vector"].label_stats.label_count == 1


def test_char_order_puts_misread_chars_first(tmp_path):
    run = _make_run(tmp_path, "r", [_A], "HELLO WORLX")
    _meta_, entries = load_run(run, parser=None)
    agg = score_text(entries["pdf:A"], MetricConfig()).aggregate
    assert char_order({"r": agg}, "native_to_vector")[0] == "D"
