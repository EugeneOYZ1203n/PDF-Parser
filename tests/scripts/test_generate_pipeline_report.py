"""ReportConfig validation for scripts/generate_pipeline_report.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

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


class _Variant:
    def __init__(self, engine: str) -> None:
        self.engine = engine


def test_active_artifacts_includes_paddle_detect_for_current_engine():
    """Test branch (`test/paddle-detect-post-fast`): the old segment/
    similarity/restore-gated rows are gone from `_ARTIFACTS`; one
    `paddle_detect` row replaces them."""
    active = gpr._active_artifacts(gpr.ReportConfig(input_files=["a.pdf"]), _Variant("current"))
    stems = [row[0] for row in active]
    assert "paddle_detect" in stems
    assert "segmentation" not in stems
    assert "similarity" not in stems
    assert "paddle_ocr" not in stems


def test_active_artifacts_respects_final_stage_gate():
    cfg = gpr.ReportConfig(input_files=["a.pdf"], final_stage="fast")
    stems = [row[0] for row in gpr._active_artifacts(cfg, _Variant("current"))]
    assert "fast_heatmap" in stems
    assert "paddle_detect" not in stems
    assert "drawing_vectors" not in stems


def test_active_artifacts_legacy_only_reconstructed():
    active = gpr._active_artifacts(gpr.ReportConfig(input_files=["a.pdf"]), _Variant("legacy"))
    assert [row[0] for row in active] == ["reconstructed"]


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
