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
