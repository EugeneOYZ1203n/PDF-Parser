"""Tests for rastervec.paths."""
from __future__ import annotations

import rastervec.paths as paths


def test_repo_root_holds_the_package():
    assert (paths.REPO_ROOT / "rastervec" / "paths.py").is_file()
    assert paths.OUTPUTS_DIR == paths.REPO_ROOT / "outputs"


def test_output_dir_no_create_does_not_touch_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path / "outputs")
    d = paths.output_dir("thing", "sub", create=False)
    assert d == tmp_path / "outputs" / "thing" / "sub"
    assert not d.exists()


def test_output_dir_creates_nested(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path / "outputs")
    d = paths.output_dir("benchmark_notebook", "reconstructions")
    assert d.is_dir()
    assert d == tmp_path / "outputs" / "benchmark_notebook" / "reconstructions"
