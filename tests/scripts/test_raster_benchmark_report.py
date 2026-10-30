"""End to end: `raster_benchmark: true` rasterises each benchmark page, runs the raster pass on
it, writes the raster ground truth next to the run, and the benchmark scores it (Liu & Dori)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.slow


def test_raster_benchmark_run_and_score(tmp_path):
    cfg = {
        "p2": "stub",
        "p3": "stub",
        "benchmark": True,
        "raster_benchmark": True,
        "raster_dpi": 100,
        "input_files": [str(ROOT / "tests/references/test_pdfs_1.pdf")],
        "pages": [0],
        "debug_images": False,
        "debug_layers": False,
        "output_root": str(tmp_path / "runs"),
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_pipeline_report.py",
            "--config",
            str(tmp_path / "cfg.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    (run,) = (tmp_path / "runs").iterdir()
    raster = run / "test_pdfs_1" / "raster"
    gt = json.loads((raster / "ground_truth_vector_to_raster.json").read_text(encoding="utf-8"))
    assert gt["pages"][0]["page_index"] == 0 and gt["pages"][0]["vectors"]
    assert (raster / "dump.json").is_file()

    subprocess.run(
        [
            sys.executable,
            "scripts/pipeline_report_benchmark.py",
            "--run",
            str(run),
            "--out-root",
            str(tmp_path / "bench"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    (bench,) = (tmp_path / "bench").iterdir()
    txt = (bench / "benchmark.txt").read_text(encoding="utf-8")
    assert "raster vectorization" in txt and "D_v" in txt
    assert "Raster vectorization (Liu &amp; Dori)" in (bench / "report.html").read_text(
        encoding="utf-8"
    )
