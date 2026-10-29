"""Report output guardrail: a benchmark run of the fixture config (`generate_pipeline_report.py`
then `pipeline_report_benchmark.py`) keeps the same scores, ground truth and HTML sections as the
stored snapshot. Timings, timestamps and absolute paths are left out.

Regenerate the snapshot on purpose with `REPORT_SNAPSHOT_WRITE=1` (and say why in the commit).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "tests" / "references" / "report_snapshot"

pytestmark = pytest.mark.slow


def _text_sections(benchmark_txt: str) -> list[str]:
    """`benchmark.txt` without its timing tables and run paths/timestamps."""
    out, skip = [], False
    for line in benchmark_txt.splitlines():
        if line.startswith("[") and "wall-clock" in line:
            skip = True
            continue
        if skip and (not line.strip() or line.startswith(("  ", "-"))):
            continue
        skip = False
        if line.startswith("compared runs") or re.match(r"\s+\d{8}_\d{6}__", line):
            continue
        out.append(re.sub(r"\d{8}_\d{6}__", "<ts>__", line))
    return out


def _headings(report_html: str) -> list[str]:
    return [
        re.sub(r"\d{8}_\d{6}", "<ts>", h) for h in re.findall(r"<h[1-4][^>]*>([^<]*)", report_html)
    ]


def _ground_truth(doc_dir: Path) -> dict:
    out = {}
    for path in sorted(doc_dir.glob("ground_truth_*.json")):
        entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
        out[path.stem] = sorted(
            (
                e["page_index"],
                e["text"],
                [round(c, 2) for c in e.get("bbox", e.get("cluster_bbox"))],
                e["expected_rotation"],
            )
            for e in entries
        )
    return out


def _snapshot(tmp_path: Path) -> dict:
    run_root, bench_root = tmp_path / "runs", tmp_path / "bench"
    cfg = json.loads((SNAP / "config.json").read_text(encoding="utf-8"))
    cfg["output_root"] = str(run_root)
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_pipeline_report.py",
            "--config",
            str(tmp_path / "config.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    (run,) = [p for p in run_root.iterdir() if p.is_dir()]
    subprocess.run(
        [
            sys.executable,
            "scripts/pipeline_report_benchmark.py",
            "--run",
            str(run),
            "--out-root",
            str(bench_root),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    (bench,) = [p for p in bench_root.iterdir() if p.is_dir()]
    return {
        "benchmark_txt": _text_sections((bench / "benchmark.txt").read_text(encoding="utf-8")),
        "report_headings": _headings((bench / "report.html").read_text(encoding="utf-8")),
        "ground_truth": {d.name: _ground_truth(d) for d in sorted(run.iterdir()) if d.is_dir()},
    }


def test_benchmark_report_matches_snapshot(tmp_path):
    got = json.loads(json.dumps(_snapshot(tmp_path)))
    path = SNAP / "snapshot.json"
    if os.environ.get("REPORT_SNAPSHOT_WRITE"):
        path.write_text(json.dumps(got, indent=1), encoding="utf-8")
    want = json.loads(path.read_text(encoding="utf-8"))
    for key in want:
        assert got[key] == want[key], key
