"""Canonical filesystem locations for rastervec outputs.

Every script / notebook that writes files sends them under one top-level
``outputs/`` tree, one subfolder per source (``outputs/benchmark_notebook/``,
``outputs/labels/``, ...), so generated artifacts never litter the repo root
and are all ignored by one ``.gitignore`` rule.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"


def output_dir(name: str, *subparts: str, create: bool = True) -> Path:
    """``outputs/<name>/<subparts...>``, created (parents=True) unless create=False."""
    d = OUTPUTS_DIR.joinpath(name, *subparts)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d
