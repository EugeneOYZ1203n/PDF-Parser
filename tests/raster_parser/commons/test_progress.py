"""`commons/progress.py`: stage lines and bars, gated on the `raster_parser` logger's level."""
from __future__ import annotations

import logging
import re
import subprocess
import sys

import pytest

from raster_parser.commons import progress

_ROOT = logging.getLogger("raster_parser")


@pytest.fixture
def level():
    """Set the `raster_parser` logger's level for one test; restore it afterwards."""
    before = _ROOT.level
    yield _ROOT.setLevel
    _ROOT.setLevel(before)


def test_info_level_writes_nothing(level, capsys):
    level(logging.INFO)
    assert not progress.enabled()
    with progress.stage("quiet"):
        pass
    progress.note("quiet note")
    assert list(progress.bar(range(3), desc="quiet bar")) == [0, 1, 2]
    assert capsys.readouterr().err == ""


def test_debug_level_writes_start_and_end_lines(level, capsys):
    level(logging.DEBUG)
    assert progress.enabled()
    with progress.stage("p3 test › step"):
        pass
    err = capsys.readouterr().err.splitlines()
    assert re.fullmatch(r"\d\d:\d\d:\d\d  ▶ p3 test › step", err[0])
    assert re.fullmatch(r"\d\d:\d\d:\d\d  ✔ p3 test › step  \d+\.\d\ds", err[1])


def test_failing_stage_writes_a_cross_and_reraises(level, capsys):
    level(logging.DEBUG)
    with pytest.raises(RuntimeError, match="boom"):
        with progress.stage("broken"):
            raise RuntimeError("boom")
    assert "✖ broken" in capsys.readouterr().err


def test_bar_is_disabled_below_debug(level):
    level(logging.INFO)
    assert progress.bar(range(2), desc="x").disable
    level(logging.DEBUG)
    assert not progress.bar(range(2), desc="x").disable


def _enabled_in_fresh_process(before_import: str) -> str:
    code = f"import logging\n{before_import}\nfrom raster_parser.commons import progress\nprint(progress.enabled())"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()


def test_progress_is_on_by_default():
    assert _enabled_in_fresh_process("") == "True"


def test_a_level_set_before_import_is_kept():
    assert _enabled_in_fresh_process('logging.getLogger("raster_parser").setLevel(logging.INFO)') == "False"
