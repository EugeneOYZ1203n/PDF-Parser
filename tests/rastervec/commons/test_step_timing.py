"""commons.step_timing.StepClock."""
from __future__ import annotations

import pytest

from rastervec.commons.step_timing import StepClock


def test_step_clock_writes_into_sink_and_accumulates():
    sink: dict = {}
    clock = StepClock(sink)
    for _ in range(3):
        with clock("loop"):
            pass
    with clock("once"):
        pass
    assert set(sink) == {"loop", "once"}
    assert sink["loop"] >= 0.0


def test_step_clock_without_sink_uses_private_dict():
    clock = StepClock(None)
    with clock("x"):
        pass
    assert "x" in clock.durations


def test_step_clock_records_even_when_step_raises():
    sink: dict = {}
    clock = StepClock(sink)
    with pytest.raises(ValueError):
        with clock("boom"):
            raise ValueError
    assert "boom" in sink
