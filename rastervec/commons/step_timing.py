"""`StepClock` -- wall-clock timing of a backend's own named sub-steps.

A P2/P3 backend that accepts a `step_durations: dict | None` kwarg wraps
each of its sub-steps in `with clock("name"):`; `core.pipeline` forwards
that dict and returns it as `PipelineResult.substep_durations`. Durations
**accumulate** per name, so a step run once per cluster inside a loop
(render / detect / recognize) sums across iterations. Debug-layer rendering
(`_emit(...)`) is meant to stay *outside* the timed blocks, so sub-steps
measure computation only -- the phase-level time minus their sum is that
overhead.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class StepClock:
    def __init__(self, sink: "dict[str, float] | None" = None) -> None:
        # A private dict when the caller didn't ask for timings, so backends
        # can time unconditionally.
        self.durations: "dict[str, float]" = sink if sink is not None else {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] = self.durations.get(name, 0.0) + (time.perf_counter() - start)
