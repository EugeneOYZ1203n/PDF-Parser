"""`StepClock` -- wall-clock timing of a backend's own named sub-steps.

A backend wraps each sub-step in `with clock("name"):`. Durations **accumulate** per name, so a
step run once per cluster inside a loop sums across iterations. The backend hands the totals to
`RunSettings.on_step_time` when it finishes (see `models/settings.py`); without that callback the
clock only fills its own dict, so backends can time unconditionally.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class StepClock:
    def __init__(self, sink: "dict[str, float] | None" = None) -> None:
        self.durations: "dict[str, float]" = sink if sink is not None else {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[name] = self.durations.get(name, 0.0) + (time.perf_counter() - start)
