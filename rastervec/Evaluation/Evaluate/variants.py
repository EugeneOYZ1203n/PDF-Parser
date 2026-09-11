"""Named pipeline variants for the benchmark.

The benchmark scores a few pipeline configurations against the same ground
truth -- the current pipeline (FAST on / off), the `fast_first` pipeline
(FAST on / off), and the archive/legacy pipeline. Each is a named
`PipelineVariant`; `benchmark.py --variants` and the notebook's
`VARIANTS_TO_RUN` select from `VARIANTS` by name.

Adding an ablation = one entry in `VARIANTS`:
`Reader/Parallel/benchmark_jobs.run_page_task` reads it and threads
`enable_fast` into `rastervec.pipelines.current.run_pipeline`. Note:
`benchmark_jobs.run_page_task` currently only dispatches `"current"`/
`"legacy"` engines -- a `"fast_first"` variant here is usable by
`scripts/generate_pipeline_report.py` today; wiring it into the parallel
multi-page benchmark harness too is a separate, not-yet-done step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PipelineVariant:
    """One benchmarked pipeline configuration.

    `engine="current"` runs `rastervec.pipelines.current.run_pipeline` with
    `enable_fast`. `engine="fast_first"` runs `rastervec.pipelines.
    fast_first.run_pipeline` with `enable_fast`. `engine="legacy"` runs
    `archive/raster_parser` unchanged via `legacy_adapter` (`enable_fast`
    ignored there).
    """

    name: str
    engine: Literal["current", "legacy", "fast_first"]
    enable_fast: bool = True


VARIANTS: dict[str, PipelineVariant] = {
    "current": PipelineVariant("current", "current", True),
    "current_nofast": PipelineVariant("current_nofast", "current", False),
    "fast_first": PipelineVariant("fast_first", "fast_first", True),
    "fast_first_nofast": PipelineVariant("fast_first_nofast", "fast_first", False),
    "legacy": PipelineVariant("legacy", "legacy"),
}

# Default selection for the CLI / notebook when none is given.
DEFAULT_VARIANTS = ["current", "legacy"]


def resolve_variant(name: str) -> PipelineVariant:
    """`VARIANTS[name]` with a clear error listing the valid names."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown pipeline variant {name!r}; must be one of {sorted(VARIANTS)}"
        ) from None
