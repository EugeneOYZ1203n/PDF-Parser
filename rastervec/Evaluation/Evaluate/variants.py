"""Named pipeline variants for the benchmark.

The benchmark scores the current pipeline against the archive/legacy
pipeline. Each is a named `PipelineVariant`; `benchmark.py --variants` and
the notebook's `VARIANTS_TO_RUN` select from `VARIANTS` by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PipelineVariant:
    """One benchmarked pipeline configuration.

    `engine="current"` runs `rastervec.pipelines.current.run_pipeline` with
    `enable_fast`. `engine="legacy"` runs `archive/raster_parser` unchanged
    via `legacy_adapter` (`enable_fast` ignored there).
    """

    name: str
    engine: Literal["current", "legacy"]
    enable_fast: bool = True


VARIANTS: dict[str, PipelineVariant] = {
    "current": PipelineVariant("current", "current", True),
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
