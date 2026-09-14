"""Named pipeline variants for the benchmark.

The benchmark scores the new pluggable pipeline (`core.pipeline.run_pipeline`,
`p2`/`p3` backend choice) against the archive/legacy pipeline. Each is a
named `PipelineVariant`; `benchmark.py --variants`, `generate_pipeline_
report.py`'s `ReportConfig`, and the notebook's `VARIANTS_TO_RUN` select
from `VARIANTS` by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rastervec.core.registry import DEFAULT_P2, DEFAULT_P3


@dataclass(frozen=True)
class PipelineVariant:
    """One benchmarked pipeline configuration.

    `engine="current"` runs `rastervec.core.pipeline.run_pipeline` with the
    given `p2`/`p3` backend names (`enable_fast` forwarded to whichever `p3`
    backend accepts it). `engine="legacy"` runs `archive/raster_parser`
    unchanged via `legacy_adapter` (`p2`/`p3`/`enable_fast` all ignored
    there -- legacy is a whole separate historical pipeline, not a P2/P3
    backend).
    """

    name: str
    engine: Literal["current", "legacy"]
    p2: str = DEFAULT_P2
    p3: str = DEFAULT_P3
    enable_fast: bool = True


VARIANTS: dict[str, PipelineVariant] = {
    "current": PipelineVariant("current", "current", DEFAULT_P2, DEFAULT_P3, True),
    "legacy": PipelineVariant("legacy", "legacy"),
    # Named presets for benchmark comparisons across P3 backends (all with
    # the default Stub P2, since none of them consume raster-to-vector
    # output today).
    "current_vectorclassification": PipelineVariant(
        "current_vectorclassification", "current", DEFAULT_P2, "VectorClassification", True,
    ),
    "current_legacyrecreation": PipelineVariant(
        "current_legacyrecreation", "current", DEFAULT_P2, "LegacyRecreation", True,
    ),
    "current_junction": PipelineVariant(
        "current_junction", "current", "Junction", DEFAULT_P3, True,
    ),
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
