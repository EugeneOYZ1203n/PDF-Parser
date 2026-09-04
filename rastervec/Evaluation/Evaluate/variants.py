"""Named pipeline variants for the benchmark.

The benchmark scores several pipeline configurations against the same
ground truth -- the current pipeline with the light vs. heavy OCR backend,
with FAST on vs. off, and the archive/legacy pipeline. Each is a named
`PipelineVariant` here; `benchmark.py --variants` and the benchmark
notebook's `VARIANTS_TO_RUN` both select from `VARIANTS` by name.

Adding an ablation = one entry in `VARIANTS` (no other file changes):
`Reader/Parallel/benchmark_jobs.run_page_task` reads the entry and threads
`enable_fast` / `ocr_backend` into `pipeline.run_page_context`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PipelineVariant:
    """One benchmarked pipeline configuration.

    `engine="current"` runs `pipeline.run_page_context` with `enable_fast`
    and the chosen OCR backend (`"light"` -> `LightPaddleOcrBackend`,
    `"heavy"` -> `RenderOCR`'s default `PaddleOcrBackend`).
    `engine="legacy"` runs `archive/raster_parser` unchanged via
    `legacy_adapter` -- `enable_fast` / `ocr_backend` are ignored there.
    """

    name: str
    engine: Literal["current", "legacy"]
    enable_fast: bool = True
    ocr_backend: Literal["light", "heavy"] = "light"


VARIANTS: dict[str, PipelineVariant] = {
    "current_light": PipelineVariant("current_light", "current", True, "light"),
    "current_heavy": PipelineVariant("current_heavy", "current", True, "heavy"),
    "current_light_nofast": PipelineVariant("current_light_nofast", "current", False, "light"),
    "current_heavy_nofast": PipelineVariant("current_heavy_nofast", "current", False, "heavy"),
    "legacy": PipelineVariant("legacy", "legacy"),
}

# Default selection for the CLI / notebook when none is given.
DEFAULT_VARIANTS = ["current_heavy", "current_light", "legacy"]


def resolve_variant(name: str) -> PipelineVariant:
    """`VARIANTS[name]` with a clear error listing the valid names."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown pipeline variant {name!r}; must be one of {sorted(VARIANTS)}"
        ) from None
