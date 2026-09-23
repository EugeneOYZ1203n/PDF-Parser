"""`PipelineResult` -- the one value `core.pipeline.run_pipeline` returns,
regardless of which P2/P3 backend combination ran. Final-output fields are
always populated; `extra` carries whatever verbose intermediates the chosen
backends produced (their shapes genuinely differ per backend now that P3
has three interchangeable implementations, so this stays a generic bucket
rather than one fixed schema of named Optional fields).

NOTE: the pipeline closes its `Reader` before returning, so `page.fitz_page`
is `None` on a returned result -- use `PipelineResult.open_page()` to reopen
the source PDF and get a live `fitz.Page` for rasterization."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from rastervec.commons.models import Page, Text, Vector


@dataclass
class StepOutcome:
    name: str
    status: str  # "ok" | "error"
    error: str | None = None
    duration_seconds: float | None = None


@dataclass
class PipelineResult:
    page: "Page"  # `page.fitz_page` is None -- use `open_page()` for a live one
    texts: "list[Text]"  # phase1 native + phase2 + phase3 OCR text, flat
    vectors: "list[Vector]"  # final drawing content, flat
    step_durations: dict
    p2: str  # the P2_REGISTRY name that ran
    p3: str  # the P3_REGISTRY name that ran

    # verbose-only: whatever intermediates the phase1/p2/p3 calls produced,
    # keyed by phase ("phase1" / "phase2" / "phase3") -- shape is
    # backend-specific, see that backend's own module for what it puts here.
    extra: dict = field(default_factory=dict)
    step_outputs: dict | None = None
    # The P3 backend's own named sub-step wall-clock seconds (only for a
    # backend that accepts `step_durations`; `{}` otherwise). Kept separate
    # from `step_durations` (phase-level only) so summing that stays a
    # correct per-page total.
    substep_durations: dict = field(default_factory=dict)

    @contextmanager
    def open_page(self) -> "Iterator[Page]":
        """Reopen the source PDF and yield a live `models.Page` for this
        run's page."""
        from rastervec.P1_Reading_Native.reader import Reader

        with Reader(self.page.doc_path) as reader:
            yield reader.get_page(self.page.meta.index)
