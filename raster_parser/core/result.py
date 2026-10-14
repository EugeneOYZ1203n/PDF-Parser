"""`PipelineResult` -- what `core.pipeline.run_pipeline` returns for one page."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from raster_parser.commons.models import Page, Text, Vector


@dataclass
class PipelineResult:
    page: "Page"  # the PDF is closed again: `page.fitz_page` is unusable, see `open_page()`
    texts: "list[Text]"  # Phase 4 output: native + Phase 3 + Phase 2 text
    vectors: "list[Vector]"  # Phase 4 output: Phase 3 + Phase 2 vectors
    p2: str  # the P2_REGISTRY name that ran ("Stub" when the raster pass was skipped)
    p3: str  # the P3_REGISTRY name that ran
    step_durations: dict[str, float] = field(default_factory=dict)  # seconds per phase

    @contextmanager
    def open_page(self) -> "Iterator[Page]":
        """Reopen the source PDF and yield a live `Page` for this run's page."""
        from raster_parser.P1_Reading_Native.reader import Reader

        with Reader(self.page.doc_path) as reader:
            yield reader.get_page(self.page.meta.index)
