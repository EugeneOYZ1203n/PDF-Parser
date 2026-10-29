"""`dump.json` -- a pipeline report's reloadable record of one document's run.

`scripts/generate_pipeline_report.py` writes one `dump.json` per input document: for every page
processed, the final `Text` + `Vector` output (the same `ExtractResult` layout `extract()`
returns) and the timings. `scripts/pipeline_report_benchmark.py` reads it back and scores the
OCR text against ground truth without re-running the pipeline.

    {"schema": 1, "pdf_path": ..., "doc_durations": {...},
     "pages": [{"page": PageMeta, "texts": [...], "vectors": [...],
                "p2": ..., "p3": ..., "step_durations": {...}, "sub_durations": {...},
                "debug_durations": {...}}, ...]}

`sub_durations` (`{phase: {sub-step: seconds}}`) holds the sub-step times a backend reported
through `RunSettings.on_step_time`; they are already inside their phase's `step_durations`.

"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from raster_parser.models.result import ExtractResult

_SCHEMA = 1


@dataclasses.dataclass
class PageDump:
    # The page's output; the pipeline's seconds per phase are `result.step_durations`.
    result: ExtractResult
    p2: str = ""
    p3: str = ""
    # Report-generation seconds spent on this page outside the pipeline itself: input
    # conversion, the fixed layers, extra-prediction layers.
    debug_durations: dict = dataclasses.field(default_factory=dict)
    # Backend sub-step seconds per phase (`RunSettings.on_step_time`), part of their phase's
    # `result.step_durations`.
    sub_durations: dict = dataclasses.field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        # `ExtractResult` equality leaves out `step_durations`; a dump compares them too.
        return isinstance(other, PageDump) and (
            self.result,
            self.result.step_durations,
            self.p2,
            self.p3,
            self.debug_durations,
            self.sub_durations,
        ) == (
            other.result,
            other.result.step_durations,
            other.p2,
            other.p3,
            other.debug_durations,
            other.sub_durations,
        )


@dataclasses.dataclass
class Dump:
    pdf_path: str
    pages: list[PageDump]
    # Report-generation seconds for the whole document (label overlays, saving the layer PDFs).
    doc_durations: dict = dataclasses.field(default_factory=dict)


def _page_to_json(pd: PageDump) -> dict:
    return {
        **pd.result.to_dict(),
        "p2": pd.p2,
        "p3": pd.p3,
        "step_durations": pd.result.step_durations,
        "sub_durations": pd.sub_durations,
        "debug_durations": pd.debug_durations,
    }


def _page_from_json(d: dict) -> PageDump:
    result = ExtractResult.from_dict(d)
    result.step_durations = dict(d.get("step_durations", {}))
    return PageDump(
        result=result,
        p2=d.get("p2", ""),
        p3=d.get("p3", ""),
        debug_durations=d.get("debug_durations", {}),
        sub_durations=dict(d.get("sub_durations", {})),
    )


def write_dump(
    path: str | Path,
    pdf_path: str,
    pages: list[PageDump],
    doc_durations: "dict | None" = None,
) -> None:
    payload = {
        "schema": _SCHEMA,
        "pdf_path": str(pdf_path),
        "pages": [_page_to_json(p) for p in pages],
        "doc_durations": dict(doc_durations or {}),
    }
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_dump(path: str | Path) -> Dump:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Dump(
        pdf_path=payload.get("pdf_path", ""),
        pages=[_page_from_json(p) for p in payload.get("pages", [])],
        doc_durations=payload.get("doc_durations", {}),
    )
