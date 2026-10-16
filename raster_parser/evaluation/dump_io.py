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

from raster_parser.commons.models import PageMeta, Text, Vector
from raster_parser.commons.models.result import ExtractResult

_SCHEMA = 1


@dataclasses.dataclass
class PageDump:
    page_meta: PageMeta
    texts: list[Text]
    vectors: list[Vector]
    p2: str = ""
    p3: str = ""
    # Pipeline seconds per phase (`PipelineResult.step_durations`).
    step_durations: dict = dataclasses.field(default_factory=dict)
    # Report-generation seconds spent on this page outside the pipeline itself: input
    # conversion, the fixed layers, extra-prediction layers.
    debug_durations: dict = dataclasses.field(default_factory=dict)
    # Backend sub-step seconds per phase (`RunSettings.on_step_time`), part of their phase's
    # `step_durations` above. Last, so positional construction is unchanged.
    sub_durations: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Dump:
    pdf_path: str
    pages: list[PageDump]
    # Report-generation seconds for the whole document (label overlays, saving the layer PDFs).
    doc_durations: dict = dataclasses.field(default_factory=dict)


def _page_to_json(pd: PageDump) -> dict:
    return {
        **ExtractResult(pd.page_meta, pd.texts, pd.vectors).to_dict(),
        "p2": pd.p2,
        "p3": pd.p3,
        "step_durations": pd.step_durations,
        "sub_durations": pd.sub_durations,
        "debug_durations": pd.debug_durations,
    }


def _page_from_json(d: dict) -> PageDump:
    result = ExtractResult.from_dict(d)
    return PageDump(
        page_meta=result.page, texts=result.texts, vectors=result.vectors,
        p2=d.get("p2", ""), p3=d.get("p3", ""),
        step_durations=d.get("step_durations", {}),
        sub_durations=d.get("sub_durations", {}),
        debug_durations=d.get("debug_durations", {}),
    )


def write_dump(
    path: str | Path, pdf_path: str, pages: list[PageDump], doc_durations: "dict | None" = None,
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
