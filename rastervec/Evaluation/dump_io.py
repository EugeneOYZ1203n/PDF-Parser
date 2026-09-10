"""JSON dump / reload of a pipeline run's final `Text` + `Vector` output.

`generate_pipeline_report.py` writes one `dump.json` per source PDF holding
every processed page's `res.texts` (native + restored OCR) and `res.vectors`
(drawing output). `pipeline_report_benchmark.py` reads it back into real
`Text`/`Vector` dataclasses and scores the OCR text against ground truth
without re-running the pipeline.

`Text`/`Vector` are plain dataclasses, so `dataclasses.asdict` + a
tuple-coercion pass on reload is a clean, human-readable round trip --
`Text.to_pymupdf` is native-only and would lose every OCR reading, so it is
deliberately not used here.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from rastervec.models import PageMeta, Text, Vector

_SCHEMA = 1


def _to_tuples(obj):
    if isinstance(obj, list):
        return tuple(_to_tuples(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_tuples(v) for k, v in obj.items()}
    return obj


def text_to_json(t: Text) -> dict:
    return dataclasses.asdict(t)


def text_from_json(d: dict) -> Text:
    d = dict(d)
    for key in ("bbox", "direction", "origin"):
        if d.get(key) is not None:
            d[key] = tuple(d[key])
    if d.get("raw_word") is not None:
        d["raw_word"] = tuple(d["raw_word"])
    # raw_span stays a plain dict (its nested tuples become lists in JSON;
    # nothing downstream of a reload depends on their exact type).
    field_names = {f.name for f in dataclasses.fields(Text)}
    return Text(**{k: v for k, v in d.items() if k in field_names})


def vector_to_json(v: Vector) -> dict:
    return dataclasses.asdict(v)


def vector_from_json(d: dict) -> Vector:
    d = dict(d)
    for key in ("color", "fill"):
        if d.get(key) is not None:
            d[key] = tuple(d[key])
    for key in ("rect", "scissor"):
        if d.get(key) is not None:
            d[key] = tuple(d[key])
    if d.get("items") is not None:
        d["items"] = [_to_tuples(item) for item in d["items"]]
    field_names = {f.name for f in dataclasses.fields(Vector)}
    return Vector(**{k: v for k, v in d.items() if k in field_names})


@dataclasses.dataclass
class PageDump:
    page_meta: PageMeta
    texts: list[Text]
    vectors: list[Vector]
    engine: str
    step_durations: dict


def _page_dump_to_json(pd: PageDump) -> dict:
    return {
        "page_meta": dataclasses.asdict(pd.page_meta),
        "texts": [text_to_json(t) for t in pd.texts],
        "vectors": [vector_to_json(v) for v in pd.vectors],
        "engine": pd.engine,
        "step_durations": pd.step_durations,
    }


def _page_dump_from_json(d: dict) -> PageDump:
    meta = d["page_meta"]
    return PageDump(
        page_meta=PageMeta(
            index=meta["index"], number=meta["number"],
            mediabox=tuple(meta["mediabox"]), rotation=meta["rotation"],
            width=meta["width"], height=meta["height"],
        ),
        texts=[text_from_json(t) for t in d["texts"]],
        vectors=[vector_from_json(v) for v in d["vectors"]],
        engine=d.get("engine", "current"),
        step_durations=d.get("step_durations", {}),
    )


def write_dump(path: str | Path, pdf_path: str, pages: list[PageDump]) -> None:
    payload = {
        "schema": _SCHEMA,
        "pdf_path": str(pdf_path),
        "pages": [_page_dump_to_json(p) for p in pages],
    }
    Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")


@dataclasses.dataclass
class Dump:
    pdf_path: str
    pages: list[PageDump]


def load_dump(path: str | Path) -> Dump:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Dump(
        pdf_path=payload.get("pdf_path", ""),
        pages=[_page_dump_from_json(p) for p in payload.get("pages", [])],
    )
