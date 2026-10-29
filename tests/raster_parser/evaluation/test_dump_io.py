"""Round-trip tests for raster_parser.evaluation.dump_io."""

from __future__ import annotations

import dataclasses
import json

from raster_parser.models.result import ExtractResult
from raster_parser.evaluation import dump_io


def _page(page_meta, texts=(), vectors=(), **durations) -> ExtractResult:
    return ExtractResult(
        page=page_meta, texts=list(texts), vectors=list(vectors), step_durations=durations
    )


def test_write_load_dump(tmp_path, text, vector, page_meta):
    clip_items = (
        ("c", (0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0)),
        ("l", (4.0, 0.0), (0.0, 0.0)),
    )
    v = dataclasses.replace(
        vector(kind="l", bbox=(0.0, 0.0, 10.0, 5.0)), clips=((clip_items, True),)
    )
    pd = dump_io.PageDump(
        result=_page(
            page_meta(index=2, width=300, height=400),
            [text(text="Hello", bbox=(1.0, 2.0, 3.0, 4.0), direction=(0.0, 1.0), source="ocr")],
            [v],
            native_read=0.5,
        ),
        p2="stub",
        p3="v1_vector_class",
        debug_durations={"conversion": 0.1},
    )
    path = tmp_path / "dump.json"
    dump_io.write_dump(path, "doc.pdf", [pd], {"layer_save": 0.2})

    dump = dump_io.load_dump(path)
    assert dump.pdf_path == "doc.pdf"
    assert dump.doc_durations == {"layer_save": 0.2}
    assert dump.pages == [pd]
    assert dump.pages[0].result.step_durations == {"native_read": 0.5}
    assert isinstance(dump.pages[0].result.texts[0].bbox, tuple)
    assert isinstance(dump.pages[0].result.vectors[0].clips[0][0][0][1], tuple)


def test_page_json_layout_is_unchanged(tmp_path, page_meta):
    pd = dump_io.PageDump(result=_page(page_meta(), native_read=0.5), p2="stub", p3="stub")
    path = tmp_path / "dump.json"
    dump_io.write_dump(path, "doc.pdf", [pd])
    page = json.loads(path.read_text(encoding="utf-8"))["pages"][0]
    assert list(page) == [
        "page",
        "texts",
        "vectors",
        "p2",
        "p3",
        "step_durations",
        "debug_durations",
    ]
    assert page["step_durations"] == {"native_read": 0.5}


def test_missing_optional_fields_default(tmp_path, page_meta):
    pd = dump_io.PageDump(result=_page(page_meta()))
    path = tmp_path / "dump.json"
    dump_io.write_dump(path, "doc.pdf", [pd])
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("p2", "p3", "step_durations", "debug_durations"):
        payload["pages"][0].pop(key)
    payload.pop("doc_durations")
    path.write_text(json.dumps(payload), encoding="utf-8")

    dump = dump_io.load_dump(path)
    assert dump.pages[0].result.step_durations == {} and dump.pages[0].p2 == ""
    assert dump.doc_durations == {}
