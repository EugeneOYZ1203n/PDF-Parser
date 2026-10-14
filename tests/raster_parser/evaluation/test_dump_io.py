"""Round-trip tests for raster_parser.evaluation.dump_io."""
from __future__ import annotations

import dataclasses
import json

from raster_parser.evaluation import dump_io


def test_write_load_dump(tmp_path, text, vector, page_meta):
    clip_items = (("c", (0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0)), ("l", (4.0, 0.0), (0.0, 0.0)))
    v = dataclasses.replace(vector(kind="l", bbox=(0.0, 0.0, 10.0, 5.0)), clips=((clip_items, True),))
    pd = dump_io.PageDump(
        page_meta=page_meta(index=2, width=300, height=400),
        texts=[text(text="Hello", bbox=(1.0, 2.0, 3.0, 4.0), direction=(0.0, 1.0), source="ocr")],
        vectors=[v], p2="Stub", p3="v1VectorClass",
        step_durations={"phase1": 0.5}, debug_durations={"conversion": 0.1},
    )
    path = tmp_path / "dump.json"
    dump_io.write_dump(path, "doc.pdf", [pd], {"layer_save": 0.2})

    dump = dump_io.load_dump(path)
    assert dump.pdf_path == "doc.pdf"
    assert dump.doc_durations == {"layer_save": 0.2}
    assert dump.pages == [pd]
    assert isinstance(dump.pages[0].texts[0].bbox, tuple)
    assert isinstance(dump.pages[0].vectors[0].clips[0][0][0][1], tuple)


def test_missing_optional_fields_default(tmp_path, page_meta):
    pd = dump_io.PageDump(page_meta=page_meta(), texts=[], vectors=[])
    path = tmp_path / "dump.json"
    dump_io.write_dump(path, "doc.pdf", [pd])
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("p2", "p3", "step_durations", "debug_durations"):
        payload["pages"][0].pop(key)
    payload.pop("doc_durations")
    path.write_text(json.dumps(payload), encoding="utf-8")

    dump = dump_io.load_dump(path)
    assert dump.pages[0].step_durations == {} and dump.pages[0].p2 == ""
    assert dump.doc_durations == {}
