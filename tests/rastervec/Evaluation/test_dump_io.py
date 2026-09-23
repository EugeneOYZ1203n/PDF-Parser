"""Round-trip tests for rastervec.Evaluation.dump_io."""
from __future__ import annotations

from rastervec.Evaluation import dump_io


def test_text_round_trip(text):
    t = text(text="Hello world", bbox=(1.0, 2.0, 3.0, 4.0), direction=(0.0, 1.0), source="ocr")
    back = dump_io.text_from_json(dump_io.text_to_json(t))
    assert back == t
    assert isinstance(back.bbox, tuple) and isinstance(back.direction, tuple)


def test_vector_round_trip(vector):
    v = vector(kind="l", bbox=(0.0, 0.0, 10.0, 5.0), color=(0.1, 0.2, 0.3), layer="L1", seqno=7)
    back = dump_io.vector_from_json(dump_io.vector_to_json(v))
    assert back == v
    assert isinstance(back.items[0], tuple)
    assert isinstance(back.color, tuple)


def test_write_load_dump(tmp_path, text, vector, page_meta):
    pd = dump_io.PageDump(
        page_meta=page_meta(index=2, width=300, height=400),
        texts=[text(source="native"), text(text="ocr'd", source="ocr")],
        vectors=[vector(seqno=1), vector(seqno=2)],
        engine="current",
        step_durations={"read": 0.1, "native": 0.2},
    )
    path = tmp_path / "dump.json"
    dump_io.write_dump(path, "some/input.pdf", [pd])

    loaded = dump_io.load_dump(path)
    assert loaded.pdf_path == "some/input.pdf"
    assert len(loaded.pages) == 1
    got = loaded.pages[0]
    assert got.page_meta == pd.page_meta
    assert got.texts == pd.texts
    assert got.vectors == pd.vectors
    assert got.engine == "current"
    assert got.step_durations == {"read": 0.1, "native": 0.2}


def test_dump_timing_fields_roundtrip_and_default(tmp_path, page_meta):
    pd = dump_io.PageDump(
        page_meta=page_meta(), texts=[], vectors=[], engine="current",
        step_durations={"phase3": 2.0}, substep_durations={"ocr": 1.5},
        raster_step_durations={"phase3": 1.0}, raster_substep_durations={"ocr": 0.5},
    )
    path = tmp_path / "dump.json"
    dump_io.write_dump(path, "x.pdf", [pd])
    got = dump_io.load_dump(path).pages[0]
    assert got.substep_durations == {"ocr": 1.5}
    assert got.raster_step_durations == {"phase3": 1.0}
    assert got.raster_substep_durations == {"ocr": 0.5}

    # an older dump without the timing fields still loads
    import json
    raw = json.loads(path.read_text(encoding="utf-8"))
    for page in raw["pages"]:
        for k in ("substep_durations", "raster_step_durations", "raster_substep_durations"):
            page.pop(k)
    path.write_text(json.dumps(raw), encoding="utf-8")
    old = dump_io.load_dump(path).pages[0]
    assert old.substep_durations == {} and old.raster_step_durations == {}
