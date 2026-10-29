"""`label_schema`: a label file holds hand-labelled original vector text, one entry per word or
line, tied to its `Vector`s by `path_signature` and convertible to / from a `Text`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raster_parser.evaluation.labelling.label_schema import (
    SCHEMA_VERSION,
    LabelEntry,
    LabelSet,
    load_labels,
    path_signature,
    save_labels,
    vector_signatures_for,
)

V1_FORMAT = (
    Path(__file__).resolve().parents[3] / "references" / "labels" / "test_pdfs_1_v1_format.json"
)


def test_path_signature_stable_across_rebuilds(vector):
    a = vector(bbox=(1, 2, 3, 4), seqno=7, items=[("l", (1, 2), (3, 4))])
    b = vector(bbox=(1, 2, 3, 4), seqno=7, items=[("l", (1, 2), (3, 4))])
    assert path_signature(a) == path_signature(b)


def test_path_signature_differs_for_different_vectors(vector):
    a = vector(bbox=(1, 2, 3, 4), seqno=7)
    assert path_signature(a) != path_signature(vector(bbox=(1, 2, 3, 4), seqno=8))
    assert path_signature(a) != path_signature(vector(bbox=(5, 6, 7, 8), seqno=7))
    assert path_signature(a) != path_signature(
        vector(bbox=(1, 2, 3, 4), seqno=7, color=(1.0, 0.0, 0.0))
    )


def test_save_and_load_round_trip(tmp_path):
    labels = LabelSet(
        pdf_path="foo.pdf",
        entries=[
            LabelEntry(
                page_index=0,
                label_id="abc",
                text="Hello",
                bbox=(0, 0, 10, 10),
                expected_rotation=90,
                vector_signatures=["abc123", "def456"],
            ),
        ],
    )
    out = tmp_path / "labels.json"
    save_labels(labels, str(out))
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION == 2
    assert set(raw) == {"schema_version", "pdf_path", "entries"}
    assert set(raw["entries"][0]) == {
        "page_index",
        "label_id",
        "text",
        "bbox",
        "expected_rotation",
        "vector_signatures",
        "source",
    }
    assert load_labels(str(out)) == labels
    assert labels.entries[0].source == "vector"


def test_loads_a_version_1_file(tmp_path):
    labels = load_labels(str(V1_FORMAT))
    assert labels.schema_version == SCHEMA_VERSION
    assert [e.text for e in labels.entries] == ["Beta12", "Gamma-X", "Alpha"]
    assert labels.entries[0].bbox == pytest.approx((306.342, 169.146, 323.742, 209.276), abs=1e-3)
    assert labels.entries[0].expected_rotation == 90
    assert len(labels.entries[0].vector_signatures) == 6


def test_version_1_raster_and_cad_font_entries_are_dropped(tmp_path):
    raw = json.loads(V1_FORMAT.read_text(encoding="utf-8"))
    raw["entries"][0]["source"] = "raster"
    raw["entries"][1]["source"] = "cad_font"
    raw["geometry_entries"] = [{"page_index": 0, "kind": "l", "points": [[0, 0], [1, 1]]}]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert [e.text for e in load_labels(str(path)).entries] == ["Alpha"]


def test_entry_from_text_and_back(text, vector):
    a, b = vector(seqno=1), vector(seqno=2, bbox=(30, 30, 40, 40))
    t = text(text="M12", bbox=(1.0, 2.0, 30.0, 9.0), direction=(0.0, -1.0), page_index=3)
    entry = LabelEntry.from_text(t, [b, a])
    assert (entry.page_index, entry.text, entry.bbox, entry.expected_rotation) == (
        3,
        "M12",
        (1.0, 2.0, 30.0, 9.0),
        270,
    )
    assert entry.vector_signatures == vector_signatures_for([a, b]) and entry.source == "vector"
    assert entry.label_id
    back = entry.to_text()
    assert (back.text, back.bbox, back.page_index) == ("M12", (1.0, 2.0, 30.0, 9.0), 3)
    assert back.direction == pytest.approx((0.0, -1.0))
    assert back.angle() % 360 == pytest.approx(270.0)
