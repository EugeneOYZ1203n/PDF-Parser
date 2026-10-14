"""JSON round trip of `Text`, `Vector` and `ExtractResult` (`to_dict` / `from_dict`)."""
from __future__ import annotations

import json
from pathlib import Path

from raster_parser.commons.models import ExtractResult, Text, Vector
from raster_parser.P1_Reading_Native.phase1 import read_and_extract

REFERENCES = Path(__file__).resolve().parents[3] / "references"


def _round_trip(obj):
    return type(obj).from_dict(json.loads(json.dumps(obj.to_dict())))


def test_vector_and_text_round_trip_on_real_extraction() -> None:
    for pdf in sorted(REFERENCES.glob("test_pdfs_*.pdf")):
        phase1 = read_and_extract(str(pdf), 0)
        assert phase1.vectors and phase1.texts
        for v in phase1.vectors:
            assert _round_trip(v) == v
        for t in phase1.texts:
            assert _round_trip(t) == t


def test_every_item_kind_and_clip_round_trips(vector) -> None:
    v = vector(kind="l", items=[
        ("l", (0.0, 0.0), (1.0, 1.0)),
        ("c", (0.0, 0.0), (1.0, 2.0), (2.0, 2.0), (3.0, 0.0)),
        ("re", (0.0, 0.0, 5.0, 5.0), 1),
        ("qu", ((0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0))),
    ], scissor=(0.0, 0.0, 9.0, 9.0), fill=(1.0, 0.0, 0.0))
    v.clips = (((("l", (0.0, 0.0), (1.0, 1.0)),), True),)
    assert _round_trip(v) == v


def test_extract_result_json_round_trip(synthetic_pdf: Path) -> None:
    phase1 = read_and_extract(str(synthetic_pdf), 0)
    result = ExtractResult(page=phase1.page.meta, texts=phase1.texts, vectors=phase1.vectors)
    again = ExtractResult.from_json(result.to_json())
    assert again == result
    assert isinstance(again.texts[0], Text) and isinstance(again.vectors[0], Vector)
