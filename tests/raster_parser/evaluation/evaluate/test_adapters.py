"""Tests for adapters.py's text-type bucketing: a mixed LabelSet split into the scored types
(native_to_vector / original_vector)."""

from __future__ import annotations

from raster_parser.evaluation.evaluate.adapters import entries_by_text_type, gt_regions_by_text_type
from raster_parser.evaluation.labelling.label_schema import LabelEntry, LabelSet


def _entry(label_id, text, source):
    return LabelEntry(
        page_index=0,
        bbox=(0, 0, 10, 10),
        label_id=label_id,
        text=text,
        source=source,
    )


def _mixed() -> LabelSet:
    return LabelSet(
        pdf_path="x.pdf",
        entries=[
            _entry("line:0:0:0", "native", "native"),
            _entry("uuid-1", "vector", "vector"),
        ],
    )


def test_entries_by_text_type_buckets_by_source():
    buckets = entries_by_text_type(_mixed())
    assert set(buckets) == {"native_to_vector", "original_vector"}
    assert [e.text for e in buckets["native_to_vector"]] == ["native"]
    assert [e.text for e in buckets["original_vector"]] == ["vector"]


def test_gt_regions_carry_their_text_type():
    regions = gt_regions_by_text_type(_mixed())
    assert [(g.text, g.text_type) for g in regions["original_vector"]] == [
        ("vector", "original_vector")
    ]
