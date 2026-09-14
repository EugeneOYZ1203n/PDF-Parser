"""Tests for adapters.py's text-type bucketing -- the 5-way split of a
mixed LabelSet into native_to_vector/original_vector/vector_to_raster/
original_raster/native_to_raster."""
from __future__ import annotations

from rastervec.Evaluation.Evaluate.adapters import entries_by_text_type
from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet


def _entry(label_id, text, source):
    return LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 10), cluster_signature="s",
        label_id=label_id, text=text, source=source,
    )


def test_entries_by_text_type_buckets_all_five_types():
    labels = LabelSet(pdf_path="x.pdf", entries=[
        _entry("line:0:0:0", "native", "native"),
        _entry("uuid-1", "vector", "vector"),
        _entry("vecsync:uuid-1", "vec-copy", "raster"),
        _entry("natsync:line:0:0:0", "nat-copy", "raster"),
        _entry("uuid-2", "hand-drawn", "raster"),
    ])
    buckets = entries_by_text_type(labels)
    assert [e.text for e in buckets["native_to_vector"]] == ["native"]
    assert [e.text for e in buckets["original_vector"]] == ["vector"]
    assert [e.text for e in buckets["vector_to_raster"]] == ["vec-copy"]
    assert [e.text for e in buckets["native_to_raster"]] == ["nat-copy"]
    assert [e.text for e in buckets["original_raster"]] == ["hand-drawn"]


def test_natsync_prefix_takes_precedence_over_source():
    # natsync entries carry source="raster" like vecsync/hand-drawn ones --
    # only the label_id prefix distinguishes them.
    labels = LabelSet(pdf_path="x.pdf", entries=[_entry("natsync:x", "t", "raster")])
    buckets = entries_by_text_type(labels)
    assert buckets["native_to_raster"] and not buckets["original_raster"]
