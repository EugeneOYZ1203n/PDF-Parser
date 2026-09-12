from __future__ import annotations

import pytest

from rastervec.Evaluation.conversion import convert_page_text_only
from rastervec.Evaluation.Labelling.native_label import (
    attach_vector_signatures,
    native_label_pdf,
)


def test_native_label_pdf_recovers_known_text(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory(
        [{"texts": [{"point": (10, 50), "text": "Hello", "fontsize": 20}]}]
    )
    path = tmp_pdf_path(doc)

    labels = native_label_pdf(path, 0)

    assert labels.pdf_path == path
    assert len(labels.entries) == 1
    entry = labels.entries[0]
    assert entry.text == "Hello"
    assert entry.source == "native"
    assert entry.page_index == 0
    assert entry.expected_rotation == 0
    assert entry.vector_signatures == []


def test_native_label_pdf_groups_separate_lines_independently(
    synthetic_pdf_factory, tmp_pdf_path,
):
    doc = synthetic_pdf_factory(
        [
            {
                "width": 200,
                "height": 100,
                "texts": [
                    {"point": (10, 20), "text": "First"},
                    {"point": (10, 60), "text": "Second"},
                ],
            }
        ]
    )
    path = tmp_pdf_path(doc)

    labels = native_label_pdf(path, 0)

    assert len(labels.entries) == 2
    texts = {e.text for e in labels.entries}
    assert texts == {"First", "Second"}
    signatures = {e.cluster_signature for e in labels.entries}
    assert len(signatures) == 2
    assert all(sig.startswith("line:0:") for sig in signatures)
    label_ids = {e.label_id for e in labels.entries}
    assert label_ids == signatures  # native_label_pdf reuses the line id as label_id


def test_native_label_pdf_multi_word_line_joined_left_to_right(
    synthetic_pdf_factory, tmp_pdf_path,
):
    doc = synthetic_pdf_factory(
        [{"texts": [{"point": (10, 50), "text": "Hello World"}]}]
    )
    path = tmp_pdf_path(doc)

    labels = native_label_pdf(path, 0)

    assert len(labels.entries) == 1
    assert labels.entries[0].text == "Hello World"


def test_native_label_pdf_rotated_text_sets_expected_rotation(
    synthetic_pdf_factory, tmp_pdf_path,
):
    doc = synthetic_pdf_factory(
        [
            {
                "width": 200,
                "height": 200,
                "texts": [{"point": (50, 150), "text": "VertText", "rotate": 90}],
            }
        ]
    )
    path = tmp_pdf_path(doc)

    labels = native_label_pdf(path, 0)

    assert len(labels.entries) == 1
    # pymupdf's rotate=90 in insert_text corresponds to a -90 degree
    # direction vector -- rounds to 270 under (round(angle/90)%4*90).
    assert labels.entries[0].expected_rotation in (90, 270)


def test_native_label_pdf_empty_page_returns_no_entries(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{"texts": []}])
    path = tmp_pdf_path(doc)

    labels = native_label_pdf(path, 0)

    assert labels.entries == []


def test_attach_vector_signatures_assigns_vectors_to_matching_lines(
    synthetic_pdf_factory, tmp_pdf_path, tmp_path,
):
    doc = synthetic_pdf_factory(
        [
            {
                "width": 200,
                "height": 100,
                "texts": [
                    {"point": (10, 20), "text": "First"},
                    {"point": (10, 60), "text": "Second"},
                ],
            }
        ]
    )
    path = tmp_pdf_path(doc)
    labels = native_label_pdf(path, 0)
    assert len(labels.entries) == 2
    assert all(e.vector_signatures == [] for e in labels.entries)

    vectors_pdf_path = str(tmp_path / "vectors.pdf")
    convert_page_text_only(path, 0, vectors_pdf_path)
    attach_vector_signatures(labels, 0, vectors_pdf_path)

    by_text = {e.text: e for e in labels.entries}
    assert by_text["First"].vector_signatures != []
    assert by_text["Second"].vector_signatures != []
    # each glyph's vector goes to exactly one line, never split across both
    assert set(by_text["First"].vector_signatures).isdisjoint(
        by_text["Second"].vector_signatures
    )
