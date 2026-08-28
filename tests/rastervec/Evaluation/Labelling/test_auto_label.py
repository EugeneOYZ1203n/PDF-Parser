from __future__ import annotations

import pytest

from rastervec.Evaluation.Labelling.auto_label import auto_label_pdf


def test_auto_label_pdf_recovers_known_text(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory(
        [{"texts": [{"point": (10, 50), "text": "Hello", "fontsize": 20}]}]
    )
    path = tmp_pdf_path(doc)

    labels = auto_label_pdf(path, 0)

    assert labels.pdf_path == path
    assert len(labels.entries) == 1
    entry = labels.entries[0]
    assert entry.text == "Hello"
    assert entry.source == "auto"
    assert entry.page_index == 0
    assert entry.expected_rotation == 0


def test_auto_label_pdf_groups_separate_lines_independently(
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

    labels = auto_label_pdf(path, 0)

    assert len(labels.entries) == 2
    texts = {e.text for e in labels.entries}
    assert texts == {"First", "Second"}
    signatures = {e.cluster_signature for e in labels.entries}
    assert len(signatures) == 2
    assert all(sig.startswith("line:0:") for sig in signatures)


def test_auto_label_pdf_multi_word_line_joined_left_to_right(
    synthetic_pdf_factory, tmp_pdf_path,
):
    doc = synthetic_pdf_factory(
        [{"texts": [{"point": (10, 50), "text": "Hello World"}]}]
    )
    path = tmp_pdf_path(doc)

    labels = auto_label_pdf(path, 0)

    assert len(labels.entries) == 1
    assert labels.entries[0].text == "Hello World"


def test_auto_label_pdf_rotated_text_sets_expected_rotation(
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

    labels = auto_label_pdf(path, 0)

    assert len(labels.entries) == 1
    # pymupdf's rotate=90 in insert_text corresponds to a -90 degree
    # direction vector -- rounds to 270 under (round(angle/90)%4*90).
    assert labels.entries[0].expected_rotation in (90, 270)


def test_auto_label_pdf_empty_page_returns_no_entries(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{"texts": []}])
    path = tmp_pdf_path(doc)

    labels = auto_label_pdf(path, 0)

    assert labels.entries == []
