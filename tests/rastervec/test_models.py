from __future__ import annotations

from rastervec.models import TextWord, VectorRecord


def test_text_word_carries_full_pymupdf_word_field_surface():
    word = TextWord(
        text="Hello", bbox=(0, 0, 10, 5),
        quad=((0, 5), (10, 5), (10, 0), (0, 0)),
        angle=0.0, direction=(1.0, 0.0), font="helv", font_size=10.0,
        color=None, flags=0, origin=(0, 5), ascender=None, descender=None,
        orientation_source="text-span", page_index=0, seq=0,
        wmode=0, block_no=1, line_no=2, word_no=3,
    )

    assert word.text == "Hello"
    assert word.wmode == 0
    assert (word.block_no, word.line_no, word.word_no) == (1, 2, 3)


def test_text_word_block_line_word_default_to_zero():
    word = TextWord(
        text="x", bbox=(0, 0, 1, 1), quad=((0, 0), (1, 0), (1, 1), (0, 1)),
        angle=0.0, direction=(1.0, 0.0), font="", font_size=0.0, color=None,
        flags=0, origin=None, ascender=None, descender=None,
        orientation_source="fallback", page_index=0, seq=0,
    )
    assert (word.wmode, word.block_no, word.line_no, word.word_no) == (0, 0, 0, 0)


def test_vector_record_carries_full_pymupdf_drawing_field_surface(vector_path):
    path = vector_path()
    record = VectorRecord(
        items=[path], bbox=path.bbox, stroke_color=(0, 0, 0), fill_color=None,
        stroke_width=1.0, dashed=False, page_index=0,
        even_odd=False, line_cap=0, line_join=0, seqno=7,
        rect=path.bbox, scissor=None, blendmode=None,
        isolated=False, knockout=False, opacity=1.0,
    )

    assert record.items == [path]
    assert record.seqno == 7
    assert record.even_odd is False
    assert record.groups is None
    assert record.role is None


def test_vector_record_optional_group_lineage_fields(vector_path):
    path_a = vector_path(seq=0)
    path_b = vector_path(seq=1, item_index=1, bbox=(2, 2, 3, 3))
    record = VectorRecord(
        items=[path_a, path_b], bbox=(0, 0, 3, 3), stroke_color=(0, 0, 0),
        fill_color=None, stroke_width=1.0, dashed=False, page_index=0,
        even_odd=False, line_cap=0, line_join=0, seqno=0,
        rect=(0, 0, 3, 3), scissor=None, blendmode=None,
        isolated=False, knockout=False, opacity=None,
        groups=[[path_a], [path_b]], role="kept",
    )

    assert record.groups == [[path_a], [path_b]]
    assert record.role == "kept"
