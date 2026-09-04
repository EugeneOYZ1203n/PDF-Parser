from __future__ import annotations

import numpy as np

from rastervec.OCR.Paddle_OCR.ink_segment import (
    _ink_runs,
    _split_on_gaps,
    split_lines_by_ink,
    split_words_by_ink,
)


def test_ink_runs_finds_contiguous_blocks():
    row = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert _ink_runs(row) == [(2, 4), (7, 7)]
    assert _ink_runs(np.zeros(5, dtype=bool)) == []


def test_split_on_gaps_single_run_is_one_span():
    has_ink = np.zeros(20, dtype=bool)
    has_ink[5:10] = True
    assert _split_on_gaps(has_ink, gap_factor=1.9, min_gap=2.0) == [(5, 9)]


def test_split_on_gaps_breaks_on_wide_gap_relative_to_median():
    # runs at 0-1, 3-4, 6-7 (gaps of 1) then a big jump to 40-41.
    has_ink = np.zeros(50, dtype=bool)
    for start in (0, 3, 6, 40):
        has_ink[start:start + 2] = True
    spans = _split_on_gaps(has_ink, gap_factor=1.9, min_gap=2.0)
    assert spans == [(0, 7), (40, 41)]


def test_split_words_by_ink_two_words():
    gray = np.full((30, 200), 255, np.uint8)
    # word 1: three thin strokes tightly spaced; word 2: same, far away.
    for x in (10, 15, 20, 25):
        gray[8:22, x:x + 2] = 0
    for x in (120, 125, 130, 135):
        gray[8:22, x:x + 2] = 0
    boxes = split_words_by_ink(gray)
    assert len(boxes) == 2
    (x0a, _y0a, x1a, _y1a), (x0b, _y0b, x1b, _y1b) = boxes
    assert x1a < x0b  # non-overlapping, left-to-right
    assert x0a <= 10 and x1b >= 137


def test_split_words_by_ink_single_blob_is_one_box():
    gray = np.full((20, 50), 255, np.uint8)
    gray[5:15, 10:40] = 0
    assert len(split_words_by_ink(gray)) == 1


def test_split_words_by_ink_blank_is_empty():
    assert split_words_by_ink(np.full((10, 10), 255, np.uint8)) == []


def test_split_lines_by_ink_two_lines():
    gray = np.full((120, 60), 255, np.uint8)
    gray[5:15, 3:55] = 0
    gray[60:75, 3:55] = 0
    boxes = split_lines_by_ink(gray)
    assert len(boxes) == 2
    assert boxes[0][3] < boxes[1][1]  # first line ends above second line's start


def test_split_lines_by_ink_blank_is_empty():
    assert split_lines_by_ink(np.full((10, 10), 255, np.uint8)) == []
