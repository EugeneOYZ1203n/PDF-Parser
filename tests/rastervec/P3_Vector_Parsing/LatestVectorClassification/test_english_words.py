from __future__ import annotations

import pytest

from rastervec.P3_Vector_Parsing.LatestVectorClassification.english_words import (
    english_word_count,
    selection_score,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.paddle_engine import OcrBox


@pytest.mark.parametrize("text, n", [
    ("TYPICAL DETAIL", 2),
    ("typical detail", 2),
    ("WALL, DOOR.", 2),  # edge punctuation stripped
    ("DOOR DOOR", 2),  # every occurrence counts
    ("300MM B-12", 0),  # not purely alphabetic
    ("A", 0),  # below the 2-letter minimum
    ("XQZV QZX", 0),  # not in the dictionary
    ("", 0),
])
def test_english_word_count(text, n):
    assert english_word_count(text) == n


def test_selection_score_multiplies_per_english_word():
    assert selection_score(OcrBox(text="DOOR FRAME", confidence=0.6)) == (pytest.approx(0.6 * 1.2 ** 2), 2)
    assert selection_score(OcrBox(text="XQZV", confidence=0.7)) == (pytest.approx(0.7), 0)
    assert selection_score(OcrBox(text="", confidence=0.0)) == (0.0, 0)
