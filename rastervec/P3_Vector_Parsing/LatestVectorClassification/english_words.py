"""English-word boost for the recognition retry's winner selection.

`selection_score` multiplies `paddle_engine.score` by
`ENGLISH_WORD_MULTIPLIER ** n`, n = the number of dictionary English words
(pyenchant, `ENGLISH_DICT_LANG`) in the read. Used only to compare a retried
crop's attempts against each other -- never for the retry trigger itself.
Kept out of `paddle_engine.py` so Pool-2 jobs never import enchant; scoring
runs in the calling process after recognition returns.
"""
from __future__ import annotations

import string

import enchant

from rastervec.P3_Vector_Parsing.LatestVectorClassification.config import (
    ENGLISH_DICT_LANG,
    ENGLISH_WORD_MIN_LEN,
    ENGLISH_WORD_MULTIPLIER,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.paddle_engine import (
    OcrBox,
    score,
)

_DICT: enchant.Dict | None = None


def _dict() -> enchant.Dict:
    global _DICT
    if _DICT is None:
        _DICT = enchant.Dict(ENGLISH_DICT_LANG)
    return _DICT


def _is_english_word(token: str) -> bool:
    if len(token) < ENGLISH_WORD_MIN_LEN or not token.isalpha() or not token.isascii():
        return False
    d = _dict()
    return d.check(token) or d.check(token.lower())


def english_word_count(text: str) -> int:
    """Whitespace tokens (leading/trailing punctuation stripped) that are
    purely alphabetic, >= `ENGLISH_WORD_MIN_LEN` letters and in the
    dictionary. Every occurrence counts."""
    return sum(_is_english_word(tok.strip(string.punctuation)) for tok in text.split())


def selection_score(box: OcrBox) -> tuple[float, int]:
    """`(score(box) * ENGLISH_WORD_MULTIPLIER ** n, n)`."""
    n = english_word_count(box.text)
    return score(box) * ENGLISH_WORD_MULTIPLIER ** n, n
