from __future__ import annotations

import pytest

from raster_parser.evaluation.evaluate.text_metrics import (
    char_multiset,
    levenshtein,
    normalize_text,
    word_tokens,
)


def test_normalize_text_uppercases_and_collapses_whitespace():
    assert normalize_text("  Setback\tline ") == "SETBACK LINE"
    assert normalize_text("Foo   Bar\nBaz") == "FOO BAR BAZ"
    assert normalize_text("5 mm") == "5 MM"
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_char_multiset_drops_spaces_and_folds_case():
    assert char_multiset("A a  b") == char_multiset("aAB")
    assert set(char_multiset("a b").elements()) == {"A", "B"}


def test_word_tokens():
    assert word_tokens("  the   quick brown ") == ["THE", "QUICK", "BROWN"]
    assert word_tokens("") == []


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("kitten", "sitting", 3),
        ("abc", "abd", 1),  # substitution
        ("abc", "abxc", 1),  # insertion
        ("abc", "ac", 1),  # deletion
    ],
)
def test_levenshtein_strings(a, b, expected):
    assert levenshtein(a, b) == expected


def test_levenshtein_token_lists():
    assert levenshtein(["foo", "bar", "baz"], ["foo", "baz"]) == 1
    assert levenshtein(["a", "b"], ["b", "a"]) == 2
