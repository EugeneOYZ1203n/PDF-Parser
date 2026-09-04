from __future__ import annotations

import pytest

from rastervec.Evaluation.Evaluate.text_metrics import (
    char_error_rate,
    char_multiset,
    levenshtein,
    normalize_text,
    word_error_rate,
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


def test_char_error_rate():
    assert char_error_rate("kitten", "kitten") == 0.0
    assert char_error_rate("kitten", "sitting") == pytest.approx(3 / 6)
    assert char_error_rate("", "abc") == pytest.approx(3.0)  # min-1 denominator, not clamped


def test_word_error_rate():
    assert word_error_rate("the quick brown fox", "the quick brown fox") == 0.0
    assert word_error_rate("the quick brown fox", "the quick fox") == pytest.approx(1 / 4)


