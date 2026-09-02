"""Text normalisation + comparison primitives shared by the evaluation
metric suite (`metrics.py`) and the legacy scorer (`evaluate.py`).

Two things live here:

- **Normalisation** -- `normalize_text` is applied to every ground-truth and
  every predicted string before any character/word comparison anywhere in
  the metric suite, so casing and whitespace never count as errors:
  upper-case, strip both ends, collapse every internal whitespace run to a
  single space. `char_multiset` then drops the spaces (character metrics
  compare glyphs); `word_tokens` splits on them (word metrics compare
  tokens).

- **Edit distance** -- `levenshtein` (pure-Python, no dependency) is the
  standard text-diff for every accuracy metric that needs one.
  `difflib.SequenceMatcher` is deliberately *not* used: it measures longest
  matching blocks, not edits, so a transposition or a run of single-char
  substitutions scores very differently from the intuitive error count.
  (`evaluate.py`'s legacy `evaluate_pipeline` still uses `difflib` -- that
  path is frozen and not worth changing.)
"""
from __future__ import annotations

import collections
from typing import Hashable, Sequence

__all__ = [
    "normalize_text",
    "char_multiset",
    "word_tokens",
    "levenshtein",
    "char_error_rate",
    "word_error_rate",
]


def normalize_text(s: str) -> str:
    """Upper-case, strip both ends, collapse every internal whitespace run
    (spaces, tabs, newlines) to a single space.

    So ``"  Setback\tline "`` and ``"SETBACK LINE"`` normalise equal.
    """
    return " ".join(s.split()).upper()


def char_multiset(s: str) -> "collections.Counter[str]":
    """Multiset (Counter) of the non-space characters of ``normalize_text(s)``.

    Character-accuracy metrics compare these directly -- spaces are dropped
    so word spacing/segmentation doesn't leak into a character score.
    """
    return collections.Counter(c for c in normalize_text(s) if c != " ")


def word_tokens(s: str) -> list[str]:
    """``normalize_text(s)`` split into whitespace-separated tokens (empties
    dropped). Since `normalize_text` already collapses whitespace, this is
    just ``normalize_text(s).split(" ")`` minus empty strings."""
    return [tok for tok in normalize_text(s).split(" ") if tok]


def levenshtein(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Levenshtein edit distance (insertions + deletions + substitutions,
    unit cost) between two sequences. Works on strings for character
    distance and on token lists for word distance. O(len(a) * len(b)) time,
    O(min(len(a), len(b))) space (two-row DP)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + cost,  # substitution / match
            )
        previous = current
    return previous[-1]


def char_error_rate(ref: str, hyp: str) -> float:
    """Character edit distance between ``normalize_text(ref)`` and
    ``normalize_text(hyp)``, divided by ``len(normalize_text(ref))`` (min 1
    so an empty reference doesn't divide by zero). Not clamped -- a
    hypothesis much longer than the reference can exceed 1.0."""
    ref_n = normalize_text(ref)
    hyp_n = normalize_text(hyp)
    return levenshtein(ref_n, hyp_n) / max(len(ref_n), 1)


def word_error_rate(ref: str, hyp: str) -> float:
    """Token edit distance between ``word_tokens(ref)`` and
    ``word_tokens(hyp)``, divided by ``len(word_tokens(ref))`` (min 1)."""
    ref_toks = word_tokens(ref)
    hyp_toks = word_tokens(hyp)
    return levenshtein(ref_toks, hyp_toks) / max(len(ref_toks), 1)
