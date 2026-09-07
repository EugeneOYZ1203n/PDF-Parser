from __future__ import annotations

import pytest

from rastervec.Evaluation.Evaluate.variants import (
    DEFAULT_VARIANTS,
    VARIANTS,
    resolve_variant,
)


def test_every_variant_name_matches_its_key():
    for key, variant in VARIANTS.items():
        assert variant.name == key


def test_default_variants_are_all_registered():
    assert set(DEFAULT_VARIANTS) <= set(VARIANTS)


def test_engines_and_flags():
    assert VARIANTS["legacy"].engine == "legacy"
    assert VARIANTS["current"].engine == "current"
    assert VARIANTS["current"].enable_fast is True
    assert VARIANTS["current_nofast"].enable_fast is False


def test_resolve_variant_rejects_unknown():
    assert resolve_variant("current") is VARIANTS["current"]
    with pytest.raises(ValueError, match="unknown pipeline variant"):
        resolve_variant("does_not_exist")
