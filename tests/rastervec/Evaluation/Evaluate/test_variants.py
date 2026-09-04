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
    assert VARIANTS["current_heavy"].engine == "current"
    assert VARIANTS["current_heavy"].ocr_backend == "heavy"
    assert VARIANTS["current_light"].ocr_backend == "light"
    assert VARIANTS["current_light_nofast"].enable_fast is False
    assert VARIANTS["current_heavy"].enable_fast is True


def test_resolve_variant_rejects_unknown():
    assert resolve_variant("current_light") is VARIANTS["current_light"]
    with pytest.raises(ValueError, match="unknown pipeline variant"):
        resolve_variant("does_not_exist")
