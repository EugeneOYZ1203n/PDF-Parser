"""The FAST-running P3 backend feeds FAST 640 px tiles 1:1 (effective dpi =
render dpi = 300), and warns once when a caller's tile size would resample."""
from __future__ import annotations

import importlib
import logging

import pytest
from PIL import Image

BACKENDS = ("VectorClassification",)


def _mods(backend):
    base = f"rastervec.P3_Vector_Parsing.{backend}"
    return importlib.import_module(f"{base}.config"), importlib.import_module(f"{base}.fast_detect")


@pytest.mark.parametrize("backend", BACKENDS)
def test_tiles_match_fast_input_size_at_300_dpi(backend):
    config, fast_detect = _mods(backend)
    assert config.FAST_TILE_BLOCK_SIZE == fast_detect._SHORT_SIDE
    assert config.FAST_PAGE_RENDER_DPI * config.FAST_TILE_SCALE_FACTOR == 300
    tile = Image.new("RGB", (config.FAST_TILE_BLOCK_SIZE, config.FAST_TILE_BLOCK_SIZE))
    assert fast_detect._scale_aligned_short(tile).size == tile.size  # fed 1:1, no resample


@pytest.mark.parametrize("backend", BACKENDS)
def test_detect_tiled_warns_once_for_resampling_block_size(backend, monkeypatch, caplog):
    import numpy as np

    _config, fast_detect = _mods(backend)
    monkeypatch.setattr(fast_detect, "_WARNED_BLOCK_SIZES", set())
    monkeypatch.setattr(
        fast_detect.FastDetector, "detect",
        lambda self, block: np.zeros((block.height, block.width), dtype=np.float32),
    )
    det = fast_detect.FastDetector("unused.pth")
    image = Image.new("RGB", (64, 32), (255, 255, 255))
    with caplog.at_level(logging.WARNING):
        det.detect_tiled(image, block_size=fast_detect._SHORT_SIDE, show_progress=False)
        assert "resampled" not in caplog.text
        for _ in range(2):
            det.detect_tiled(image, block_size=32, show_progress=False)
    assert caplog.text.count("resampled") == 1
