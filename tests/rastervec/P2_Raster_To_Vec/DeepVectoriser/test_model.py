from __future__ import annotations

import numpy as np
import pytest
import torch

from rastervec.P2_Raster_To_Vec.DeepVectoriser import inference
from rastervec.P2_Raster_To_Vec.DeepVectoriser.model.layers import prepare_input
from rastervec.P2_Raster_To_Vec.DeepVectoriser.model.liu_model import build_model, load_weights, save_weights
from rastervec.P2_Raster_To_Vec.DeepVectoriser.train import TINY_CONFIG


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(TINY_CONFIG).eval()


@pytest.mark.parametrize("size", [64, 96, 128, 256])
def test_forward_shapes_at_variable_tile_sizes(model, size):
    cfg = TINY_CONFIG
    x = prepare_input(torch.full((2, size, size), 255, dtype=torch.uint8))
    assert x.shape == (2, 3, size, size)
    enc = model.encoder(x)
    assert enc["endpoints"].shape == (2, cfg["n_stroke"], 4)
    assert enc["emb"].shape == (2, cfg["n_stroke"], cfg["d_emb"])
    assert enc["logit"].shape == (2, cfg["n_stroke"])
    m = 3
    cond = model.make_cond(torch.rand(m, 4), torch.randn(m, cfg["d_emb"]), torch.rand(m))
    skips = model.unet.encode(x)
    idx = torch.tensor([0, 1, 1])
    levels = model.unet.decode(skips, cond, idx, stop_level=0, return_levels=True)
    assert model.decoder_head(levels[0]).shape == (m, 1, size, size)
    mem = model.vectorizer.memory(levels[model.vec_level])
    t = cfg["max_prims"]
    curves, eos = model.vectorizer(mem, cond, torch.rand(m, t, 6))
    assert curves.shape == (m, t, 6) and eos.shape == (m, t)
    out = model.vectorizer.generate(mem, cond, torch.rand(m, 2))
    assert len(out) == m and all(1 <= len(c) <= t for c in out)


def test_weights_roundtrip(tmp_path, model):
    path = tmp_path / "w.pth"
    save_weights(path, model, {"note": "test"})
    again = load_weights(path).eval()
    x = prepare_input(torch.randint(0, 255, (1, 64, 64), dtype=torch.uint8))
    assert torch.allclose(model.encoder(x)["endpoints"], again.encoder(x)["endpoints"])


def test_predict_tiles_returns_pixel_strokes(model):
    tiles = [np.full((64, 64), 255, np.uint8)]
    preds = inference.predict_tiles(model, tiles, conf_thresh=0.0)
    assert len(preds) == 1 and preds[0].n_queries == TINY_CONFIG["n_stroke"]
    assert len(preds[0].strokes) == TINY_CONFIG["n_stroke"]
    s = preds[0].strokes[0]
    assert s.ndim == 3 and s.shape[1:] == (4, 2)
    for a, b in zip(s, s[1:]):
        assert np.allclose(a[3], b[0])  # partial format: pieces chain


def test_missing_weights_raise_with_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPVEC_WEIGHTS_PATH", str(tmp_path / "nope.pth"))
    with pytest.raises(FileNotFoundError, match="DeepVectoriser weights not found"):
        inference.load_model()
    inference.warmup()  # best effort: swallowed
