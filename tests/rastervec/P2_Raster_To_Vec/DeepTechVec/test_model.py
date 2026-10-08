from __future__ import annotations

import numpy as np
import pytest
import torch

from rastervec.P2_Raster_To_Vec.DeepTechVec import inference
from rastervec.P2_Raster_To_Vec.DeepTechVec.model.losses import primitive_loss
from rastervec.P2_Raster_To_Vec.DeepTechVec.model.network import build_model, load_weights, save_weights
from rastervec.P2_Raster_To_Vec.DeepTechVec.train import TINY_CONFIG


@pytest.mark.parametrize("kind,d_emb", [("line", 6), ("curve", 8)])
def test_forward_shapes_follow_the_paper(kind, d_emb):
    model = build_model({**TINY_CONFIG, "prim_kind": kind})
    assert model.d_emb == d_emb  # paper: d_emb = 6 (lines) / 8 (curves)
    y = model(torch.rand(3, 1, 64, 64)).detach()
    assert y.shape == (3, TINY_CONFIG["n_prim"], d_emb)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_paper_sized_model_builds():
    from rastervec.P2_Raster_To_Vec.DeepTechVec.config import MODEL_DEFAULTS

    model = build_model(MODEL_DEFAULTS)
    assert len(model.blocks) == 8 and model.cfg["n_prim"] == 10


def test_loss_on_perfect_prediction_and_placeholders():
    target = torch.zeros(1, 4, 6)
    target[0, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.05, 1.0])
    loss, terms = primitive_loss(target.clone(), target)
    assert float(loss) < 1e-4 and terms["L_loc"] == 0.0
    # wrong parameters on a placeholder row are penalised too (targets are zero there)
    pred = target.clone()
    pred[0, 3, :5] = 0.5
    _, terms = primitive_loss(pred, target)
    assert terms["L_loc"] > 0


def test_save_load_round_trip_and_predict(tmp_path):
    torch.manual_seed(0)
    model = build_model({**TINY_CONFIG, "prim_kind": "curve"}).eval()
    path = tmp_path / "w.pth"
    save_weights(path, model)
    loaded = load_weights(path).eval()
    x = torch.rand(2, 1, 64, 64)
    assert torch.allclose(model(x), loaded(x))
    preds = inference.predict_patches(loaded, [np.full((64, 64), 255, np.uint8)] * 2, conf_thresh=0.0)
    assert len(preds) == 2 and all(len(p) == TINY_CONFIG["n_prim"] for p in preds)
    assert all(not prim.is_line for prim in preds[0])


def test_missing_weights_raise_with_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPTECHVEC_WEIGHTS_PATH", str(tmp_path / "nope.pth"))
    with pytest.raises(FileNotFoundError, match="DeepTechVec weights not found"):
        inference.load_model()
    inference.warmup()  # best effort: swallowed
