from __future__ import annotations

import numpy as np
import pytest
import torch

from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import inference, train_data as td
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import geometry as geo
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec.model.losses import rec_loss, udf_loss
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec.model.network import (
    build_model, load_submodule, load_weights, save_weights,
)
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec.train import TINY_CONFIG, _to_device


def test_shapes_follow_the_paper_grids():
    model = build_model(TINY_CONFIG).eval()
    with torch.no_grad():
        udf, out = model(torch.rand(2, 1, 40, 48))
    assert udf.shape == (2, 6, 81, 97)  # (2H+1) x (2W+1) UDF lattice, six fields
    assert out["edge"].shape == (2, 4, 80, 96) and out["vertex"].shape == (2, 2, 80, 96)
    assert out["skeleton"].shape == (2, 1, 80, 96)
    assert float(udf.min()) >= 0 and float(udf.max()) <= 1


def test_paper_sized_models_build():
    from rastervec.P2_Raster_To_Vec.ImplicitSketchVec.config import BASIC_OVERRIDES, MODEL_DEFAULTS

    full = build_model(MODEL_DEFAULTS)
    basic = build_model({**MODEL_DEFAULTS, **BASIC_OVERRIDES})
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(basic) < n(full)
    assert len(full.ndc.branches) == 3


def test_losses_vanish_on_ground_truth():
    t = td.make_targets([geo.line_to_cubic((2, 10.3), (30, 10.3))[None]], 32)
    b = _to_device({k: t[k][None] for k in ("udf", "mask", "edge", "emask", "vert", "vmask", "skel")}, "cpu")
    loss, terms = udf_loss(b["udf"] / 8.0, b["udf"], b["mask"])
    assert float(loss) < 1e-5
    logits = torch.nn.functional.one_hot(b["edge"], 4).permute(0, 3, 1, 2).float() * 50
    out = {"edge": logits, "vertex": b["vert"], "skeleton": (b["skel"][:, None] * 2 - 1) * 50}
    loss, terms = rec_loss(out, b)
    assert terms["L_edge"] < 1e-4 and terms["L_vertex"] < 1e-9 and terms["L_skel"] < 1e-4


def test_save_load_and_submodule_transfer(tmp_path):
    torch.manual_seed(0)
    a = build_model(TINY_CONFIG).eval()
    path = tmp_path / "a.pth"
    save_weights(path, a)
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(a(x)[0], load_weights(path).eval()(x)[0])
    torch.manual_seed(1)
    b = build_model(TINY_CONFIG).eval()
    load_submodule(b, path, "ndc")
    for pa, pb in zip(a.ndc.parameters(), b.ndc.parameters()):
        assert torch.equal(pa, pb)
    assert not all(torch.equal(pa, pb) for pa, pb in zip(a.dfp.parameters(), b.dfp.parameters()))


def test_predict_tiles_returns_px_fields():
    fields = inference.predict_tiles(build_model(TINY_CONFIG).eval(), [np.full((32, 32), 255, np.uint8)])
    (f,) = fields
    assert f["udf"].shape == (6, 65, 65) and f["udf"].max() <= 8.0
    assert f["edge"].shape == (64, 64) and f["edge"].max() <= 3 and f["vertex"].shape == (2, 64, 64)


def test_missing_weights_raise_with_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("IMPLICITSKETCHVEC_WEIGHTS_PATH", str(tmp_path / "nope.pth"))
    with pytest.raises(FileNotFoundError, match="ImplicitSketchVec weights not found"):
        inference.load_model()
    inference.warmup()
