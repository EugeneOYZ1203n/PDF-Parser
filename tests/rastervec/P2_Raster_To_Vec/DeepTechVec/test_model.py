from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from rastervec.P2_Raster_To_Vec.DeepTechVec import inference
from rastervec.P2_Raster_To_Vec.DeepTechVec.model.network import build_model, load_weights, prepare_input


@pytest.mark.parametrize("kind,out", [("line", 6), ("curve", 8)])
def test_forward_shapes_follow_the_released_specs(kind, out):
    model = build_model(kind).eval()
    with torch.inference_mode():
        y = model(torch.rand(3, 3, 64, 64), 10)
    assert y.shape == (3, 10, out)
    assert float(y[..., -1].min()) >= 0.0 and float(y[..., -1].max()) <= 1.0  # sigmoid confidence
    if kind == "line":  # (tanh + 1) / 2 coordinates
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_prepare_input_masks_the_coordinate_channels_to_the_ink():
    gray = np.full((1, 64, 64), 255, np.uint8)
    gray[0, 10, 20] = 0
    x = prepare_input(torch.from_numpy(gray))
    assert x.shape == (1, 3, 64, 64)
    assert float(x[0, 0].sum()) == pytest.approx(1.0)
    assert float(x[0, 1, 10, 20]) == pytest.approx(21 / 64) and float(x[0, 2, 10, 20]) == pytest.approx(11 / 64)
    assert float(x[0, 1].sum()) == pytest.approx(21 / 64)  # zero off the ink


@pytest.mark.parametrize("kind", ["line", "curve"])
def test_upstream_style_checkpoint_loads_strict_and_knows_its_kind(tmp_path, kind):
    torch.manual_seed(0)
    model = build_model(kind).eval()
    # pre-refactor key names, as upstream's run_pipeline.serialize expects
    state = {k.replace("hidden.decoder.transformer", "hidden.transformer"): v for k, v in model.state_dict().items()}
    assert any(k.startswith("hidden.transformer.") for k in state)
    path = tmp_path / "w.weights"
    torch.save({"model_state_dict": state}, path)
    loaded = load_weights(path).eval()
    assert loaded.kind == kind
    x = torch.rand(2, 3, 64, 64)
    with torch.inference_mode():
        assert torch.allclose(model(x, 10), loaded(x, 10))


def test_not_a_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "w.weights"
    torch.save({"model_state_dict": {"foo": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="not a Deep-Vectorization"):
        load_weights(path)


def test_decode_output_scales_by_the_patch_and_drops_low_confidence():
    row = np.array([[0.25, 0.5, 0.75, 0.5, 0.05, 0.9],
                    [0.1, 0.1, 0.2, 0.2, 0.05, 0.3]])
    (prim,) = inference.decode_output(row, "line")
    assert np.allclose(prim.pts, [[16, 32], [48, 32]]) and prim.width == pytest.approx(3.2)
    curve = np.array([[-5.0, 0.5, 0.5, 0.5, 0.75, 0.5, 0.05, 0.9]])
    (c,) = inference.decode_output(curve, "curve")
    assert c.pts.shape == (3, 2) and c.pts[0, 0] == pytest.approx(-16.0)  # unbounded coords clamped


def test_predict_patches_runs_the_network(monkeypatch):
    model = build_model("line").eval()
    preds = inference.predict_patches(model, [np.full((64, 64), 255, np.uint8)] * 2, conf_thresh=0.0)
    assert len(preds) == 2 and all(len(p) == 10 and all(q.is_line for q in p) for p in preds)
    with pytest.raises(ValueError, match="64 px patches"):
        inference.predict_patches(model, [np.full((32, 32), 255, np.uint8)])


def test_missing_weights_raise_with_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPTECHVEC_WEIGHTS_PATH", str(tmp_path / "nope.weights"))
    with pytest.raises(FileNotFoundError, match="download_weights"):
        inference.load_model()
    inference.warmup()  # best effort: swallowed


@pytest.mark.skipif(os.environ.get("RASTERVEC_RUN_DEEPTECHVEC_TESTS") != "1",
                    reason="needs the downloaded pretrained weights (RASTERVEC_RUN_DEEPTECHVEC_TESTS=1)")
def test_pretrained_lines_model_finds_a_horizontal_line():
    path = inference.default_weights_path("line")
    model = inference.load_model(path)
    assert model.kind == "line"
    patch = np.full((64, 64), 255, np.uint8)
    patch[30:33, 8:56] = 0
    (prims,) = inference.predict_patches(model, [patch])
    assert prims
    best = max(prims, key=lambda p: p.length())
    assert best.length() > 30
    assert np.abs(best.pts[:, 1] - 31.5).max() < 3.0
