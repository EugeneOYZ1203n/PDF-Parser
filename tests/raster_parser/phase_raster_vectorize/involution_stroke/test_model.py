from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the optional `ml` extra

from raster_parser.phase_raster_vectorize.involution_stroke import inference
from raster_parser.phase_raster_vectorize.involution_stroke.model.centerline import ResNextUNetLarge
from raster_parser.phase_raster_vectorize.involution_stroke.model.intersection import (
    SceneGraphVectorModel,
)
from raster_parser.phase_raster_vectorize.involution_stroke.model.weights import (
    load_centerline,
    load_intersection,
)

# A sample of the released checkpoints' own key names / shapes: the vendored
# classes (incl. the re-implemented torchvision Bottleneck) must match them.
_CENTERLINE_KEYS = {
    "conv_in.0.weight": (64, 1, 7, 7),
    "enc1.0.conv2.weight": (32, 1, 3, 3),  # ResNeXt: groups 32, base width 4
    "enc1.0.conv3.weight": (64, 32, 1, 1),
    "down4.weight": (1024, 512, 2, 2),
    "center.3.bn3.running_var": (1024,),
    "up4.expand_conv.weight": (2048, 1024, 1, 1),
    "dec1.0.downsample.0.weight": (96, 64, 1, 1),
    "final_conv.weight": (1, 96, 1, 1),
}
_INTERSECTION_KEYS = {
    "backbone.mlp.0.weight": (256, 4200),  # 100 points x (2 + 4 x 10 Fourier features)
    "transformer.transformer.layers.0.self_attn.in_proj_weight": (96, 32),
    "transformer.transformer.layers.3.linear1.weight": (128, 32),
    "head.key.weight": (32, 32),
}


def test_centerline_keys_match_release():
    sd = ResNextUNetLarge().state_dict()
    assert len(sd) == 462
    for k, shape in _CENTERLINE_KEYS.items():
        assert tuple(sd[k].shape) == shape, k


def test_intersection_keys_match_release():
    sd = SceneGraphVectorModel(d_dim=32).state_dict()
    assert len(sd) == 61
    for k, shape in _INTERSECTION_KEYS.items():
        assert tuple(sd[k].shape) == shape, k


def test_centerline_forward_shape():
    m = ResNextUNetLarge().eval()
    with torch.inference_mode():
        out = m(torch.zeros(1, 1, 32, 48))
    assert out.shape == (1, 1, 32, 48)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_intersection_scores_symmetric_and_masked():
    m = SceneGraphVectorModel(d_dim=32).eval()
    R = torch.ones(1, 5, 5)
    R[0, 1, 2] = R[0, 2, 1] = 0
    with torch.inference_mode():
        s = m({"lines": torch.randn(1, 5, 100, 2), "R": R, "n_objects": torch.tensor([5])})[0]
    assert torch.equal(torch.isinf(s), R[0] == 0)
    finite = torch.isfinite(s)
    assert torch.allclose(s[finite], s.T[finite])


def test_load_checkpoints_like_the_release(tmp_path):
    c = ResNextUNetLarge()
    torch.save(
        {"model_class": "ResNextUNetLarge", "state_dict": c.state_dict(), "epoch": 9},
        tmp_path / "c.pt",
    )
    i = SceneGraphVectorModel(d_dim=16)
    lightning_sd = {f"model.{k}": v for k, v in i.state_dict().items()}
    torch.save(
        {"state_dict": lightning_sd, "pytorch-lightning_version": "2.2.5"}, tmp_path / "i.pt"
    )

    c2 = load_centerline(tmp_path / "c.pt")
    i2 = load_intersection(tmp_path / "i.pt")
    assert i2.d_dim == 16
    assert torch.equal(c2.final_conv.weight, c.final_conv.weight)
    assert torch.equal(i2.head.key.weight, i.head.key.weight)


def test_load_centerline_rejects_other_classes(tmp_path):
    torch.save({"model_class": "ResNextUNet", "state_dict": {}}, tmp_path / "c.pt")
    with pytest.raises(ValueError, match="ResNextUNetLarge"):
        load_centerline(tmp_path / "c.pt")


def test_missing_weights_name_the_download_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_weights"):
        inference.load_centerline_model(tmp_path / "nope.pt")


def test_predict_centerline_pads_and_crops(tmp_path):
    class Echo(torch.nn.Module):
        def forward(self, x):
            assert x.shape[-1] % 16 == 0 and x.shape[-2] % 16 == 0
            return x

    tile = np.zeros((20, 37), np.uint8)
    tile[5, 3:30] = 255
    out = inference.predict_centerline(Echo(), tile)
    assert out.shape == tile.shape
    assert np.array_equal(out, tile)


@pytest.mark.skipif(
    not os.environ.get("RASTER_PARSER_RUN_INVOLUTION_TESTS")
    or not inference.default_centerline_path().is_file()
    or not inference.default_intersection_path().is_file(),
    reason="set RASTER_PARSER_RUN_INVOLUTION_TESTS=1 with the pretrained weights downloaded",
)
def test_real_weights_resolve_crossings():
    """Two diagonals, a circle and a long rectangle, all crossing each other
    (~25 graph polylines): the pretrained pairing joins them back into the
    four drawn strokes. (A lone `+` is *not* paired: the released model
    prefers to end both strokes there -- its training data lets a stroke
    end at any intersection.)"""
    import cv2

    w = 512
    gray = np.full((w, w), 255, np.uint8)
    cv2.line(gray, (40, 40), (w - 40, w - 60), 0, 3)
    cv2.line(gray, (40, w - 40), (w - 40, 60), 0, 3)
    cv2.circle(gray, (w // 2, w // 2), w // 4, 0, 3)
    cv2.rectangle(gray, (60, w // 2 - 30), (w - 60, w // 2 + 30), 0, 3)
    res = inference.vectorize_layer(gray)
    assert res.n_graph_polylines > 20
    assert len(res.strokes) == 4
