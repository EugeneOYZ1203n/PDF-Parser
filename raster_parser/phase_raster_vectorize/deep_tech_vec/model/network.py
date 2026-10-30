# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Derived from github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings
# @ 63cbc673 (vectorization/models/generic.py, vectorization/modules/
# {conv_modules,transformer,_transformer_modules,output,maybe_module}.py and
# run_pipeline.py::preprocess_image/serialize). Changes: the spec-driven
# loader is replaced by the two fixed specs in `config.UPSTREAM_SPECS`,
# torchvision's conv3x3/conv1x1 by plain bias-free `nn.Conv2d`, unused module
# kinds (VGG, Bottleneck, LSTM, FC, discriminator) dropped.
"""Egiazarian et al.'s primitive extraction network, as released by the
authors (`GenericVectorizationNet`), so their pretrained checkpoints load
with `strict=True`. Module/attribute names mirror upstream exactly -- they
*are* the state_dict keys.

    patch (B, 3, 64, 64): [ink, x * mask, y * mask]   (`prepare_input`)
      -> ResNet18 stem + `blocks` layers -> conv1x1 adapter to 256 ch
      -> `num_layers` Transformer decoder layers over n sine queries
      -> conv1x1 adapter to 512 -> Linear -> (B, n, out)

Output rows: coordinates (lines: `(tanh + 1) / 2`, curves: raw), width,
sigmoid confidence -- all normalised by the patch size (x 64 to get px)."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import N_PRIM_QUERIES, UPSTREAM_SPECS, WEIGHTS_SHA256

# output.final_fc out_features -> primitive kind
_KIND_BY_OUT = {spec["out_features"]: kind for kind, spec in UPSTREAM_SPECS.items()}


def _conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False)


class MaybeModule(nn.Module):
    """`layer` applied only when `maybe` -- but always registered, so its
    parameters/buffers are in the state_dict either way (as upstream)."""

    def __init__(self, maybe: bool = False, layer: "nn.Module | None" = None) -> None:
        super().__init__()
        self.maybe = maybe
        self.layer = layer

    def forward(self, x):
        return self.layer(x) if self.maybe else x


# ---------------------------------------------------------------------------
# ResNet feature extractor
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        resample: "nn.Module | None" = None,
        bn: bool = True,
    ) -> None:
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.resample = resample
        self.conv = nn.Sequential(
            _conv3x3(inplanes, planes, stride),
            MaybeModule(bn, nn.BatchNorm2d(planes)),
            nn.LeakyReLU(inplace=True),
            _conv3x3(planes, planes),
            MaybeModule(bn, nn.BatchNorm2d(planes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.resample is None else self.resample(x)
        return self.relu(self.conv(x) + identity)


# ResNet18: per layer (out channels, blocks, stride)
_RESNET18 = ([64, 128, 256, 512], [2, 2, 2, 2], [1, 2, 2, 2])


def resnet_features(
    in_channels: int, convmap_channels: int, blocks: int, bn: bool = True
) -> nn.Sequential:
    """Upstream `resnet_model_creator(conf='18')`: stem + `blocks` layers, the
    last block of the last layer projecting to `convmap_channels`."""
    outs, n_blocks, strides = _RESNET18
    pre = 64
    layers: list[nn.Module] = [
        nn.Sequential(
            nn.Conv2d(in_channels, pre, kernel_size=7, stride=2, padding=3, bias=False),
            MaybeModule(bn, nn.BatchNorm2d(pre)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
    ]
    cin = pre
    for idx in range(blocks):
        out, nb, stride = outs[idx], n_blocks[idx], strides[idx]
        convmap = convmap_channels if idx == blocks - 1 else None
        res1 = None
        if stride != 1 or cin != out:
            res1 = nn.Sequential(_conv1x1(cin, out, stride), MaybeModule(bn, nn.BatchNorm2d(out)))
        layer = [BasicBlock(cin, out, stride, res1, bn=bn)]
        c = out
        for _ in range(1, nb - 1):
            layer.append(BasicBlock(c, out, bn=bn))
        last_out, res_n = out, None
        if convmap and convmap != out:
            last_out = convmap
            res_n = nn.Sequential(_conv1x1(c, last_out), MaybeModule(bn, nn.BatchNorm2d(last_out)))
        layer.append(BasicBlock(c, last_out, 1, res_n, bn=bn))
        layers.append(nn.Sequential(*layer))
        cin = out
    return nn.Sequential(*layers)


class ResnetConv(nn.Module):
    def __init__(self, in_channels: int, convmap_channels: int, blocks: int) -> None:
        super().__init__()
        self.conv = resnet_features(in_channels, convmap_channels, blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvAdapter(nn.Module):
    """conv1x1 (2D) + BN + LeakyReLU: conv features -> transformer width."""

    def __init__(self, cin: int, cout: int, bn: bool = True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, (1, 1)),
            MaybeModule(bn, nn.BatchNorm2d(cout)),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransformerAdapter(nn.Module):
    """conv1x1 (1D) + LeakyReLU over the query rows (BN registered, unused)."""

    def __init__(self, cin: int, cout: int, bn: bool = False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(cin, cout, 1),
            MaybeModule(bn, nn.BatchNorm1d(cout)),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------------------
# Transformer decoder (upstream's copy of jadore801120/attention-is-all-you-need-pytorch)
# ---------------------------------------------------------------------------
class _Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LayerNormalization(nn.Module):
    """Upstream's layer norm: unbiased std, eps added to sigma, and a no-op
    when dim 1 has size 1."""

    def __init__(self, d_hid: int, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid))
        self.b_2 = nn.Parameter(torch.zeros(d_hid))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(1) == 1:
            return z
        mu = z.mean(dim=-1, keepdim=True)
        sigma = z.std(dim=-1, keepdim=True)
        return (z - mu) / (sigma + self.eps) * self.a_2 + self.b_2


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_head, self.d = n_head, d_model
        self.w_qs = nn.Parameter(torch.empty(n_head, d_model, d_model))
        self.w_ks = nn.Parameter(torch.empty(n_head, d_model, d_model))
        self.w_vs = nn.Parameter(torch.empty(n_head, d_model, d_model))
        for w in (self.w_qs, self.w_ks, self.w_vs):
            nn.init.xavier_normal_(w)
        self.temper = float(np.power(d_model, 0.5))
        self.proj = _Linear(n_head * d_model, d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: "torch.Tensor | None" = None, v: "torch.Tensor | None" = None
    ) -> torch.Tensor:
        k = q if k is None else k
        v = q if v is None else v
        h, d = self.n_head, self.d
        b, len_q, _ = q.shape
        len_k, len_v = k.shape[1], v.shape[1]
        q_s = torch.bmm(q.repeat(h, 1, 1).view(h, -1, d), self.w_qs).view(-1, len_q, d)
        k_s = torch.bmm(k.repeat(h, 1, 1).view(h, -1, d), self.w_ks).view(-1, len_k, d)
        v_s = torch.bmm(v.repeat(h, 1, 1).view(h, -1, d), self.w_vs).view(-1, len_v, d)
        attn = torch.softmax(torch.bmm(q_s, k_s.transpose(1, 2)) / self.temper, dim=-1)
        out = torch.bmm(self.attn_dropout(attn), v_s)
        out = torch.cat(torch.split(out, b, dim=0), dim=-1)
        out = self.dropout(self.proj(out)) + q
        return self.layer_norm(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid: int, d_inner: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner, 1)
        self.w_2 = nn.Conv1d(d_inner, d_hid, 1)
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.w_2(self.relu(self.w_1(x.transpose(1, 2)))).transpose(2, 1)
        return self.layer_norm(self.dropout(out) + x)


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_inner: int, n_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self, dec: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        dec = self.slf_attn(dec)
        dec = self.enc_attn(dec, enc, enc)
        return self.pos_ffn(dec)


def sinusoid_table(n_position: int, d_hid: int, scale: float = 1.0) -> torch.Tensor:
    """Upstream `get_sinusoid_encoding_table(n, d, scale=1)`."""
    pos = np.arange(n_position, dtype=np.float64)[:, None]
    j = np.arange(d_hid)
    table = (pos / np.power(10000, float(scale) * (j // 2) / d_hid)).astype(np.float32)
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.from_numpy(table)


class _SequentialDecoder(nn.Module):
    def __init__(self, feature_dim: int, ffn_dim: int, n_head: int, num_layers: int) -> None:
        super().__init__()
        self.transformer = nn.Sequential(
            *(TransformerLayer(feature_dim, ffn_dim, n_head) for _ in range(num_layers))
        )

    def forward(self, conv_features: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        h = queries
        for layer in self.transformer:
            h = layer(h, conv_features)
        return h


class TransformerDecoder(nn.Module):
    def __init__(self, feature_dim: int, ffn_dim: int, n_head: int, num_layers: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.decoder = _SequentialDecoder(feature_dim, ffn_dim, n_head, num_layers)
        self._query_cache: dict[int, torch.Tensor] = {}

    def forward(self, conv_features: torch.Tensor, n: int) -> torch.Tensor:
        if n not in self._query_cache:
            self._query_cache[n] = sinusoid_table(n, self.feature_dim)
        q = (
            self._query_cache[n]
            .to(conv_features.device)[None]
            .expand(conv_features.shape[0], -1, -1)
        )
        return self.decoder(conv_features, q.contiguous())


class Output(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.final_fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc = self.final_fc(x)
        coord = (torch.tanh(fc[..., :-1]) + 1.0) / 2.0 if fc.shape[2] <= 6 else fc[..., :-1]
        return torch.cat((coord, torch.sigmoid(fc[..., -1:])), dim=-1)


class UpstreamVectorizationNet(nn.Module):
    """Upstream `GenericVectorizationNet` with `use_fc_from_conv` /
    `use_fc_from_hidden` both on (as in both released specs)."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        spec = UPSTREAM_SPECS[kind]
        self.kind = kind
        self.n_prim = N_PRIM_QUERIES
        self.features = ResnetConv(spec["in_channels"], spec["convmap_channels"], spec["blocks"])
        self.fc_from_conv = MaybeModule(
            True, ConvAdapter(spec["convmap_channels"], spec["feature_dim"])
        )
        self.hidden = TransformerDecoder(
            spec["feature_dim"], spec["ffn_dim"], spec["n_head"], spec["num_layers"]
        )
        self.fc_from_hidden = MaybeModule(
            True, TransformerAdapter(spec["feature_dim"], spec["in_features"])
        )
        self.output = Output(spec["in_features"], spec["out_features"])

    def forward(self, images: torch.Tensor, n: "int | None" = None) -> torch.Tensor:
        """`images` (B, 3, S, S) from `prepare_input` -> `(B, n, out)`."""
        x = self.fc_from_conv(self.features(images))
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)  # (B, h*w, c)
        x = self.fc_from_hidden(self.hidden(x, n or self.n_prim))
        return self.output(x)


def prepare_input(gray: torch.Tensor) -> torch.Tensor:
    """uint8/float gray `(B, S, S)` (255 = paper) -> `(B, 3, S, S)`: ink in
    [0, 1] plus x/y coordinate channels (`1/S .. 1`) masked to the ink --
    upstream `run_pipeline.preprocess_image`."""
    b, h, w = gray.shape
    ink = 1.0 - gray.float() / 255.0
    mask = (ink > 0).float()
    xs = (torch.arange(1, w + 1, dtype=torch.float32, device=gray.device) / w)[
        None, None, :
    ].expand(b, h, w)
    ys = (torch.arange(1, h + 1, dtype=torch.float32, device=gray.device) / h)[
        None, :, None
    ].expand(b, h, w)
    return torch.stack([ink, xs * mask, ys * mask], dim=1)


def build_model(kind: str) -> UpstreamVectorizationNet:
    return UpstreamVectorizationNet(kind)


def _rename_keys(state: dict) -> dict:
    """Upstream `run_pipeline.serialize`: checkpoints saved before a refactor
    name the decoder `hidden.transformer.*`."""
    old = "hidden.transformer"
    return {
        ("hidden.decoder.transformer" + k[len(old) :] if k.startswith(old) else k): v
        for k, v in state.items()
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # The released checkpoints may pickle non-tensor objects; only unpickle
        # them fully when the file is byte-identical to a known release.
        if _sha256(path) not in WEIGHTS_SHA256.values():
            raise
        return torch.load(str(path), map_location="cpu", weights_only=False)


def load_weights(path: "str | Path") -> UpstreamVectorizationNet:
    """An upstream checkpoint (`{"model_state_dict": ...}`) -> the network,
    its kind (line / curve) read off the output layer's width."""
    path = Path(path)
    blob = _load_checkpoint(path)
    state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else None
    if not isinstance(state, dict) or "output.final_fc.weight" not in state:
        raise ValueError(f"{path}: not a Deep-Vectorization-of-Technical-Drawings checkpoint")
    state = _rename_keys(state)
    out = int(state["output.final_fc.weight"].shape[0])
    if out not in _KIND_BY_OUT:
        raise ValueError(
            f"{path}: output width {out} matches no known spec ({sorted(_KIND_BY_OUT)})"
        )
    model = build_model(_KIND_BY_OUT[out])
    # strict, except that checkpoints from torch < 0.4.1 have no BatchNorm
    # `num_batches_tracked` buffers (unused in eval mode)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(
            f"{path}: state_dict mismatch -- missing {missing[:5]}, unexpected {unexpected[:5]}"
        )
    return model
