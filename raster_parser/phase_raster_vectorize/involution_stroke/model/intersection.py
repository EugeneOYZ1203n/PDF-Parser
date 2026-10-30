"""Stage 3 intersection network: upstream `intersection_model/mymodel.py`'s
`SceneGraphVectorModel` (the `--model` the README exports), vendored @
178eede602bfb488b46cfc38ea2176ff902a4da2.

It reads only the half-edge polylines: `lines` (B, K, 100, 2) -- row 0 the
background slot, rows 1..K-1 one half-edge each, resampled to 100 points in
tile pixel coordinates (unnormalized, exactly as upstream's `get_line_tensor`
builds them) -> per-polyline Fourier-feature MLP -> 4-layer set transformer
(no positional encoding) -> symmetric bilinear scores, `-inf` wherever the
constraint matrix `R` (B, K, K) is 0. `n_objects` (B,) masks padding.

Changes from upstream: the image backbone / torchvision import and the mock
/ sinkhorn training helpers are dropped, and the `(B, max_objects, d)`
shape assert is gone -- upstream's own `prepare_batch` never pads to
`max_objects` either (its traced export just didn't keep the assert)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

N_POINTS = 100  # samples per half-edge polyline
NUM_FREQS = 10


class FourierFeatures2d(nn.Module):
    def __init__(self, num_freqs: int = NUM_FREQS) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs)
        self.register_buffer("freqs", freqs * torch.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x.unsqueeze(-1) * self.freqs
        scaled = scaled.view(*x.shape[:-1], -1)
        return torch.cat([x, torch.sin(scaled), torch.cos(scaled)], dim=-1)


class PolylineBackbone(nn.Module):
    def __init__(self, N: int, C: int, num_freqs: int = NUM_FREQS) -> None:
        super().__init__()
        self.N = N
        self.C = C
        self.fourier = FourierFeatures2d(num_freqs=num_freqs)
        ff_dim = 2 + (4 * num_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(N * ff_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, C),
        )

    def forward(self, batch: dict, mask: "torch.Tensor | None" = None) -> torch.Tensor:
        x = batch["lines"]
        B, K, N, _ = x.shape
        assert N == self.N
        out = self.mlp(self.fourier(x).view(B, K, -1))
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out


class RelationSetTransformer(nn.Module):
    def __init__(
        self, input_dim: int, num_layers: int = 2, nhead: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        # nested tensors are unavailable with norm_first anyway (torch warns otherwise)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.input_dim = input_dim

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x.ndim == 3 and x.shape[2] == self.input_dim
        return self.transformer(
            x, src_key_padding_mask=key_padding_mask, mask=None, is_causal=False
        )


class SymmetricOutputHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)

    def forward(self, nodes: torch.Tensor, constraint_mask: torch.Tensor) -> torch.Tensor:
        d = nodes.shape[2]
        scores = torch.bmm(self.query(nodes), self.key(nodes).transpose(1, 2)) / (d**0.5)
        scores = scores.masked_fill(~(constraint_mask != 0), float("-inf"))
        return 0.5 * (scores + scores.transpose(1, 2))


class SceneGraphVectorModel(nn.Module):
    def __init__(self, max_objects: int = 1000, d_dim: int = 32) -> None:
        super().__init__()
        self.max_objects = max_objects
        self.d_dim = d_dim
        self.backbone = PolylineBackbone(N_POINTS, d_dim, num_freqs=NUM_FREQS)
        self.transformer = RelationSetTransformer(d_dim, num_layers=4, dropout=0.1)
        self.head = SymmetricOutputHead(d_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        """`batch`: `lines` (B, K, 100, 2), `R` (B, K, K), `n_objects` (B,).
        Returns symmetric scores (B, K, K)."""
        constraint_matrix = batch["R"]
        K = constraint_matrix.shape[-1]
        obj_embeddings = self.backbone(batch)
        range_vec = torch.arange(K, device=obj_embeddings.device).unsqueeze(0)
        key_padding_mask = range_vec >= batch["n_objects"].unsqueeze(1)
        context = self.transformer(obj_embeddings, key_padding_mask=key_padding_mask)
        return self.head(context, constraint_matrix)
