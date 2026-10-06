"""Stroke Vectorizer (paper Algorithm 1): auto-regressive Bezier tracing.

    lst_0 = {curve_0};  while not eos: curve_t, eos = Transformer(lst_{t-1}, X_s)

`curve_t = (xc1, yc1, xc2, yc2, xe, ye)` -- the "partial format": curve t
starts where curve t-1 ended; `curve_0 = (xs, ys, xs, ys, xs, ys)` with
(xs, ys) the stroke's start endpoint. Coordinates are tile-normalized but
unconstrained (a control point may lie outside the tile).

X_s comes from the shared UNet at a coarse level (see `unet.py`), is
projected to `d_model` tokens (+ 2D sine positions) and encoded by a
TransformerEncoder; the curve sequence (+ 1D positions + the projected
stroke feature F_i) is decoded causally by a TransformerDecoder. An MLP
predicts the next curve and the end-of-sequence flag (a logit) per step."""
from __future__ import annotations

import torch
from torch import nn

from .layers import pos1d, pos2d


def _causal_mask(t: int, device) -> torch.Tensor:
    """Bool (True = may not attend), the same type as the padding mask."""
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)


class StrokeVectorizer(nn.Module):
    def __init__(self, cfg: dict, feat_ch: int, cond_dim: int) -> None:
        super().__init__()
        d = cfg["d_model"]
        self.d = d
        self.max_prims = cfg["max_prims"]
        self.proj = nn.Conv2d(feat_ch, d, 1)
        enc_layer = nn.TransformerEncoderLayer(d, cfg["n_heads"], cfg["ff_dim"], cfg["dropout"],
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg["vec_enc_layers"], enable_nested_tensor=False)
        dec_layer = nn.TransformerDecoderLayer(d, cfg["n_heads"], cfg["ff_dim"], cfg["dropout"],
                                               batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg["vec_dec_layers"])
        self.curve_in = nn.Linear(6, d)
        self.cond_in = nn.Linear(cond_dim, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, 7))

    def memory(self, xs: torch.Tensor) -> torch.Tensor:
        f = self.proj(xs)
        m, d, h, w = f.shape
        tokens = f.flatten(2).transpose(1, 2) + pos2d(h, w, d, f.device, f.dtype)[None]
        return self.encoder(tokens)

    def _decode(self, memory: torch.Tensor, cond: torch.Tensor, seq: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        t = seq.shape[1]
        tok = self.curve_in(seq) + pos1d(t, self.d, seq.device, seq.dtype)[None] + self.cond_in(cond)[:, None, :]
        out = self.decoder(tok, memory, tgt_mask=_causal_mask(t, seq.device),
                           tgt_key_padding_mask=pad_mask)
        y = self.head(out)
        return y[..., :6], y[..., 6]

    def forward(self, memory: torch.Tensor, cond: torch.Tensor, seq_in: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher forcing: `seq_in (M, T, 6)` = [curve_0, curve_1 .. curve_{T-1}]
        -> predictions for [curve_1 .. curve_T] `(M, T, 6)` + eos logits `(M, T)`."""
        return self._decode(memory, cond, seq_in, pad_mask)

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, cond: torch.Tensor, start: torch.Tensor,
                 max_prims: int | None = None) -> list[torch.Tensor]:
        """Algorithm 1 for `M` strokes at once. `start (M, 2)`. Returns one
        `(K_i, 6)` tensor of curves per stroke (K_i >= 1)."""
        max_prims = max_prims or self.max_prims
        m = start.shape[0]
        seq = start.repeat(1, 3)[:, None, :]  # curve_0
        done = torch.zeros(m, dtype=torch.bool, device=start.device)
        lengths = torch.full((m,), max_prims, dtype=torch.long, device=start.device)
        for step in range(max_prims):
            curves, eos = self._decode(memory, cond, seq)
            nxt = curves[:, -1:, :]
            seq = torch.cat([seq, nxt], dim=1)
            stop = (torch.sigmoid(eos[:, -1]) > 0.5) & ~done
            lengths[stop] = step + 1
            done |= stop
            if bool(done.all()):
                break
        return [seq[i, 1:1 + int(lengths[i])] for i in range(m)]
