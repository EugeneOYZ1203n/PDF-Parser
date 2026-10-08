"""Primitive refinement (paper Sec. 3.3 + Appendix I): the network's
primitives are aligned to the patch raster by minimising a charge-
interaction energy with Adam, batched over many patches in torch.

Physics (Appendix I.1): filled pixels carry fixed negative charge (their
ink intensity q^), primitives positive charge spread over their area, so
primitives are attracted to ink and repel each other. Discretised (I.3),
the energy of primitive k with a charge grid q is

    E_k(q) = sum_i q_i * (phi (*) q_k)_i                      (Eq. 8 / 33)

with q_k the k-th primitive's pixel coverage and phi the potential

    phi(r) = exp(-r^2 / R_c^2) + lambda_f exp(-r^2 / R_f^2)   (Eq. 47)

(R_c = 1 px, R_f = 32 px, lambda_f = 0.02; phi is a sum of Gaussians, so
the convolution is separable). Mean-field optimisation (I.2, Eq. 28-30):

    E* = sum_k E_k(q^pos_k)|size frozen + E_k(q^size_k)|pos frozen + E^rdn_k

where every charge grid q^... is frozen (detached) at each step, so the
gradient only flows through the k-th primitive's own coverage -- and only
into its position parameters (midpoint + angles) for E^pos, its size
parameters (lengths + width) for E^size / E^rdn. Charges, with q_i = max
over primitives' coverages (charge saturation, Eq. 38/39) and the
connected-area mask c_k (Fig. 31):

    q^size_k = q - q^  inside c_k,   q_k outside               (Eq. 45)
    q^pos_k  = lambda_pos (q - q_k - q^) inside c_k,
               q - q_k - q^ outside                            (Eq. 40 / 46)
    q^rdn_k  = |m_k| exp(-(|l_k . m^_k| - 1)^2 beta),
               beta = (cos alpha_col - 1)^-2, phi truncated at r*, lambda_f = 0
                                                               (Eq. 41-44)

Every `REFINE_JOIN_EVERY` steps (Sec. 3.3, "every few iterations we join
lined up primitives by stretching one and collapsing the rest, and move
collapsed primitives into uncovered raster pixels"); primitives still
collapsed at the end are dropped.

Ours (the paper leaves these open, or we simplify):
  * coverage is the soft rasterisation clamp(w/2 + 1/2 - dist, 0, 1);
    quadratics are flattened into `CURVE_FLATTEN_SEGMENTS` segments for the
    distance (the paper flattens curves for its integrals too);
  * a curve is parameterised by its point at t = 1/2 (the paper uses the
    curve's intersection with the control-angle bisector) plus the two arm
    lengths and angles from there to the endpoints;
  * c_k: the contiguous run of filled pixels along the primitive's own
    (extended) line / parabola through its midpoint, widened by the filled
    run along its normal -- symmetric (max of the two sides) rather than the
    paper's two-sided rectangle;
  * directions in m_k are summed in doubled-angle form, so a segment and its
    reverse count the same;
  * the join / move heuristics' exact rules (join only lines, using the
    merge thresholds; move a collapsed primitive onto the most uncovered
    ink pixel with a 2 px length).
"""
from __future__ import annotations

import math

import numpy as np

from . import geometry as geo
from .config import (
    CURVE_FLATTEN_SEGMENTS, MERGE_LINE_ANGLE_DEG, MERGE_LINE_DIST_PX, MERGE_LINE_GAP_PX, REFINE_BATCH,
    REFINE_COLLAPSED_PX, REFINE_COLLINEAR_DEG, REFINE_ITERS, REFINE_JOIN_EVERY, REFINE_KERNEL_RADIUS,
    REFINE_LAMBDA_FAR, REFINE_LAMBDA_POS, REFINE_LR_ANGLE, REFINE_LR_PX, REFINE_MASK_EVERY, REFINE_MAX_WIDTH_PX,
    REFINE_MIN_WIDTH_PX, REFINE_R_CLOSE, REFINE_R_FAR, REFINE_RDN_RADIUS,
)


def refine_patches(grays: list[np.ndarray], prims: list[list[geo.Prim]], *, iters: int = REFINE_ITERS,
                   batch: int = REFINE_BATCH) -> list[list[geo.Prim]]:
    """Refine every patch's primitives against its own raster. `grays` are
    same-size uint8 patches (255 = paper); primitives in patch px. Lines and
    quadratics may not be mixed within one call."""
    if iters <= 0 or not grays:
        return [list(p) for p in prims]
    out: list[list[geo.Prim]] = [list(p) for p in prims]
    # similar primitive counts per chunk -> little padding to K; empty patches skipped
    order = sorted((i for i in range(len(grays)) if prims[i]), key=lambda i: len(prims[i]))
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        for i, ps in zip(idx, _refine_chunk([grays[i] for i in idx], [prims[i] for i in idx], iters)):
            out[i] = ps
    return out


# ---------------------------------------------------------------------------
# Parameterisation
# ---------------------------------------------------------------------------
def _to_params(chunk: list[list[geo.Prim]], is_line: bool, k_max: int):
    b = len(chunk)
    n_arm = 1 if is_line else 2
    mid = np.zeros((b, k_max, 2))
    ang = np.zeros((b, k_max, n_arm))
    lens = np.zeros((b, k_max, n_arm))
    width = np.ones((b, k_max))
    conf = np.zeros((b, k_max))
    valid = np.zeros((b, k_max), bool)
    for i, ps in enumerate(chunk):
        for k, p in enumerate(ps):
            pts = np.asarray(p.pts, float)
            if is_line:
                d = pts[1] - pts[0]
                mid[i, k] = 0.5 * (pts[0] + pts[1])
                ang[i, k, 0] = math.atan2(d[1], d[0])
                lens[i, k, 0] = float(np.hypot(*d))
            else:
                m = 0.25 * (pts[0] + 2 * pts[1] + pts[2])  # Q(1/2)
                mid[i, k] = m
                for a, e in enumerate((pts[0], pts[2])):
                    v = e - m
                    ang[i, k, a] = math.atan2(v[1], v[0])
                    lens[i, k, a] = float(np.hypot(*v))
            width[i, k] = p.width
            conf[i, k] = p.conf
            valid[i, k] = True
    return mid, ang, lens, width, conf, valid


def _ctrl(mid, ang, lens, is_line: bool):
    """Torch params -> control points `(B, K, 2|3, 2)`."""
    import torch

    u = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)  # (B, K, n_arm, 2)
    if is_line:
        half = 0.5 * lens[..., 0:1] * u[..., 0, :]
        return torch.stack([mid - half, mid + half], dim=2)
    e1 = mid + lens[..., 0:1] * u[..., 0, :]
    e2 = mid + lens[..., 1:2] * u[..., 1, :]
    c = 2.0 * mid - 0.5 * (e1 + e2)
    return torch.stack([e1, c, e2], dim=2)


def _flatten(ctrl, is_line: bool, n_seg: int = CURVE_FLATTEN_SEGMENTS):
    """Control points -> polyline vertices `(B, K, n_seg + 1, 2)`."""
    import torch

    if is_line:
        return ctrl
    t = torch.linspace(0.0, 1.0, n_seg + 1, dtype=ctrl.dtype, device=ctrl.device)[:, None]
    mt = 1.0 - t
    p0, p1, p2 = ctrl[:, :, None, 0], ctrl[:, :, None, 1], ctrl[:, :, None, 2]
    return mt * mt * p0 + 2 * mt * t * p1 + t * t * p2


def _seg_distance(verts, grid, want_dir: bool = False):
    """Distance from every pixel centre `grid (HW, 2)` to the polyline
    `verts (B, K, V, 2)` -> `(B, K, HW)` (+ doubled-angle unit direction of
    the nearest segment `(B, K, HW, 2)`)."""
    import torch

    a = verts[:, :, :-1, None, :]                     # (B, K, S, 1, 2)
    d = verts[:, :, 1:, None, :] - a
    g = grid[None, None, None]                        # (1, 1, 1, HW, 2)
    dd = (d * d).sum(-1).clamp_min(1e-9)
    t = (((g - a) * d).sum(-1) / dd).clamp(0.0, 1.0)
    diff = g - a - t[..., None] * d
    dist2 = (diff * diff).sum(-1)                     # (B, K, S, HW)
    dmin, idx = dist2.min(dim=2)
    dist = torch.sqrt(dmin + 1e-12)
    if not want_dir:
        return dist, None
    ang = torch.atan2(d[..., 0, 1], d[..., 0, 0])    # (B, K, S)
    seg_ang = torch.gather(ang, 2, idx)               # (B, K, HW)
    return dist, torch.stack([torch.cos(2 * seg_ang), torch.sin(2 * seg_ang)], dim=-1)


class _Fields:
    """Everything one step needs about every primitive's soft coverage,
    computed without autograd: coverage `(B, K, HW)`, the nearest segment
    per pixel, the parameter t on it, the offset `diff = pixel - closest
    point`, and the doubled-angle segment direction."""

    def __init__(self, verts, width, valid, grid) -> None:
        import torch

        a = verts[:, :, :-1]                                  # (B, K, S, 2)
        d = verts[:, :, 1:] - a
        g = grid[None, None, None]                            # (1, 1, 1, HW, 2)
        dd = (d * d).sum(-1).clamp_min(1e-9)[..., None]       # (B, K, S, 1)
        t = (((g - a[..., None, :]) * d[..., None, :]).sum(-1) / dd).clamp(0.0, 1.0)
        diff = g - a[..., None, :] - t[..., None] * d[..., None, :]
        dmin, idx = (diff * diff).sum(-1).min(dim=2)          # (B, K, HW)
        self.idx = idx
        self.t = torch.gather(t, 2, idx[:, :, None]).squeeze(2)
        sel = idx[..., None].expand(*idx.shape, 2)
        a_sel = torch.gather(a, 2, sel)
        d_sel = torch.gather(d, 2, sel)
        self.diff = grid[None, None] - a_sel - self.t[..., None] * d_sel
        self.dist = torch.sqrt(dmin + 1e-12)
        val = 0.5 * width[..., None] + 0.5 - self.dist
        self.cov = val.clamp(0.0, 1.0) * valid[..., None]
        self.ramp = ((val > 0) & (val < 1)).to(verts.dtype) * valid[..., None]
        seg_ang = torch.gather(torch.atan2(d[..., 1], d[..., 0]), 2, idx)
        self.dir2 = torch.stack([torch.cos(2 * seg_ang), torch.sin(2 * seg_ang)], dim=-1)
        self.n_verts = verts.shape[2]

    def vjp(self, f):
        """Gradients of sum_i f_i * cov_i w.r.t. the polyline vertices
        `(B, K, V, 2)` and the widths `(B, K)`, analytically: on the ramp,
        dcov/dw = 1/2 and dcov/dd = -1, and (envelope theorem) the distance
        to the nearest segment (a, b) at parameter t moves with
        dd/da = -(1 - t) diff / d, dd/db = -t diff / d."""
        import torch

        w_ = f * self.ramp
        unit = self.diff / self.dist[..., None].clamp_min(1e-6)
        ga = (w_ * (1.0 - self.t))[..., None] * unit          # dE/da (B, K, HW, 2)
        gb = (w_ * self.t)[..., None] * unit
        out = torch.zeros(*self.idx.shape[:2], self.n_verts, 2, dtype=f.dtype)
        sel = self.idx[..., None].expand(*self.idx.shape, 2)
        out.scatter_add_(2, sel, ga)
        out.scatter_add_(2, sel + 1, gb)
        return out, 0.5 * w_.sum(-1)


# ---------------------------------------------------------------------------
# Potential (separable Gaussians)
# ---------------------------------------------------------------------------
class _Phi:
    """Convolution with phi. The close Gaussian (R_c = 1 px) runs at full
    resolution; the far one (R_f = 32 px) is smooth enough to run on a 4x
    average-pooled grid and be upsampled back (ours, for speed)."""

    POOL = 4

    def __init__(self, device, dtype) -> None:
        import torch

        def kernel(sigma: float, radius: int):
            x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
            return torch.exp(-(x * x) / (sigma * sigma))

        self.close = kernel(REFINE_R_CLOSE, max(1, int(math.ceil(3 * REFINE_R_CLOSE))))
        self.far = kernel(REFINE_R_FAR / self.POOL, max(1, REFINE_KERNEL_RADIUS // self.POOL))
        self.rdn = kernel(REFINE_R_CLOSE, REFINE_RDN_RADIUS)

    @staticmethod
    def _sep(y, k):
        import torch.nn.functional as F

        r = (len(k) - 1) // 2
        y = F.conv2d(y, k.view(1, 1, 1, -1), padding=(0, r))
        return F.conv2d(y, k.view(1, 1, -1, 1), padding=(r, 0))

    def full(self, x):
        import torch.nn.functional as F

        shape = x.shape
        y = x.reshape(-1, 1, shape[-2], shape[-1])
        out = self._sep(y, self.close)
        if REFINE_LAMBDA_FAR:
            coarse = F.avg_pool2d(y, self.POOL, ceil_mode=True) * (self.POOL * self.POOL)
            far = F.interpolate(self._sep(coarse, self.far), scale_factor=self.POOL, mode="bilinear",
                                align_corners=False)[..., :shape[-2], :shape[-1]]
            out = out + REFINE_LAMBDA_FAR * far
        return out.reshape(shape)

    def truncated(self, x):
        shape = x.shape
        return self._sep(x.reshape(-1, 1, shape[-2], shape[-1]), self.rdn).reshape(shape)


# ---------------------------------------------------------------------------
# Connected-area mask c_k (Appendix I, Fig. 31)
# ---------------------------------------------------------------------------
def _run_bounds(filled, center: int):
    """For bool `(..., N)` samples, the [lo, hi] index run of True through
    `center` (lo > hi when the centre itself is unfilled)."""
    import torch

    n = filled.shape[-1]
    idx = torch.arange(n, device=filled.device)
    right = (~filled) & (idx >= center)
    left = (~filled) & (idx <= center)
    hi = torch.where(right, idx, torch.full_like(idx, n)).min(-1).values - 1
    lo = torch.where(left, idx, torch.full_like(idx, -1)).max(-1).values + 1
    return lo, hi


def _sample(img, pts):
    """Nearest-pixel lookup of `img (B, H, W)` at `pts (B, K, N, 2)`; out of
    bounds reads 0."""
    import torch

    b, h, w = img.shape
    ij = torch.floor(pts).long()
    x, y = ij[..., 0], ij[..., 1]
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    flat = (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)).reshape(b, -1)
    vals = torch.gather(img.reshape(b, -1), 1, flat).reshape(x.shape)
    return torch.where(ok, vals, torch.zeros_like(vals))


def _area_mask(mid, ang, lens, is_line: bool, qhat, grid, size: int):
    """`(B, K, HW)` bool: the filled region primitive k could cover from its
    current position and orientation, at any size."""
    import torch

    n = 4 * size + 1
    center = n // 2
    filled_img = qhat >= 0.5
    if is_line:
        u = torch.stack([torch.cos(ang[..., 0]), torch.sin(ang[..., 0])], dim=-1)   # (B, K, 2)
        s = torch.linspace(-size, size, n, dtype=mid.dtype, device=mid.device)
        along = mid[:, :, None] + s[None, None, :, None] * u[:, :, None]
    else:
        ctrl = _ctrl(mid, ang, lens, False)
        t = torch.linspace(-1.5, 2.5, n, dtype=mid.dtype, device=mid.device)[:, None]  # the whole parabola
        mt = 1.0 - t
        along = mt * mt * ctrl[:, :, None, 0] + 2 * mt * t * ctrl[:, :, None, 1] + t * t * ctrl[:, :, None, 2]
        u = ctrl[..., 2, :] - ctrl[..., 0, :]
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    lo, hi = _run_bounds(_sample(filled_img.to(mid.dtype), along) > 0.5, center)
    nrm = torch.stack([-u[..., 1], u[..., 0]], dim=-1)
    s = torch.linspace(-size, size, n, dtype=mid.dtype, device=mid.device)
    across = mid[:, :, None] + s[None, None, :, None] * nrm[:, :, None]
    nlo, nhi = _run_bounds(_sample(filled_img.to(mid.dtype), across) > 0.5, center)
    step = 2.0 * size / (n - 1)
    half = torch.maximum((center - nlo).clamp_min(0), (nhi - center).clamp_min(0)).to(mid.dtype) * step + 0.5
    empty = lo > hi
    # the run as a polyline of n_sub vertices
    n_sub = 2 if is_line else 9
    frac = torch.linspace(0.0, 1.0, n_sub, dtype=mid.dtype, device=mid.device)
    lo_f, hi_f = lo.clamp_min(0).to(mid.dtype), hi.clamp_min(0).to(mid.dtype)
    sel = (lo_f[..., None] + (hi_f - lo_f)[..., None] * frac).round().long().clamp(0, n - 1)
    verts = torch.gather(along, 2, sel[..., None].expand(*sel.shape, 2))
    dist, _ = _seg_distance(verts, grid)
    return (dist <= half[..., None]) & ~empty[..., None]


# ---------------------------------------------------------------------------
# The optimisation
# ---------------------------------------------------------------------------
def _refine_chunk(grays: list[np.ndarray], chunk: list[list[geo.Prim]], iters: int) -> list[list[geo.Prim]]:
    import torch

    kinds = {p.is_line for ps in chunk for p in ps}
    if not kinds:
        return [[] for _ in chunk]
    if len(kinds) > 1:
        raise ValueError("refine_patches: lines and curves can't be refined in one call")
    is_line = kinds.pop()
    size = grays[0].shape[0]
    k_max = max(len(ps) for ps in chunk)
    mid0, ang0, lens0, w0, conf, valid_np = _to_params(chunk, is_line, k_max)

    dt = torch.float32
    with torch.enable_grad():
        mid = torch.tensor(mid0, dtype=dt, requires_grad=True)
        ang = torch.tensor(ang0, dtype=dt, requires_grad=True)
        lens = torch.tensor(lens0, dtype=dt, requires_grad=True)
        width = torch.tensor(w0, dtype=dt, requires_grad=True)
        valid = torch.tensor(valid_np, dtype=dt)
        qhat = torch.tensor(1.0 - np.stack(grays).astype(np.float32) / 255.0)        # (B, S, S)
        ys, xs = torch.meshgrid(torch.arange(size, dtype=dt) + 0.5, torch.arange(size, dtype=dt) + 0.5, indexing="ij")
        grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)                 # (HW, 2)
        phi = _Phi(None, dt)
        beta = 1.0 / (math.cos(math.radians(REFINE_COLLINEAR_DEG)) - 1.0) ** 2
        opt = torch.optim.Adam([
            {"params": [mid, lens, width], "lr": REFINE_LR_PX},
            {"params": [ang], "lr": REFINE_LR_ANGLE},
        ])
        b = len(grays)
        qhat_flat = qhat.reshape(b, 1, -1)
        mask = None
        for it in range(iters):
            if it % REFINE_MASK_EVERY == 0:
                with torch.no_grad():
                    mask = _area_mask(mid.detach(), ang.detach(), lens.detach(), is_line, qhat, grid, size)
            # the mean-field split -- E^pos moves only the position parameters,
            # E^size / E^rdn only the size ones -- is two VJPs: per-pixel ones
            # analytic (`_Fields.vjp`), params -> vertices by autograd (tiny)
            verts = _flatten(_ctrl(mid, ang, lens, is_line), is_line)
            with torch.no_grad():
                fld = _Fields(verts.detach(), width.detach(), valid, grid)
                qk, dir2 = fld.cov, fld.dir2
                q = qk.max(dim=1, keepdim=True).values                               # saturation
                q_size = torch.where(mask, q - qhat_flat, qk)
                base = q - qk - qhat_flat
                q_pos = torch.where(mask, REFINE_LAMBDA_POS * base, base)
                # collinearity charges (doubled-angle direction sums of the others)
                d2 = dir2 * qk[..., None]                                             # (B, K, HW, 2)
                m2 = d2.sum(dim=1, keepdim=True) - d2
                m_norm = m2.norm(dim=-1)
                cos2 = (dir2 * m2).sum(-1) / m_norm.clamp_min(1e-9)
                align = torch.sqrt(((1.0 + cos2) * 0.5).clamp(0.0, 1.0))
                q_rdn = m_norm * torch.exp(-((align - 1.0) ** 2) * beta)
                shape = (b, k_max, size, size)
                f_pos = phi.full(q_pos.reshape(shape)).reshape(b, k_max, -1)
                f_size = (phi.full(q_size.reshape(shape)) + phi.truncated(q_rdn.reshape(shape))).reshape(b, k_max, -1)
                gv_pos, _ = fld.vjp(f_pos)
                gv_size, g_width = fld.vjp(f_size)
            g_mid, g_ang = torch.autograd.grad(verts, [mid, ang], grad_outputs=gv_pos, retain_graph=True)
            (g_lens,) = torch.autograd.grad(verts, [lens], grad_outputs=gv_size)
            opt.zero_grad(set_to_none=True)
            mid.grad, ang.grad, lens.grad, width.grad = g_mid, g_ang, g_lens, g_width
            opt.step()
            with torch.no_grad():
                lens.clamp_(min=0.0)
                width.clamp_(REFINE_MIN_WIDTH_PX, REFINE_MAX_WIDTH_PX)
                if REFINE_JOIN_EVERY and (it + 1) % REFINE_JOIN_EVERY == 0 and it + 1 < iters:
                    _join_and_move(mid, ang, lens, width, valid, qhat, grid, is_line, size)

        ctrl = _ctrl(mid, ang, lens, is_line).detach().numpy()
        wid = width.detach().numpy()
        ln = lens.detach().numpy().sum(-1)
    out: list[list[geo.Prim]] = []
    for i in range(len(chunk)):
        ps = []
        for k in range(k_max):
            if valid_np[i, k] and ln[i, k] >= REFINE_COLLAPSED_PX:
                ps.append(geo.Prim(ctrl[i, k].astype(float), float(wid[i, k]), float(conf[i, k])))
        out.append(ps)
    return out


def _join_and_move(mid, ang, lens, width, valid, qhat, grid, is_line: bool, size: int) -> None:
    """In place (torch params): (1) lines lined up with a longer one are
    absorbed -- the longer is stretched over both, the other collapsed;
    (2) collapsed primitives jump onto the most uncovered ink pixel (coverage
    recomputed after step 1)."""
    m, a, ln = mid.detach().numpy(), ang.detach().numpy(), lens.detach().numpy()  # shared memory
    ok = valid.numpy() > 0
    b, k_max = ok.shape
    if is_line:
        cos_tol = math.cos(math.radians(MERGE_LINE_ANGLE_DEG))
        for i in range(b):
            ks = [k for k in np.argsort(-ln[i, :, 0]) if ok[i, k] and ln[i, k, 0] >= REFINE_COLLAPSED_PX]
            for n_a, ka in enumerate(ks):
                if ln[i, ka, 0] < REFINE_COLLAPSED_PX:
                    continue
                ua = np.array([math.cos(a[i, ka, 0]), math.sin(a[i, ka, 0])])
                na = np.array([-ua[1], ua[0]])
                for kc in ks[n_a + 1:]:
                    if ln[i, kc, 0] < REFINE_COLLAPSED_PX:
                        continue
                    uc = np.array([math.cos(a[i, kc, 0]), math.sin(a[i, kc, 0])])
                    if abs(float(ua @ uc)) < cos_tol:
                        continue
                    ends = m[i, kc] + np.outer([-0.5, 0.5], uc) * ln[i, kc, 0] - m[i, ka]
                    if np.abs(ends @ na).max() > MERGE_LINE_DIST_PX:
                        continue
                    ta = 0.5 * ln[i, ka, 0]
                    tc = ends @ ua
                    if tc.min() > ta + MERGE_LINE_GAP_PX or tc.max() < -ta - MERGE_LINE_GAP_PX:
                        continue
                    lo, hi = min(tc.min(), -ta), max(tc.max(), ta)
                    m[i, ka] += 0.5 * (lo + hi) * ua
                    ln[i, ka, 0] = hi - lo
                    ln[i, kc, 0] = 0.0
    collapsed = ok & (ln.sum(-1) < REFINE_COLLAPSED_PX)
    if not collapsed.any():
        return
    fld = _Fields(_flatten(_ctrl(mid.detach(), ang.detach(), lens.detach(), is_line), is_line),
                  width.detach(), valid, grid)
    uncovered = (qhat.reshape(b, -1) - fld.cov.max(dim=1).values).numpy().copy()
    for i, k in zip(*np.nonzero(collapsed)):
        j = int(np.argmax(uncovered[i]))
        if uncovered[i, j] <= 0.5:
            continue
        y, x = divmod(j, size)
        m[i, k] = (x + 0.5, y + 0.5)
        ln[i, k] = 2.0 if is_line else 1.0  # 2 px line / two 1 px arms
        u2 = uncovered[i].reshape(size, size)
        u2[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = 0.0
