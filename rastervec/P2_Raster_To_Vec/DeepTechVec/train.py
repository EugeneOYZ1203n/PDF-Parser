"""Train DeepTechVec's primitive network (Egiazarian et al., ECCV 2020) on the
dataset DeepVectoriser's `prep_dataset.py` writes (this backend has no prep
script of its own -- see `train_data.py`).

    python -m rastervec.P2_Raster_To_Vec.DeepTechVec.train --data data/deepvec --prim line \\
        [--epochs 17] [--batch 128] [--no-amp] [--threads N] [--resume] \\
        [--out rastervec/weights/deep_tech_vec_line.pth]

How to read what this prints: `TRAINING.md` next to this file.

The paper trains separate line and quadratic-curve models (`--prim line` /
`--prim curve`); both can run at the same time as separate processes, and
alongside DeepVectoriser's own trainer, on the same `--data`.

Paper Appendix B: batch 128, Adam "with a scheduler with the same
hyperparameters as the original Transformer paper" (beta = 0.9 / 0.98,
eps = 1e-9, lr = factor * d^-0.5 * min(step^-0.5, step * warmup^-1.5),
warmup 4000), 15-17 epochs, random 64 x 64 crops with random rotation and
scaling. An epoch here is `--steps-per-epoch` batches of fresh random
patches (the paper pre-computes its augmented crops; we draw them on the
fly from the shared vector ground truth).

Progress: one tqdm bar per epoch with the live loss terms, validation after
every epoch on a fixed set of un-augmented patches (loss, raster IoU of the
predicted vs GT primitives drawn at their widths -- the paper's training
plot -- and Chamfer distance in px), all rows appended to
`<out stem>_train_log.csv` next to `--out` (named after the weights file so
a line and a curve run writing to the same folder don't share a log).
`<out>.last.ckpt` is saved every epoch (`--resume` continues from it);
`--out` gets the model with the best validation IoU -- the file the P2
adapter loads. A non-finite loss saves `<out>.last.ckpt` and exits 2.

On CUDA: mixed precision by default (bf16 where the GPU supports it, else
fp16 + loss scaling; --no-amp for fp32; the loss itself is always fp32),
fused Adam, cuDNN autotuning, TF32 matmuls, pinned host memory, and loss
terms summed on the GPU, read back only every --log-every steps.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from rastervec.P2_Raster_To_Vec.DeepTechVec import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepTechVec import train_data as td
from rastervec.P2_Raster_To_Vec.DeepTechVec.config import MODEL_DEFAULTS, PRIM_KINDS, WEIGHTS_FILENAMES
from rastervec.P2_Raster_To_Vec.DeepTechVec.model.losses import primitive_loss
from rastervec.P2_Raster_To_Vec.DeepTechVec.model.network import PrimitiveNet, build_model, prepare_input, save_weights

TINY_CONFIG = {**MODEL_DEFAULTS, "res_ch": 16, "n_dec": 1, "n_heads": 2, "ff_dim": 32}
_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"


class TrainError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class _Layers:
    """Lazily loaded dataset layers (masks memory-mapped from the decode cache)."""

    def __init__(self, data_dir: Path, keys: list[str], cache_dir: "Path | None" = None) -> None:
        self.dir = Path(data_dir) / "layers"
        self.keys = list(keys)
        self.cache_dir = cache_dir
        self._cache: dict[str, td.LayerData] = {}

    def get(self, key: str) -> td.LayerData:
        if key not in self._cache:
            self._cache[key] = td.load_layer(self.dir, key, self.cache_dir)
        return self._cache[key]

    def prepare(self, desc: str) -> None:
        """Decode every layer's PNG into the (shared) cache now."""
        for key in tqdm(self.keys, desc=desc, unit="layer", leave=False):
            td.gray_path(self.dir, key, self.cache_dir)


def _weights(index: dict, keys: list[str]) -> np.ndarray:
    w = np.array([math.sqrt(max(1, index.get("n_strokes", {}).get(k, 1))) for k in keys], float)
    return w / w.sum()


class PatchStream(torch.utils.data.IterableDataset):
    """`steps` random augmented batches per epoch."""

    def __init__(self, data_dir: Path, keys: list[str], probs: np.ndarray, cfg: dict, batch: int, steps: int,
                 seed: int, cache_dir: "Path | None" = None) -> None:
        super().__init__()
        self.data_dir, self.keys, self.probs = Path(data_dir), keys, probs
        self.cfg, self.batch, self.steps, self.seed = cfg, batch, steps, seed
        self.cache_dir = cache_dir
        self.epoch = 0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, self.epoch, wid])
        layers = _Layers(self.data_dir, self.keys, self.cache_dir)
        for _ in range(wid, self.steps, nw):
            patches = [td.sample_patch(layers.get(self.keys[int(rng.choice(len(self.keys), p=self.probs))]), rng,
                                       self.cfg["n_prim"], self.cfg["prim_kind"])
                       for _b in range(self.batch)]
            yield td.build_batch(patches, self.cfg["n_prim"], self.cfg["prim_kind"])


def fixed_patches(layers: _Layers, probs: np.ndarray, n: int, cfg: dict, seed: int, desc: str) -> list[td.Patch]:
    rng = np.random.default_rng(seed)
    return [td.sample_patch(layers.get(layers.keys[int(rng.choice(len(layers.keys), p=probs))]), rng,
                            cfg["n_prim"], cfg["prim_kind"], augment=False)
            for _ in tqdm(range(n), desc=desc, unit="patch", leave=False)]


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
def _setup_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True        # fixed patch size: autotuned kernels pay off
        torch.set_float32_matmul_precision("high")   # TF32 for the fp32 parts


def _amp_dtype(device: torch.device, enabled: bool):
    """bf16 where the GPU has it (no loss scaling needed), else fp16; None
    = full precision (always on CPU)."""
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _loader(stream, workers: int, device: torch.device) -> torch.utils.data.DataLoader:
    # persistent_workers stays off: each epoch's workers must see the new `stream.epoch`
    extra = {"prefetch_factor": 4} if workers > 0 else {}
    return torch.utils.data.DataLoader(stream, batch_size=None, num_workers=workers, persistent_workers=False,
                                       pin_memory=device.type == "cuda", **extra)


def _accumulate(sums: dict, terms: dict) -> None:
    """Sum loss terms without a host sync (tensors stay on the device)."""
    for k, v in terms.items():
        v = v.detach().float() if torch.is_tensor(v) else float(v)
        sums[k] = sums[k] + v if k in sums else v


def _means(sums: dict, n: int) -> dict[str, float]:
    return {k: float(v) / max(n, 1) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Schedule + step
# ---------------------------------------------------------------------------
def noam_lr(step: int, d_model: int, warmup: int, factor: float) -> float:
    """The original Transformer's learning-rate schedule."""
    step = max(1, step)
    return factor * d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def train_step(model: PrimitiveNet, batch: dict, device, amp_dtype=None) -> tuple[torch.Tensor, dict]:
    device = torch.device(device)
    gray = torch.as_tensor(batch["gray"]).to(device, non_blocking=True)
    target = torch.as_tensor(batch["target"]).to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        pred = model(prepare_input(gray))
    return primitive_loss(pred.float(), target.float())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _chamfer(pred: list[geo.Prim], gt: list[geo.Prim], size: int) -> float:
    """Symmetric mean nearest distance (px) between the sampled primitives."""
    if not pred and not gt:
        return 0.0
    if not pred or not gt:
        return size / 2.0

    def one_way(a, b):
        dist = cv2.distanceTransform((~td.prim_mask(b, size, 1)).astype(np.uint8), cv2.DIST_L2, 3)
        pts = np.concatenate([geo.prim_points(p, 8) for p in a])
        ij = np.clip(np.floor(pts).astype(int), 0, size - 1)
        return float(dist[ij[:, 1], ij[:, 0]].mean())

    return 0.5 * (one_way(pred, gt) + one_way(gt, pred))


@torch.inference_mode()
def validate(model: PrimitiveNet, patches: list[td.Patch], device, batch: int) -> dict:
    model.eval()
    losses, ious, chams = [], [], []
    try:
        for b0 in range(0, len(patches), batch):
            chunk = patches[b0:b0 + batch]
            bt = td.build_batch(chunk, model.cfg["n_prim"], model.kind)
            pred = model(prepare_input(torch.as_tensor(bt["gray"]).to(device)))
            loss, _ = primitive_loss(pred, torch.as_tensor(bt["target"]).to(device))
            losses.append(float(loss) * len(chunk))
            for p, row in zip(chunk, pred.cpu().numpy()):
                size = p.gray.shape[0]
                prims = td.decode_output(row, model.kind, size)
                a, g = td.prim_mask(prims, size), td.prim_mask(p.prims, size)
                union = (a | g).sum()
                ious.append(float((a & g).sum() / union) if union else 1.0)
                chams.append(_chamfer(prims, p.prims, size))
    finally:
        model.train()
    return {"val_loss": sum(losses) / max(len(patches), 1),
            "val_iou": float(np.mean(ious)) if ious else float("nan"),
            "val_chamfer_px": float(np.mean(chams)) if chams else float("nan")}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_last(path: Path, model, opt, step: int, epoch: int, best: float, scaler=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": model.cfg, "model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else {},
                "step": step, "epoch": epoch, "best": best}, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="DeepVectoriser prep_dataset.py output directory")
    ap.add_argument("--prim", choices=PRIM_KINDS, default="line", help="primitive kind (paper: separate models)")
    ap.add_argument("--out", default=None,
                    help="weights file the adapter loads (default rastervec/weights/deep_tech_vec_<prim>.pth)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=17, help="paper: 15 (ABC) / 17 (PFP)")
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128, help="paper: 128")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--warmup", type=int, default=4000, help="Noam warmup steps (Transformer paper)")
    ap.add_argument("--lr-factor", type=float, default=1.0, help="Noam schedule factor")
    ap.add_argument("--val-crops", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="mixed precision on CUDA (bf16 where supported, else fp16); --no-amp for fp32")
    ap.add_argument("--log-every", type=int, default=20, help="progress-bar refresh interval (steps)")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (default: torch's own choice, ~all cores)")
    ap.add_argument("--cache-dir", default=None,
                    help="where layer PNGs are decoded once (default: <data>/cache, shared with other trainers)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>.last.ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="tiny model (smoke tests only)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data = Path(args.data)
    out = Path(args.out) if args.out else _WEIGHTS_DIR / WEIGHTS_FILENAMES[args.prim]
    last = out.with_name(out.name + ".last.ckpt")
    log_csv = out.with_name(f"{out.stem}_train_log.csv")
    device = torch.device(args.device)
    _setup_device(device)
    amp_dtype = _amp_dtype(device, args.amp)
    if args.threads:
        torch.set_num_threads(args.threads)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    torch.manual_seed(args.seed)

    def banner(msg: str) -> None:
        tqdm.write("=" * 72 + f"\n{msg}\n" + "=" * 72)

    index = td.load_index(data)
    if not index["train"]:
        raise SystemExit(f"{data}/index.json has no training layers -- run DeepVectoriser's prep_dataset.py first")
    resume_blob = None
    if args.resume:
        if not last.is_file():
            raise SystemExit(f"--resume: {last} not found")
        resume_blob = torch.load(last, map_location="cpu", weights_only=False)
    cfg = resume_blob["config"] if resume_blob else {**(TINY_CONFIG if args.tiny else MODEL_DEFAULTS),
                                                     "prim_kind": args.prim}
    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    banner(f"DeepTechVec training ({cfg['prim_kind']}) | device {device}{gpu}"
           f"{f' | AMP ' + str(amp_dtype).split('.')[-1] if amp_dtype else ''}"
           f"{f' | {torch.get_num_threads()} threads' if device.type == 'cpu' else ''} | data {data}\n"
           f"train layers {len(index['train'])}, val layers {len(index['val'])} | out {out}\nmodel {cfg}")

    train_keys = index["train"]
    probs = _weights(index, train_keys)
    _Layers(data, train_keys, cache_dir).prepare("decode train layers")
    if index["val"]:
        val_layers = _Layers(data, index["val"], cache_dir)
        val_layers.prepare("decode val layers")
        val = fixed_patches(val_layers, _weights(index, index["val"]), args.val_crops, cfg, args.seed + 1, "val patches")
    else:
        tqdm.write("WARNING: no val pages in index.json -- validating on training-page patches")
        val = fixed_patches(_Layers(data, train_keys, cache_dir), probs, args.val_crops, cfg, args.seed + 1,
                            "val patches")
    tqdm.write(f"val: {len(val)} patches, {sum(len(p.prims) for p in val)} GT primitives, "
               f"{sum(p.overflow for p in val)} truncated to n_prim={cfg['n_prim']}")

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9,
                           fused=device.type == "cuda" or None)
    # loss scaling only for fp16 (bf16 has fp32's range); disabled = pass-through
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    step, epoch0, best = 0, 0, -1.0
    if resume_blob:
        model.load_state_dict(resume_blob["model"])
        opt.load_state_dict(resume_blob["opt"])
        if resume_blob.get("scaler"):  # absent / empty for older or fp32 checkpoints
            scaler.load_state_dict(resume_blob["scaler"])
        step, epoch0, best = resume_blob["step"], resume_blob["epoch"], resume_blob["best"]
        tqdm.write(f"resumed from {last}: epoch {epoch0}, step {step}, best val IoU {best:.4f}")

    new_csv = not log_csv.exists()
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(log_csv, "a", newline="", encoding="utf-8")
    fields = ["time", "prim", "epoch", "steps", "lr", "loss", "L_cls", "L_loc", "overflow",
              "val_loss", "val_iou", "val_chamfer_px", "seconds"]
    writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
    if new_csv:
        writer.writeheader()

    stream = PatchStream(data, train_keys, probs, cfg, args.batch, args.steps_per_epoch, args.seed, cache_dir)
    ctx = {"epoch": None, "step": None}
    try:
        model.train()
        for ep in range(epoch0, args.epochs):
            ctx["epoch"] = ep
            stream.epoch = ep
            t_ep = time.perf_counter()
            loader = _loader(stream, args.workers, device)
            sums: dict[str, float] = {}
            n = 0
            opt.zero_grad(set_to_none=True)
            bar = tqdm(loader, total=args.steps_per_epoch, desc=f"{cfg['prim_kind']} ep {ep + 1}/{args.epochs}",
                       unit="batch")
            for i, batch in enumerate(bar):
                ctx["step"] = i
                loss, terms = train_step(model, batch, device, amp_dtype)
                if not torch.isfinite(loss):
                    raise TrainError(f"non-finite loss {loss.item()} (terms {terms})")
                scaler.scale(loss / args.accum).backward()
                if (i + 1) % args.accum == 0:
                    step += 1
                    lr = noam_lr(step, model.d_emb, args.warmup, args.lr_factor)
                    for g in opt.param_groups:
                        g["lr"] = lr
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                n += 1
                _accumulate(sums, {"loss": loss, **terms, "overflow": batch["overflow"]})
                if n % args.log_every == 0:  # a host sync -- not every step
                    bar.set_postfix({k: f"{v:.4f}" for k, v in _means(sums, n).items()}, refresh=False)
            bar.close()
            ctx["step"] = "validation"
            metrics = validate(model, val, device, args.batch)
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "prim": cfg["prim_kind"], "epoch": ep, "steps": n,
                   "lr": f"{noam_lr(step, model.d_emb, args.warmup, args.lr_factor):.3e}",
                   **{k: round(v, 6) for k, v in _means(sums, n).items()},
                   **{k: round(v, 4) for k, v in metrics.items()},
                   "seconds": round(time.perf_counter() - t_ep, 1)}
            writer.writerow(row)
            csv_file.flush()
            tqdm.write(f"[ep {ep + 1}/{args.epochs}] " + ", ".join(
                f"{k}={v}" for k, v in row.items() if k not in ("time", "prim", "epoch")))
            if metrics["val_iou"] > best:
                best = metrics["val_iou"]
                save_weights(out, model, {"epoch": ep, "step": step, **metrics})
                tqdm.write(f"  new best val IoU {best:.4f} -> {out}")
            save_last(last, model, opt, step, ep + 1, best, scaler)
        if not out.exists():
            save_weights(out, model, {"note": "no validation improvement"})
            tqdm.write(f"saved final model -> {out}")
        banner(f"done. best val IoU {best:.4f} | weights {out} | log {log_csv}")
        return 0
    except (TrainError, torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and not isinstance(exc, (TrainError, torch.cuda.OutOfMemoryError)) \
                and "out of memory" not in str(exc):
            raise
        tqdm.write(f"\nTRAINING FAILED at epoch={ctx['epoch']} step={ctx['step']}: {type(exc).__name__}: {exc}")
        try:
            save_last(last, model, opt, step, ctx["epoch"] or 0, best, scaler)
            tqdm.write(f"state saved to {last} (resume with --resume; the failed epoch restarts)")
        except Exception as save_exc:  # noqa: BLE001
            tqdm.write(f"could not save {last}: {save_exc}")
        return 2
    finally:
        csv_file.close()


if __name__ == "__main__":
    sys.exit(main())
