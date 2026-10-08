"""Train ImplicitSketchVec (Yan et al., SIGGRAPH 2024) on the dataset
DeepVectoriser's `prep_dataset.py` writes (this backend has no prep script of
its own -- see `train_data.py`).

    python -m rastervec.P2_Raster_To_Vec.ImplicitSketchVec.train --data data/deepvec --stage dfp   [--threads N]
    python -m rastervec.P2_Raster_To_Vec.ImplicitSketchVec.train --data data/deepvec --stage ndc   [--threads N]
    python -m rastervec.P2_Raster_To_Vec.ImplicitSketchVec.train --data data/deepvec --stage joint [--threads N]

How to read what this prints: `TRAINING.md` next to this file.

The paper's schedule (Sec. 6, Eq. 5): "after the two sub-networks have been
properly trained, we continue fine-tuning the end-to-end pipeline":

  dfp    the Distance Field Prediction network alone, on L_line (Eq. 2)
         summed over its six UDFs, from the raster;
  ndc    the Line Reconstruction network alone, on L_rec (Eq. 4), from the
         *ground-truth* centerline UDF plus Gaussian noise (std = 1 % of the
         UDF's own std -- the level of the paper's Appendix C robustness
         test; training on it is ours). Independent of `dfp`, so the two can
         run at the same time as separate processes;
  joint  both, end to end -- the NDC reads the DFP's predicted centerline
         UDF -- on L_rec + sum L_line, starting from the two stage files
         (`--init-dfp` / `--init-ndc`, default the stages' default outputs).
         Writes `implicit_sketch_vec.pth`, the file the adapter loads.

Each stage writes its own `--out` (default `rastervec/weights/
implicit_sketch_vec_<stage>.pth`, joint: `implicit_sketch_vec.pth`), its
`<out>.last.ckpt` (every epoch; `--resume`) and `<out stem>_train_log.csv`.
Crops: random 128-256 px (step 32, one size per batch), random rotation +
flip, targets generated on the fly. Adam (lr 1e-4, ours -- the paper gives
no optimiser settings). A non-finite loss saves `<out>.last.ckpt` and
exits 2.

Validation after every epoch on fixed un-augmented 128 px crops: the masked
centerline UDF error in px (dfp, joint) and the edge-flag precision / recall
/ F1 over flagged edges (ndc, joint). `--out` keeps the best: the lowest
validation L_line for `dfp`, the highest edge F1 for `ndc` / `joint`.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import train_data as td
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec.config import (
    BASIC_OVERRIDES, MODEL_DEFAULTS, NDC_INPUT_NOISE_FRAC, STAGE_WEIGHTS_FILENAMES, UDF_TRUNC_PX,
)
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec.model.losses import rec_loss, udf_loss
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec.model.network import (
    ImplicitSketchNet, build_model, load_submodule, prepare_input, save_weights,
)

STAGES = ("dfp", "ndc", "joint")
TINY_CONFIG = {**MODEL_DEFAULTS, "dfp_ch": 16, "dfp_cardinality": 2, "dfp_dilations": [1, 2], "dfp_hr_ch": 8,
               "ndc_ch": 8, "ndc_branch_layers": 1, "ndc_trunk_layers": 1}
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
        for key in tqdm(self.keys, desc=desc, unit="layer", leave=False):
            td.gray_path(self.dir, key, self.cache_dir)


def _weights(index: dict, keys: list[str]) -> np.ndarray:
    w = np.array([math.sqrt(max(1, index.get("n_strokes", {}).get(k, 1))) for k in keys], float)
    return w / w.sum()


class CropStream(torch.utils.data.IterableDataset):
    """`steps` random augmented batches per epoch (one crop size per batch)."""

    def __init__(self, data_dir: Path, keys: list[str], probs: np.ndarray, batch: int, steps: int, seed: int,
                 sizes=td.SIZES, cache_dir: "Path | None" = None) -> None:
        super().__init__()
        self.data_dir, self.keys, self.probs = Path(data_dir), keys, probs
        self.batch, self.steps, self.seed, self.sizes = batch, steps, seed, tuple(sizes)
        self.cache_dir = cache_dir
        self.epoch = 0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, self.epoch, wid])
        layers = _Layers(self.data_dir, self.keys, self.cache_dir)
        for _ in range(wid, self.steps, nw):
            size = int(rng.choice(self.sizes))
            crops = [td.sample_crop(layers.get(self.keys[int(rng.choice(len(self.keys), p=self.probs))]), size, rng)
                     for _b in range(self.batch)]
            yield td.build_batch(crops)


def fixed_batches(layers: _Layers, probs: np.ndarray, n: int, size: int, batch: int, seed: int,
                  desc: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    crops = [td.sample_crop(layers.get(layers.keys[int(rng.choice(len(layers.keys), p=probs))]), size, rng,
                            augment=False) for _ in tqdm(range(n), desc=desc, unit="crop", leave=False)]
    return [td.build_batch(crops[b0:b0 + batch]) for b0 in range(0, len(crops), batch)]


def _to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            t = torch.as_tensor(v)
            if k == "edge":
                t = t.long()
            elif k != "gray":
                t = t.float()
            out[k] = t.to(device)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def ndc_input_from_gt(udf_px: torch.Tensor, noise_frac: float = NDC_INPUT_NOISE_FRAC) -> torch.Tensor:
    """GT centerline UDF `(B, 1, L, L)` px -> normalised NDC input with
    Gaussian noise (std = `noise_frac` x the UDF's own std, per sample)."""
    u = udf_px / UDF_TRUNC_PX
    if noise_frac > 0:
        std = u.flatten(1).std(dim=1).view(-1, 1, 1, 1)
        u = u + torch.randn_like(u) * std * noise_frac
    return u.clamp(0.0, 1.0)


def step(model: ImplicitSketchNet, stage: str, b: dict) -> tuple[torch.Tensor, dict]:
    if stage == "dfp":
        return udf_loss(model.dfp(prepare_input(b["gray"])), b["udf"], b["mask"])
    if stage == "ndc":
        return rec_loss(model.ndc(ndc_input_from_gt(b["udf"][:, :1])), b)
    udf, out = model(prepare_input(b["gray"]))
    l_line, t_line = udf_loss(udf, b["udf"], b["mask"])
    l_rec, t_rec = rec_loss(out, b)
    return l_rec + l_line, {**t_line, **t_rec}


def stage_params(model: ImplicitSketchNet, stage: str):
    return list(model.dfp.parameters()) if stage == "dfp" else (
        list(model.ndc.parameters()) if stage == "ndc" else list(model.parameters()))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.inference_mode()
def validate(model: ImplicitSketchNet, stage: str, batches: list[dict], device) -> dict:
    model.eval()
    err_sum, err_n = 0.0, 0.0
    tp = fp = fn = 0
    try:
        for batch in batches:
            b = _to_device(batch, device)
            udf = None
            if stage in ("dfp", "joint"):
                udf = model.dfp(prepare_input(b["gray"]))
                m = b["mask"]
                err_sum += float(((udf[:, :1] * UDF_TRUNC_PX - b["udf"][:, :1]).abs() * m).sum())
                err_n += float(m.sum())
            if stage in ("ndc", "joint"):
                inp = udf[:, :1] if udf is not None else ndc_input_from_gt(b["udf"][:, :1], 0.0)
                pred = model.ndc(inp)["edge"].argmax(dim=1)
                gt = b["edge"]
                for bit in (1, 2):
                    p = (pred & bit) > 0
                    g = (gt & bit) > 0
                    tp += int((p & g).sum())
                    fp += int((p & ~g).sum())
                    fn += int((~p & g).sum())
    finally:
        model.train()
    out = {}
    if err_n:
        out["val_center_err_px"] = err_sum / err_n
    if stage in ("ndc", "joint"):
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out.update(val_edge_precision=prec, val_edge_recall=rec,
                   val_edge_f1=2 * prec * rec / max(prec + rec, 1e-9))
    return out


def _score(stage: str, metrics: dict) -> float:
    """Higher is better."""
    return -metrics["val_center_err_px"] if stage == "dfp" else metrics["val_edge_f1"]


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_last(path: Path, model, opt, epoch: int, best: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": model.cfg, "model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch, "best": best}, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="DeepVectoriser prep_dataset.py output directory")
    ap.add_argument("--stage", choices=STAGES, required=True, help="dfp / ndc (independent) then joint")
    ap.add_argument("--out", default=None,
                    help="weights file (default rastervec/weights/implicit_sketch_vec[_<stage>].pth)")
    ap.add_argument("--init-dfp", default=None, help="joint: DFP weights (default the dfp stage's default --out)")
    ap.add_argument("--init-ndc", default=None, help="joint: NDC weights (default the ndc stage's default --out)")
    ap.add_argument("--basic", action="store_true", help="the paper's reduced-size model (128-channel DFP)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-crops", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (targets are built on the fly)")
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
    stage = args.stage
    out = Path(args.out) if args.out else _WEIGHTS_DIR / STAGE_WEIGHTS_FILENAMES[stage]
    last = out.with_name(out.name + ".last.ckpt")
    log_csv = out.with_name(f"{out.stem}_train_log.csv")
    device = torch.device(args.device)
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
    if resume_blob:
        cfg = resume_blob["config"]
    else:
        cfg = dict(TINY_CONFIG if args.tiny else MODEL_DEFAULTS)
        if args.basic and not args.tiny:
            cfg.update(BASIC_OVERRIDES)
    model = build_model(cfg).to(device)
    if stage == "joint" and not resume_blob:
        for name, given in (("dfp", args.init_dfp), ("ndc", args.init_ndc)):
            path = Path(given) if given else _WEIGHTS_DIR / STAGE_WEIGHTS_FILENAMES[name]
            if not path.is_file():
                raise SystemExit(f"--stage joint needs trained {name.upper()} weights: {path} not found "
                                 f"(run --stage {name} first, or pass --init-{name})")
            load_submodule(model, path, name)
            tqdm.write(f"{name.upper()} initialised from {path}")
    banner(f"ImplicitSketchVec training, stage {stage} | device {device}"
           f"{f' | {torch.get_num_threads()} threads' if device.type == 'cpu' else ''} | data {data}\n"
           f"train layers {len(index['train'])}, val layers {len(index['val'])} | out {out}\nmodel {cfg}")

    train_keys = index["train"]
    probs = _weights(index, train_keys)
    _Layers(data, train_keys, cache_dir).prepare("decode train layers")
    val_keys = index["val"] or train_keys
    if not index["val"]:
        tqdm.write("WARNING: no val pages in index.json -- validating on training-page crops")
    val_layers = _Layers(data, val_keys, cache_dir)
    val_layers.prepare("decode val layers")
    val = fixed_batches(val_layers, _weights(index, val_keys), args.val_crops, 128, args.batch, args.seed + 1,
                        "val crops")

    opt = torch.optim.Adam(stage_params(model, stage), lr=args.lr)
    epoch0, best = 0, -float("inf")
    if resume_blob:
        model.load_state_dict(resume_blob["model"])
        opt.load_state_dict(resume_blob["opt"])
        epoch0, best = resume_blob["epoch"], resume_blob["best"]
        tqdm.write(f"resumed from {last}: epoch {epoch0}, best score {best:.4f}")

    new_csv = not log_csv.exists()
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(log_csv, "a", newline="", encoding="utf-8")
    fields = ["time", "stage", "epoch", "steps", "loss", "L_line", "L_line_center_px", "L_edge", "L_vertex",
              "L_skel", "val_center_err_px", "val_edge_precision", "val_edge_recall", "val_edge_f1", "seconds"]
    writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
    if new_csv:
        writer.writeheader()

    stream = CropStream(data, train_keys, probs, args.batch, args.steps_per_epoch, args.seed, cache_dir=cache_dir)
    ctx = {"epoch": None, "step": None, "size": None}
    params = stage_params(model, stage)
    try:
        model.train()
        for ep in range(epoch0, args.epochs):
            ctx["epoch"] = ep
            stream.epoch = ep
            t_ep = time.perf_counter()
            loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers,
                                                 persistent_workers=False)
            sums: dict[str, float] = {}
            n = 0
            opt.zero_grad(set_to_none=True)
            bar = tqdm(loader, total=args.steps_per_epoch, desc=f"{stage} ep {ep + 1}/{args.epochs}", unit="batch")
            for i, batch in enumerate(bar):
                ctx.update(step=i, size=int(batch["size"]))
                loss, terms = step(model, stage, _to_device(batch, device))
                if not torch.isfinite(loss):
                    raise TrainError(f"non-finite loss {loss.item()} (terms {terms})")
                (loss / args.accum).backward()
                if (i + 1) % args.accum == 0:
                    nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                n += 1
                for k, v in {"loss": loss.item(), **terms}.items():
                    sums[k] = sums.get(k, 0.0) + v
                bar.set_postfix({k: f"{v / n:.4f}" for k, v in sums.items()}, refresh=False)
            bar.close()
            ctx["step"] = "validation"
            metrics = validate(model, stage, val, device)
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage, "epoch": ep, "steps": n,
                   **{k: round(v / max(n, 1), 6) for k, v in sums.items()},
                   **{k: round(v, 4) for k, v in metrics.items()},
                   "seconds": round(time.perf_counter() - t_ep, 1)}
            writer.writerow(row)
            csv_file.flush()
            tqdm.write(f"[{stage} ep {ep + 1}/{args.epochs}] " + ", ".join(
                f"{k}={v}" for k, v in row.items() if k not in ("time", "stage", "epoch")))
            score = _score(stage, metrics)
            if score > best:
                best = score
                save_weights(out, model, {"stage": stage, "epoch": ep, **metrics})
                tqdm.write(f"  new best ({'-val_center_err_px' if stage == 'dfp' else 'val_edge_f1'} "
                           f"{best:.4f}) -> {out}")
            save_last(last, model, opt, ep + 1, best)
        if not out.exists():
            save_weights(out, model, {"stage": stage, "note": "no validation improvement"})
            tqdm.write(f"saved final model -> {out}")
        banner(f"done. stage {stage} | best {best:.4f} | weights {out} | log {log_csv}")
        return 0
    except (TrainError, torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and not isinstance(exc, (TrainError, torch.cuda.OutOfMemoryError)) \
                and "out of memory" not in str(exc):
            raise
        tqdm.write(f"\nTRAINING FAILED at stage={stage} epoch={ctx['epoch']} step={ctx['step']} "
                   f"crop size={ctx['size']}: {type(exc).__name__}: {exc}")
        try:
            save_last(last, model, opt, ctx["epoch"] or 0, best)
            tqdm.write(f"state saved to {last} (resume with --resume; the failed epoch restarts)")
        except Exception as save_exc:  # noqa: BLE001
            tqdm.write(f"could not save {last}: {save_exc}")
        return 2
    finally:
        csv_file.close()


if __name__ == "__main__":
    sys.exit(main())
