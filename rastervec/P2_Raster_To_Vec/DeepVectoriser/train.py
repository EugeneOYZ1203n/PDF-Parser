"""Train DeepVectoriser (Liu et al., AAAI-22) on a `prep_dataset.py` output.

    python -m rastervec.P2_Raster_To_Vec.DeepVectoriser.train --data data/deepvec \\
        [--device cuda] [--amp] [--epochs 50] [--batch 16] [--accum 1] [--resume] \\
        [--out rastervec/weights/deep_vectoriser.pth]

The paper's schedule ("Training" section), run automatically, stage by stage:

  bootstrap  a fixed set of --bootstrap-crops crops; every GT stroke in it gets
             a learnable embedding, optimized jointly with the Stroke Decoder
             (shared UNet + head) against the stroke's raster (L_R).
  supervise  same crops: the Stroke Encoder is supervised with the GT
             endpoints + the learned embeddings (L_F + emb MSE) and the
             Stroke Vectorizer with the embeddings + GT primitives (L_P).
  joint      fresh random crops (64-256 px, step 32, augmented) every
             step; all three modules end to end on Eq. 10, conditioned on
             the encoder's own predicted F_i.

Three Adam optimizers (encoder / decoder+UNet / vectorizer, lr 1e-4), AMP
(--amp, CUDA only), gradient accumulation (--accum), gradient checkpointing
of the UNet decoder (--grad-checkpoint).

Progress: one tqdm bar per epoch with the live loss terms, a banner per
stage, validation after every epoch (endpoint L1 px; for the supervise and
joint stages also full inference: Chamfer distance px + raster IoU), all
rows appended to `<out dir>/train_log.csv`. `<out>.last.ckpt` is saved
every epoch (`--resume` continues from it); `--out` gets the best joint-
stage model by validation Chamfer distance -- the file the P2 adapter
loads. A non-finite loss or an out-of-memory error logs where it happened
(stage / epoch / step / crop size), saves `<out>.last.ckpt` and exits 2.
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

from rastervec.P2_Raster_To_Vec.DeepVectoriser import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepVectoriser import train_data as td
from rastervec.P2_Raster_To_Vec.DeepVectoriser.config import LAMBDA_PRIM, LAMBDA_RECON, MODEL_DEFAULTS
from rastervec.P2_Raster_To_Vec.DeepVectoriser.inference import default_weights_path, predict_tiles
from rastervec.P2_Raster_To_Vec.DeepVectoriser.model.layers import prepare_input
from rastervec.P2_Raster_To_Vec.DeepVectoriser.model.liu_model import LiuVectorizer, build_model, save_weights
from rastervec.P2_Raster_To_Vec.DeepVectoriser.model import losses as L

STAGES = ("bootstrap", "supervise", "joint")
TINY_CONFIG = {**MODEL_DEFAULTS, "d_model": 32, "d_emb": 8, "n_stroke": 16, "enc_layers": 1,
               "vec_enc_layers": 1, "vec_dec_layers": 1, "n_heads": 2, "ff_dim": 64,
               "unet_base": 8, "unet_depth": 3, "max_prims": 4}


class TrainError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class _Layers:
    """Lazily loaded prep layers (gray images are memory-mapped)."""

    def __init__(self, data_dir: Path, keys: list[str]) -> None:
        self.dir = Path(data_dir) / "layers"
        self.keys = list(keys)
        self._cache: dict[str, td.LayerData] = {}

    def get(self, key: str) -> td.LayerData:
        if key not in self._cache:
            self._cache[key] = td.load_layer(self.dir, key)
        return self._cache[key]


def _weights(index: dict, keys: list[str]) -> np.ndarray:
    w = np.array([math.sqrt(max(1, index["n_strokes"].get(k, 1))) for k in keys], float)
    return w / w.sum()


def fixed_crops(layers: _Layers, probs: np.ndarray, n: int, size: int, cfg: dict, seed: int,
                desc: str, augment_once: bool) -> list[td.Crop]:
    rng = np.random.default_rng(seed)
    out: list[td.Crop] = []
    for _ in tqdm(range(n), desc=desc, unit="crop", leave=False):
        layer = layers.get(layers.keys[int(rng.choice(len(layers.keys), p=probs))])
        crop = td.sample_crop(layer, size, rng, cfg["n_stroke"], cfg["max_prims"])
        if augment_once:
            crop = td.augment(crop, rng)
        out.append(crop)
    return out


class CropStream(torch.utils.data.IterableDataset):
    """Endless random same-size batches for the joint stage."""

    def __init__(self, data_dir: Path, keys: list[str], probs: np.ndarray, cfg: dict, batch: int,
                 steps: int, seed: int) -> None:
        super().__init__()
        self.data_dir, self.keys, self.probs = Path(data_dir), keys, probs
        self.cfg, self.batch, self.steps, self.seed = cfg, batch, steps, seed
        self.epoch = 0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, self.epoch, wid])
        layers = _Layers(self.data_dir, self.keys)
        for _ in range(wid, self.steps, nw):
            size = int(rng.choice(td.SIZES))
            crops = []
            for _b in range(self.batch):
                layer = layers.get(self.keys[int(rng.choice(len(self.keys), p=self.probs))])
                crop = td.sample_crop(layer, size, rng, self.cfg["n_stroke"], self.cfg["max_prims"])
                crops.append(td.augment(crop, rng))
            yield td.build_batch(crops, self.cfg["n_stroke"], self.cfg["max_prims"])


def _batches_from(crops: list[td.Crop], batch: int, cfg: dict, rng, id_offsets: list[int] | None):
    order = rng.permutation(len(crops))
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        offs = None if id_offsets is None else [id_offsets[i] for i in idx]
        yield td.build_batch([crops[i] for i in idx], cfg["n_stroke"], cfg["max_prims"], id_offsets=offs)


def _has_grad(opt) -> bool:
    return any(p.grad is not None for g in opt.param_groups for p in g["params"])


def _to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = torch.as_tensor(v).to(device, non_blocking=True) if isinstance(v, np.ndarray) else v
    return out


# ---------------------------------------------------------------------------
# Steps (each returns (loss, {term: float}))
# ---------------------------------------------------------------------------
def _subsample(m: int, cap: int, device) -> torch.Tensor:
    if m <= cap:
        return torch.arange(m, device=device)
    return torch.randperm(m, device=device)[:cap]


def step_bootstrap(model: LiuVectorizer, emb: nn.Embedding, b: dict, cap: int):
    m = len(b["stroke_img"])
    if m == 0:
        return None, {}
    sel = _subsample(m, cap, b["gray"].device)
    x = prepare_input(b["gray"])
    img, slot = b["stroke_img"][sel], b["stroke_slot"][sel]
    ep = b["endpoints"][img, slot]
    cond = model.make_cond(ep, emb(b["stroke_ids"][sel]), torch.ones(len(sel), device=ep.device))
    skips = model.unet.encode(x)
    logits = model.decoder_head(model.unet.decode(skips, cond, img, stop_level=0))
    l_r = L.recon_loss_sum(logits, b["raster"][sel][:, None]) / len(sel)
    return LAMBDA_RECON * l_r, {"L_R": l_r.item()}


def step_supervise(model: LiuVectorizer, emb: nn.Embedding, b: dict, cap: int, beta: float):
    x = prepare_input(b["gray"])
    enc = model.encoder(x)
    l_f = L.feature_loss(enc, b["endpoints"], b["valid"], beta)
    m = len(b["stroke_img"])
    if m == 0:
        return l_f, {"L_F": l_f.item()}
    img, slot = b["stroke_img"], b["stroke_slot"]
    target = torch.zeros_like(enc["emb"])
    target[img, slot] = emb(b["stroke_ids"]).detach().to(target.dtype)
    l_emb = L.emb_loss(enc["emb"], target, b["valid"])
    sel = _subsample(m, cap, x.device)
    img_s, slot_s = img[sel], slot[sel]
    cond = model.make_cond(b["endpoints"][img_s, slot_s], emb(b["stroke_ids"][sel]).detach(),
                           torch.ones(len(sel), device=x.device))
    skips = model.unet.encode(x)
    mem = model.vectorizer.memory(model.unet.decode(skips, cond, img_s, stop_level=model.vec_level))
    pad = ~b["prim_mask"][sel]
    pc, pe = model.vectorizer(mem, cond, b["seq_in"][sel], pad_mask=pad)
    l_p = L.primitive_loss_sum(pc, pe, b["curves"][sel], b["prim_mask"][sel], b["n_prims"][sel], beta) / len(sel)
    loss = l_f + l_emb + LAMBDA_PRIM * l_p
    return loss, {"L_F": l_f.item(), "L_emb": l_emb.item(), "L_P": l_p.item()}


def step_joint(model: LiuVectorizer, b: dict, cap: int, beta: float):
    x = prepare_input(b["gray"])
    enc = model.encoder(x)
    l_f = L.feature_loss(enc, b["endpoints"], b["valid"], beta)
    m = len(b["stroke_img"])
    if m == 0:
        return l_f, {"L_F": l_f.item()}
    sel = _subsample(m, cap, x.device)
    img, slot = b["stroke_img"][sel], b["stroke_slot"][sel]
    cond = model.make_cond(enc["endpoints"][img, slot], enc["emb"][img, slot],
                           torch.sigmoid(enc["logit"][img, slot]))
    skips = model.unet.encode(x)
    levels = model.unet.decode(skips, cond, img, stop_level=0, return_levels=True)
    logits = model.decoder_head(levels[0])
    l_r_sum = L.recon_loss_sum(logits, b["raster"][sel][:, None])
    mem = model.vectorizer.memory(levels[model.vec_level])
    pc, pe = model.vectorizer(mem, cond, b["seq_in"][sel], pad_mask=~b["prim_mask"][sel])
    l_p_sum = L.primitive_loss_sum(pc, pe, b["curves"][sel], b["prim_mask"][sel], b["n_prims"][sel], beta)
    loss = L.total_loss(l_f, l_r_sum, l_p_sum, len(sel))
    n = len(sel)
    return loss, {"L_F": l_f.item(), "L_R": l_r_sum.item() / n, "L_P": l_p_sum.item() / n}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _mask(strokes, size: int, width: int = 2) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    for s in strokes:
        pts = np.round((geo.sample_stroke(s, 16) - 0.5) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(m, [pts], False, 1, width, cv2.LINE_8, shift=4)
    return m > 0


def _chamfer(pred, gt, size: int) -> float:
    """Symmetric mean nearest distance (px) between sampled strokes."""
    if not pred and not gt:
        return 0.0
    if not pred or not gt:
        return size / 2.0

    def one_way(a, b_strokes):
        dist = cv2.distanceTransform((~_mask(b_strokes, size, 1)).astype(np.uint8), cv2.DIST_L2, 3)
        pts = np.concatenate([geo.sample_stroke(s, 8) for s in a])
        ij = np.clip(np.floor(pts).astype(int), 0, size - 1)
        return float(dist[ij[:, 1], ij[:, 0]].mean())

    return 0.5 * (one_way(pred, gt) + one_way(gt, pred))


@torch.no_grad()
def validate(model: LiuVectorizer, crops: list[td.Crop], device, cfg: dict, full: bool, batch: int) -> dict:
    model.eval()
    ep_err, ep_n = 0.0, 0
    cham, iou = [], []
    try:
        for b0 in range(0, len(crops), batch):
            chunk = crops[b0:b0 + batch]
            b = _to_device(td.build_batch(chunk, cfg["n_stroke"], cfg["max_prims"], with_raster=False), device)
            enc = model.encoder(prepare_input(b["gray"]))
            v = b["valid"]
            if bool(v.any()):
                size = chunk[0].gray.shape[0]
                ep_err += float((enc["endpoints"][v] - b["endpoints"][v]).abs().sum()) * size / 4
                ep_n += int(v.sum())
            if full:
                preds = predict_tiles(model, [c.gray for c in chunk])
                for c, p in zip(chunk, preds):
                    size = c.gray.shape[0]
                    cham.append(_chamfer(p.strokes, c.strokes, size))
                    a, g = _mask(p.strokes, size), _mask(c.strokes, size)
                    union = (a | g).sum()
                    iou.append(float((a & g).sum() / union) if union else 1.0)
    finally:
        model.train()
    out = {"val_endpoint_l1_px": ep_err / max(ep_n, 1)}
    if full:
        out["val_chamfer_px"] = float(np.mean(cham)) if cham else float("nan")
        out["val_raster_iou"] = float(np.mean(iou)) if iou else float("nan")
    return out


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_last(path: Path, model, emb, optims, scaler, stage_i: int, epoch: int, best: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "config": model.cfg, "model": model.state_dict(),
        "emb": emb.state_dict() if emb is not None else None,
        "optims": {k: o.state_dict() for k, o in optims.items()},
        "scaler": scaler.state_dict(), "stage_i": stage_i, "epoch": epoch, "best": best,
    }, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="prep_dataset.py output directory")
    ap.add_argument("--out", default=str(default_weights_path()), help="weights file the adapter loads")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=50, help="joint-stage epochs")
    ap.add_argument("--bootstrap-epochs", type=int, default=20)
    ap.add_argument("--supervise-epochs", type=int, default=20)
    ap.add_argument("--bootstrap-crops", type=int, default=5000, help="paper: 5,000")
    ap.add_argument("--steps-per-epoch", type=int, default=1000, help="joint-stage batches per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--grad-checkpoint", action="store_true", help="checkpoint the UNet decoder")
    ap.add_argument("--max-decode-strokes", type=int, default=128,
                    help="strokes per batch through the raster/vector branches (memory cap)")
    ap.add_argument("--val-crops", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers for the joint stage")
    ap.add_argument("--resume", action="store_true", help="continue from <out>.last.ckpt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="tiny model (smoke tests only)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data = Path(args.data)
    out = Path(args.out)
    last = out.with_name(out.name + ".last.ckpt")
    log_csv = out.parent / "train_log.csv"
    device = torch.device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    torch.manual_seed(args.seed)

    def banner(msg: str) -> None:
        tqdm.write("=" * 72 + f"\n{msg}\n" + "=" * 72)

    index = td.load_index(data)
    if not index["train"]:
        raise SystemExit(f"{data}/index.json has no training layers -- run prep_dataset.py first")
    resume_blob = None
    if args.resume:
        if not last.is_file():
            raise SystemExit(f"--resume: {last} not found")
        resume_blob = torch.load(last, map_location="cpu", weights_only=False)
    cfg = resume_blob["config"] if resume_blob else dict(TINY_CONFIG if args.tiny else MODEL_DEFAULTS)
    banner(f"DeepVectoriser training | device {device}{' (AMP)' if amp else ''} | data {data}\n"
           f"train layers {len(index['train'])}, val layers {len(index['val'])} | out {out}\n"
           f"model {cfg}")

    train_layers = _Layers(data, index["train"])
    probs = _weights(index, index["train"])
    tqdm.write("building fixed crop sets ...")
    boot = fixed_crops(train_layers, probs, args.bootstrap_crops, 128, cfg, args.seed, "bootstrap crops", True)
    if index["val"]:
        val_layers = _Layers(data, index["val"])
        val = fixed_crops(val_layers, _weights(index, index["val"]), args.val_crops, 128, cfg,
                          args.seed + 1, "val crops", False)
    else:
        tqdm.write("WARNING: no val pages in index.json -- validating on training-page crops")
        val = fixed_crops(train_layers, probs, args.val_crops, 128, cfg, args.seed + 1, "val crops", False)
    boot_offsets = list(np.cumsum([0] + [len(c.strokes) for c in boot])[:-1])
    n_boot_strokes = int(sum(len(c.strokes) for c in boot))
    tqdm.write(f"bootstrap: {len(boot)} crops, {n_boot_strokes} strokes | val: {len(val)} crops")

    model = build_model(cfg).to(device)
    model.unet.grad_checkpoint = args.grad_checkpoint
    emb = nn.Embedding(max(1, n_boot_strokes), cfg["d_emb"]).to(device)
    nn.init.normal_(emb.weight, std=0.1)
    groups = model.module_groups()
    optims = {
        "encoder": torch.optim.Adam(groups["encoder"], lr=args.lr),
        "decoder": torch.optim.Adam(groups["decoder"] + list(emb.parameters()), lr=args.lr),
        "vectorizer": torch.optim.Adam(groups["vectorizer"], lr=args.lr),
    }
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    stage_i, epoch0, best = 0, 0, float("inf")
    if resume_blob:
        model.load_state_dict(resume_blob["model"])
        if resume_blob.get("emb") is not None and resume_blob["emb"]["weight"].shape == emb.weight.shape:
            emb.load_state_dict(resume_blob["emb"])
        for k, o in optims.items():
            try:
                o.load_state_dict(resume_blob["optims"][k])
            except (KeyError, ValueError) as exc:
                tqdm.write(f"WARNING: optimizer '{k}' state not restored ({exc})")
        scaler.load_state_dict(resume_blob["scaler"])
        stage_i, epoch0, best = resume_blob["stage_i"], resume_blob["epoch"], resume_blob["best"]
        tqdm.write(f"resumed from {last}: stage {STAGES[stage_i]}, epoch {epoch0}, best {best:.3f}")

    stage_epochs = {"bootstrap": args.bootstrap_epochs, "supervise": args.supervise_epochs, "joint": args.epochs}
    stage_optims = {"bootstrap": ["decoder"], "supervise": ["encoder", "vectorizer"],
                    "joint": ["encoder", "decoder", "vectorizer"]}
    new_csv = not log_csv.exists()
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(log_csv, "a", newline="", encoding="utf-8")
    fields = ["time", "stage", "epoch", "steps", "loss", "L_F", "L_R", "L_P", "L_emb",
              "val_endpoint_l1_px", "val_chamfer_px", "val_raster_iou", "seconds"]
    writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
    if new_csv:
        writer.writeheader()

    ctx = {"stage": None, "epoch": None, "step": None, "size": None}
    rng = np.random.default_rng(args.seed + 7)
    try:
        model.train()
        for si in range(stage_i, len(STAGES)):
            stage = STAGES[si]
            n_ep = stage_epochs[stage]
            start_ep = epoch0 if si == stage_i else 0
            banner(f"stage {si + 1}/3: {stage} -- epochs {start_ep}..{n_ep - 1}, optimizers {stage_optims[stage]}")
            stream = None
            if stage == "joint" and n_ep > start_ep:
                stream = CropStream(data, index["train"], probs, cfg, args.batch, args.steps_per_epoch, args.seed)
            for ep in range(start_ep, n_ep):
                ctx.update(stage=stage, epoch=ep)
                t_ep = time.perf_counter()
                if stage == "joint":
                    stream.epoch = ep
                    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers,
                                                         persistent_workers=False)
                    total = args.steps_per_epoch
                else:
                    loader = _batches_from(boot, args.batch, cfg, rng, boot_offsets)
                    total = math.ceil(len(boot) / args.batch)
                sums: dict[str, float] = {}
                n_steps = 0
                for o in optims.values():
                    o.zero_grad(set_to_none=True)
                bar = tqdm(loader, total=total, desc=f"{stage} ep {ep + 1}/{n_ep}", unit="batch")
                for step, batch in enumerate(bar):
                    ctx.update(step=step, size=int(batch["size"]))
                    b = _to_device(batch, device)
                    beta = float(batch["size"])
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        if stage == "bootstrap":
                            loss, terms = step_bootstrap(model, emb, b, args.max_decode_strokes)
                        elif stage == "supervise":
                            loss, terms = step_supervise(model, emb, b, args.max_decode_strokes, beta)
                        else:
                            loss, terms = step_joint(model, b, args.max_decode_strokes, beta)
                    if loss is None:
                        continue
                    if not torch.isfinite(loss):
                        raise TrainError(f"non-finite loss {loss.item()} (terms {terms})")
                    scaler.scale(loss / args.accum).backward()
                    if (step + 1) % args.accum == 0:
                        # an optimizer whose params got no gradient this step
                        # (e.g. no GT strokes in the batch) is skipped
                        active = [n for n in stage_optims[stage] if _has_grad(optims[n])]
                        for name in active:
                            scaler.unscale_(optims[name])
                        nn.utils.clip_grad_norm_([p for name in active
                                                  for g in optims[name].param_groups for p in g["params"]], 1.0)
                        for name in active:
                            scaler.step(optims[name])
                        scaler.update()
                        for o in optims.values():
                            o.zero_grad(set_to_none=True)
                    n_steps += 1
                    sums["loss"] = sums.get("loss", 0.0) + loss.item()
                    for k, v in terms.items():
                        sums[k] = sums.get(k, 0.0) + v
                    bar.set_postfix({k: f"{v / n_steps:.4f}" for k, v in sums.items()}, refresh=False)
                bar.close()
                ctx.update(step="validation")
                metrics = validate(model, val, device, cfg, full=stage != "bootstrap", batch=args.batch)
                row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage, "epoch": ep, "steps": n_steps,
                       **{k: round(v / max(n_steps, 1), 6) for k, v in sums.items()},
                       **{k: round(v, 4) for k, v in metrics.items()},
                       "seconds": round(time.perf_counter() - t_ep, 1)}
                writer.writerow(row)
                csv_file.flush()
                tqdm.write(f"[{stage} ep {ep + 1}/{n_ep}] " + ", ".join(
                    f"{k}={v}" for k, v in row.items() if k not in ("time", "stage", "epoch")))
                if stage == "joint" and metrics.get("val_chamfer_px", float("inf")) < best:
                    best = metrics["val_chamfer_px"]
                    save_weights(out, model, {"stage": stage, "epoch": ep, **metrics})
                    tqdm.write(f"  new best val Chamfer {best:.3f} px -> {out}")
                save_last(last, model, emb, optims, scaler, si, ep + 1, best)
        if not out.exists():
            save_weights(out, model, {"stage": "final", "note": "no joint-stage validation improvement"})
            tqdm.write(f"saved final model -> {out}")
        banner(f"done. best val Chamfer {best:.3f} px | weights {out} | log {log_csv}")
        return 0
    except (TrainError, torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and not isinstance(exc, (TrainError, torch.cuda.OutOfMemoryError)) \
                and "out of memory" not in str(exc):
            raise
        tqdm.write(f"\nTRAINING FAILED at stage={ctx['stage']} epoch={ctx['epoch']} step={ctx['step']} "
                   f"crop size={ctx['size']}: {type(exc).__name__}: {exc}")
        try:
            save_last(last, model, emb, optims, scaler, STAGES.index(ctx["stage"]) if ctx["stage"] else 0,
                      ctx["epoch"] or 0, best)
            tqdm.write(f"state saved to {last} (resume with --resume; the failed epoch restarts)")
        except Exception as save_exc:  # noqa: BLE001
            tqdm.write(f"could not save {last}: {save_exc}")
        return 2
    finally:
        csv_file.close()


if __name__ == "__main__":
    sys.exit(main())
