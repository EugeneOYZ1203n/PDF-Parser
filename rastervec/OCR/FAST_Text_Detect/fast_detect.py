"""FAST text detector: a faithful, inference-only port of czczup/FAST's
TextNet-Tiny variant (github.com/czczup/FAST, `fast_tiny_ic17mlt_640`
config) -- backbone (TextNet, built from FAST's own NAS layer config),
neck (FASTNeck), and head (FASTHead)'s conv path, plus the exact test-time
preprocessing (`scale_aligned_short` + ImageNet normalize) and score-map
postprocessing (`FASTHead.get_results`' interpolate/max-pool/sigmoid/
interpolate recipe) FAST itself uses -- ported directly from the upstream
source so a real `fast_tiny_ic17mlt_640.pth` checkpoint's `state_dict`
loads and runs correctly here. Only TextNet-**Tiny**-shaped checkpoints
are supported -- a Small/Base checkpoint has a different NAS config
(different channel widths/kernel sizes per stage) and would need
`_FAST_TINY_CONFIG` swapped for that variant's own config.

Not ported: FAST's embedding-based instance separation (pixel aggregation
via a CUDA/OpenCV connected-components post-process, turning the score map
into per-instance polygon boxes). This project only needs the per-pixel
text-probability mask -- `FASTHead.get_results`' `score_maps` tensor,
computed *before* that connected-components step -- which
`pipeline.py`'s `_run_fast_text_detect` averages over each candidate
cluster's own bbox region. Skipping instance separation avoids vendoring FAST's
CUDA/OpenCV postprocessing code entirely, with no loss of the signal this
project actually uses.

`detect()` runs one direct pass over a whole image. `detect_tiled()` is
what `pipeline.py`'s fast_text_detect stage actually calls -- it upscales
the image, splits it into fixed-size square tiles, and detects each tile
at several rotations (summed then averaged), since `detect()`'s own
`_scale_aligned_short` preprocessing always downsizes to a 640px short
side regardless of input size and so throws away most of a large
whole-page render's resolution in one direct pass.

torch is only imported lazily, inside FastDetector._model()/_preprocess/
detect(), so importing this module (and constructing a FastDetector) stays
cheap and doesn't require torch to be installed unless detect() is
actually called -- e.g. a page with zero text candidates never touches
torch at all.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from rastervec.config import (
    FAST_TILE_BLOCK_SIZE as TILED_BLOCK_SIZE,
    FAST_TILE_ROTATION_COUNT as TILED_ROTATION_COUNT,
    FAST_TILE_SCALE_FACTOR as TILED_SCALE_FACTOR,
)

_MODEL_CACHE: dict[str, object] = {}

_DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "weights",
    "fast_tiny_ic17mlt_640.pth",
)

# Ported verbatim from config/fast/nas-configs/fast_tiny.config (the NAS
# layer config the fast_tiny_ic17mlt_640 checkpoint was trained with).
_FAST_TINY_CONFIG = {
    "first_conv": {"kernel_size": 3, "stride": 2, "in_channels": 3, "out_channels": 64},
    "stage1": [
        {"in_channels": 64, "out_channels": 64, "kernel_size": (3, 3), "stride": 1},
        {"in_channels": 64, "out_channels": 64, "kernel_size": (3, 3), "stride": 2},
        {"in_channels": 64, "out_channels": 64, "kernel_size": (3, 3), "stride": 1},
    ],
    "stage2": [
        {"in_channels": 64, "out_channels": 128, "kernel_size": (3, 3), "stride": 2},
        {"in_channels": 128, "out_channels": 128, "kernel_size": (1, 3), "stride": 1},
        {"in_channels": 128, "out_channels": 128, "kernel_size": (3, 3), "stride": 1},
        {"in_channels": 128, "out_channels": 128, "kernel_size": (3, 1), "stride": 1},
    ],
    "stage3": [
        {"in_channels": 128, "out_channels": 256, "kernel_size": (3, 3), "stride": 2},
        {"in_channels": 256, "out_channels": 256, "kernel_size": (3, 3), "stride": 1},
        {"in_channels": 256, "out_channels": 256, "kernel_size": (3, 1), "stride": 1},
        {"in_channels": 256, "out_channels": 256, "kernel_size": (1, 3), "stride": 1},
    ],
    "stage4": [
        {"in_channels": 256, "out_channels": 512, "kernel_size": (3, 3), "stride": 2},
        {"in_channels": 512, "out_channels": 512, "kernel_size": (3, 1), "stride": 1},
        {"in_channels": 512, "out_channels": 512, "kernel_size": (1, 3), "stride": 1},
        {"in_channels": 512, "out_channels": 512, "kernel_size": (3, 3), "stride": 1},
    ],
    "neck": {
        "reduce_layer1": {"in_channels": 64, "out_channels": 128, "kernel_size": (3, 3), "stride": 1},
        "reduce_layer2": {"in_channels": 128, "out_channels": 128, "kernel_size": (3, 3), "stride": 1},
        "reduce_layer3": {"in_channels": 256, "out_channels": 128, "kernel_size": (3, 3), "stride": 1},
        "reduce_layer4": {"in_channels": 512, "out_channels": 128, "kernel_size": (3, 3), "stride": 1},
    },
    "head": {
        "conv": {"in_channels": 512, "out_channels": 128, "kernel_size": (3, 3), "stride": 1},
        "final": {"in_channels": 128, "out_channels": 5, "kernel_size": 1, "stride": 1},
    },
}

# FASTHead's own pooling_size (config/fast/ic17mlt/fast_tiny_ic17mlt_640.py's
# detection_head.pooling_size) -- the 2s-scale max-pool kernel used by
# get_results' score-map recipe is pooling_size // 2 + 1.
_POOLING_SIZE = 9
_SHORT_SIDE = 640
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_same_padding(kernel_size: int) -> int:
    assert kernel_size % 2 == 1, "kernel size should be an odd number"
    return kernel_size // 2


def _build_conv_layer(cfg: dict, use_bn: bool = True, act: bool = True, bias: bool = False):
    """Ported from nas_utils.py's ConvLayer -- a plain conv, optionally
    followed by BatchNorm2d and/or ReLU, with submodule names ("conv",
    "bn") matching the upstream checkpoint's own state_dict keys. `bias`
    matches each config entry's own "bias" field exactly (both this
    module's two ConvLayer uses -- first_conv and the head's final layer
    -- set it to False in fast_tiny.config, independent of use_bn)."""
    import torch.nn as nn

    class _ConvLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            padding = _get_same_padding(cfg["kernel_size"])
            self.conv = nn.Conv2d(
                cfg["in_channels"], cfg["out_channels"], cfg["kernel_size"],
                stride=cfg["stride"], padding=padding, bias=bias,
            )
            self.bn = nn.BatchNorm2d(cfg["out_channels"]) if use_bn else None
            self.act = nn.ReLU(inplace=True) if act else None

        def forward(self, x):
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.act is not None:
                x = self.act(x)
            return x

    return _ConvLayer()


def _build_rep_conv_layer(cfg: dict):
    """Ported near-verbatim from nas_utils.py's RepConvLayer, kept in its
    training-mode (non-reparameterized) multi-branch form -- the released
    checkpoint's state_dict was saved in that form (main_conv/main_bn plus
    optional ver_conv/ver_bn, hor_conv/hor_bn, rbr_identity), not the
    switch_to_deploy()-fused single-conv form, so this must match it
    exactly to load correctly."""
    import torch.nn as nn

    in_ch, out_ch = cfg["in_channels"], cfg["out_channels"]
    kh, kw = cfg["kernel_size"]
    stride = cfg["stride"]

    class _RepConvLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.main_conv = nn.Conv2d(
                in_ch, out_ch, (kh, kw), stride=stride,
                padding=((kh - 1) // 2, (kw - 1) // 2), bias=False,
            )
            self.main_bn = nn.BatchNorm2d(out_ch)

            if kw != 1:
                self.ver_conv = nn.Conv2d(
                    in_ch, out_ch, (kh, 1), stride=stride, padding=((kh - 1) // 2, 0), bias=False,
                )
                self.ver_bn = nn.BatchNorm2d(out_ch)
            else:
                self.ver_conv, self.ver_bn = None, None

            if kh != 1:
                self.hor_conv = nn.Conv2d(
                    in_ch, out_ch, (1, kw), stride=stride, padding=(0, (kw - 1) // 2), bias=False,
                )
                self.hor_bn = nn.BatchNorm2d(out_ch)
            else:
                self.hor_conv, self.hor_bn = None, None

            self.rbr_identity = (
                nn.BatchNorm2d(in_ch) if out_ch == in_ch and stride == 1 else None
            )
            self.nonlinearity = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.main_bn(self.main_conv(x))
            if self.ver_conv is not None:
                out = out + self.ver_bn(self.ver_conv(x))
            if self.hor_conv is not None:
                out = out + self.hor_bn(self.hor_conv(x))
            if self.rbr_identity is not None:
                out = out + self.rbr_identity(x)
            return self.nonlinearity(out)

    return _RepConvLayer()


def _build_backbone(config: dict):
    """Ported from models/backbone/textnet.py's TextNet: a stride-2 first
    conv, then 4 stages of RepConvLayer built from the NAS config, each
    stage's final feature map collected for the neck."""
    import torch.nn as nn

    first_conv = _build_conv_layer(config["first_conv"], use_bn=True, act=True)
    stages = [
        nn.ModuleList([_build_rep_conv_layer(layer_cfg) for layer_cfg in config[f"stage{i}"]])
        for i in (1, 2, 3, 4)
    ]

    class _TextNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first_conv = first_conv
            self.stage1, self.stage2, self.stage3, self.stage4 = stages

        def forward(self, x):
            x = self.first_conv(x)
            outputs = []
            for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
                for block in stage:
                    x = block(x)
                outputs.append(x)
            return outputs

    return _TextNet()


def _build_neck(config: dict):
    """Ported from models/neck/fast_neck.py's FASTNeck: each stage's
    feature map reduced to 128 channels, upsampled to the first (highest-
    resolution) stage's size, and concatenated into one 512-channel map."""
    import torch.nn as nn
    import torch.nn.functional as F

    reduce_layers = [_build_rep_conv_layer(config[f"reduce_layer{i}"]) for i in (1, 2, 3, 4)]

    class _FastNeck(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.reduce_layer1, self.reduce_layer2, self.reduce_layer3, self.reduce_layer4 = reduce_layers

        def forward(self, x):
            f1, f2, f3, f4 = x
            f1 = self.reduce_layer1(f1)
            f2 = self.reduce_layer2(f2)
            f3 = self.reduce_layer3(f3)
            f4 = self.reduce_layer4(f4)
            size = f1.shape[2:]
            f2 = F.interpolate(f2, size=size, mode="bilinear", align_corners=False)
            f3 = F.interpolate(f3, size=size, mode="bilinear", align_corners=False)
            f4 = F.interpolate(f4, size=size, mode="bilinear", align_corners=False)
            return torch.cat((f1, f2, f3, f4), 1)

    import torch  # noqa: E402 -- used by _FastNeck.forward's torch.cat, imported here to keep this function's own lazy-import contract

    return _FastNeck()


def _build_head(config: dict):
    """Ported from models/head/fast_head.py's FASTHead -- just the conv
    path (`conv` then `final`, no `blocks` for the Tiny config), not
    `.loss`/`.get_results`'s connected-components instance separation."""
    import torch.nn as nn

    conv = _build_rep_conv_layer(config["conv"])
    final = _build_conv_layer(config["final"], use_bn=False, act=False)

    class _FastHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = conv
            self.final = final

        def forward(self, x):
            return self.final(self.conv(x))

    return _FastHead()


def _build_model():
    import torch.nn as nn
    import torch.nn.functional as F

    backbone = _build_backbone(_FAST_TINY_CONFIG)
    neck = _build_neck(_FAST_TINY_CONFIG["neck"])
    det_head = _build_head(_FAST_TINY_CONFIG["head"])

    class _FastNet(nn.Module):
        """Top-level module, ported from models/fast.py's FAST.forward
        (inference branch): backbone -> neck -> det_head -> bilinear
        upsample to (H//4, W//4) -- matches FAST's own `scale=4` inference
        upsample exactly. Attribute names (backbone/neck/det_head) match
        the checkpoint's own state_dict prefixes (after stripping the
        "module." DataParallel prefix test.py strips)."""

        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.neck = neck
            self.det_head = det_head

        def forward(self, imgs):
            f = self.backbone(imgs)
            f = self.neck(f)
            det_out = self.det_head(f)
            h, w = imgs.shape[2], imgs.shape[3]
            return F.interpolate(det_out, size=(h // 4, w // 4), mode="bilinear", align_corners=False)

    return _FastNet()


def _scale_aligned_short(image: "Image.Image", short_size: int = _SHORT_SIDE) -> "Image.Image":
    """Ported from dataset/utils.py's scale_aligned_short: resizes so the
    shorter side is `short_size`, then rounds each dimension up to the
    next multiple of 32 (so every backbone stride divides it evenly)."""
    w, h = image.size
    scale = short_size / min(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    if new_h % 32:
        new_h += 32 - new_h % 32
    if new_w % 32:
        new_w += 32 - new_w % 32
    return image.resize((new_w, new_h), Image.BILINEAR)


def _preprocess(image: "Image.Image") -> tuple["torch.Tensor", tuple[int, int]]:
    """Ported from dataset/fast/fast_ic17mlt.py's test-time transform:
    scale_aligned_short + ImageNet Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]). Returns the 1x3xHxW input tensor plus its
    own (H, W) for _score_map's postprocessing."""
    import torch

    resized = _scale_aligned_short(image.convert("RGB"))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std = np.array(_IMAGENET_STD, dtype=np.float32)
    array = (array - mean) / std
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, (resized.height, resized.width)


def _score_map(det_out: "torch.Tensor", out_hw: tuple[int, int], pooling_size: int = _POOLING_SIZE) -> "np.ndarray":
    """Ported from models/head/fast_head.py's FASTHead.get_results' score-
    map recipe (everything up to, but not including, the connected-
    components instance-separation step this project doesn't need):
    channel 0 -> interpolate to half `out_hw` (nearest) -> max-pool
    (smooths/dilates the raw logits) -> sigmoid -> interpolate back up to
    `out_hw` (nearest)."""
    import torch.nn.functional as F

    h, w = out_hw
    texts = det_out[:, 0:1, :, :]
    texts = F.interpolate(texts, size=(h // 2, w // 2), mode="nearest")
    kernel = pooling_size // 2 + 1
    texts = F.max_pool2d(texts, kernel_size=kernel, stride=1, padding=(kernel - 1) // 2)
    score = texts.sigmoid()
    score = F.interpolate(score, size=(h, w), mode="nearest")
    return score[0, 0].detach().numpy()


class FastDetector:
    """Lazily-built, path-keyed-cached FAST text detector -- mirrors
    PaddleOcrBackend's engine caching (OCR/Paddle_OCR/ocr_backend.py)."""

    def __init__(self, weights_path: str | None = None) -> None:
        self.weights_path = weights_path or os.environ.get(
            "FAST_WEIGHTS_PATH", _DEFAULT_WEIGHTS_PATH
        )

    def warmup(self) -> None:
        """Build + cache the model now, in the calling process, so a
        process pool spawned next doesn't each pay the load. A missing
        weights file is swallowed (pages with no vector paths skip FAST
        anyway) rather than raised."""
        try:
            self._model()
        except FileNotFoundError:
            pass

    def _model(self):
        if self.weights_path not in _MODEL_CACHE:
            import torch

            if not os.path.isfile(self.weights_path):
                raise FileNotFoundError(
                    f"FAST detector weights not found at {self.weights_path!r} -- "
                    "set FAST_WEIGHTS_PATH or pass weights_path= to FastDetector() "
                    "(download fast_tiny_ic17mlt_640.pth from the czczup/FAST releases)."
                )
            model = _build_model()
            checkpoint = torch.load(self.weights_path, map_location="cpu")
            # test.py: checkpoint['state_dict'] (or checkpoint['ema']) with a
            # "module." DataParallel prefix on every key. Released FAST
            # checkpoints often ship only the EMA copy (no 'state_dict' key
            # at all) -- prefer 'state_dict' when present, fall back to
            # 'ema', and tolerate a raw state_dict too (a checkpoint
            # re-saved without either wrapper key).
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "ema" in checkpoint:
                state_dict = checkpoint["ema"]
            else:
                state_dict = checkpoint
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()
            _MODEL_CACHE[self.weights_path] = model
        return _MODEL_CACHE[self.weights_path]

    def detect(self, image: "Image.Image") -> "np.ndarray":
        """Runs FAST-Tiny on one image (a rendered whole page of surviving
        text-candidate vector paths), returning a text-probability mask
        resized back to the image's own original size, float32 in [0, 1].
        Callers threshold or aggregate it themselves -- pipeline.py's
        fast_text_detect stage averages it over each candidate cluster's
        own bbox region within the page."""
        import torch

        model = self._model()
        tensor, out_hw = _preprocess(image)
        with torch.no_grad():
            det_out = model(tensor)
        mask = _score_map(det_out, out_hw)

        if mask.shape != (image.height, image.width):
            mask_img = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)).resize(
                (image.width, image.height), Image.BILINEAR,
            )
            mask = np.asarray(mask_img, dtype=np.float32) / 255.0
        return mask

    def detect_tiled(
        self,
        image: "Image.Image",
        block_size: int = TILED_BLOCK_SIZE,
        scale: float = TILED_SCALE_FACTOR,
        n_rotations: int = TILED_ROTATION_COUNT,
        desc: str = "FAST text detection",
        show_progress: bool = True,
    ) -> "np.ndarray":
        """Runs FAST over `image` upscaled by `scale` and split into
        non-overlapping `block_size`-square tiles (the last row/column of
        tiles is right-padded with white up to `block_size` before
        detection, so an edge tile isn't massively upscaled internally by
        `detect()`'s own `_scale_aligned_short` preprocessing -- the padded
        region is simply never copied back out). Each tile is detected at
        `n_rotations` evenly-spaced rotations (0/90/180/270 by default) --
        the rotated mask is rotated back to the tile's own orientation
        before summing, then the sum is divided by `n_rotations` for that
        tile's final score. Every tile's mask is stitched back into one
        full mask at the *scaled* resolution, then resized back down to
        `image`'s own original size before returning -- callers sample it
        exactly like `detect()`'s own return value, at `image`'s own pixel
        coordinates. `show_progress` wraps the block loop in a `tqdm` bar
        (`desc`), since a large page at a real `scale` can mean hundreds of
        tiles x rotations."""
        if n_rotations <= 0:
            raise ValueError("n_rotations must be a positive integer")

        orig_w, orig_h = image.size
        scaled = image.convert("RGB").resize(
            (max(1, round(orig_w * scale)), max(1, round(orig_h * scale))), Image.BICUBIC,
        )
        sw, sh = scaled.size
        n_cols = max(1, math.ceil(sw / block_size))
        n_rows = max(1, math.ceil(sh / block_size))

        full_mask = np.zeros((sh, sw), dtype=np.float32)
        blocks = [(r, c) for r in range(n_rows) for c in range(n_cols)]
        iterator = tqdm(blocks, desc=desc, unit="block") if show_progress else blocks

        for r, c in iterator:
            x0, y0 = c * block_size, r * block_size
            x1, y1 = min(x0 + block_size, sw), min(y0 + block_size, sh)
            block = scaled.crop((x0, y0, x1, y1))
            if block.size != (block_size, block_size):
                padded = Image.new("RGB", (block_size, block_size), (255, 255, 255))
                padded.paste(block, (0, 0))
                block = padded

            block_sum = np.zeros((block_size, block_size), dtype=np.float32)
            for k in range(n_rotations):
                angle = 360.0 * k / n_rotations
                rotated = block if angle == 0.0 else block.rotate(
                    -angle, expand=False, fillcolor=(255, 255, 255), resample=Image.BICUBIC,
                )
                rotated_mask = self.detect(rotated)
                if angle:
                    mask_img = Image.fromarray(
                        (np.clip(rotated_mask, 0.0, 1.0) * 255).astype(np.uint8)
                    ).rotate(angle, expand=False, fillcolor=0, resample=Image.BICUBIC)
                    rotated_mask = np.asarray(mask_img, dtype=np.float32) / 255.0
                block_sum += rotated_mask

            block_avg = block_sum / n_rotations
            full_mask[y0:y1, x0:x1] = block_avg[: y1 - y0, : x1 - x0]

        mask_img = Image.fromarray((np.clip(full_mask, 0.0, 1.0) * 255).astype(np.uint8)).resize(
            (orig_w, orig_h), Image.BILINEAR,
        )
        return np.asarray(mask_img, dtype=np.float32) / 255.0
