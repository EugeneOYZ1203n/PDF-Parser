"""Stage 1 centerline network: upstream `centerline_model/mymodel.py`'s
`ResNextUNetLarge` (the class the pretrained checkpoint names in its
`model_class`), vendored @ 178eede602bfb488b46cfc38ea2176ff902a4da2.

Input (B, 1, H, W) in [0, 1], white strokes on black, H and W divisible by
`DOWNSAMPLE` (four stride-2 downsamplings); output (B, 1, H, W) sigmoid
centerline map.

The only change: torchvision's `Bottleneck` is reproduced below (torchvision
is not a dependency here) with identical parameter names, so the checkpoint's
state_dict keys match one to one."""

from __future__ import annotations

import torch
import torch.nn as nn

DOWNSAMPLE = 16  # total stride of the encoder


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class Bottleneck(nn.Module):
    """torchvision.models.resnet.Bottleneck (v1.5: stride on the 3x3 conv)."""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: "nn.Module | None" = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class UpsampleBlock(nn.Module):
    """(C, H, W) -> 1x1 expand to out*4 -> PixelShuffle(2) -> 3x3 conv -> (out, 2H, 2W)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.expand_conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels * 4)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_smooth = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_smooth = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_expand(self.expand_conv(x)))
        x = self.pixel_shuffle(x)
        return self.relu(self.bn_smooth(self.conv_smooth(x)))


def make_layer(
    in_ch: int, target_ch: int, block_type: str = "resnet", blocks: int = 1
) -> nn.Sequential:
    """`blocks` Bottlenecks producing `target_ch` channels (ResNeXt: groups 32,
    base width 4; ResNet: groups 1, base width 64)."""
    groups = 32 if block_type == "resnext" else 1
    base_width = 4 if block_type == "resnext" else 64
    bneck_planes = target_ch // 4
    layers = []
    for i in range(blocks):
        cur_in = in_ch if i == 0 else target_ch
        downsample = None
        if cur_in != target_ch:
            downsample = nn.Sequential(
                nn.Conv2d(cur_in, target_ch, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(target_ch),
            )
        layers.append(
            Bottleneck(
                inplanes=cur_in,
                planes=bneck_planes,
                stride=1,
                downsample=downsample,
                groups=groups,
                base_width=base_width,
            )
        )
    return nn.Sequential(*layers)


class ResNextUNetLarge(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc1 = make_layer(64, 64, block_type="resnext", blocks=2)
        self.down1 = nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False)
        self.enc2 = make_layer(128, 128, block_type="resnet", blocks=2)
        self.down2 = nn.Conv2d(128, 256, kernel_size=2, stride=2, bias=False)
        self.enc3 = make_layer(256, 256, block_type="resnet", blocks=2)
        self.down3 = nn.Conv2d(256, 512, kernel_size=2, stride=2, bias=False)
        self.enc4 = make_layer(512, 512, block_type="resnet", blocks=2)
        self.down4 = nn.Conv2d(512, 1024, kernel_size=2, stride=2, bias=False)
        self.center = make_layer(1024, 1024, block_type="resnet", blocks=4)
        self.up4 = UpsampleBlock(1024, 512)
        self.dec4 = make_layer(512, 512, block_type="resnet", blocks=2)
        self.up3 = UpsampleBlock(512, 256)
        self.dec3 = make_layer(256, 256, block_type="resnet", blocks=2)
        self.up2 = UpsampleBlock(256, 128)
        self.dec2 = make_layer(128, 128, block_type="resnet", blocks=2)
        self.up1 = UpsampleBlock(128, 64)
        self.dec1 = make_layer(64, 96, block_type="resnet", blocks=2)
        self.final_block = make_layer(96, 96, block_type="resnext", blocks=2)
        self.final_conv = nn.Conv2d(96, out_channels, kernel_size=1)
        self.out_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        e1 = self.enc1(x)
        x = self.down1(e1)
        e2 = self.enc2(x)
        x = self.down2(e2)
        e3 = self.enc3(x)
        x = self.down3(e3)
        e4 = self.enc4(x)
        x = self.down4(e4)
        x = self.center(x)
        x = self.dec4(self.up4(x) + e4)
        x = self.dec3(self.up3(x) + e3)
        x = self.dec2(self.up2(x) + e2)
        x = self.dec1(self.up1(x) + e1)
        x = self.final_conv(self.final_block(x))
        return self.out_activation(x)
