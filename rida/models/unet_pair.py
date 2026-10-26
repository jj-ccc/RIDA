"""U-Net on pairs of adjacent electron-microscopy sections.

Each section of a pair is encoded by a shared stem, the two streams are fused in the bottleneck and
decoded by two slice-specific heads plus a head for the change between the slices.  The forward
pass returns the three logit maps, the matching full-resolution decoder features and the
bottleneck feature map.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PairUNet"]


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Inconv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        return self.conv(torch.cat([x2, self.up(x1)], dim=1))


class UpCat3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch // 3, in_ch // 3, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2, x3):
        return self.conv(torch.cat([x3, x2, self.up(x1)], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class PairUNet(nn.Module):
    """Paired-slice segmentation network.

    Args:
        in_channels: channels of one slice (1 for EM).
        num_classes: number of classes (2 for mitochondria).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.inc = Inconv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(256, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3_1 = Up(256, 64)
        self.up3_2 = Up(256, 64)
        self.up4_1 = Up(128, 32)
        self.up4_2 = Up(128, 32)

        self.up1_diff = Up(1024, 256)
        self.up2_diff = Up(512, 128)
        self.up3_diff = UpCat3(128 * 3, 64)
        self.up4_diff = UpCat3(64 * 3, 32)

        self.out_1 = OutConv(32, num_classes)
        self.out_2 = OutConv(32, num_classes)
        self.out_diff = OutConv(32, num_classes)

    def forward(self, x: torch.Tensor):
        x1 = x[:, 0:1]
        x2 = x[:, 1:]
        down0_1 = self.inc(x1)
        down0_2 = self.inc(x2)
        down1_1 = self.down1(down0_1)
        down1_2 = self.down1(down0_2)
        down2 = self.down2(torch.cat([down1_1, down1_2], dim=1))
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        up1 = self.up1(down4, down3)
        up2 = self.up2(up1, down2)
        up4_1 = self.up4_1(self.up3_1(up2, down1_1), down0_1)
        up4_2 = self.up4_2(self.up3_2(up2, down1_2), down0_2)
        out_1 = self.out_1(up4_1)
        out_2 = self.out_2(up4_2)

        up1_diff = self.up1_diff(down4, down3)
        up2_diff = self.up2_diff(up1_diff, down2)
        up3_diff = self.up3_diff(up2_diff, down1_1, down1_2)
        up4_diff = self.up4_diff(up3_diff, down0_1, down0_2)
        out_diff = self.out_diff(up4_diff)
        return out_1, out_2, out_diff, up4_1, up4_2, up4_diff, down4
