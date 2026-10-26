"""Four-level U-Net for 2-D medical slices."""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock

__all__ = ["UNet"]


class UNet(nn.Module):
    """U-Net returning ``(logits, decoder_feature, bottleneck_feature)``.

    Args:
        in_channels: number of input channels.
        num_classes: number of segmentation classes.
        base: width of the first stage.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 5, base: int = 32):
        super().__init__()
        self.inc = ConvBlock(in_channels, base)
        self.down1 = ConvBlock(base, base * 2)
        self.down2 = ConvBlock(base * 2, base * 4)
        self.down3 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec2 = ConvBlock(base * 8, base * 4)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec1 = ConvBlock(base * 4, base * 2)
        self.up0 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec0 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor):
        x0 = self.inc(x)
        x1 = self.down1(self.pool(x0))
        x2 = self.down2(self.pool(x1))
        low = self.down3(self.pool(x2))
        y = self.dec2(torch.cat([self.up2(low), x2], dim=1))
        y = self.dec1(torch.cat([self.up1(y), x1], dim=1))
        feat = self.dec0(torch.cat([self.up0(y), x0], dim=1))
        return self.out(feat), feat, low
