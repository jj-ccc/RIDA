"""Shared convolutional building blocks."""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ConvBlock", "UpBlock"]


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by batch normalisation and a ReLU."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """Transposed-convolution upsampling followed by a skip concatenation and a conv block."""

    def __init__(self, cin: int, cup: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cup, 2, stride=2)
        self.conv = ConvBlock(cup + cskip, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))
