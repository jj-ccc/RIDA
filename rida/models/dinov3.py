"""DINOv3 ViT-L/16 segmentor.

Four intermediate token maps of the DINOv3 encoder are projected, fused and decoded along an
image pyramid so that the decoder sees both the semantics of the foundation encoder and
full-resolution detail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, UpBlock

__all__ = ["DINOv3Segmentor", "build_dinov3_encoder"]

ROOT = Path(__file__).resolve().parents[2]
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_dinov3_encoder(weights: str = "pretrained/dinov3_vitl16_pretrain_lvd1689m.pth",
                         repository: str = "third_party/dinov3"):
    """DINOv3 ViT-L/16 from the vendored repository and the LVD-1689M checkpoint."""
    repo = Path(repository)
    repo = repo if repo.is_absolute() else ROOT / repo
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dinov3.hub.backbones import dinov3_vitl16

    backbone = dinov3_vitl16(pretrained=False)
    path = Path(weights)
    path = path if path.is_absolute() else ROOT / path
    state = torch.load(path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(state.get("model", state), strict=True)
    return backbone, backbone.embed_dim


class DINOv3Segmentor(nn.Module):
    """DINOv3 encoder with a convolutional pyramid decoder.

    Args:
        encoder: DINOv3 vision transformer.
        encoder_channels: token width of the encoder.
        num_classes: number of segmentation classes.
        in_channels: number of input channels.
        decoder_channels: width of the first decoder stage.
        low_channels: width of the fused token map.
        enc_layers: encoder blocks whose outputs are decoded.
        freeze_encoder: run the encoder without gradients.
    """

    def __init__(self, encoder: nn.Module, encoder_channels: int = 1024, num_classes: int = 5,
                 in_channels: int = 3, decoder_channels: int = 64, low_channels: int = 256,
                 enc_layers=(5, 11, 17, 23), freeze_encoder: bool = True):
        super().__init__()
        c = decoder_channels
        self.encoder = encoder
        self.freeze_encoder = bool(freeze_encoder)
        self.enc_layers = tuple(enc_layers)
        self.encoder.requires_grad_(not self.freeze_encoder)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        self.proj = nn.ModuleList([nn.Conv2d(encoder_channels, low_channels, 1) for _ in self.enc_layers])
        self.fuse = ConvBlock(low_channels * len(self.enc_layers), low_channels)
        self.stem1 = ConvBlock(in_channels, c)
        self.stem2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c, c * 2))
        self.stem3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c * 2, c * 4))
        self.stem4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c * 4, c * 4))
        self.up1 = UpBlock(low_channels, c * 4, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 4, c * 4, c * 4)
        self.up3 = UpBlock(c * 4, c * 2, c * 2, c * 2)
        self.up4 = UpBlock(c * 2, c, c, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, x):
        return self.encoder.get_intermediate_layers((x - self.mean) / self.std, n=self.enc_layers,
                                                   reshape=True, norm=True)

    def forward(self, x: torch.Tensor):
        if self.freeze_encoder:
            with torch.no_grad():
                stages = self._encode(x)
        else:
            stages = self._encode(x)
        f = self.fuse(torch.cat([p(t) for p, t in zip(self.proj, stages)], dim=1))
        s1 = self.stem1(x)
        s2 = self.stem2(s1)
        s3 = self.stem3(s2)
        s4 = self.stem4(s3)
        d = self.up1(f, s4)
        d = self.up2(d, s3)
        d = self.up3(d, s2)
        d = self.up4(d, s1)
        logits = self.head(d)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits, d, f

    def train_last_blocks(self, n: int) -> None:
        """Make only the last ``n`` transformer blocks and the final norm of the encoder trainable."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for blk in list(self.encoder.blocks)[-int(n):]:
            for p in blk.parameters():
                p.requires_grad_(True)
        norm = getattr(self.encoder, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad_(True)
