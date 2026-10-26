"""MedSAM ViT-B segmentor with q/v LoRA adapters."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock

__all__ = ["LoRAQKV", "MedSAMSegmentor", "build_medsam_encoder"]

ROOT = Path(__file__).resolve().parents[2]
_SAM_MEAN = (123.675, 116.28, 103.53)
_SAM_STD = (58.395, 57.12, 57.375)


class LoRAQKV(nn.Module):
    """Low-rank adapters on the query and value projections of a frozen ``qkv`` layer."""

    def __init__(self, qkv: nn.Linear, rank: int = 4):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.qkv.requires_grad_(False)
        self.a_q = nn.Linear(self.dim, rank, bias=False)
        self.b_q = nn.Linear(rank, self.dim, bias=False)
        self.a_v = nn.Linear(self.dim, rank, bias=False)
        self.b_v = nn.Linear(rank, self.dim, bias=False)
        nn.init.zeros_(self.b_q.weight)
        nn.init.zeros_(self.b_v.weight)

    def forward(self, x):
        q, k, v = self.qkv(x).split(self.dim, dim=-1)
        return torch.cat((q + self.b_q(self.a_q(x)), k, v + self.b_v(self.a_v(x))), dim=-1)


def build_medsam_encoder(weights: str = "pretrained/medsam_vit_b.pth",
                         repository: str = "third_party/segment-anything",
                         lora_rank: int = 4, image_size: int = 512):
    """MedSAM image encoder with LoRA on every attention block and a trainable neck."""
    repo = Path(repository)
    repo = repo if repo.is_absolute() else ROOT / repo
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from segment_anything import sam_model_registry

    path = Path(weights)
    path = path if path.is_absolute() else ROOT / path
    sam = sam_model_registry["vit_b"](checkpoint=str(path))
    encoder = sam.image_encoder
    encoder.requires_grad_(False)
    grid = int(image_size) // 16
    if encoder.pos_embed is not None and encoder.pos_embed.shape[1] != grid:
        pos = F.interpolate(encoder.pos_embed.detach().permute(0, 3, 1, 2), size=(grid, grid),
                            mode="bicubic", align_corners=False)
        encoder.pos_embed = nn.Parameter(pos.permute(0, 2, 3, 1).contiguous(), requires_grad=False)
    for block in encoder.blocks:
        block.attn.qkv = LoRAQKV(block.attn.qkv, rank=int(lora_rank))
    encoder.neck.requires_grad_(True)
    return encoder, 256


class MedSAMSegmentor(nn.Module):
    """MedSAM embedding decoded by a light convolutional decoder with an image stem.

    Args:
        encoder: MedSAM image encoder.
        encoder_channels: width of the image embedding.
        num_classes: number of segmentation classes.
        in_channels: number of input channels.
        decoder_channels: width of the last decoder stage.
        image_size: encoder input resolution.
    """

    def __init__(self, encoder: nn.Module, encoder_channels: int = 256, num_classes: int = 5,
                 in_channels: int = 3, decoder_channels: int = 64, image_size: int = 512):
        super().__init__()
        c = decoder_channels
        self.encoder = encoder
        self.sam_size = int(image_size)
        self.register_buffer("mean", torch.tensor(_SAM_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_SAM_STD).view(1, 3, 1, 1))

        self.low_proj = ConvBlock(encoder_channels, 256)
        self.stem1 = ConvBlock(in_channels, c)
        self.stem2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c, c * 2))
        self.up1 = nn.ConvTranspose2d(256, c * 4, 2, stride=2)
        self.dec1 = ConvBlock(c * 4, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up3 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec3 = ConvBlock(c, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    @staticmethod
    def _align(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor):
        xs = F.interpolate(x, size=(self.sam_size, self.sam_size), mode="bilinear", align_corners=False)
        embedding = self.encoder((xs * 255.0 - self.mean) / self.std)
        low = self.low_proj(embedding)
        s1 = self.stem1(x)
        s2 = self.stem2(s1)
        d = self.dec1(self.up1(low))
        d = self.dec2(torch.cat([self._align(self.up2(d), s2), s2], dim=1))
        d = self._align(self.dec3(self.up3(d)), s1)
        logits = self.head(d)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits, d, low
