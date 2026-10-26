"""Segmentation networks of the three backbone families."""
from .dinov3 import DINOv3Segmentor, build_dinov3_encoder
from .unet_pair import PairUNet
from .medsam import LoRAQKV, MedSAMSegmentor, build_medsam_encoder
from .unet import UNet

__all__ = [
    "DINOv3Segmentor", "PairUNet", "LoRAQKV", "MedSAMSegmentor", "UNet",
    "build_dinov3_encoder", "build_medsam_encoder", "build_unet", "build_dinov3", "build_medsam", "build_pair_unet",
    "model_key",
]


def model_key(model_cfg) -> str:
    """Registry key of a model configuration; ``paired: true`` selects the U-Net on pairs of adjacent sections."""
    return "unet_pair" if bool(getattr(model_cfg, "paired", False)) else str(model_cfg.backbone)


def build_unet(model_cfg, num_classes: int):
    return UNet(int(model_cfg.in_channels), num_classes, int(model_cfg.base_channels))


def build_dinov3(model_cfg, num_classes: int):
    encoder, channels = build_dinov3_encoder(model_cfg.pretrained, model_cfg.repository)
    model = DINOv3Segmentor(encoder, channels, num_classes, int(model_cfg.in_channels),
                            int(model_cfg.decoder_channels), int(model_cfg.low_channels),
                            tuple(model_cfg.enc_layers), bool(model_cfg.freeze_encoder))
    if int(getattr(model_cfg, "trainable_blocks", 0)) > 0:
        model.train_last_blocks(int(model_cfg.trainable_blocks))
    return model


def build_medsam(model_cfg, num_classes: int):
    encoder, channels = build_medsam_encoder(model_cfg.pretrained, model_cfg.repository,
                                             int(model_cfg.lora_rank), int(model_cfg.image_size))
    return MedSAMSegmentor(encoder, channels, num_classes, int(model_cfg.in_channels),
                           int(model_cfg.decoder_channels), int(model_cfg.image_size))


def build_pair_unet(model_cfg, num_classes: int):
    return PairUNet(int(model_cfg.in_channels), num_classes)
