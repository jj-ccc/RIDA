"""Trainers of the three backbone families."""
from ..models import model_key
from .base import BaseTrainer
from .dinov3 import DINOv3Trainer
from .unet_pair import PairUNetTrainer
from .medsam import MedSAMTrainer
from .slice_trainer import SliceTrainer
from .unet import UNetTrainer

TRAINERS = {"unet": UNetTrainer, "dinov3": DINOv3Trainer, "medsam": MedSAMTrainer, "unet_pair": PairUNetTrainer}

__all__ = ["BaseTrainer", "SliceTrainer", "UNetTrainer", "DINOv3Trainer", "MedSAMTrainer", "PairUNetTrainer",
           "TRAINERS", "build_trainer"]


def build_trainer(cfg, work_dir, device="cuda"):
    return TRAINERS[model_key(cfg.model)](cfg, work_dir, device=device)
