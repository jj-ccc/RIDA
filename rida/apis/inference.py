"""Model loading and prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..datasets.transforms import normalize_image, to_tensor
from ..models import build_dinov3, build_medsam, build_pair_unet, build_unet, model_key
from ..utils import load_model_weights, load_weights

__all__ = ["BUILDERS", "init_model", "predict_slice", "predict_slice_dir"]

BUILDERS = {"unet": build_unet, "dinov3": build_dinov3, "medsam": build_medsam, "unet_pair": build_pair_unet}


def init_model(cfg, checkpoint: str | Path | None, device: str | torch.device = "cuda"):
    """Build the network of a setting and load ``checkpoint``."""
    model = BUILDERS[model_key(cfg.model)](cfg.model, int(cfg.data.num_classes))
    if checkpoint:
        load_model_weights(model, load_weights(checkpoint))
    return model.to(device).eval()


@torch.no_grad()
def predict_slice(model, image: np.ndarray, device: str | torch.device = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """Label map and class probabilities of one stored slice."""
    x = to_tensor(normalize_image(image)).unsqueeze(0).to(device)
    prob = torch.softmax(model(x)[0], dim=1)[0]
    return prob.argmax(0).cpu().numpy().astype(np.uint8), prob.cpu().numpy()


@torch.no_grad()
def predict_slice_dir(model, in_dir: str | Path, out_dir: str | Path, device: str | torch.device = "cuda") -> int:
    """Segment every ``.npz`` slice of ``in_dir`` and write ``<name>.npy`` label maps to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(in_dir).glob("*.npz"))
    for f in files:
        label, _ = predict_slice(model, np.load(f)["image"], device)
        np.save(out / f"{f.stem}.npy", label)
    return len(files)
