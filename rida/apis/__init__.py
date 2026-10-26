"""High-level training, inference and evaluation interfaces."""
from .evaluate import evaluate, evaluate_em, evaluate_pairs, evaluate_medical
from .inference import BUILDERS, init_model, predict_slice, predict_slice_dir
from .train import train_setting

__all__ = ["evaluate", "evaluate_em", "evaluate_pairs", "evaluate_medical", "BUILDERS", "init_model",
           "predict_slice", "predict_slice_dir", "train_setting"]
