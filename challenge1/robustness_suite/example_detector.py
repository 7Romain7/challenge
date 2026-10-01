"""Template for scoring your own detector on the robustness suite.

    uv run python -m detection.robustness eval-fn --fn example_detector:detect

(with challenge1/robustness_suite on PYTHONPATH, see README). Replace ``detect`` by your
model: raw images (N, 150, 150) float32 in, score map (N, 150, 150) in [0, 1] out. The
threshold is chosen on val by the harness, never on the perturbed sets.
"""

from pathlib import Path

import numpy as np
import torch

from detection.evalsets import predict
from detection.export_dl import load

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "unet_lowsnr.pt"
_model = None


def detect(images: np.ndarray) -> np.ndarray:
    """Example: our U-Net (part 2). Swap in your own model here."""
    global _model
    if _model is None:
        _model, _ = load(WEIGHTS, DEV)
    return predict(_model, images, DEV).astype(np.float32)
