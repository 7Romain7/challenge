"""Freeze a trained U-Net / TransUNet run into a small checkpoint for the repo.

    uv run --extra train python -m detection.export_dl runs/fast_unet_lowsnr challenge1/models/unet_lowsnr.pt

Keeps the EMA weights of ``best.pt`` (selected on val by (obj_F1 + tol_F1)/2) in float16 and
the decision threshold re-derived on val by ``detection.evaluate`` (``eval.json``). Nothing
else: no optimizer state, no data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from detection.models import build_model


def export(run: Path, out: Path) -> None:
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    ev = json.loads((run / "eval.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": ck["arch"],
        "state_dict": {k: v.half() if v.is_floating_point() else v for k, v in ck["model"].items()},
        "threshold": ev["thr"],
        "step": ck["step"],
        "train_args": {k: cfg[k] for k in ("seed", "steps", "bs", "lr", "intensity_law", "affine",
                                            "polarity", "noise_jitter", "pool_size")},
        "val": {k: ev["sets"]["val"][k] for k in ("obj_f1", "tol_f1", "f1", "iou")},
        "test": {k: ev["sets"]["test"][k] for k in ("obj_f1", "tol_f1", "f1", "iou")},
    }, out)
    print(f"{run} -> {out} ({out.stat().st_size / 1e6:.1f} MB, thr {ev['thr']})")


def load(path: str | Path, device="cpu"):
    """-> (model in eval mode, float32), threshold on the sigmoid output."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["arch"])
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})
    return model.to(device).eval(), ck["threshold"]


@torch.no_grad()
def predict_mask(model, thr: float, images, device="cpu"):
    """images: (N, H, W) raw CSDs (any H, W) -> boolean masks (N, H, W). Only the image is used."""
    from detection.evalsets import predict

    return predict(model, images, torch.device(device)) > thr


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
