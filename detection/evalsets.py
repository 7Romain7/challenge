"""Frozen evaluation sets + batched inference / scoring shared by train and evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from detection.metrics import object_scores, pixel_counts, pixel_scores
from detection.synth import robust_normalize

THRESHOLDS = tuple(np.round(np.arange(0.1, 0.91, 0.05), 2))


class EvalSet:
    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        path = Path(path)
        self.name = path.name
        self.images = np.load(path / "images.npy", mmap_mode="r")[:limit]
        self.masks = np.load(path / "masks.npy", mmap_mode="r")[:limit]
        self.sticks = []
        with (path / "sticks.jsonl").open() as f:
            for line in f:
                self.sticks.append(json.loads(line)["sticks"])
        self.sticks = self.sticks[: len(self.images)]
        self.meta = json.loads((path / "meta.json").read_text())

    def __len__(self) -> int:
        return len(self.images)


@torch.no_grad()
def predict(model, images: np.ndarray, device, bs: int = 64, amp_dtype=None) -> np.ndarray:
    """Probabilities (N, H, W) float16 on CPU."""
    model.eval()
    out = []
    for i in range(0, len(images), bs):
        x = torch.from_numpy(np.asarray(images[i : i + bs], dtype=np.float32))[:, None].to(device)
        x = robust_normalize(x)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            p = torch.sigmoid(model(x).float())
        out.append(p[:, 0].half().cpu())
    return torch.cat(out).numpy()


def score(probs: np.ndarray, es: EvalSet, thr: float | None = None, device="cpu") -> dict:
    """All metrics. If ``thr`` is None, the pixel F1-optimal threshold *of this set* is
    used — only legitimate on **val**; test/OOD must receive the val threshold."""
    counts = None
    for i in range(0, len(probs), 256):
        p = torch.from_numpy(probs[i : i + 256].astype(np.float32))[:, None].to(device)
        g = torch.from_numpy(np.asarray(es.masks[i : i + 256]))[:, None].to(device)
        c = pixel_counts(p, g, THRESHOLDS)
        counts = c if counts is None else {k: counts[k] + c[k] for k in c}
    rows = pixel_scores(counts, THRESHOLDS)
    if thr is None:
        best = max(rows, key=lambda r: r["tol_f1"])
    else:
        best = min(rows, key=lambda r: abs(r["thr"] - thr))
    h, w = probs.shape[1:]
    obj = object_scores(probs > best["thr"], es.sticks, h, w)
    return {"set": es.name, "n": len(es), **best, **obj, "pr_curve": rows}
