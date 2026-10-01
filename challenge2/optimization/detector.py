"""Frozen M5_min stick detector from challenge 1 (logistic regression on matched + s2).

The parameters come from ``detection.export_m5`` (fit on stage-1 data, threshold chosen
on stage-1 val). Only the feature maps are recomputed here, on the frame alone, so the
optimisation package stays self-contained (no import of ``detection`` / ``csd``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

PARAMS = Path(__file__).resolve().parents[2] / "challenge1/results/m5_min_params.json"


def _whiten(r: np.ndarray) -> np.ndarray:
    med = np.median(r)
    return (r - med) / (1.4826 * np.median(np.abs(r - med)) + 1e-6)


def _templates(thetas, length=5.0, width=1.0, size=11):
    r = np.arange(size) - size // 2
    yy, xx = np.meshgrid(r, r, indexing="ij")
    bank = []
    for th in thetas:
        u = xx * np.cos(th) + yy * np.sin(th)
        v = -xx * np.sin(th) + yy * np.cos(th)
        t = np.exp(-0.5 * (u / (length / 2.5)) ** 2 - 0.5 * (v / width) ** 2)
        t -= t.mean()
        bank.append(t / np.sqrt((t**2).sum()))
    return bank


BANK = _templates((np.pi / 4 - 0.2, np.pi / 4, np.pi / 4 + 0.2))


class M5Min:
    def __init__(self, path: Path = PARAMS) -> None:
        p = json.loads(Path(path).read_text())
        self.mu, self.sd = np.array(p["mu"]), np.array(p["sd"])
        self.w, self.thr = np.array(p["w"]), float(p["thr"])

    def logit(self, img: np.ndarray) -> np.ndarray:
        """Per-pixel logit for one raw frame (same preprocessing as detection.baselines.prep)."""
        x = img - np.median(img, axis=1, keepdims=True)
        z = -x / (1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6)
        matched = _whiten(np.max([fftconvolve(z, t, mode="same") for t in BANK], axis=0))
        s2 = _whiten(ndi.gaussian_filter(z, 2.0))
        X = (np.stack([matched, s2], -1) - self.mu) / self.sd
        return X @ self.w[:-1] + self.w[-1]

    def mask(self, img: np.ndarray) -> np.ndarray:
        return self.logit(img) > self.thr


class DLDetector:
    """Frozen challenge-1 segmentation network (best.pt of detection.train), same interface
    as :class:`M5Min`: ``logit(img)`` is here the stick probability, ``thr`` its val threshold.

    torch / detection.models are imported lazily (GPU extra). Same input normalisation as
    detection.evalsets.predict: per-image (x - median) / (1.4826 MAD).
    """

    def __init__(self, ckpt: str, thr: float = 0.1) -> None:
        import torch

        from detection.models import build_model

        self.torch, self.thr = torch, thr
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location=self.dev, weights_only=False)
        self.model = build_model(ck["arch"]).to(self.dev).eval()
        self.model.load_state_dict(ck["model"])

    def logit(self, img: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = np.asarray(img, np.float32)
        x = (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6)
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.from_numpy(x)[None, None].to(self.dev)).float())
        return p[0, 0].cpu().numpy()

    def mask(self, img: np.ndarray) -> np.ndarray:
        return self.logit(img) > self.thr
