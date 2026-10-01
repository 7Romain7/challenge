"""Method 2 (light) - PFN surrogate: a transformer trained only on synthetic landscapes.

The network reads the history {(b_j, y_j, se_j)} as context tokens and predicts a Gaussian
(mean, sd) of the latent score at query barriers, in one forward pass: amortised Bayesian
inference (Mueller et al., PFN). It replaces the GP of :class:`BayesOpt` unchanged (same
EI, same candidates, same recommendation rule), so any difference is the surrogate.

Trained by ``training.train_pfn`` on ``training.priors`` landscapes WITHOUT the Lorentzian
family (the simulator's shape, PROTOCOL P15); never on the simulator. torch is imported
lazily (GPU extra); inference runs on CPU.
"""

from __future__ import annotations

import os

import numpy as np

from ..session import Session
from .bo import BayesOpt

BOX = 0.6


def build_net(d: int = 128, layers: int = 4, heads: int = 4):
    import torch
    import torch.nn as nn

    class PFN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ctx_in = nn.Sequential(nn.Linear(5, d), nn.GELU(), nn.Linear(d, d))
            self.qry_in = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
            self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(d, heads, 4 * d, 0.0, batch_first=True,
                                                                   norm_first=True) for _ in range(layers))
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

        def forward(self, xc, yc, sc, cmask, xq):
            """xc (B,C,3) in [-1,1]; yc, sc (B,C) normalised; cmask (B,C) True = padding;
            xq (B,Q,3). Every token attends to the (non-padded) context only."""
            C = xc.shape[1]
            h = torch.cat([self.ctx_in(torch.cat([xc, yc[..., None], sc[..., None]], -1)),
                           self.qry_in(xq)], 1)
            key_pad = torch.cat([cmask, torch.ones(xq.shape[:2], dtype=torch.bool, device=xq.device)], 1)
            for blk in self.blocks:
                h = blk(h, src_key_padding_mask=key_pad)
            out = self.head(h[:, C:])
            return out[..., 0], out[..., 1].clamp(-7, 4)  # mean, log sd (normalised units)

    return PFN()


def normalise(y, se):
    """Context-only normalisation, as the GP does: no information from the queries."""
    mu, sd = float(np.mean(y)), max(float(np.std(y)), 0.05)  # floor: flat histories
    return (y - mu) / sd, se / sd, mu, sd


class PFNSurrogate:
    """GP-compatible surrogate: ``fit(B, y, se)`` stores the context, ``predict`` -> (mean, sd)."""

    _net = None

    def __init__(self, ckpt: str) -> None:
        import torch

        self.torch = torch
        torch.set_num_threads(1)
        if PFNSurrogate._net is None:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            net = build_net(**ck["arch"])
            net.load_state_dict(ck["model"])
            PFNSurrogate._net = net.eval()

    def fit(self, B, y, se):
        self.yn, self.sn, self.mu, self.sd = normalise(np.asarray(y, float), np.asarray(se, float))
        self.B = np.asarray(B, float) / BOX
        return self

    def predict(self, Xs):
        t = self.torch
        f = lambda a: t.as_tensor(np.asarray(a, np.float32))[None]  # noqa: E731
        with t.no_grad():
            m, ls = PFNSurrogate._net(f(self.B), f(self.yn), f(np.log(self.sn + 1e-3)),
                                      t.zeros(1, len(self.B), dtype=t.bool), f(np.asarray(Xs) / BOX))
        return m[0].numpy() * self.sd + self.mu, np.exp(ls[0].numpy()) * self.sd


class PFNBayesOpt(BayesOpt):
    """BayesOpt with the GP swapped for the PFN surrogate (checkpoint: $C12_PFN_CKPT)."""

    name = "bo_pfn"

    def __init__(self, *a, ckpt: str | None = None, **kw) -> None:
        super().__init__(*a, **kw)
        self.ckpt = ckpt or os.environ.get("C12_PFN_CKPT", "")

    def _model(self):
        return PFNSurrogate(self.ckpt)

    def _fit(self, s: Session):
        B, y, se = s.points()
        if len(y) < 3:
            return None
        self.gp = self._model().fit(B, y, se)
        return self.gp

    def recommend(self, s: Session) -> np.ndarray:
        B, y, se = s.points()
        if len(y) < 4:
            return B[int(np.argmax(y))] if len(y) else s.b0
        mean, _ = self._model().fit(B, y, se).predict(B)
        return B[int(np.argmax(mean))]
