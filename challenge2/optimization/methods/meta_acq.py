"""Meta-learned acquisition on top of the GP posterior (MetaBO-style), inside ``bo_roi``.

The GP stays the estimator of the landscape; a small network only replaces expected
improvement in the full-frame phase (the search for the bright region, where the
benchmark loses most). It never sees an image nor the simulator's peak shape, only
posterior statistics of each candidate, so what it can learn is a *strategy* (when to
explore, how far to jump given the remaining budget), not the simulator.

Features of a candidate x (y-statistics normalised by the GP output scale):
  log EI, (mu - best)/s, sd/s, distance to the incumbent, distance to the nearest
  measured point, budget fraction t, t * sd/s, drift-prediction std / sigma_max.
Score = w_lin . phi + w2 . tanh(W1 phi + b1). Initial weights (w_lin = e_logEI, rest 0)
reproduce EI exactly, so the policy starts from the baseline. Weights are trained in
``training/train_meta_acq.py`` (CMA-ES on dev 100-999, observable reward only: the
score re-measured on fresh frames, PROTOCOL 5.3); loaded from ``$C12_META_W``.
"""

from __future__ import annotations

import os

import numpy as np

from ..session import Session
from .bo_roi import ROIBayesOpt
from .gp import GP

N_FEAT, N_HID = 8, 8
N_PARAMS = N_FEAT * N_HID + N_HID + N_HID + N_FEAT


def init_weights() -> np.ndarray:
    w = np.zeros(N_PARAMS)
    w[-N_FEAT] = 1.0  # w_lin on log EI
    return w


def unpack(w):
    i = 0
    W1 = w[i:i + N_FEAT * N_HID].reshape(N_HID, N_FEAT); i += N_FEAT * N_HID
    b1 = w[i:i + N_HID]; i += N_HID
    w2 = w[i:i + N_HID]; i += N_HID
    return W1, b1, w2, w[i:i + N_FEAT]


class Phase1Done(Exception):
    """Raised in training mode once the focus region is chosen."""


class MetaAcqROI(ROIBayesOpt):
    name = "bo_roi_meta"

    def __init__(self, *a, weights=None, stop_after_phase1: bool = False, **kw) -> None:
        super().__init__(*a, **kw)
        if weights is None:
            path = os.environ.get("C12_META_W", "")
            weights = np.load(path) if path else init_weights()
        self.W = unpack(np.asarray(weights, float))
        self.stop_after_phase1 = stop_after_phase1
        self.phase1_best = None
        self.phase1_score = 0.0

    def _features(self, s: Session, cand, m, sd, B, mean_obs) -> np.ndarray:
        scale = max(float(np.std(mean_obs)), 1e-6)
        best = mean_obs.max()
        ei = GP.expected_improvement(m, sd, best) / scale
        inc = B[int(np.argmax(mean_obs))]
        d_inc = np.linalg.norm(cand - inc, axis=1) / 0.3
        d_nn = np.min(np.linalg.norm(cand[:, None, :] - B[None, :, :], axis=2), axis=1) / 0.3
        t = s.bx.n_pixels / (self.switch * s.bx.pixel_cap)
        dstd = np.array([s.drift.predict_std(c - s.b0) for c in cand]) / s.cfg.sigma_max
        z_sd = sd / scale
        return np.stack([np.log(ei + 1e-9), (m - best) / scale, z_sd, d_inc, d_nn,
                         np.full(len(cand), t), t * z_sd, dstd], axis=1)

    def _acquire(self, s: Session, cand, m, sd, B, mean_obs) -> np.ndarray:
        phi = self._features(s, cand, m, sd, B, mean_obs)
        W1, b1, w2, wl = self.W
        score = phi @ wl + np.tanh(phi @ W1.T + b1) @ w2
        return cand[int(np.argmax(score))]

    def _step_full(self, s: Session) -> None:
        if len(s.points()[1]) < self.n_init + 3:
            return super()._step_full(s)
        gp = self._fit(s)
        if gp is None:
            s.evaluate(s.sample_feasible(self.rng)[0])
        else:
            B, _, _ = s.points()
            mean_obs, _ = gp.predict(B)
            cand = self._candidates(s, B, mean_obs)
            m, sd = gp.predict(cand)
            s.evaluate(self._acquire(s, cand, m, sd, B, mean_obs))
        s.set_reco(self.recommend(s))

    def _choose_focus(self, s: Session):
        chosen = super()._choose_focus(s)
        if self.stop_after_phase1:
            # training reward = observable only: tracked score re-measured on fresh frames
            b = super(ROIBayesOpt, self).recommend(s)
            self.phase1_best = b
            s.verify(b, 2)
            _, y, _ = s.points(roles=("verify",))
            self.phase1_score = float(np.mean(y)) if len(y) else 0.0
            raise Phase1Done
        return chosen
