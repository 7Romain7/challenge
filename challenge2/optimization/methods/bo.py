"""Method 1 - Bayesian optimisation on the tracked contrast score.

GP (Matern-5/2 ARD, per-point noise from the front-end) over the 3 barriers; expected
improvement over candidates drawn *inside the trackable region* (Session.feasible), so
the explored region grows as the drift model learns. The recommendation is the
posterior-mean argmax over visited points, not the noisy raw maximum (winner's curse).
"""

from __future__ import annotations

import numpy as np

from ..session import Session
from .base import Method
from .gp import GP


class BayesOpt(Method):
    name = "bo"

    def __init__(self, *a, n_init: int = 4, n_cand: int = 3000, **kw) -> None:
        super().__init__(*a, **kw)
        self.n_init, self.n_cand = n_init, n_cand
        self.gp: GP | None = None

    def _fit(self, s: Session):
        B, y, se = s.points()
        if len(y) < 3:
            return None
        self.gp = GP(rng=self.rng).fit(B, y, se)
        return self.gp

    def recommend(self, s: Session) -> np.ndarray:
        B, y, se = s.points()
        if len(y) < 4:
            return B[int(np.argmax(y))] if len(y) else s.b0
        gp = GP(rng=self.rng).fit(B, y, se)
        mean, _ = gp.predict(B)
        return B[int(np.argmax(mean))]

    def _candidates(self, s: Session, B: np.ndarray, mean: np.ndarray) -> np.ndarray:
        cand = s.sample_feasible(self.rng, n=self.n_cand)
        top = B[np.argsort(mean)[-3:]]  # local refinement around the current best
        loc = (top[:, None, :] + self.rng.normal(0, 0.05, (3, 100, 3))).reshape(-1, 3)
        loc = np.clip(loc, -s.cfg.box, s.cfg.box)
        loc = loc[[s.feasible(c) for c in loc]]
        return np.vstack([cand, loc]) if len(loc) else cand

    def search(self, s: Session) -> None:
        # exploration: maximin among feasible random candidates
        while self.can_afford(s) and len(s.points()[1]) < self.n_init + 3:
            B = s.points()[0]
            cand = s.sample_feasible(self.rng, n=300)
            if len(B):
                d = np.min(np.linalg.norm(cand[:, None, :] - B[None, :, :], axis=2), axis=1)
                cand = cand[[int(np.argmax(d))]]
            s.evaluate(cand[0])
            s.set_reco(self.recommend(s))
        while self.can_afford(s):
            gp = self._fit(s)
            B, y, _ = s.points()
            if gp is None:
                s.evaluate(s.sample_feasible(self.rng)[0])
                continue
            mean_obs, _ = gp.predict(B)
            cand = self._candidates(s, B, mean_obs)
            m, sd = gp.predict(cand)
            ei = GP.expected_improvement(m, sd, mean_obs.max())
            s.evaluate(cand[int(np.argmax(ei))])
            s.set_reco(self.recommend(s))
