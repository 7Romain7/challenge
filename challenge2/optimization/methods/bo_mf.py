"""Multi-fidelity BO: full frames to find the bright region, then cheap zooms on it.

Lab analogue: survey the whole diagram, then zoom on the transitions that matter and
spend the remaining acquisition time there. The bottleneck measured on the benchmark is
the SNR of the contrast estimate near the floor, not detection; a zoom costs
(span / 0.3)^2 of a full frame at the *same* step (PROTOCOL P2: amplitudes stay
comparable), so the same pixel budget buys many more evaluations of the region.

Phase 1 (``switch`` of the budget): the plain BayesOpt search on full frames.
Phase 2: freeze the per-interdot floors (otherwise sampling near the peak would raise
them), pick the ``n_focus`` interdots with the largest excess around the current best
point, zoom on their bounding box, and run BO on y_F = max excess over those interdots.
"""

from __future__ import annotations

import numpy as np

from ..session import Session
from .bo import BayesOpt
from .gp import GP

ROLES = ("probe", "eval", "verify", "zoom")


class MultiFidelityBO(BayesOpt):
    name = "bo_mf"

    def __init__(self, *a, switch: float = 0.6, n_focus: int = 3, margin: float = 0.03,
                 min_span: float = 0.08, **kw) -> None:
        super().__init__(*a, **kw)
        self.switch, self.n_focus, self.margin, self.min_span = switch, n_focus, margin, min_span
        self.focus: np.ndarray | None = None

    # -- objective restricted to the focus interdots ----------------------------
    def _points(self, s: Session):
        if self.focus is None:
            return s.points()
        y, se = s.scorer.scores(cols=self.focus, min_visible=1)
        B = np.array(s.scorer.B)
        m = np.isfinite(y) & np.isin(np.array(s.roles), ROLES)
        return B[m], y[m], np.maximum(se[m], 1e-3)

    def recommend(self, s: Session) -> np.ndarray:
        B, y, se = self._points(s)
        if len(y) < 4:
            return B[int(np.argmax(y))] if len(y) else s.b0
        mean, _ = GP(rng=self.rng).fit(B, y, se).predict(B)
        return B[int(np.argmax(mean))]

    # -- phase 2 set-up ---------------------------------------------------------
    def _choose_focus(self, s: Session) -> tuple[np.ndarray, float] | None:
        B, y, se = s.points()
        if len(y) < 4:
            return None
        mean, _ = GP(rng=self.rng).fit(B, y, se).predict(B)
        top = B[np.argsort(mean)[-3:]]  # the 3 best points by posterior mean
        M, fl = s.scorer._matrix(), s.scorer.floors()
        rows = [j for j, b in enumerate(s.scorer.B) if any(np.allclose(b, t) for t in top)]
        ex = (M[rows] - fl[None, : M.shape[1]]) / fl[None, : M.shape[1]]
        score = np.nanmean(ex, axis=0)
        cand = np.flatnonzero(np.isfinite(score))
        if len(cand) == 0:
            return None
        focus = cand[np.argsort(score[cand])[-self.n_focus:]]
        pts = s.ref[focus]
        span = float(np.clip(np.ptp(pts, axis=0).max() + 2 * self.margin, self.min_span, s.cfg.span))
        span = round(span / s.step) * s.step  # whole number of native pixels
        return focus, span

    def search(self, s: Session) -> None:
        cap = s.bx.pixel_cap
        # phase 1: full-frame BO until `switch` of the budget
        while self.can_afford(s) and s.bx.n_pixels < self.switch * cap:
            self._step_full(s)
        chosen = self._choose_focus(s)
        if chosen is None:  # nothing to focus on: keep searching on full frames
            while self.can_afford(s):
                self._step_full(s)
            return
        s.scorer.freeze_floors()
        self.focus, span = chosen
        centre = s.ref[self.focus].mean(axis=0)
        cost = round(span / s.step) ** 2
        mm = min(2, len(self.focus))
        # phase 2: BO on the focus objective, measured through zooms
        while s.bx.remaining >= cost + self.reserve(s):
            B, y, se = self._points(s)
            if len(y) < 3:
                break
            gp = GP(rng=self.rng).fit(B, y, se)
            mean_obs, _ = gp.predict(B)
            cand = self._candidates(s, B, mean_obs)
            m, sd = gp.predict(cand)
            b = cand[int(np.argmax(GP.expected_improvement(m, sd, mean_obs.max())))]
            s.evaluate(b, role="zoom", span=span, centre_ref=centre, min_matches=mm)
            s.set_reco(self.recommend(s))

    def _step_full(self, s: Session) -> None:
        B = s.points()[0]
        if len(s.points()[1]) < self.n_init + 3:  # maximin exploration, as BayesOpt
            cand = s.sample_feasible(self.rng, n=300)
            if len(B):
                d = np.min(np.linalg.norm(cand[:, None, :] - B[None, :, :], axis=2), axis=1)
                cand = cand[[int(np.argmax(d))]]
            s.evaluate(cand[0])
        else:
            gp = self._fit(s)
            if gp is None:
                s.evaluate(s.sample_feasible(self.rng)[0])
            else:
                B, y, _ = s.points()
                mean_obs, _ = gp.predict(B)
                cand = self._candidates(s, B, mean_obs)
                m, sd = gp.predict(cand)
                s.evaluate(cand[int(np.argmax(GP.expected_improvement(m, sd, mean_obs.max())))])
        s.set_reco(self.recommend(s))
