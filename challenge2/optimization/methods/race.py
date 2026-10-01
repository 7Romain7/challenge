"""Racing two regions in the ROI phase instead of committing to one.

Diagnosis (dev 0-99): on 11/47 devices that lose >5 % by the region choice, the best
region had been lit (>= 20 % of its amplitude) during the full-frame phase, yet the ROI
phase zoomed on another one; once zoomed, the method cannot see it any more
("explore-then-commit" with a premature commit).

Here the ROI phase opens two arms: the usual focus (best interdots around the incumbent)
and a second focus made of interdots spatially distinct from it (> ``sep`` V away), chosen
by their own best excess over the full-frame visits. The first ``race`` share of the ROI
evaluations alternates between the arms; the method then keeps the arm with the higher
posterior maximum and spends the rest on it. Arms are compared on the *floor-normalised*
excess (a_i - floor_i) / floor_i (PROTOCOL P6): raw amplitudes differ between interdots by
their geometric gain. Parameters fixed a priori (race = 0.4, sep = 0.05 V), not tuned.
"""

from __future__ import annotations

import numpy as np

from ..session import Session
from .bo_coarse import CoarseExploreROI
from .bo_roi import ROIBayesOpt
from .gp import GP


class RaceMixin:
    race: float = 0.4
    sep: float = 0.05

    def _second_focus(self, s: Session, focus):
        M, fl = s.scorer._matrix(), s.scorer.floors()
        if M.size == 0:
            return None
        ok = np.array(s.scorer.ok)
        ex = (M[ok] - fl[None, : M.shape[1]]) / fl[None, : M.shape[1]]
        best_ex = np.nanmax(np.where(np.isfinite(ex), ex, -np.inf), axis=0)
        far = np.array([np.min(np.linalg.norm(s.ref[focus] - p, axis=1)) > self.sep for p in s.ref])
        cand = np.flatnonzero(far & np.isfinite(best_ex) & (best_ex > 0))
        if len(cand) == 0:
            return None
        top = cand[int(np.argmax(best_ex[cand]))]
        near = cand[np.linalg.norm(s.ref[cand] - s.ref[top], axis=1) <= 0.08]
        f2 = near[np.argsort(best_ex[near])[-self.n_focus:]]
        rows = np.flatnonzero(ok)
        j = rows[int(np.nanargmax(np.where(np.isfinite(ex[:, top]), ex[:, top], -np.inf)))]
        return f2, np.array(s.scorer.B)[j]

    def _arm_eval(self, s: Session, arm: dict, b) -> None:
        """One ROI evaluation of ``arm``; stored as floor-normalised excess (comparable)."""
        self.focus, self.X2, self.Y2, self.S2 = arm["focus"], arm["X"], arm["Yraw"], arm["S"]
        self._roi_eval(s, b)
        fl = arm["floors"]
        arm["Y"].append(float(self.Y2[-1] / np.mean(fl) - 1.0))
        arm["Sn"].append(float(self.S2[-1] / np.mean(fl)))

    def _arm_next(self, arm: dict, box: float) -> np.ndarray:
        if len(arm["Y"]) < 4:
            return arm["anchors"].pop(0)
        X, y, se = np.array(arm["X"]), np.array(arm["Y"]), np.array(arm["Sn"])
        gp = GP(ls_prior=self.local_sd, rng=self.rng).fit(X, y, se)
        mean_obs, _ = gp.predict(X)
        top = X[int(np.argmax(mean_obs))]
        cand = np.vstack([top + self.rng.normal(0, self.local_sd, (1500, 3)),
                          arm["b0"] + self.rng.normal(0, 2 * self.local_sd, (1500, 3))])
        cand = np.clip(cand, -box, box)
        m, sd = gp.predict(cand)
        return cand[int(np.argmax(GP.expected_improvement(m, sd, mean_obs.max())))]

    @staticmethod
    def _arm_value(arm: dict, rng) -> float:
        if len(arm["Y"]) < 4:
            return max(arm["Y"], default=-np.inf)
        X = np.array(arm["X"])
        mean, _ = GP(rng=rng).fit(X, np.array(arm["Y"]), np.array(arm["Sn"])).predict(X)
        return float(mean.max())

    def _phase2(self, s: Session, b_best) -> None:
        second = self._second_focus(s, self.focus)
        if second is None:
            return super()._phase2(s, b_best)
        s.scorer.freeze_floors()
        fl = s.scorer.floors()

        def arm(focus, b0):
            jit = [b0 + self.rng.normal(0, self.local_sd / 2, 3) for _ in range(3)]
            return {"focus": np.asarray(focus), "b0": np.asarray(b0, float), "X": [], "Yraw": [],
                    "S": [], "Y": [], "Sn": [], "floors": fl[np.asarray(focus)],
                    "anchors": [np.asarray(b0, float)] + jit}

        arms = [arm(self.focus, b_best), arm(*second)]
        self.arms, self.winner = arms, None
        patch = round(self.patch / s.step) ** 2
        left = lambda: s.bx.meas_cap - s.bx.n_meas - self.verify_frames  # noqa: E731
        n_total = max(2, left() // max(len(a["focus"]) for a in arms))
        n_race, k = int(self.race * n_total), 0
        while True:
            a = arms[k % 2] if k < n_race else self.winner
            if a is None:
                self.winner = max(arms, key=lambda x: self._arm_value(x, self.rng))
                a = self.winner
            cost = len(a["focus"]) * patch
            if s.bx.remaining < cost + self.reserve(s) or left() < len(a["focus"]):
                break
            self._arm_eval(s, a, np.clip(self._arm_next(a, s.cfg.box), -s.cfg.box, s.cfg.box))
            s.set_reco(self.recommend(s))
            k += 1
        if self.winner is None:
            self.winner = max(arms, key=lambda x: self._arm_value(x, self.rng))

    def recommend(self, s: Session) -> np.ndarray:
        arms = getattr(self, "arms", None)
        if not arms:
            return super().recommend(s)
        a = self.winner or max(arms, key=lambda x: max(x["Y"], default=-np.inf))
        if len(a["Y"]) < 4:
            return a["X"][int(np.argmax(a["Y"]))] if a["Y"] else a["b0"]
        X = np.array(a["X"])
        mean, _ = GP(rng=self.rng).fit(X, np.array(a["Y"]), np.array(a["Sn"])).predict(X)
        return X[int(np.argmax(mean))]


class RaceROI(RaceMixin, ROIBayesOpt):
    name = "bo_roi_race"


class RaceCoarse(RaceMixin, CoarseExploreROI):
    name = "bo_coarse_race"
