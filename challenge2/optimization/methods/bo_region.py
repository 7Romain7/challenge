"""Per-region BO: one GP per spatial group of interdots instead of one GP on the max.

Diagnosis (dev 0-99, ``evaluation.truth.region_diagnosis``): the max-over-interdots score
hides every region but the brightest one. On 11/47 devices that lose >5 % by the region
choice, the best region was lit to >= 20 % of its amplitude at some visited point and was
not pursued, because another region dominated the max there. Each full frame sees every
interdot, so it observes every region's landscape at once: modelling them separately uses
that information (PROTOCOL 5.1, variant B).

Groups: k-means on the reference interdot positions (k = 6, deliberately more than the
number of regions one would expect: splitting a region in two is harmless, merging two
regions brings back the max). No group label from the simulator is used. Objective of
group c: max shrunk excess over its interdots (``Scorer.scores(cols=c)``). Acquisition:
max over groups of EI against the best posterior mean over all groups. Phase 2 (ROI
patches) focuses on the best interdots of the winning group.
"""

from __future__ import annotations

import numpy as np

from ..session import Session
from .bo_mf import ROLES
from .bo_roi import ROIBayesOpt
from .gp import GP


def kmeans(X: np.ndarray, k: int, rng, n_iter: int = 30) -> np.ndarray:
    k = min(k, len(X))
    c = X[rng.choice(len(X), k, replace=False)]
    lab = np.zeros(len(X), int)
    for _ in range(n_iter):
        lab = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        new = np.array([X[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return lab


class RegionBO(ROIBayesOpt):
    name = "bo_region"

    def __init__(self, *a, n_groups: int = 6, **kw) -> None:
        kw.setdefault("switch", "auto")
        super().__init__(*a, **kw)
        self.n_groups = n_groups

    # -- groups and per-group data ------------------------------------------------
    def _groups(self, s: Session) -> list[np.ndarray]:
        if len(s.ref) < 2:
            return [np.arange(len(s.ref))]
        lab = kmeans(s.ref, self.n_groups, np.random.default_rng(0))  # deterministic split
        return [np.flatnonzero(lab == j) for j in np.unique(lab)]

    def _group_points(self, s: Session, cols):
        y, se = s.scorer.scores(cols=cols, min_visible=1)
        B = np.array(s.scorer.B)
        m = np.isfinite(y) & np.isin(np.array(s.roles), ROLES)
        return B[m], y[m], np.maximum(se[m], 1e-3)

    def _fit_groups(self, s: Session):
        fits = []
        for cols in self._groups(s):
            B, y, se = self._group_points(s, cols)
            if len(y) >= 4 and np.ptp(y) > 0:
                fits.append((cols, GP(rng=self.rng).fit(B, y, se)))
        return fits

    def _reco_groups(self, s: Session, fits=None):
        """(barriers, group cols) maximising the best group posterior mean over visited points."""
        fits = self._fit_groups(s) if fits is None else fits
        B = np.array(s.scorer.B)
        ok = np.isin(np.array(s.roles), ROLES)
        if not fits or not ok.any():
            return super(ROIBayesOpt, self).recommend(s), None
        B = B[ok]
        means = np.array([gp.predict(B)[0] for _, gp in fits])  # (groups, points)
        g, j = np.unravel_index(int(np.argmax(means)), means.shape)
        return B[j], fits[g][0]

    def recommend(self, s: Session) -> np.ndarray:
        if len(self.Y2) >= 4:
            return super().recommend(s)
        return self._reco_groups(s)[0]

    # -- phase 1 step -------------------------------------------------------------
    def _step_full(self, s: Session) -> None:
        if len(s.points()[1]) < self.n_init + 3:
            return super()._step_full(s)
        fits = self._fit_groups(s)
        if not fits:
            return super()._step_full(s)
        B = np.array(s.scorer.B)[np.isin(np.array(s.roles), ROLES)]
        inc = max(float(gp.predict(B)[0].max()) for _, gp in fits)
        B_all, _, _ = s.points()
        mean_any = np.max([gp.predict(B_all)[0] for _, gp in fits], axis=0)
        cand = self._candidates(s, B_all, mean_any)
        if self.acq == "ucb":  # sqrt(beta) = 2, as in phase 1 of bo_roi_auto_ucb
            acq = np.max([(lambda m, sd: m + 2.0 * sd)(*gp.predict(cand)) for _, gp in fits], axis=0)
        else:
            acq = np.max([GP.expected_improvement(*gp.predict(cand), inc) for _, gp in fits], axis=0)
        s.evaluate(cand[int(np.argmax(acq))])
        s.set_reco(self.recommend(s))

    # -- phase 2 focus ------------------------------------------------------------
    def _choose_focus(self, s: Session):
        fits = self._fit_groups(s)
        b_best, cols = self._reco_groups(s, fits)
        if cols is None or len(cols) == 0:
            return super()._choose_focus(s)
        M, fl = s.scorer._matrix(), s.scorer.floors()
        Ball = np.array(s.scorer.B)
        rows = np.argsort(np.linalg.norm(Ball - b_best, axis=1))[:3]  # the reco and its neighbours
        ex = (M[rows][:, cols] - fl[cols]) / fl[cols]
        score = np.nanmean(ex, axis=0)
        okc = np.isfinite(score)
        if not okc.any():
            return super()._choose_focus(s)
        focus = np.asarray(cols)[okc][np.argsort(score[okc])[-self.n_focus:]]
        self._b_best = b_best
        return focus, s.cfg.span

    def search(self, s: Session) -> None:
        while self.can_afford(s) and self._phase1(s):
            self._step_full(s)
        chosen = self._choose_focus(s)
        if chosen is None:
            while self.can_afford(s):
                self._step_full(s)
            return
        self.focus = chosen[0]
        b_best = getattr(self, "_b_best", None)
        if b_best is None:
            b_best = super(ROIBayesOpt, self).recommend(s)
        self._phase2(s, b_best)
