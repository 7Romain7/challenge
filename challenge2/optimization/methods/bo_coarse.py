"""Coarse-to-fine exploration: survey with 2x-step frames, confirm and track with native ones.

Diagnosis (dev 0-99): on 26/47 devices that lose >5 % by the region choice, the best
region was never lit to 10 % of its amplitude at any visited point: a coverage problem.
Measured on the simulator (training seeds, diagnostic only): at twice the native step a
frame costs 4x fewer pixels and a lit region stays far above the noise (brightest-interdot
SNR ~30 lit vs ~5 at the floor), while half-lit interdots fade. The current max-score
already cannot tell <~20 % lit from noise (y ~ 1.8 floor units at the floor), so the coarse
survey loses little sensitivity for 4x more points. Lab analogue: fast low-resolution scans
to search, full resolution to measure.

Comparability (PROTOCOL P2): coarse scores are only compared with coarse scores (own
Scorer, own floors, own GP); fine frames keep the session's scorer. Coarse frames cannot be
registered at the floor (interdots fade), so they are centred on the drift *prediction*
inside the same trust region, and amplitudes are read as the maximum filter response in a
disk sized by the predicted drift uncertainty.

Allocation (from both caps, no tuned share): ROI patches keep ``roi_meas`` of the measurement cap;
the rest of phase 1 is cycles of ``ratio`` coarse frames + 1 native frame, placed on the
best coarse point (confirmation + registration), or on a maximin feasible point while
nothing is lit, which keeps the trust region growing.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..perception import Perception
from ..scoring import Scorer
from ..session import Session
from .bo_roi import ROIBayesOpt
from .gp import GP


class CoarseExploreROI(ROIBayesOpt):
    name = "bo_coarse"

    def __init__(self, *a, coarse: int = 2, ratio: int = 4, roi_meas: float = 0.3, **kw) -> None:
        kw.setdefault("switch", "auto")
        super().__init__(*a, **kw)
        self.coarse, self.ratio, self.roi_meas = coarse, ratio, roi_meas
        self.cscore = Scorer()
        self.cB: list = []
        self.cpercep: Perception | None = None

    # -- coarse frame ---------------------------------------------------------------
    def _coarse_eval(self, s: Session, b) -> None:
        b = np.clip(np.asarray(b, float), -s.cfg.box, s.cfg.box)
        db = b - s.b0
        span, step = s.cfg.span, self.coarse * s.step
        centre = s._focus_ref(span) + s.drift.predict(db)
        img = s.bx.measure(g1=b[0], g2=centre[0], g3=b[1], g4=centre[1], g5=b[2],
                           span_h=span, span_v=span, step_h=step, step_v=step)
        if self.cpercep is None:
            self.cpercep = Perception()
            self.cpercep.polarity = s.percep.polarity
        fr = self.cpercep.process(img)
        origin = centre - span / 2
        rc = (s.ref + s.drift.predict(db) - origin) / step  # (x=col, y=row)
        rad = int(np.clip(np.ceil(3 * s.drift.predict_std(db) / step), 1, 6))
        mx = ndi.maximum_filter(fr.resp, size=2 * rad + 1)
        n_v, n_h = mx.shape
        amps = np.full(len(s.ref), np.nan)
        for i, (c, r) in enumerate(rc):
            ri, ci = int(round(r)), int(round(c))
            if rad <= ri < n_v - rad and rad <= ci < n_h - rad:
                amps[i] = mx[ri, ci]
        self.cscore.add(b, amps, fr.sigma, True)
        self.cB.append(b)

    def _coarse_points(self):
        y, se = self.cscore.scores(min_visible=2)
        B = np.array(self.cB)
        m = np.isfinite(y)
        return B[m], y[m], np.maximum(se[m], 1e-3)

    def _coarse_next(self, s: Session) -> np.ndarray:
        B, y, se = self._coarse_points()
        if len(y) < 6:  # space-filling start, inside the trust region
            cand = s.sample_feasible(self.rng, n=300)
            allB = np.vstack([np.array(s.scorer.B)] + ([np.array(self.cB)] if self.cB else []))
            d = np.min(np.linalg.norm(cand[:, None, :] - allB[None], axis=2), axis=1)
            return cand[int(np.argmax(d))]
        gp = GP(rng=self.rng).fit(B, y, se)
        mean_obs, _ = gp.predict(B)
        cand = self._candidates(s, B, mean_obs)
        m, sd = gp.predict(cand)
        if self.acq == "ucb":
            return cand[int(np.argmax(m + 2.0 * sd))]
        return cand[int(np.argmax(GP.expected_improvement(m, sd, mean_obs.max())))]

    def _fine_next(self, s: Session) -> np.ndarray:
        """Native frame on the best coarse point if it is lit and not yet confirmed,
        else a maximin feasible point (grows the trust region)."""
        B, y, se = self._coarse_points()
        Bf = np.array(s.scorer.B)
        if len(y) >= 4:
            mean, _ = GP(rng=self.rng).fit(B, y, se).predict(B)
            for j in np.argsort(mean)[::-1][:3]:
                lit = y[j] - se[j] > np.nanmedian(y) + 2 * np.nanmedian(se)
                if lit and np.min(np.linalg.norm(Bf - B[j], axis=1)) > 0.02 and s.feasible(B[j]):
                    return B[j]
        cand = s.sample_feasible(self.rng, n=300)
        d = np.min(np.linalg.norm(cand[:, None, :] - Bf[None], axis=2), axis=1)
        return cand[int(np.argmax(d))]

    # -- driver ---------------------------------------------------------------------------
    def search(self, s: Session) -> None:
        frame = round(s.cfg.span / s.step) ** 2
        cframe = round(s.cfg.span / (self.coarse * s.step)) ** 2
        patch = round(self.patch / s.step) ** 2
        roi_meas = int(round(self.roi_meas * s.bx.meas_cap))  # share of the cap (90 of 300)
        roi_px = roi_meas * patch
        k = 0
        while True:
            meas_left = s.bx.meas_cap - s.bx.n_meas - self.verify_frames - roi_meas
            px_left = s.bx.remaining - self.reserve(s) - roi_px
            fine_turn = k % (self.ratio + 1) == self.ratio
            cost = frame if fine_turn else cframe
            if meas_left < 1 or px_left < cost:
                break
            if fine_turn:
                s.evaluate(self._fine_next(s))
            else:
                self._coarse_eval(s, self._coarse_next(s))
            s.set_reco(self.recommend(s))
            k += 1
        chosen = self._choose_focus(s)
        if chosen is None:
            while self.can_afford(s):
                self._step_full(s)
            return
        self.focus = chosen[0]
        self._phase2(s, super(ROIBayesOpt, self).recommend(s))


class CoarseRegionROI(CoarseExploreROI):
    """Coarse survey scored per spatial group of interdots (one GP per group, max of UCB).

    Combines the two diagnosis-driven ideas: more survey points (coarse frames) and no
    masking of a second region by the brightest one (groups as in ``bo_region``)."""

    name = "bo_coarse_region"

    def __init__(self, *a, n_groups: int = 6, **kw) -> None:
        super().__init__(*a, **kw)
        self.n_groups = n_groups

    def _coarse_next(self, s: Session) -> np.ndarray:
        from .bo_region import kmeans
        if len(self.cB) < 6 or len(s.ref) < 2:
            return super()._coarse_next(s)
        lab = kmeans(s.ref, self.n_groups, np.random.default_rng(0))
        Ball = np.array(self.cB)
        width = max(len(a) for a in self.cscore.A)  # the reference grows; older rows are shorter
        fits = []
        for j in np.unique(lab):
            cols = np.flatnonzero(lab == j)
            cols = cols[cols < width]
            if len(cols) == 0:
                continue
            y, se = self.cscore.scores(cols=cols, min_visible=1)
            m = np.isfinite(y)
            if m.sum() >= 4 and np.ptp(y[m]) > 0:
                fits.append(GP(rng=self.rng).fit(Ball[m], y[m], np.maximum(se[m], 1e-3)))
        if not fits:
            return super()._coarse_next(s)
        mean_any = np.max([gp.predict(Ball)[0] for gp in fits], axis=0)
        cand = self._candidates(s, Ball, mean_any)
        if self.acq == "ucb":
            acq = np.max([(lambda m, sd: m + 2.0 * sd)(*gp.predict(cand)) for gp in fits], axis=0)
        else:
            inc = float(mean_any.max())
            acq = np.max([GP.expected_improvement(*gp.predict(cand), inc) for gp in fits], axis=0)
        return cand[int(np.argmax(acq))]
