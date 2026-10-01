"""ROI-sampled BO: after the survey, measure only small patches on the interdots that matter.

Borrowed from active / adaptive imaging (sparse dynamic sampling in microscopy; Lennon et
al. 2019 and ray-based acquisition for quantum dots): do not image the whole diagram, spend
the acquisition time where the information is. A 25x25 px patch on each focus interdot
costs ~3 % of a full frame, so the remaining budget buys ~10x more evaluations.

Phase 1 (``switch`` of the budget): full-frame BayesOpt, as ``MultiFidelityBO``.
Phase 2: pick the focus interdots (largest excess around the current best point), then a
*local* BO whose objective is measured only through patches centred on their predicted
positions. Estimator, identical for every phase-2 point (so values are comparable):
amplitude = max of the filter readout within a fixed 5 px disk of the predicted position,
y = mean over the focus interdots (they belong to the bright region and co-modulate, so
averaging them adds their SNR). Each found interdot also refines the drift model.
"""

from __future__ import annotations

import numpy as np

from ..session import Session
from .bo_mf import MultiFidelityBO
from .gp import GP


class ROIBayesOpt(MultiFidelityBO):
    name = "bo_roi"

    def __init__(self, *a, switch: float = 0.4, patch: float = 0.05, disk_px: int = 5,
                 local_sd: float = 0.08, lit_thr: float | None = None, max_switch: float = 0.8,
                 acq: str = "ei", **kw) -> None:
        super().__init__(*a, switch=switch, **kw)
        self.acq = acq  # phase-1 acquisition: "ei" or "ucb"
        # adaptive switch: stay on full frames past `switch` until some point is lit
        # (lower bound y - se > lit_thr, in floor units), at most until `max_switch`
        self.lit_thr, self.max_switch = lit_thr, max_switch
        self.patch, self.disk_px, self.local_sd = patch, disk_px, local_sd
        self.X2: list = []
        self.Y2: list = []
        self.S2: list = []

    def recommend(self, s: Session) -> np.ndarray:
        if len(self.Y2) >= 4:
            X, y, se = np.array(self.X2), np.array(self.Y2), np.array(self.S2)
            mean, _ = GP(rng=self.rng).fit(X, y, se).predict(X)
            return X[int(np.argmax(mean))]
        return super().recommend(s)

    def _roi_eval(self, s: Session, b) -> None:
        b = np.clip(np.asarray(b, float), -s.cfg.box, s.cfg.box)
        db = b - s.b0
        d_pred = s.drift.predict(db)
        n = round(self.patch / s.step)
        span = n * s.step
        yy, xx = np.mgrid[:n, :n]
        disk = (yy - (n - 1) / 2) ** 2 + (xx - (n - 1) / 2) ** 2 <= self.disk_px ** 2
        amps, sig, shifts = [], [], []
        for i in self.focus:
            c = s.ref[i] + d_pred
            img = s.bx.measure(g1=b[0], g2=c[0], g3=b[1], g4=c[1], g5=b[2],
                               span_h=span, span_v=span, step_h=s.step, step_v=s.step)
            fr = s.percep.process(img)
            ro = np.where(disk, fr.readout, -np.inf)
            k = np.unravel_index(int(np.argmax(ro)), ro.shape)
            amps.append(float(ro[k]))
            sig.append(fr.sigma)
            if ro[k] > 5.0 * fr.sigma:  # clearly found: its position measures the drift
                shifts.append((np.array([k[1], k[0]]) - (n - 1) / 2) * s.step)
        if shifts:
            s.drift.add(db, d_pred + np.mean(shifts, axis=0))
        self.X2.append(b)
        self.Y2.append(float(np.mean(amps)))
        self.S2.append(float(np.median(sig) / np.sqrt(len(amps))))

    def _lit(self, s: Session) -> bool:
        _, y, se = s.points()
        return bool(len(y) and np.max(y - se) > self.lit_thr)

    def _phase1(self, s: Session) -> bool:
        cap, px = s.bx.pixel_cap, s.bx.n_pixels
        if self.switch == "auto":
            # split derived from both caps: keep full frames while the pixels left after
            # one more frame exceed what the patch phase can spend with the measurements left
            frame = round(s.cfg.span / s.step) ** 2
            patch = round(self.patch / s.step) ** 2
            meas_after = s.bx.meas_cap - s.bx.n_meas - 1 - self.verify_frames
            return s.bx.remaining - frame - self.reserve(s) > meas_after * patch
        if px < self.switch * cap:
            return True
        return self.lit_thr is not None and px < self.max_switch * cap and not self._lit(s)

    def search(self, s: Session) -> None:
        while self.can_afford(s) and self._phase1(s):
            self._step_full(s)
        chosen = self._choose_focus(s)
        if chosen is None:
            while self.can_afford(s):
                self._step_full(s)
            return
        self.focus = chosen[0]
        self._phase2(s, super().recommend(s))

    def _phase2(self, s: Session, b_best) -> None:
        """Local BO on the ROI-patch objective of the focus interdots, around ``b_best``."""
        self.focus_best = b_best
        cost = len(self.focus) * round(self.patch / s.step) ** 2
        n_meas_left = lambda: s.bx.meas_cap - s.bx.n_meas - self.verify_frames  # noqa: E731
        anchors = [b_best] + [b_best + self.rng.normal(0, self.local_sd / 2, 3) for _ in range(3)]
        while (s.bx.remaining >= cost + self.reserve(s) and n_meas_left() >= len(self.focus)):
            if anchors:
                b = anchors.pop(0)
            else:
                X, y, se = np.array(self.X2), np.array(self.Y2), np.array(self.S2)
                gp = GP(ls_prior=self.local_sd, rng=self.rng).fit(X, y, se)
                mean_obs, _ = gp.predict(X)
                top = X[int(np.argmax(mean_obs))]
                cand = np.vstack([top + self.rng.normal(0, self.local_sd, (1500, 3)),
                                  b_best + self.rng.normal(0, 2 * self.local_sd, (1500, 3))])
                cand = np.clip(cand, -s.cfg.box, s.cfg.box)
                m, sd = gp.predict(cand)
                b = cand[int(np.argmax(GP.expected_improvement(m, sd, mean_obs.max())))]
            self._roi_eval(s, b)
            s.set_reco(self.recommend(s))
