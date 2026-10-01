"""Classical model-free baselines, on the same tracked front-end as the BO family.

* ``coord_official`` (B1): the organisers' coordinate ascent on full-frame ``img.std()``
  over the five gates, no tracking - the starting point to beat, as shipped.
* ``spsa`` (B3): SPSA gradient estimate (2 evaluations per step, any dimension) + Adam
  on the tracked score - the sound way to "use Adam" on a noisy black box.
* ``cma`` : CMA-ES (Hansen's (mu/mu_w, lambda) with rank-mu and rank-one updates) on
  the tracked score, full frames for the whole budget.
* ``roi_cma``: the ``bo_roi`` pipeline with the phase-2 GP-EI replaced by CMA-ES on the
  ROI-patch objective. Isolates "does the GP help once the region is chosen?".

The recommendation rule is the same as the BO family's (posterior-mean argmax over the
visited points), so the comparison isolates the *search* strategy.
"""

from __future__ import annotations

import numpy as np

from ..blind import BudgetExceeded
from ..session import Session
from .base import Method
from .bo import BayesOpt
from .bo_roi import ROIBayesOpt


class CMAES:
    """Minimal CMA-ES (ask / tell), maximisation."""

    def __init__(self, x0, sigma0: float, rng, lam: int | None = None) -> None:
        self.n = n = len(x0)
        self.rng = rng
        self.lam = lam or 4 + int(3 * np.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.ds = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.asarray(x0, float).copy()
        self.sigma = sigma0
        self.C = np.eye(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)

    def ask(self) -> np.ndarray:
        A = np.linalg.cholesky(self.C + 1e-12 * np.eye(self.n))
        return self.m + self.sigma * self.rng.standard_normal((self.lam, self.n)) @ A.T

    def tell(self, X: np.ndarray, f: np.ndarray) -> None:
        idx = np.argsort(-np.asarray(f))[: self.mu]
        Y = (X[idx] - self.m) / self.sigma
        yw = self.w @ Y
        self.m = self.m + self.sigma * yw
        ev, B = np.linalg.eigh(self.C)
        Cinvsqrt = B @ np.diag(1 / np.sqrt(np.maximum(ev, 1e-20))) @ B.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * Cinvsqrt @ yw
        hs = np.linalg.norm(self.ps) / self.chin < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * np.outer(self.pc, self.pc)
                  + self.cmu * (Y.T * self.w) @ Y)
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chin - 1))


def _score_of(s: Session, b) -> float:
    """Current tracked score of the latest evaluation at ``b`` (floors move: re-read)."""
    B, y, _ = s.points()
    if not len(y):
        return -np.inf
    d = np.linalg.norm(B - np.asarray(b), axis=1)
    j = int(np.argmin(d))
    return float(y[j]) if d[j] < 1e-9 else -np.inf


class CMAESSearch(BayesOpt):
    """CMA-ES on full frames; infeasible proposals are resampled inside the trust region."""

    name = "cma"

    def __init__(self, *a, sigma0: float = 0.15, **kw) -> None:
        super().__init__(*a, **kw)
        self.sigma0 = sigma0

    def _feasible_draw(self, s: Session, es: CMAES) -> np.ndarray:
        for _ in range(50):
            X = np.clip(es.ask(), -s.cfg.box, s.cfg.box)
            ok = [s.feasible(x) for x in X]
            if all(ok):
                return X
        X = np.clip(es.ask(), -s.cfg.box, s.cfg.box)
        for i, x in enumerate(X):  # pull infeasible points back toward the mean
            t = 1.0
            while not s.feasible(x) and t > 0.05:
                t *= 0.6
                x = es.m + t * (X[i] - es.m)
            X[i] = x
        return X

    def search(self, s: Session) -> None:
        es = CMAES(s.b0, self.sigma0, self.rng)
        while self.can_afford(s):
            X = self._feasible_draw(s, es)
            done = []
            for x in X:
                if not self.can_afford(s):
                    break
                s.evaluate(x)
                done.append(x)
                s.set_reco(self.recommend(s))
            if len(done) < len(X):
                return
            es.tell(X, np.array([_score_of(s, x) for x in X]))


class SPSAAdam(BayesOpt):
    """SPSA gradient (Rademacher perturbation, 2 evaluations) + Adam, ascent."""

    name = "spsa"

    def __init__(self, *a, lr: float = 0.06, c: float = 0.06, **kw) -> None:
        super().__init__(*a, **kw)
        self.lr, self.c = lr, c

    def search(self, s: Session) -> None:
        x = s.b0.copy()
        m = np.zeros(3)
        v = np.zeros(3)
        t = 0
        while self.can_afford(s, 2):
            t += 1
            ck = self.c / t ** 0.101
            d = self.rng.choice([-1.0, 1.0], 3)
            xp, xm = np.clip(x + ck * d, -s.cfg.box, s.cfg.box), np.clip(x - ck * d, -s.cfg.box, s.cfg.box)
            s.evaluate(xp)
            s.evaluate(xm)
            fp, fm = _score_of(s, xp), _score_of(s, xm)
            if not (np.isfinite(fp) and np.isfinite(fm)):
                s.set_reco(self.recommend(s))
                continue
            g = (fp - fm) / (2 * ck) * d  # d_i = +-1, so 1/d_i = d_i
            g /= max(1.0, np.linalg.norm(g) / 50.0)  # clip the noisy estimate
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            step = self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            x_new = np.clip(x + step, -s.cfg.box, s.cfg.box)
            if s.feasible(x_new):
                x = x_new
            s.set_reco(self.recommend(s))


class ROICMAES(ROIBayesOpt):
    """``bo_roi`` with CMA-ES instead of GP-EI in the ROI phase."""

    name = "roi_cma"

    def search(self, s: Session) -> None:
        cap = s.bx.pixel_cap
        while self.can_afford(s) and s.bx.n_pixels < self.switch * cap:
            self._step_full(s)
        chosen = self._choose_focus(s)
        if chosen is None:
            while self.can_afford(s):
                self._step_full(s)
            return
        self.focus = chosen[0]
        b_best = super(ROIBayesOpt, self).recommend(s)
        cost = len(self.focus) * round(self.patch / s.step) ** 2
        left = lambda: s.bx.meas_cap - s.bx.n_meas - self.verify_frames  # noqa: E731
        es = CMAES(b_best, self.local_sd, self.rng)
        while True:
            X = np.clip(es.ask(), -s.cfg.box, s.cfg.box)
            f = []
            for x in X:
                if not (s.bx.remaining >= cost + self.reserve(s) and left() >= len(self.focus)):
                    return
                self._roi_eval(s, x)
                f.append(self.Y2[-1])
                s.set_reco(self.recommend(s))
            es.tell(X, np.array(f))


GATES = ("g1", "g2", "g3", "g4", "g5")


class OfficialCoordAscent(Method):
    """B1: the organisers' starter (coordinate ascent on full-frame std), budget-capped."""

    name = "coord_official"

    def run(self, bx) -> dict:
        log: list[dict] = []
        img = bx.reference_frame()
        span, step = bx.native_span, bx.native_step
        point = dict(bx.start)
        log.append({"pixels": bx.n_pixels, "wp": dict(point), "span": span})

        def score(p):
            return float(np.mean([bx.measure(**p, span_h=span, span_v=span,
                                             step_h=step, step_v=step).std() for _ in range(3)]))

        try:
            best = float(np.mean([img.std()] + [bx.measure(**point, span_h=span, span_v=span,
                                                           step_h=step, step_v=step).std()
                                                for _ in range(2)]))
            st = 0.04
            for _ in range(6):
                for g in GATES:
                    for sgn in (+1, -1):
                        trial = dict(point)
                        trial[g] += sgn * st
                        sc = score(trial)
                        if sc > best:
                            best, point = sc, trial
                            log.append({"pixels": bx.n_pixels, "wp": dict(point), "span": span})
                st *= 0.6
        except BudgetExceeded:
            pass
        log.append({"pixels": bx.n_pixels, "wp": dict(point), "span": span})
        bx.commit(point)
        return {"b_hat": [point["g1"], point["g3"], point["g5"]], "reco_log": log,
                "n_track_fail": 0, "n_pixels": bx.n_pixels, "n_meas": bx.n_meas}
