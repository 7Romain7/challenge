"""Common interface of every method (baselines, BO, learned policies)."""

from __future__ import annotations

import numpy as np

from ..blind import BlindExperiment, BudgetExceeded
from ..session import Session, SessionConfig


class Method:
    """A method decides *where to measure next* and *what to recommend*.

    Subclasses implement :meth:`search`, which repeatedly calls ``session.evaluate`` and
    keeps ``session.set_reco`` up to date (the evaluator scores the logged
    recommendations, never the visited points). The base ``run`` handles the budget:
    a reserve is kept for the final verification frames.
    """

    name = "base"

    def __init__(self, run_seed: int = 0, session_cfg: SessionConfig | None = None,
                 verify_frames: int = 2) -> None:
        self.rng = np.random.default_rng(run_seed)  # algorithm RNG - never np.random.seed
        self.session_cfg = session_cfg or SessionConfig()
        self.verify_frames = verify_frames

    # -- to override --------------------------------------------------------
    def search(self, s: Session) -> None:
        raise NotImplementedError

    def recommend(self, s: Session) -> np.ndarray:
        """Best barriers according to the method's own (smoothed) belief."""
        B, y, _ = s.points()
        return B[int(np.argmax(y))] if len(y) else s.b0

    # -- driver --------------------------------------------------------------
    def run(self, bx: BlindExperiment) -> dict:
        s = Session(bx, self.session_cfg)
        try:
            s.start()
            s.set_reco(s.b0)
            self.search(s)
        except BudgetExceeded:
            pass
        # final: verify the recommendation with fresh frames if the budget allows
        b_hat = self.recommend(s)
        try:
            cost = round(s.cfg.span / s.step) ** 2 if s.step else 0
            if cost and bx.remaining >= self.verify_frames * cost:
                s.verify(b_hat, self.verify_frames)
                b_hat = self.recommend(s)
        except BudgetExceeded:
            pass
        s.set_reco(b_hat)
        bx.commit(s.wp_for(b_hat))
        return {"b_hat": [float(v) for v in b_hat], "reco_log": s.reco_log,
                "n_track_fail": s.n_track_fail, "n_pixels": bx.n_pixels, "n_meas": bx.n_meas}

    def reserve(self, s: Session) -> int:
        """Pixels to keep for the final verification."""
        return self.verify_frames * round(s.cfg.span / s.step) ** 2

    def can_afford(self, s: Session, n_frames: int = 1) -> bool:
        cost = round(s.cfg.span / s.step) ** 2
        return s.bx.remaining >= (n_frames * cost + self.reserve(s))
