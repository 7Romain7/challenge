"""Per-interdot amplitudes -> a scalar contrast score robust to the optimiser's curse."""

from __future__ import annotations

import numpy as np


class Scorer:
    """Keeps the (point x interdot) amplitude matrix and derives the objective.

    For every tracked interdot i we estimate its own *floor* (low quantile of its
    amplitudes over visited points), so that the per-interdot geometric gain (width,
    aliasing, intensity jitter) cancels: excess_ij = (a_ij - floor_i) / floor_i. The
    score of a point is the max excess over its interdots, shrunk by ``z`` noise
    standard deviations against the winner's curse.
    """

    def __init__(self, floor_q: float = 0.3, z: float = 1.0, min_visible: int = 3,
                 min_floor_snr: float = 2.0) -> None:
        self.floor_q, self.z = floor_q, z
        self.min_visible, self.min_floor_snr = min_visible, min_floor_snr
        self.B: list = []
        self.A: list = []
        self.sig: list = []
        self.ok: list = []
        self.frozen: np.ndarray | None = None  # floors fixed by freeze_floors()

    def add(self, b, amps, sigma: float, ok: bool) -> None:
        self.B.append(np.asarray(b, float))
        self.A.append(np.asarray(amps, float))
        self.sig.append(float(sigma))
        self.ok.append(bool(ok))

    def _matrix(self) -> np.ndarray:
        n = max((len(a) for a in self.A), default=0)
        M = np.full((len(self.A), n), np.nan)
        for j, a in enumerate(self.A):
            M[j, : len(a)] = a
        return M

    def freeze_floors(self) -> None:
        """Fix the per-interdot floors from the rows seen so far. Needed before a phase that
        samples a few interdots near their peak: the low quantile would otherwise drift up."""
        self.frozen = self.floors()

    def floors(self) -> np.ndarray:
        if self.frozen is not None:
            n = max((len(a) for a in self.A), default=0)
            sig = np.median(self.sig) if self.sig else 0.0
            extra = np.full(n - len(self.frozen), np.nanmedian(self.frozen))
            return np.maximum(np.r_[self.frozen, extra], self.min_floor_snr * sig)
        M = self._matrix()
        M = np.where(np.array(self.ok)[:, None], M, np.nan)
        n = M.shape[1]
        fl = np.full(n, np.nan)
        for i in range(n):
            col = M[:, i][np.isfinite(M[:, i])]
            if len(col) >= 3:
                fl[i] = np.quantile(col, self.floor_q)
        typical = np.nanmedian(fl) if np.isfinite(fl).any() else np.nanmedian(M)
        fl = np.where(np.isfinite(fl), fl, typical)
        sig = np.median(self.sig) if self.sig else 0.0
        return np.maximum(fl, self.min_floor_snr * sig)

    def scores(self, cols=None, min_visible: int | None = None):
        """(y, se) per recorded point; NaN where tracking failed / too few interdots.

        ``cols`` restricts the max to a subset of interdots (zoom phase)."""
        M, fl = self._matrix(), self.floors()
        if cols is not None:
            keep = np.zeros(M.shape[1], bool)
            keep[np.asarray(cols, int)] = True
            M = np.where(keep[None, :], M, np.nan)
        min_vis = self.min_visible if min_visible is None else min_visible
        y = np.full(len(self.A), np.nan)
        se = np.full(len(self.A), np.nan)
        for j in range(len(self.A)):
            if not self.ok[j]:
                continue
            a = M[j]
            vis = np.isfinite(a)
            if vis.sum() < min_vis:
                continue
            ex = (a[vis] - fl[vis] - self.z * self.sig[j]) / fl[vis]
            k = int(np.argmax(ex))
            y[j] = ex[k]
            se[j] = self.sig[j] / fl[vis][k]
        return y, se
