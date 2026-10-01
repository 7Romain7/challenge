"""Drift model (online lever-arm calibration) and stick-set registration."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


class DriftModel:
    """delta(db) = phi(db) @ W with phi = [db, db_i*db_j (i<=j)] (2nd-order Taylor).

    Bayesian ridge: prior std ``tau_lin`` (V/V) on lever arms and ``tau_quad`` (V/V^2)
    on curvature, observation noise ``sigma_n`` (registration accuracy, a few mV). The
    posterior also gives a predictive std, used to size the registration search radius.
    ``tau_*`` are generic dimensional-analysis scales, tuned on dev seeds only.
    """

    def __init__(self, sigma_n: float = 2e-3, tau_lin: float = 2.0, tau_quad: float = 1.0) -> None:
        self.sigma_n = sigma_n
        self.pen = (sigma_n ** 2) * np.r_[np.full(3, tau_lin ** -2), np.full(6, tau_quad ** -2)]
        self.db: list = []
        self.delta: list = []
        self.W = np.zeros((9, 2))
        self._Ainv = np.linalg.inv(np.diag(self.pen))

    @staticmethod
    def _phi(db: np.ndarray) -> np.ndarray:
        db = np.asarray(db, float)
        q = [db[i] * db[j] for i in range(3) for j in range(i, 3)]
        return np.concatenate([db, q])

    def add(self, db, delta) -> None:
        self.db.append(np.asarray(db, float))
        self.delta.append(np.asarray(delta, float))
        X = np.array([self._phi(d) for d in self.db])
        A = X.T @ X + np.diag(self.pen)
        self._Ainv = np.linalg.inv(A)
        self.W = self._Ainv @ X.T @ np.array(self.delta)

    def predict(self, db) -> np.ndarray:
        return self._phi(db) @ self.W

    def predict_std(self, db) -> float:
        """Predictive std (V) of one drift component at ``db``."""
        p = self._phi(db)
        return float(self.sigma_n * np.sqrt(1.0 + p @ self._Ainv @ p))


def register(ref: np.ndarray, det: np.ndarray, pred: np.ndarray, radius: float,
             bin_: float = 2e-3, tol: float = 5e-3):
    """Translation of the reference stick set (abs volts) explaining detections.

    Hough-style voting on all pairwise offsets within ``radius`` of ``pred``, then a
    nearest-neighbour refinement. Returns (delta, n_matches); n_matches == 0 on failure.
    Only *positions* are used, so changes in brightness (the thing we optimise) do
    not disturb registration.
    """
    if len(ref) == 0 or len(det) == 0:
        return pred, 0
    off = (det[None, :, :] - ref[:, None, :] - pred).reshape(-1, 2)
    off = off[np.linalg.norm(off, axis=1) <= radius]
    if len(off) == 0:
        return pred, 0
    n = int(np.ceil(2 * radius / bin_)) + 1
    idx = np.clip(((off + radius) / bin_).round().astype(int), 0, n - 1)
    H = np.zeros((n, n))
    np.add.at(H, (idx[:, 1], idx[:, 0]), 1.0)
    H = ndi.gaussian_filter(H, 1.2)
    iy, ix = np.unravel_index(np.argmax(H), H.shape)
    coarse = pred + np.array([ix, iy]) * bin_ - radius
    shifted = ref + coarse
    d = np.linalg.norm(shifted[:, None, :] - det[None, :, :], axis=2)
    j = d.argmin(axis=1)
    ok = d[np.arange(len(ref)), j] <= tol
    if ok.sum() == 0:
        return pred, 0
    delta = coarse + np.median(det[j[ok]] - shifted[ok], axis=0)
    return delta, int(ok.sum())
