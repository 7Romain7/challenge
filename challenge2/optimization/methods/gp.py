"""Small exact GP (Matern-5/2 ARD, heteroscedastic noise) in numpy/scipy.

n <= ~100 points: CPU is plenty. The BoTorch/GPU variant can replace this class unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm


def matern52(X1, X2, ls, var):
    d = cdist(X1 / ls, X2 / ls)
    s5 = np.sqrt(5.0) * d
    return var * (1.0 + s5 + (5.0 / 3.0) * d * d) * np.exp(-s5)


class GP:
    def __init__(self, ls_prior: float = 0.2, rng=None) -> None:
        self.ls_prior = ls_prior
        self.rng = rng or np.random.default_rng(0)

    def _unpack(self, th):
        return np.exp(th[:3]), np.exp(th[3]), np.exp(th[4])

    def _nll(self, th, X, y, nv):
        ls, var, extra = self._unpack(th)
        K = matern52(X, X, ls, var) + np.diag(nv + extra)
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve(c, y)
        nll = 0.5 * y @ a + np.log(np.diag(c[0])).sum()
        nll += 0.5 * np.sum(((th[:3] - np.log(self.ls_prior)) / 0.7) ** 2)  # weak prior on ls
        nll += 0.5 * ((th[3]) / 1.5) ** 2 + 0.5 * ((th[4] - np.log(0.05)) / 1.5) ** 2
        return nll

    def fit(self, X, y, se):
        self.X = np.asarray(X, float)
        self.mu, self.sd = float(np.mean(y)), float(np.std(y) + 1e-9)
        self.y = (np.asarray(y, float) - self.mu) / self.sd
        self.nv = (np.asarray(se, float) / self.sd) ** 2
        best = None
        for _ in range(4):
            th0 = np.r_[np.log(self.ls_prior) + self.rng.normal(0, 0.4, 3),
                        self.rng.normal(0, 0.5), np.log(0.05) + self.rng.normal(0, 0.5)]
            r = minimize(self._nll, th0, args=(self.X, self.y, self.nv), method="L-BFGS-B",
                         bounds=[(np.log(0.04), np.log(1.5))] * 3
                         + [(np.log(0.05), np.log(10))] + [(np.log(1e-4), np.log(1.0))])
            if best is None or r.fun < best.fun:
                best = r
        self.th = best.x
        ls, var, extra = self._unpack(self.th)
        K = matern52(self.X, self.X, ls, var) + np.diag(self.nv + extra)
        self.c = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.c, self.y)
        return self

    def predict(self, Xs):
        ls, var, _ = self._unpack(self.th)
        Ks = matern52(np.asarray(Xs, float), self.X, ls, var)
        mean = Ks @ self.alpha
        v = cho_solve(self.c, Ks.T)
        sd = np.sqrt(np.maximum(var - np.sum(Ks * v.T, axis=1), 1e-12))
        return mean * self.sd + self.mu, sd * self.sd

    @staticmethod
    def expected_improvement(mean, sd, best):
        z = (mean - best) / np.maximum(sd, 1e-12)
        return sd * (z * norm.cdf(z) + norm.pdf(z))
