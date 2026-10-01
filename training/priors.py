"""Synthetic contrast landscapes for pre-training PFN / RL (methods 2-3).

Deliberately BROADER than the simulator (PROTOCOL P15): a random number of bumps with
mixed shape families and widths, random floor offsets and heteroscedastic noise. The
simulator is then a held-out "real device" the learned method never saw. Nothing here
reads csd.config.

A landscape is y(b) = max_k  A_k * shape_k(b), b in [-box, box]^3 (excess over the floor,
in floor units, i.e. the quantity optimisation.scoring.Scorer produces).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHAPES = ("lorentz", "gauss", "expo")


@dataclass
class Landscape:
    centres: np.ndarray  # (K, 3)
    widths: np.ndarray  # (K, 3)
    amps: np.ndarray  # (K,)
    shapes: list
    noise_se: float
    box: float

    def f(self, B: np.ndarray) -> np.ndarray:
        B = np.atleast_2d(B)
        out = np.zeros((len(B), len(self.amps)))
        for k, (c, w, a, sh) in enumerate(zip(self.centres, self.widths, self.amps, self.shapes)):
            z = ((B - c) / w) ** 2
            if sh == "lorentz":
                v = np.prod(1.0 / (1.0 + z), axis=1)
            elif sh == "gauss":
                v = np.exp(-0.5 * z.sum(axis=1))
            else:
                v = np.exp(-np.sqrt(z.sum(axis=1)))
            out[:, k] = a * v
        return out.max(axis=1)

    def observe(self, B: np.ndarray, rng: np.random.Generator):
        """Noisy score and its standard error, as the front-end would report them."""
        y = self.f(B)
        se = self.noise_se * np.exp(rng.normal(0, 0.2, len(y)))
        return y + rng.normal(0, 1, len(y)) * se, se


def sample_landscape(rng: np.random.Generator, box: float = 0.6) -> Landscape:
    K = int(rng.integers(1, 7))
    return Landscape(
        centres=rng.uniform(-1.15 * box, 1.15 * box, (K, 3)),
        widths=np.exp(rng.uniform(np.log(0.03), np.log(0.4), (K, 3))),
        amps=np.exp(rng.uniform(np.log(1.0), np.log(12.0), K)),
        shapes=[SHAPES[i] for i in rng.integers(0, len(SHAPES), K)],
        noise_se=float(np.exp(rng.uniform(np.log(0.05), np.log(0.5)))),
        box=box,
    )
