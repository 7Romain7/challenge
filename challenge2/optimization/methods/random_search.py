"""B2 - random search inside the feasible (trackable) region. Strong baseline (fact 6)."""

from __future__ import annotations

from ..session import Session
from .base import Method


class RandomSearch(Method):
    name = "random"

    def search(self, s: Session) -> None:
        while self.can_afford(s):
            b = s.sample_feasible(self.rng)[0]
            s.evaluate(b)
            s.set_reco(self.recommend(s))
