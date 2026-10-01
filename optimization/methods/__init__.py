"""Optimisation methods. All share the :class:`Method` interface and the same Session."""
from .base import Method
from .bo import BayesOpt
from .random_search import RandomSearch

REGISTRY = {"random": RandomSearch, "bo": BayesOpt}

__all__ = ["Method", "RandomSearch", "BayesOpt", "REGISTRY"]
