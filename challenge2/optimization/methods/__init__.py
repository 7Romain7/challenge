"""Optimisation methods. All share the :class:`Method` interface and the same Session."""
from functools import partial

from ..session import SessionConfig
from .base import Method
from .bo import BayesOpt
from .random_search import RandomSearch

_M5 = SessionConfig(detector="m5_min")  # front-end with the frozen challenge-1 detector

REGISTRY = {"random": RandomSearch, "bo": BayesOpt,
            "random_m5": partial(RandomSearch, session_cfg=_M5),
            "bo_m5": partial(BayesOpt, session_cfg=_M5),
            # operating points fixed by the false-alarm rate on stick-free scenes (challenge2/transfer_m5/)
            "bo_m5_fa5": partial(BayesOpt, session_cfg=SessionConfig(detector="m5_min", m5_thr=-2.0)),
            "bo_m5_fa20": partial(BayesOpt, session_cfg=SessionConfig(detector="m5_min", m5_thr=-2.5))}

__all__ = ["Method", "RandomSearch", "BayesOpt", "REGISTRY"]
