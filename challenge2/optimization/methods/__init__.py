"""Optimisation methods. All share the :class:`Method` interface and the same Session."""
import os
from functools import partial

from ..session import SessionConfig
from .base import Method
from .bo import BayesOpt
from .bo_mf import MultiFidelityBO
from .bo_roi import ROIBayesOpt

_DL = dict(detector="dl", dl_ckpt=os.environ.get("C12_DL_CKPT", ""),
           dl_thr=float(os.environ.get("C12_DL_THR", "0.1")))  # threshold from the run's val
from .pfn import PFNBayesOpt
from .random_search import RandomSearch

_M5 = SessionConfig(detector="m5_min")  # front-end with the frozen challenge-1 detector

REGISTRY = {"random": RandomSearch, "bo": BayesOpt,
            # multi-fidelity BO; switch = budget share of the full-frame phase (dev sensitivity)
            "bo_mf": MultiFidelityBO,
            "bo_roi": ROIBayesOpt,  # active-imaging ROI patches on the focus interdots
            # reference architecture with the final challenge-1 network ($C12_DL_CKPT / $C12_DL_THR)
            "bo_dlf": partial(BayesOpt, session_cfg=SessionConfig(**_DL)),
            "bo_roi_dlf": partial(ROIBayesOpt, session_cfg=SessionConfig(**_DL)),
            "bo_pfn": PFNBayesOpt,  # GP swapped for the synthetic-only PFN ($C12_PFN_CKPT)
            "bo_mf_s40": partial(MultiFidelityBO, switch=0.4),
            "bo_mf_s80": partial(MultiFidelityBO, switch=0.8),
            "random_m5": partial(RandomSearch, session_cfg=_M5),
            "bo_m5": partial(BayesOpt, session_cfg=_M5),
            # operating points fixed by the false-alarm rate on stick-free scenes (challenge2/transfer_m5/)
            "bo_m5_fa5": partial(BayesOpt, session_cfg=SessionConfig(detector="m5_min", m5_thr=-2.0)),
            "bo_m5_fa20": partial(BayesOpt, session_cfg=SessionConfig(detector="m5_min", m5_thr=-2.5)),
            # frozen challenge-1 network; checkpoint path from $C12_DL_CKPT (GPU extra)
            "bo_dl": partial(BayesOpt, session_cfg=SessionConfig(
                detector="dl", dl_ckpt=os.environ.get("C12_DL_CKPT", ""), dl_thr=0.1)),
            "bo_dl_lo": partial(BayesOpt, session_cfg=SessionConfig(
                detector="dl", dl_ckpt=os.environ.get("C12_DL_CKPT", ""), dl_thr=0.03))}

__all__ = ["Method", "RandomSearch", "BayesOpt", "REGISTRY"]
