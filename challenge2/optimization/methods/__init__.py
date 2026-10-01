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
from .meta_acq import MetaAcqROI
from .bo_region import RegionBO
from .bo_coarse import CoarseExploreROI
from .classic import CMAESSearch, OfficialCoordAscent, ROICMAES, SPSAAdam

_M5 = SessionConfig(detector="m5_min")  # front-end with the frozen challenge-1 detector

REGISTRY = {"random": RandomSearch, "bo": BayesOpt,
            # multi-fidelity BO; switch = budget share of the full-frame phase (dev sensitivity)
            "bo_mf": MultiFidelityBO,
            "bo_roi": ROIBayesOpt,  # active-imaging ROI patches on the focus interdots
            # reference architecture with the final challenge-1 network ($C12_DL_CKPT / $C12_DL_THR)
            "bo_dlf": partial(BayesOpt, session_cfg=SessionConfig(**_DL)),
            "bo_roi_dlf": partial(ROIBayesOpt, session_cfg=SessionConfig(**_DL)),
            # switch to ROI patches only once a point is lit (threshold fixed a priori)
            # phase split computed from the pixel and measurement caps (no tuned share)
            "bo_roi_auto_dlf": partial(ROIBayesOpt, switch="auto", session_cfg=SessionConfig(**_DL)),
            # exploration variants on top of the computed split (parameters fixed a priori)
            "bo_roi_auto_ucb_dlf": partial(ROIBayesOpt, switch="auto", acq="ucb",
                                           session_cfg=SessionConfig(**_DL)),
            # half of the ~35 phase-1 frames space-filling (maximin) before EI
            "bo_roi_auto_fill_dlf": partial(ROIBayesOpt, switch="auto", n_init=14,
                                            session_cfg=SessionConfig(**_DL)),
            # one GP per spatial group of interdots (k-means, k = 6), computed split
            "bo_region_dlf": partial(RegionBO, session_cfg=SessionConfig(**_DL)),
            "bo_region": RegionBO,
            # coarse (2x step) survey + native confirmation frames, then ROI patches
            "bo_coarse_dlf": partial(CoarseExploreROI, session_cfg=SessionConfig(**_DL)),
            "bo_coarse": CoarseExploreROI,
            "bo_region_ucb_dlf": partial(RegionBO, acq="ucb", session_cfg=SessionConfig(**_DL)),
            "bo_coarse_ucb_dlf": partial(CoarseExploreROI, acq="ucb", session_cfg=SessionConfig(**_DL)),
            "bo_roi_lit_dlf": partial(ROIBayesOpt, lit_thr=1.0, session_cfg=SessionConfig(**_DL)),
            "bo_pfn": PFNBayesOpt,
            # classical model-free baselines (optimization/methods/classic.py)
            "coord_official": OfficialCoordAscent,
            "cma_dlf": partial(CMAESSearch, session_cfg=SessionConfig(**_DL)),
            "spsa_dlf": partial(SPSAAdam, session_cfg=SessionConfig(**_DL)),
            "roi_cma_dlf": partial(ROICMAES, session_cfg=SessionConfig(**_DL)),
            # meta-learned acquisition in phase 1 of bo_roi ($C12_META_W; EI if unset)
            "bo_roi_meta_dlf": partial(MetaAcqROI, session_cfg=SessionConfig(**_DL)),  # GP swapped for the synthetic-only PFN ($C12_PFN_CKPT)
            "bo_mf_s40": partial(MultiFidelityBO, switch=0.4),
            "bo_mf_s80": partial(MultiFidelityBO, switch=0.8),
            "random_m5": partial(RandomSearch, session_cfg=_M5),
            "random_dlf": partial(RandomSearch, session_cfg=SessionConfig(**_DL)),
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
