"""Entry point of challenge 2: ``optimize(exp)`` returns the gate voltages for a fresh device.

    from csd import new_experiment
    from solve import optimize

    exp = new_experiment()
    gates = optimize(exp)          # {"g1": ..., "g2": ..., "g3": ..., "g4": ..., "g5": ...}

The method is ``bo_roi_auto_ucb`` (see README): tracked Bayesian optimisation with GP-UCB
on full frames, then ROI patches, phase split computed from the two caps. The
challenge-1 U-Net is used for perception when ``$C12_DL_CKPT`` points to a checkpoint
(GPU extra ``uv sync --extra train``); otherwise the training-free matched filter is used,
which gives the same regret on the benchmark (README, section U-Net).
"""

from __future__ import annotations

import os

from optimization import BlindExperiment
from optimization.methods.bo_roi import ROIBayesOpt
from optimization.session import SessionConfig


def optimize(exp, pixel_budget: int = 1_000_000, meas_budget: int = 300, seed: int = 0) -> dict:
    """Tune a fresh device within the budget; never reads the hidden state of ``exp``."""
    ckpt = os.environ.get("C12_DL_CKPT", "")
    cfg = (SessionConfig(detector="dl", dl_ckpt=ckpt, dl_thr=float(os.environ.get("C12_DL_THR", "0.5")))
           if ckpt else SessionConfig())
    bx = BlindExperiment(exp, pixel_cap=pixel_budget, meas_cap=meas_budget)
    ROIBayesOpt(run_seed=seed, switch="auto", acq="ucb", session_cfg=cfg).run(bx)
    return dict(bx.events[-1].gates)  # the committed recommendation


if __name__ == "__main__":
    from csd import new_experiment

    exp = new_experiment(seed=12345)
    gates = optimize(exp)
    truth = exp.reveal()  # only to print how good the answer is
    print("gates      :", {k: round(v, 3) for k, v in gates.items()})
    print("contrast   :", round(exp._sim.contrast(gates), 2), "/ best", round(truth["max_contrast_factor"], 2))
    print("budget used:", truth["n_pixels"], "px,", truth["n_measurements"], "measurements")
