"""Diagnostics of the perception/tracking front-end against ground truth.

    uv run python -m evaluation.diagnose_tracking --n 20 --reach 0.4

Reports: detection recall/precision on the reference frame, drift-prediction error (before
registration), registration error (after), tracking failure rate, and the Spearman
correlation between the score y and the true region factor of the *best visible stick*.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import spearmanr

from csd import new_experiment
from optimization import BlindExperiment
from optimization.session import Session

from .truth import true_factor


def run(seed: int, reach: float, n_pts: int, cap: int):
    exp = new_experiment(seed=seed)
    sim = exp._sim
    bx = BlindExperiment(exp, pixel_cap=cap)
    s = Session(bx)
    s.start()
    left, _, bottom, _ = exp.extent
    truth_pos = np.array([m.middle for m in sim.base_interdots]) + np.array([left, bottom])
    d = np.linalg.norm(s.ref[:, None] - truth_pos[None], axis=2)
    recall = float((d.min(axis=0) < 4e-3).mean())
    precision = float((d.min(axis=1) < 4e-3).mean())
    rng = np.random.default_rng(seed)
    pred_err, reg_err, fails, ys, fs, sd, nb = [], [], 0, [], [], [], []
    for _ in range(n_pts):
        b = s.sample_feasible(rng)[0]
        sd.append(s.drift.predict_std(b - s.b0))
        db = b - s.b0
        nb.append(np.linalg.norm(db))
        true_d = sim.drift_matrix.drift(db)
        pred = s.drift.predict(db)
        info = s.evaluate(b)
        if info["ok"]:
            pred_err.append(np.linalg.norm(pred - true_d))
            reg_err.append(np.linalg.norm(s.drift.delta[-1] - true_d))
        else:
            fails += 1
        y, _ = s.scorer.scores()
        ys.append(y[-1])
        fs.append(true_factor(exp, {"g1": b[0], "g2": 0, "g3": b[1], "g4": 0, "g5": b[2]}))
    probe_err = [np.linalg.norm(dd - sim.drift_matrix.drift(db))
                 for db, dd in zip(s.drift.db[:4], s.drift.delta[:4])]
    return dict(recall=recall, precision=precision, pred=np.array(pred_err),
                reg=np.array(reg_err), fails=fails, n=n_pts, y=np.array(ys), f=np.array(fs),
                n_ref=len(s.ref), n_true=len(truth_pos), sd=np.array(sd), nb=np.array(nb), px=bx.n_pixels, probe_err=np.array(probe_err))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--reach", type=float, default=0.4)
    ap.add_argument("--pts", type=int, default=12)
    a = ap.parse_args()
    R = [run(s, a.reach, a.pts, cap=10_000_000) for s in range(a.n)]
    pr = np.concatenate([r["pred"] for r in R])
    rg = np.concatenate([r["reg"] for r in R])
    pe = np.concatenate([r["probe_err"] for r in R])
    print(f"ref detection  recall {np.mean([r['recall'] for r in R]):.2f}  "
          f"precision {np.mean([r['precision'] for r in R]):.2f}")
    print(f"probe registration error [mV]: median {np.median(pe)*1e3:.1f}  p90 {np.percentile(pe, 90)*1e3:.1f}")
    print(f"drift PREDICTION error [mV]: median {np.median(pr)*1e3:.1f}  p90 {np.percentile(pr, 90)*1e3:.1f}")
    print(f"registration error     [mV]: median {np.median(rg)*1e3:.1f}  p90 {np.percentile(rg, 90)*1e3:.1f}")
    print(f"mean |db| of sampled points {np.mean(np.concatenate([r['nb'] for r in R])):.2f} V ; px/device {np.mean([r['px'] for r in R]):.0f}")
    print(f"tracking failures: {sum(r['fails'] for r in R)}/{sum(r['n'] for r in R)}")
    rho = [spearmanr(r["y"][np.isfinite(r["y"])], r["f"][np.isfinite(r["y"])])[0] for r in R
           if np.isfinite(r["y"]).sum() > 4]
    print(f"Spearman(y, true factor) per device: median {np.median(rho):.2f}  min {np.min(rho):.2f}")


if __name__ == "__main__":
    main()
