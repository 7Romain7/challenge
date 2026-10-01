import os
import numpy as np
from optimization.session import SessionConfig
from csd import new_experiment
from optimization import BlindExperiment
from optimization.methods.bo_roi import ROIBayesOpt
from evaluation import truth
import sys
for seed in map(int, sys.argv[1:]):
    exp = new_experiment(seed=seed); base, fs = truth.base_and_best(exp)
    m = ROIBayesOpt(run_seed=0, session_cfg=SessionConfig(detector="dl", dl_ckpt=os.environ["C12_DL_CKPT"], dl_thr=0.5))
    keep = {}
    orig = m._choose_focus
    def cf(s, orig=orig):
        keep["s"] = s
        return orig(s)
    m._choose_focus = cf
    bx = BlindExperiment(exp, pixel_cap=1_000_000)
    res = m.run(bx)
    s = keep["s"]
    B, y, se = s.points()
    f = np.array([truth.true_factor(exp, {"g1": b[0], "g2": 0, "g3": b[1], "g4": 0, "g5": b[2]}) for b in B])
    o = np.argsort(-y)[:5]
    print(f"seed {seed} f*={fs:.1f} base={base} n_ref={len(s.ref)} | top-y points: y, se, true f")
    for j in o:
        print(f"   y={y[j]:6.2f} se={se[j]:.2f} f={f[j]:6.2f}")
    print("   spearman(y,f) =", round(float(np.corrcoef(np.argsort(np.argsort(y)), np.argsort(np.argsort(f)))[0,1]),2),
          " max f visited =", round(f.max(),2), " focus:", m.__dict__.get("focus"))
    reg = lambda wp: (fs - truth.true_factor(exp, wp)) / (fs - base)
    b1 = super(ROIBayesOpt, m).recommend(s)
    print("   R(phase-1 reco) =", round(reg({"g1": b1[0], "g2": 0, "g3": b1[1], "g4": 0, "g5": b1[2]}), 2),
          " R(final reco) =", round(reg(s.reco_log[-1]["wp"]), 2),
          " phase-2 evals:", len(m.Y2), " Y2 range:", round(min(m.Y2), 2), round(max(m.Y2), 2))
    # phase-2 objective vs truth
    f2 = np.array([truth.true_factor(exp, {"g1": b[0], "g2": 0, "g3": b[1], "g4": 0, "g5": b[2]}) for b in m.X2])
    print("   phase-2: corr(Y2, f) =", round(float(np.corrcoef(m.Y2, f2)[0, 1]), 2), " max f visited in phase 2 =", round(f2.max(), 2),
          " true f at focus anchor =", round(f2[0], 2))
    sim = exp._sim
    live = sorted(set(int(k) for k in sim._region_of))
    best = max(live, key=lambda k: sim.contrast_model.regions[k].amplitude)
    c = np.array(sim.contrast_model.regions[best].center)
    d_vis = np.linalg.norm(B - s.b0, axis=1)
    feas_c = s.feasible(c)
    gam = np.array(sim.contrast_model.regions[best].gamma)
    print(f"   best-region centre {np.round(c, 2)} (|c-b0|={np.linalg.norm(c - s.b0):.2f}, gamma {np.round(gam, 3)}),"
          f" feasible now: {feas_c}; visited |b-b0| max {d_vis.max():.2f};"
          f" closest visit to centre {np.min(np.linalg.norm(B - c, axis=1)):.2f}; n full-frame pts {len(B)};"
          f" frac feasible box {np.mean([s.feasible(x) for x in np.random.default_rng(0).uniform(-.6, .6, (400, 3))]):.2f}")
