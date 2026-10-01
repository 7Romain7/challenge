"""Run methods on many devices and score them with the hidden truth.

    uv run python -m evaluation.run --methods random bo --split dev --n 20 \
        --budget 1000000 --workers 6 --out challenge2/results/pilot.jsonl

One device per process (the simulator draws noise from the global NumPy RNG, so devices
must never be interleaved in one process). Metrics are computed on the algorithm's
*logged recommendations*, at fixed pixel checkpoints (anytime curve).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import time

import numpy as np

from csd import new_experiment
from optimization import BlindExperiment
from optimization.methods import REGISTRY

from . import truth
from .seeds import SPLITS

CHECKPOINTS = (100_000, 250_000, 500_000, 1_000_000, 2_000_000)


def regret(f_star: float, base: float, f: float) -> float:
    return float((f_star - f) / max(f_star - base, 1e-9))


def run_one(task: tuple) -> dict:
    method, seed, budget, run_seed = task
    t0 = time.time()
    exp = new_experiment(seed=seed)
    base, f_star = truth.base_and_best(exp)
    desc = truth.device_descriptors(exp)
    out = {"method": method, "seed": seed, "run_seed": run_seed, "budget": budget,
           "base": base, "f_star": f_star, **desc}
    start_wp = dict(exp.start)
    out["R_start"] = regret(f_star, base, truth.true_factor(exp, start_wp))
    bx = BlindExperiment(exp, pixel_cap=budget)
    try:
        res = REGISTRY[method](run_seed=run_seed).run(bx)
    except Exception as e:  # keep the sweep alive, record the failure
        out.update(error=repr(e), n_pixels=bx.n_pixels)
        return out
    log = res["reco_log"]
    out.update(n_pixels=res["n_pixels"], n_meas=res["n_meas"], n_track_fail=res["n_track_fail"])
    # anytime curve: last recommendation logged at or before each checkpoint
    curve = {}
    for ck in CHECKPOINTS:
        if ck > budget:
            continue
        past = [r for r in log if r["pixels"] <= ck]
        if not past:
            curve[ck] = out["R_start"]
            continue
        wp = past[-1]["wp"]
        curve[ck] = regret(f_star, base, truth.true_factor(exp, wp))
    final = log[-1]
    out["R_curve"] = {str(k): v for k, v in curve.items()}
    out["R_final"] = regret(f_star, base, truth.true_factor(exp, final["wp"]))
    out["R_vis_final"] = regret(f_star, base, truth.visible_factor(exp, final["wp"], final["span"]))
    # gate path length over the barrier moves (hardware-safety proxy)
    pts = [(e.gates["g1"], e.gates["g3"], e.gates["g5"]) for e in bx.events
           if e.kind == "measure" and e.gates.get("g1") is not None]
    out["path_len"] = float(np.sum(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1))) if len(pts) > 1 else 0.0
    out.update(truth.region_diagnosis(exp, final["wp"], pts))
    out["secs"] = time.time() - t0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["random", "bo"])
    ap.add_argument("--split", default="dev", choices=list(SPLITS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start", type=int, default=0, help="offset inside the split")
    ap.add_argument("--budget", type=int, default=1_000_000)
    ap.add_argument("--run-seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="challenge2/results/run.jsonl")
    a = ap.parse_args()
    seeds = list(SPLITS[a.split])[a.start : a.start + a.n]
    tasks = [(m, s, a.budget, rs) for m in a.methods for s in seeds for rs in range(a.run_seeds)]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, "w") as f:
        for r in pool.imap_unordered(run_one, tasks):
            f.write(json.dumps(r) + "\n")
            f.flush()
    summarize(a.out)


def summarize(path: str) -> None:
    rows = [json.loads(line) for line in open(path)]
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    print(f"{len(ok)} runs ok, {len(errs)} errors")
    for r in errs[:3]:
        print("  error:", r["method"], r["seed"], r["error"])
    for m in sorted({r["method"] for r in ok}):
        rr = [r for r in ok if r["method"] == m]
        line = f"{m:8s} n={len(rr):3d}"
        for ck in CHECKPOINTS:
            v = [r["R_curve"][str(ck)] for r in rr if str(ck) in r["R_curve"]]
            if v:
                line += f" | R@{ck // 1000}k med {np.median(v):.3f} (succ<=.05: {np.mean(np.array(v) <= .05):.2f})"
        if "region_ok" in rr[0]:
            line += (f" | region ok {np.mean([r['region_ok'] for r in rr]):.2f}"
                     f" best seen {np.mean([r['best_visited'] for r in rr]):.2f}"
                     f" R_region med {np.median([r['R_region'] for r in rr]):.3f}")
        line += f" | Rvis {np.median([r['R_vis_final'] for r in rr]):.3f} | fail {np.mean([r['n_track_fail'] for r in rr]):.1f}"
        print(line)
    print("start R median:", np.median([r["R_start"] for r in ok]))


if __name__ == "__main__":
    main()
