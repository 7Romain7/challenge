"""Train the meta-learned acquisition of ``optimization.methods.meta_acq`` with CMA-ES.

    python -m training.train_meta_acq --hours 2 --workers 12 --out ckpt/meta_acq

Episodes run the full-frame phase of ``bo_roi`` only (``stop_after_phase1``) on training
devices (dev 100-999; dev 0-99 stays for design comparisons, val/test untouched). The
loss is about the *region* chosen at the end of that phase, the measured failure mode:
    loss = R_region(b_phase1) + 0.25 * R(b_phase1)
Each generation draws a fresh batch of devices shared by the whole population (common
random numbers) and also scores the current mean and the EI initialisation on it, so the
log shows the gain over EI on the same devices. Training uses the matched-filter
perception (CPU); the GP statistics the policy sees do not depend on it much.
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
from optimization.methods.classic import CMAES
from optimization.methods.meta_acq import N_FEAT, MetaAcqROI, Phase1Done, init_weights

TRAIN = range(100, 1000)


def episode(task) -> float:
    w, seed = task
    exp = new_experiment(seed=seed)
    bx = BlindExperiment(exp, pixel_cap=1_000_000)
    meth = MetaAcqROI(run_seed=0, weights=w, stop_after_phase1=True)
    try:
        meth.run(bx)  # returns normally only if the focus phase was never reached
    except Phase1Done:
        return -meth.phase1_score
    except Exception:
        pass
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt/meta_acq")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    w0 = init_weights()
    # linear policy only (8 weights, hidden layer kept at 0): CMA-ES converges in the
    # time box; the full network (88 weights) would need ~10x more generations
    full = lambda z: np.r_[w0[:-N_FEAT], z]  # noqa: E731
    es = CMAES(w0[-N_FEAT:], a.sigma0, rng)
    t_end = time.time() + a.hours * 3600
    log = open(out / "train_log.jsonl", "a")
    gen = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=20) as pool:
        while time.time() < t_end:
            gen += 1
            seeds = rng.choice(np.array(TRAIN), a.batch, replace=False)
            X = es.ask()
            cands = [full(x) for x in X] + [full(es.m), w0]
            tasks = [(w, int(sd)) for w in cands for sd in seeds]
            L = np.array(pool.map(episode, tasks)).reshape(len(cands), a.batch)
            loss = L.mean(axis=1)
            es.tell(X, -loss[: len(X)])
            gain = float(loss[-1] - loss[-2])  # EI loss - mean loss, same devices (>0 = better)
            rec = {"gen": gen, "loss_mean": float(loss[-2]), "loss_ei": float(loss[-1]),
                   "loss_best_pop": float(loss[: len(X)].min()), "gain_vs_ei": gain,
                   "sigma": float(es.sigma), "t": time.time()}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
            np.save(out / "w_last.npy", full(es.m))  # the checkpoint used is the final mean (fixed rule)
    np.save(out / "w_final.npy", full(es.m))


if __name__ == "__main__":
    main()
