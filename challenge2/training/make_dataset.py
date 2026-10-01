"""Generate frames + gates (+ observable scores) for JEPA pre-training (methods 4-5).

    uv run python -m training.make_dataset --split dev --n-devices 200 --frames 40 \
        --workers 6 --out data/jepa_dev

Frames come from a random walk inside the trackable region (so every frame has a known
registration), one device per process. Labels are OBSERVABLES only (barriers, window
centre, front-end score y); no hidden state is stored, so the dataset can never leak it.
A device-level split (``seeds`` below) prevents train/val/test contamination.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib

import numpy as np

from csd import new_experiment
from evaluation.seeds import SPLITS
from optimization import BlindExperiment
from optimization.session import Session, SessionConfig


def make_device(task: tuple) -> dict:
    seed, n_frames = task
    exp = new_experiment(seed=seed)
    bx = BlindExperiment(exp, pixel_cap=10**9, meas_cap=10**6)
    s = Session(bx, SessionConfig(keep_frames=True))
    s.start()
    rng = np.random.default_rng(seed)
    for _ in range(n_frames):
        s.evaluate(s.sample_feasible(rng)[0])
    y, se = s.scorer.scores()
    F = s.frames
    return {"seed": seed,
            "imgs": np.stack([f["img"] for f in F]),
            "b": np.stack([f["b"] for f in F]),
            "centre": np.stack([f["centre"] for f in F]),
            "y": y[-len(F):].astype(np.float32), "se": se[-len(F):].astype(np.float32)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=list(SPLITS))
    ap.add_argument("--n-devices", type=int, default=100)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="data/jepa_dev")
    a = ap.parse_args()
    seeds = list(SPLITS[a.split])[: a.n_devices]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        res = list(pool.imap(make_device, [(s, a.frames) for s in seeds]))
    np.savez_compressed(
        out / "frames.npz",
        imgs=np.concatenate([r["imgs"] for r in res]),
        b=np.concatenate([r["b"] for r in res]),
        centre=np.concatenate([r["centre"] for r in res]),
        y=np.concatenate([r["y"] for r in res]), se=np.concatenate([r["se"] for r in res]),
        device=np.concatenate([[r["seed"]] * len(r["b"]) for r in res]),
    )
    print(f"saved {sum(len(r['b']) for r in res)} frames from {len(res)} devices to {out}")


if __name__ == "__main__":
    main()
