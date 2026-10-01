"""Train the PFN surrogate of ``optimization.methods.pfn`` on synthetic landscapes only.

    python -m training.train_pfn --out ckpt/pfn --steps 12000

Protocol (PROTOCOL P15 / section 5.2):
  - tasks come from ``training.priors`` (broader than the simulator); the **Lorentzian
    family is excluded from training** - it is the simulator's shape and is kept as a
    held-out generalisation test;
  - the checkpoint is selected on the NLL of an in-family synthetic val set only;
  - report: NLL on in-family val, on Lorentz-only tasks, and a GP fitted on the same
    contexts (the method must at least match the GP it replaces). The simulator is never
    touched here.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from optimization.methods.gp import GP
from optimization.methods.pfn import BOX, build_net, normalise
from training.priors import sample_landscape

C_MAX, Q = 60, 64
TRAIN_SHAPES = ("gauss", "expo")


def make_task(seed: int, shapes) -> tuple:
    rng = np.random.default_rng(seed)
    L = sample_landscape(rng, box=BOX)
    L.shapes = [shapes[i] for i in rng.integers(0, len(shapes), len(L.amps))]
    n = int(rng.integers(3, C_MAX + 1))
    # BO-like histories: part uniform, part clustered around an attractor
    n_loc = int(rng.uniform(0, 0.7) * n)
    att = L.centres[rng.integers(len(L.centres))] if rng.random() < 0.5 else rng.uniform(-BOX, BOX, 3)
    sd = np.exp(rng.uniform(np.log(0.02), np.log(0.15)))
    xc = np.vstack([rng.uniform(-BOX, BOX, (n - n_loc, 3)), att + rng.normal(0, sd, (n_loc, 3))])
    xc = np.clip(rng.permutation(xc), -BOX, BOX)
    yc, se = L.observe(xc, rng)
    xq = np.vstack([rng.uniform(-BOX, BOX, (Q // 2, 3)),
                    xc[rng.integers(n, size=Q - Q // 2)] + rng.normal(0, 0.05, (Q - Q // 2, 3))])
    xq = np.clip(xq, -BOX, BOX)
    return xc, yc, se, xq, L.f(xq)


def encode(task) -> dict:
    xc, yc, se, xq, fq = task
    n = len(xc)
    yn, sn, mu, sd = normalise(yc, se)
    pad = C_MAX - n
    return {"xc": np.pad(xc / BOX, ((0, pad), (0, 0))).astype(np.float32),
            "yc": np.pad(yn, (0, pad)).astype(np.float32),
            "sc": np.pad(np.log(sn + 1e-3), (0, pad)).astype(np.float32),
            "cm": np.r_[np.zeros(n, bool), np.ones(pad, bool)],
            "xq": (xq / BOX).astype(np.float32), "tq": ((fq - mu) / sd).astype(np.float32)}


def _gen(args):
    seed, shapes = args
    return encode(make_task(seed, shapes))


def dataset(seeds, shapes, workers=16) -> dict:
    with Pool(workers) as p:
        rows = p.map(_gen, [(s, shapes) for s in seeds], chunksize=256)
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def nll(m, ls, t):
    return 0.5 * ((t - m) / np.exp(ls)) ** 2 + ls + 0.5 * np.log(2 * np.pi)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # disjoint seed ranges: train / in-family val / Lorentz-only test
    tr = dataset(range(0, a.n_train), TRAIN_SHAPES)
    va = dataset(range(10**8, 10**8 + 4000), TRAIN_SHAPES)
    lo = dataset(range(2 * 10**8, 2 * 10**8 + 4000), ("lorentz",))
    print(f"data {time.time() - t0:.0f}s", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = {"d": 128, "layers": 4, "heads": 4}
    net = build_net(**arch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    T = lambda d, i: [torch.as_tensor(d[k][i]).to(dev) for k in ("xc", "yc", "sc", "cm", "xq", "tq")]  # noqa: E731

    def evaluate(d):
        net.eval()
        tot = []
        with torch.no_grad():
            for i in range(0, len(d["xc"]), 1000):
                xc, yc, sc, cm, xq, tq = T(d, slice(i, i + 1000))
                m, ls = net(xc, yc, sc, cm, xq)
                tot.append(nll(m.cpu().numpy(), ls.cpu().numpy(), tq.cpu().numpy()).mean())
        net.train()
        return float(np.mean(tot))

    rng = np.random.default_rng(0)
    best, log = np.inf, []
    for step in range(1, a.steps + 1):
        xc, yc, sc, cm, xq, tq = T(tr, rng.integers(0, len(tr["xc"]), a.bs))
        m, ls = net(xc, yc, sc, cm, xq)
        loss = (0.5 * ((tq - m) * torch.exp(-ls)) ** 2 + ls).clamp(max=50).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == a.steps:
            v = evaluate(va)  # selection on in-family val ONLY
            row = {"step": step, "loss": float(loss), "val_nll": v, "s": round(time.time() - t0)}
            if v < best:
                best = v
                torch.save({"model": net.state_dict(), "arch": arch, "step": step}, out / "best.pt")
            log.append(row)
            print(row, flush=True)
    # report-only diagnostics with the selected checkpoint
    ck = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    net.load_state_dict(ck["model"])
    rep = {"selected_step": ck["step"], "val_nll_in_family": evaluate(va),
           "nll_lorentz_heldout": evaluate(lo)}
    # GP fitted on the same contexts (300 Lorentz tasks), same normalised units
    g = []
    for i in range(300):
        n = int((~lo["cm"][i]).sum())
        X = lo["xc"][i, :n] * BOX
        se = np.exp(lo["sc"][i, :n]) - 1e-3
        gp = GP(rng=np.random.default_rng(i)).fit(X, lo["yc"][i, :n], np.maximum(se, 1e-3))
        mq, sq = gp.predict(lo["xq"][i] * BOX)
        g.append(nll(mq, np.log(np.maximum(sq, 1e-3)), lo["tq"][i]).mean())
    sub = {k: v[:300] for k, v in lo.items()}
    rep["nll_lorentz_heldout_first300"] = evaluate(sub)
    rep["gp_nll_lorentz_heldout_first300"] = float(np.mean(g))
    rep["train_shapes"] = TRAIN_SHAPES
    (out / "report.json").write_text(json.dumps({"log": log, "report": rep, "args": vars(a)}, indent=1))
    print(rep)


if __name__ == "__main__":
    main()
