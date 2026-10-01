"""Paired comparison of methods against a reference (PROTOCOL section 4).

    python -m evaluation.compare --ref bo_roi_dlf results/a.jsonl results/b.jsonl

For each method: median R at the budget checkpoints, success R <= 0.05, regret split
(region choice vs refinement), then on the devices shared with the reference: bootstrap
95 % CI of the median paired dR, Wilcoxon signed-rank p, Holm-adjusted over all
comparisons in the call. Several run seeds of a method are averaged per device first.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon


def load(paths):
    by = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if "error" not in r:
                by[r["method"]][r["seed"]].append(r)
    return by


def per_device(runs, key):
    return float(np.mean([key(r) for r in runs]))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    by = load(a.files)
    rng = np.random.default_rng(0)
    ref = by[a.ref]
    rows, pvals = [], []
    for m in sorted(k for k in by if by[k]):
        d = by[m]
        R = np.array([per_device(v, lambda r: r["R_final"]) for v in d.values()])
        R5 = np.array([per_device(v, lambda r: r["R_curve"].get("500000", np.nan)) for v in d.values()])
        row = {"method": m, "n": len(d), "R@500k": np.nanmedian(R5), "R@1M": np.median(R),
               "succ": np.mean(R <= 0.05), "fail>0.3": np.mean(R > 0.3)}
        some = next(iter(d.values()))[0]
        if "region_ok" in some:
            row["region_ok"] = np.mean([per_device(v, lambda r: r["region_ok"]) for v in d.values()])
            row["best_seen"] = np.mean([per_device(v, lambda r: r["best_visited"]) for v in d.values()])
            Rr = np.array([per_device(v, lambda r: r["R_region"]) for v in d.values()])
            row["R_region_mean"] = Rr.mean()
            row["R_refine_mean"] = R.mean() - Rr.mean()
        if m != a.ref and ref:
            common = sorted(set(d) & set(ref))
            dR = np.array([per_device(d[s], lambda r: r["R_final"]) - per_device(ref[s], lambda r: r["R_final"])
                           for s in common])
            bs = np.median(dR[rng.integers(0, len(dR), (a.boot, len(dR)))], axis=1)
            row.update(n_pair=len(common), dR_med=np.median(dR), ci_lo=np.percentile(bs, 2.5),
                       ci_hi=np.percentile(bs, 97.5), better=np.mean(dR < 0))
            p = wilcoxon(dR).pvalue if np.any(dR != 0) else 1.0
            row["p"] = p
            pvals.append((len(rows), p))
        rows.append(row)
    if pvals:
        adj = holm([p for _, p in pvals])
        for (i, _), q in zip(pvals, adj):
            rows[i]["p_holm"] = q
    keys = ["method", "n", "R@500k", "R@1M", "succ", "fail>0.3", "region_ok", "best_seen",
            "R_region_mean", "R_refine_mean", "dR_med", "ci_lo", "ci_hi", "better", "p", "p_holm"]
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(f"{r[k]:.3g}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                                for k in keys) + " |")


if __name__ == "__main__":
    main()
