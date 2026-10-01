"""Operating-point sweep of the frozen M5_min logit across stage-1 and stage-2 frames.

    uv run python -m transfer_m5.sweep

For a grid of logit thresholds: object P / R / F1 on val, test and ood_stage2, and false
blobs per image on the stick-free scenes (null). Also prints the background-logit
distribution of each set (is the score itself dataset-independent?). Ground truth is used
for evaluation only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from detection.baselines import FEATS, counts, features, load_split, prep, scores_from, stick_table

OUT = Path("challenge2/transfer_m5/results")
P = json.loads(Path("challenge1/results/m5_min_params.json").read_text())
COLS = [FEATS.index(c) for c in P["feats"]]


def logit(x):
    F = features(prep(x))[..., COLS]
    return ((F - P["mu"]) / P["sd"]) @ np.array(P["w"][:-1]) + P["w"][-1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    root = Path("data/eval_light")
    sets = {s: load_split(root / s, n) for s, n in
            [("val", 300), ("test", 400), ("ood_stage2", 150), ("null", 100)]}
    L, bg_stats = {}, {}
    for s, sp in sets.items():
        sp["st"] = stick_table(sp["sticks"])
        L[s] = logit(sp["x"])
        bg = L[s][~sp["m"]]
        med = float(np.median(bg))
        bg_stats[s] = {"median": med, "mad_sigma": float(1.4826 * np.median(np.abs(bg - med))),
                       "p99.9": float(np.percentile(bg, 99.9))}
        print(f"{s:10s} background logit: median {med:+.2f}  MAD-sigma {bg_stats[s]['mad_sigma']:.2f}"
              f"  p99.9 {bg_stats[s]['p99.9']:+.2f}")
    rows = []
    for t in np.round(np.arange(-5.0, 2.01, 0.25), 2):
        row = {"thr": float(t)}
        for s, sp in sets.items():
            r = scores_from(counts(L[s] > t, sp["m"], sp["st"])[0])
            if s == "null":
                row["null_blobs_per_img"] = r["blobs_per_image"]
            else:
                row[s] = {k: r[k] for k in ("obj_precision", "obj_recall", "obj_f1", "tol_f1")}
        rows.append(row)
        print(f"thr {t:+.2f} | null FA/img {row['null_blobs_per_img']:6.2f} | "
              + " | ".join(f"{s} P {row[s]['obj_precision']:.2f} R {row[s]['obj_recall']:.2f} "
                           f"F1 {row[s]['obj_f1']:.3f}" for s in ("val", "test", "ood_stage2")))
    (OUT / "sweep.json").write_text(json.dumps({"background": bg_stats, "sweep": rows}, indent=1))


if __name__ == "__main__":
    main()
