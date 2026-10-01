"""P4 (robustness) applied to the classical M5 — is it generator-specific?

    uv run python -m detection.p4_m5            # ~5 min CPU, no torch

Same protocol as baselines.py (fit -> threshold on val by sel = (obj_F1 + tol_F1)/2 ->
frozen everywhere else). One factor changes: the data M5's weights are fitted on.

  official  data/train (csd.generate_dataset, seed 0)  — the v2 baseline
  base      pool synth, generator law                   — same law, other sampler (control)
  lowsnr    pool synth, |i| log-uniform                 — P4b
  shift     pool synth, affine + noise level x U(0.7, 1.3) — P4a (no polarity: prep() fixes
            the sign, an a-priori physics choice)

Report-only sets add two noise types the generator never produces (make_real_noise.py):
1/f + slow drift along the raster, and random telegraph charge jumps. M2 (matched filter,
nothing fitted) is the no-learning reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from detection.baselines import (
    N_FIT, N_NULL, N_OOD, N_TEST, N_VAL, build_methods, counts, features, load_split, prep,
    scores_from, select_threshold, stick_table,
)

FITS = {"official": "data/train", "base": "data/fit_p4/base", "lowsnr": "data/fit_p4/lowsnr",
        "shift": "data/fit_p4/shift"}
SETS = [("val", N_VAL), ("test", N_TEST), ("ood_noise_up", N_OOD), ("ood_noise_1f", N_TEST),
        ("ood_charge_jump", N_TEST), ("ood_stage2", N_OOD), ("ood_theta_shift", N_OOD),
        ("ood_zoom_out", N_OOD), ("null", N_NULL)]
KEEP = ("M2_matched", "M5_logreg", "M5_min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval_light")
    ap.add_argument("--out", default="challenge1/results/p4_m5")
    args = ap.parse_args()
    root, out = Path(args.eval), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {s: load_split(root / s, n) for s, n in SETS}
    for sp in splits.values():
        sp["st"] = stick_table(sp["sticks"])
    feats = {s: features(prep(sp["x"])) for s, sp in splits.items()}
    print("features ready", flush=True)

    res = {}
    for fname, fpath in FITS.items():
        tr = load_split(Path(fpath), N_FIT, f"fit_{fname}")
        methods, weights = build_methods(features(prep(tr["x"])), tr["m"], (0,))
        for mname in KEEP:
            if mname == "M2_matched" and fname != "official":
                continue  # nothing fitted: identical for every fit set
            fn, binz = methods[mname]
            thr, _ = select_threshold(fn(feats["val"]), binz, splits["val"])
            row = {"thr": float(thr), "weights": weights.get(mname)}
            for s, sp in splits.items():
                C, A = counts(binz(fn(feats[s]), thr), sp["m"], sp["st"])
                r = scores_from(C)
                a = A.sum(0)
                r["recall_by_amplitude"] = [float(a[0, i] / a[1, i]) if a[1, i] else None
                                            for i in range(a.shape[1])]
                row[s] = r
            key = mname if mname == "M2_matched" else f"{mname}[{fname}]"
            res[key] = row
            t = row["test"]
            print(f"{key:22s} thr={thr:7.3f} test obj={t['obj_f1']:.3f}  "
                  f"1f={row['ood_noise_1f']['obj_f1']:.3f} jump={row['ood_charge_jump']['obj_f1']:.3f} "
                  f"stage2={row['ood_stage2']['obj_f1']:.3f} null_fp={row['null']['blobs_per_image']:.2f}",
                  flush=True)

    (out / "p4_m5.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    cols = [s for s, _ in SETS if s not in ("val", "null")]
    lines = ["| méthode [fit] | " + " | ".join(cols) + " | faux blobs / img vide |",
             "|---|" + "---|" * (len(cols) + 1)]
    for k, row in res.items():
        cells = [f"{row[s]['obj_f1']:.3f}" for s in cols]
        lines.append(f"| {k} | " + " | ".join(cells) + f" | {row['null']['blobs_per_image']:.2f} |")
    lines += ["", "Rappel par amplitude |i| [0,2) [2,4) [4,8) [8,16) [16,inf) :", ""]
    for k, row in res.items():
        for s in ("test", "ood_noise_1f", "ood_charge_jump"):
            lines.append(f"- {k} {s}: " + " ".join("—" if v is None else f"{v:.2f}"
                                                   for v in row[s]["recall_by_amplitude"]))
    (out / "p4_m5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
