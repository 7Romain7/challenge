"""Interdot masks on 6 random test images: raw, generator mask, then each method's binary mask.
    uv run python results/challenge1_methodes_faciles/make_masks.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detection.baselines import METHODS, LogReg, load_split, pixel_sweep, prep

OUT = Path(__file__).parent
res = json.loads((OUT / "baselines.json").read_text())
root = Path("data")
train = load_split(root / "train", 500)
test = load_split(root / "test_baseline", 400)
zt, zte = prep(train["x"]), prep(test["x"])
meths = [c().fit(zt, train["m"]) if c is LogReg else c() for c in METHODS if c.__name__ != "Empty"]
ids = sorted(np.random.default_rng(3).choice(len(zte), 6, replace=False).tolist())
sc = {m.name: m.scores(zte[ids]) for m in meths}
cols = ["image test (σ du bruit)", "masque interdots\n(générateur)"] + [m.name for m in meths]
fig, axs = plt.subplots(len(ids), len(cols), figsize=(2.5 * len(cols), 2.5 * len(ids)))
for r, i in enumerate(ids):
    axs[r, 0].imshow(zte[i], cmap="gray", vmin=-3, vmax=10)
    axs[r, 1].imshow(test["m"][i], cmap="gray_r")
    axs[r, 0].set_ylabel(f"test #{i}\n{len(test['sticks'][i])} sticks", fontsize=8)
    for c, m in enumerate(meths):
        thr = res[m.name]["thr"]
        p = m.binarize(sc[m.name][r:r + 1], thr)[0]
        f1 = pixel_sweep(sc[m.name][r:r + 1], test["m"][i:i + 1], m, [thr])[0]["tol_f1"]
        axs[r, 2 + c].imshow(p, cmap="gray_r")
        axs[r, 2 + c].set_xlabel(f"tol-F1 {f1:.2f}", fontsize=8)
for c, t in enumerate(cols): axs[0, c].set_title(t, fontsize=9)
for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(OUT / "fig7_masques_interdots.png", dpi=100); plt.close()
print(ids)
