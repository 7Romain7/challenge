"""Figures + fitted weights for the M5 write-up (feeds make_m5_pdf.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detection.baselines import LogReg, features, load_split, prep

OUT = Path(__file__).parent
res = json.loads((OUT / "baselines.json").read_text())
root = Path("data")
train = load_split(root / "train", 500)
test = load_split(root / "test_baseline", 400)
zt, zte = prep(train["x"]), prep(test["x"])
m = LogReg().fit(zt, train["m"])
names = ["z (prétraité)", "smooth σ=1", "smooth σ=2", "matched", "ridge", "écart-type local 7×7"]
n_pos = int(train["m"].sum())
info = {"names": names, "w": m.w[:-1].tolist(), "b": float(m.w[-1]), "mu": m.mu.tolist(), "sd": m.sd.tolist(),
        "n_pos": n_pos, "n_pix": int(train["m"].size), "thr": res["M5_logreg"]["thr"]}
(OUT / "m5_weights.json").write_text(json.dumps(info, indent=1))

i = 172
F = features(zte[i:i + 1])[0]
logit = m.scores(zte[i:i + 1])[0]
fig, ax = plt.subplots(2, 4, figsize=(13, 6.6))
for k in range(6):
    a = ax.flat[k]; f = F[..., k]
    a.imshow(f, cmap="magma" if k else "gray", vmin=np.percentile(f, 1), vmax=np.percentile(f, 99.8))
    a.set_title(f"feature {k+1} : {names[k]}\npoids w = {m.w[k]:+.2f}", fontsize=9)
a = ax.flat[6]; a.imshow(logit, cmap="magma", vmin=np.percentile(logit, 1), vmax=np.percentile(logit, 99.8))
a.set_title("logit  = Σ wₖ·(fₖ−μₖ)/σₖ + b", fontsize=9)
a = ax.flat[7]; a.imshow(logit > info["thr"], cmap="gray_r"); a.set_title(f"logit > {info['thr']}  (masque)", fontsize=9)
for a in ax.flat: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(OUT / "m5_features.png", dpi=110); plt.close()

# contribution of each feature (w_k * standardised value) on the same scene
Z = (F - m.mu) / m.sd
fig, ax = plt.subplots(1, 6, figsize=(15, 2.8))
for k in range(6):
    c = m.w[k] * Z[..., k]
    v = np.percentile(np.abs(c), 99.5)
    ax[k].imshow(c, cmap="RdBu_r", vmin=-v, vmax=v); ax[k].set_title(f"{names[k]}\nw·z", fontsize=8)
    ax[k].set_xticks([]); ax[k].set_yticks([])
plt.tight_layout(); plt.savefig(OUT / "m5_contributions.png", dpi=110); plt.close()
print(json.dumps(info, indent=1))
