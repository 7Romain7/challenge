"""Figures for challenge 1 easy methods: metrics + comparison against generator ground truth.
    uv run python results/challenge1_methodes_faciles/make_plots.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detection.baselines import (METHODS, THRS, LogReg, load_split, pixel_sweep, prep)

OUT = Path(__file__).parent
res = json.loads((OUT / "baselines.json").read_text())
names = [k for k in res if k != "M0_empty"]
col = dict(zip(names, plt.cm.tab10(np.arange(len(names)))))

# ---- 1. headline metrics
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
w = 0.38; x = np.arange(len(names))
ax[0].bar(x - w/2, [res[k]["test"]["tol_f1"] for k in names], w, label="tol-F1 (1 px)")
ax[0].bar(x + w/2, [res[k]["test"]["iou"] for k in names], w, label="IoU strict")
ax[0].set_title("Pixel : tolérant vs strict (test)")
ax[1].bar(x - w/2, [res[k]["test"]["obj_f1"] for k in names], w, label="test")
ax[1].bar(x + w/2, [res[k]["ood_noise"]["obj_f1"] for k in names], w, label="test + bruit (OOD)")
ax[1].set_title("F1 par objet (stick)")
for a in ax[:2]:
    a.set_xticks(x, [k.split("_")[0] for k in names]); a.set_ylim(0, 1); a.legend(); a.grid(axis="y", alpha=.3)
for i, k in enumerate(names):
    t = res[k]["test"]
    ax[2].scatter(t["obj_recall"], t["obj_precision"], s=90, color=col[k], label=k)
ax[2].set_xlabel("recall objet"); ax[2].set_ylabel("précision objet"); ax[2].set_title("Précision / rappel par objet")
ax[2].set_xlim(.5, 1); ax[2].set_ylim(.8, 1); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "fig1_metriques.png", dpi=130); plt.close()

# ---- 2. recall vs amplitude (limite de détection physique)
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = list(res[names[0]]["test"]["recall_by_amplitude"])
for k in names:
    rb = res[k]["test"]["recall_by_amplitude"]
    ax.plot(bins, [rb[b][0] for b in bins], "o-", color=col[k], label=k)
n = [res[names[0]]["test"]["recall_by_amplitude"][b][1] for b in bins]
ax.set_xticks(range(len(bins)), [f"{b}\n(n={m})" for b, m in zip(bins, n)])
ax.set_xlabel("|amplitude| du stick (générateur)"); ax.set_ylabel("rappel objet"); ax.set_title("Rappel vs amplitude réelle")
ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(OUT / "fig2_rappel_vs_amplitude.png", dpi=130); plt.close()

# ---- 3. F1 vs seuil (val) : choix du seuil
root = Path("data")
train = load_split(root / "train", 500); val = load_split(root / "val", 300)
test = load_split(root / "test_baseline", 400)
zt, zv, zte = prep(train["x"]), prep(val["x"]), prep(test["x"])
meths = [c().fit(zt, train["m"]) if c is LogReg else c() for c in METHODS if c.__name__ != "Empty"]
fig, ax = plt.subplots(figsize=(7, 4.5))
scores = {}
for m in meths:
    s = m.scores(zv)
    rows = pixel_sweep(s, val["m"], m, THRS)
    ax.plot([r["thr"] for r in rows], [r["tol_f1"] for r in rows], color=col[m.name], label=m.name)
    ax.axvline(res[m.name]["thr"], color=col[m.name], ls=":", alpha=.6)
    scores[m.name] = m.scores(zte)
ax.set_xscale("log"); ax.set_xlabel("seuil (σ ou logit)"); ax.set_ylabel("tol-F1 (val)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("Sélection du seuil sur val (pointillés = seuil retenu)")
plt.tight_layout(); plt.savefig(OUT / "fig3_f1_vs_seuil.png", dpi=130); plt.close()

# ---- 4. comparaison qualitative avec le générateur : vérité vs prédiction (TP/FP/FN)
from scipy import ndimage
def overlay(pred, gt):
    gtd = ndimage.binary_dilation(gt, np.ones((3, 3))); pd_ = ndimage.binary_dilation(pred, np.ones((3, 3)))
    img = np.ones(gt.shape + (3,))
    img[pred & gtd] = (0.1, 0.7, 0.2)      # TP
    img[pred & ~gtd] = (0.9, 0.15, 0.15)   # FP
    img[gt & ~pd_] = (0.15, 0.3, 0.95)     # FN
    return img
amp = [max((abs(s["intensity"]) for s in st), default=0) for st in test["sticks"]]
ids = [int(np.argsort(amp)[len(amp) * q // 4]) for q in (1, 2, 3)] + [int(np.argmax(amp))]
cols = ["image brute", "vérité (générateur)"] + names
fig, axs = plt.subplots(len(ids), len(cols), figsize=(2.6 * len(cols), 2.6 * len(ids)))
for r, i in enumerate(ids):
    axs[r, 0].imshow(zte[i], cmap="gray", vmin=-3, vmax=10)
    axs[r, 1].imshow(test["m"][i], cmap="gray_r")
    axs[r, 0].set_ylabel(f"scène {i}\n|i|max={amp[i]:.0f}", fontsize=8)
    for c, m in enumerate(meths):
        p = m.binarize(scores[m.name][i:i+1], res[m.name]["thr"])[0]
        axs[r, 2 + c].imshow(overlay(p, test["m"][i]))
        if r == 0: pass
for c, t in enumerate(cols): axs[0, c].set_title(t, fontsize=9)
for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("Vert = détecté (≤1 px de la vérité) · Rouge = faux positif · Bleu = stick manqué", y=1.0)
plt.tight_layout(); plt.savefig(OUT / "fig4_comparaison_generateur.png", dpi=110); plt.close()

# ---- 5. densité de sticks prédite vs vraie (comptage par scène)
from scipy import ndimage as ndi
fig, ax = plt.subplots(1, len(meths), figsize=(3.2 * len(meths), 3.4), sharey=True)
gt_n = np.array([len(s) for s in test["sticks"]])
for a, m in zip(ax, meths):
    P = m.binarize(scores[m.name], res[m.name]["thr"])
    pn = np.array([ndi.label(p, np.ones((3, 3)))[1] for p in P])
    a.scatter(gt_n + np.random.default_rng(0).uniform(-.2, .2, len(gt_n)), pn, s=6, alpha=.4, color=col[m.name])
    mx = max(gt_n.max(), pn.max()); a.plot([0, mx], [0, mx], "k--", lw=1)
    a.set_title(m.name, fontsize=9); a.set_xlabel("nb sticks (générateur)"); a.grid(alpha=.3)
ax[0].set_ylabel("nb blobs prédits")
plt.tight_layout(); plt.savefig(OUT / "fig5_comptage_vs_generateur.png", dpi=120); plt.close()
print("ok")
