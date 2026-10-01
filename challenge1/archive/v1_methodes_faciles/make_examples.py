"""Per-scene walkthrough: what each method computes, and the numbers on that very scene.
    uv run python results/challenge1_methodes_faciles/make_examples.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from detection.baselines import METHODS, LogReg, load_split, pixel_sweep, prep
from detection.metrics import object_scores

OUT = Path(__file__).parent
res = json.loads((OUT / "baselines.json").read_text())
root = Path("data")
train = load_split(root / "train", 500)
test = load_split(root / "test_baseline", 400)
zt, zte = prep(train["x"]), prep(test["x"])
meths = [c().fit(zt, train["m"]) if c is LogReg else c() for c in METHODS if c.__name__ != "Empty"]
DESC = {
    "M1_smooth": "lissage gaussien\n(σ=1 px) puis seuil",
    "M2_matched": "corrélation avec des\nbâtonnets orientés π/4",
    "M3_ridge": "courbure Hessienne\n(crête étroite)",
    "M4_hyst": "M1 + seuil à hystérésis\n+ retrait des petits blobs",
    "M5_logreg": "régression logistique\nsur 6 cartes multi-échelles",
}


def overlay(pred, gt):
    gtd = ndimage.binary_dilation(gt, np.ones((3, 3))); pdl = ndimage.binary_dilation(pred, np.ones((3, 3)))
    img = np.ones(gt.shape + (3,))
    img[pred & gtd] = (0.1, 0.7, 0.2); img[pred & ~gtd] = (0.9, 0.15, 0.15); img[gt & ~pdl] = (0.15, 0.3, 0.95)
    return img


amp = np.array([max((abs(s["intensity"]) for s in st), default=0) for st in test["sticks"]])
for tag, i in {"A_facile": int(np.argsort(amp)[300]), "B_lignes_de_charge": 150}.items():
    sc = {m.name: m.scores(zte[i:i + 1]) for m in meths}
    fig, axs = plt.subplots(len(meths), 5, figsize=(15, 3.1 * len(meths)))
    for r, m in enumerate(meths):
        thr = res[m.name]["thr"]
        pred = m.binarize(sc[m.name], thr)[0]
        gt = test["m"][i]
        px = pixel_sweep(sc[m.name], test["m"][i:i + 1], m, [thr])[0]
        ob = object_scores(pred[None], [test["sticks"][i]], *pred.shape)
        axs[r, 0].imshow(zte[i], cmap="gray", vmin=-3, vmax=10)
        axs[r, 0].set_ylabel(m.name + "\n" + DESC[m.name], fontsize=9)
        s = sc[m.name][0]
        im = axs[r, 1].imshow(s, cmap="magma", vmin=np.percentile(s, 1), vmax=np.percentile(s, 99.8))
        axs[r, 2].imshow(pred, cmap="gray_r"); axs[r, 3].imshow(gt, cmap="gray_r")
        axs[r, 4].imshow(overlay(pred, gt))
        axs[r, 2].set_xlabel(f"seuil = {thr:g}", fontsize=8)
        axs[r, 4].set_xlabel(f"sticks trouvés {round(ob['obj_recall']*ob['n_gt_sticks'])}/{ob['n_gt_sticks']}\n"
                             f"blobs justes {ob['obj_precision']:.2f} · tol-F1 {px['tol_f1']:.2f} · IoU {px['iou']:.2f}",
                             fontsize=8)
    for c, t in enumerate(["image prétraitée (σ du bruit)", "carte de score", "masque prédit", "vérité (générateur)",
                           "superposition (vert=bon,\nrouge=FP, bleu=manqué)"]):
        axs[0, c].set_title(t, fontsize=10)
    for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Scène test #{i} — {len(test['sticks'][i])} sticks, amplitude max {amp[i]:.0f}", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / f"fig6_exemple_{tag}.png", dpi=90); plt.close()
print("ok")
