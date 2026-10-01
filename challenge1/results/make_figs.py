"""Figures for the v2 classical baselines (run after `uv run python -m detection.baselines`
and `... --select dice --out challenge1/results/selection_dice`).

    uv run python challenge1/results/make_figs.py

fig1_masques_test.png    image, generator mask, then each method's binary mask (frozen val threshold)
fig2_masques_stage2.png  same on challenge-2 frames (the low-SNR regime the detector must serve)
fig3_metriques.png       obj F1 / tol F1 / Dice on test, 95 % bootstrap CI, two threshold criteria
fig4_robustesse.png      obj F1 per method x shifted set (report-only sets)
fig5_rappel_amplitude.png object recall vs stick amplitude |i| (detection limit)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detection.baselines import FEATS, N_FIT, build_methods, features, load_split, prep

OUT = Path(__file__).parent
EVAL = Path("data/eval_light")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dc"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

R = json.loads((OUT / "baselines.json").read_text(encoding="utf-8"))["results"]
RD = json.loads((OUT / "selection_dice" / "baselines.json").read_text(encoding="utf-8"))["results"]
SHOW = ["M1_smooth", "M2_matched", "M3_ridge", "M4_hyst", "M5_logreg", "M5_full", "M5_min"]


def masks_figure(split_name, n, seed, fname, title):
    train = load_split(Path("data/train"), N_FIT, "train_fit")
    methods, _ = build_methods(features(prep(train["x"])), train["m"], (0,))
    sp = load_split(EVAL / split_name)
    rng = np.random.default_rng(seed)
    ids = sorted(rng.choice(len(sp["x"]), n, replace=False).tolist())
    z = prep(sp["x"][ids])
    F = features(z)
    cols = ["image (σ du bruit)", "masque interdots\n(générateur)"] + SHOW
    fig, axs = plt.subplots(n, len(cols), figsize=(2.1 * len(cols), 2.2 * n))
    for r, i in enumerate(ids):
        axs[r, 0].imshow(z[r], cmap="gray", vmin=-3, vmax=10, origin="lower")
        axs[r, 1].imshow(sp["m"][i], cmap="gray_r", origin="lower")
        n_in = sum(0 <= s["row"] < 150 and 0 <= s["col"] < 150 for s in sp["sticks"][i])
        amp = np.median([abs(s["intensity"]) for s in sp["sticks"][i]]) if sp["sticks"][i] else 0
        axs[r, 0].set_ylabel(f"#{i}  {n_in} sticks\n|i| médian {amp:.1f}", fontsize=8)
        for c, m in enumerate(SHOW):
            fn, binz = methods[m]
            p = binz(fn(F[r : r + 1]), R[m]["thr"])[0]
            g = sp["m"][i]
            dice = 2 * (p & g).sum() / max(p.sum() + g.sum(), 1)
            axs[r, 2 + c].imshow(p, cmap="gray_r", origin="lower")
            axs[r, 2 + c].set_xlabel(f"Dice {dice:.2f}", fontsize=8, color=INK2)
    for c, t in enumerate(cols):
        axs[0, c].set_title(t, fontsize=9)
    for a in axs.ravel():
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(OUT / fname, dpi=110)
    plt.close()
    print(fname, ids)


def metrics_figure():
    keys = [("obj_f1", "obj F1 (interdots trouvés)"), ("tol_f1", "tol F1 (pixel ±1 px)"),
            ("dice", "Dice (pixel strict, global)")]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    y = np.arange(len(SHOW))[::-1]
    for ax, (k, lab) in zip(axs, keys):
        for res, col, off, name, mk in ((R, BLUE, 0.15, "seuil choisi par (obj F1 + tol F1)/2", "o"),
                                        (RD, ORANGE, -0.15, "seuil choisi par Dice", "s")):
            v = np.array([res[m]["test"][k] for m in SHOW])
            ci = np.array([res[m]["test"][k + "_ci"] for m in SHOW])
            ax.errorbar(v, y + off, xerr=[v - ci[:, 0], ci[:, 1] - v], fmt=mk, ms=6, color=col,
                        ecolor=col, elinewidth=2, capsize=0, label=name)
        ax.set_title(lab, fontsize=10, loc="left")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axs[0].set_yticks(y, SHOW)
    axs[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Test (400 scènes, seeds 2e7+k) — IC 95 % bootstrap sur les images", fontsize=11,
                 x=0.01, ha="left")
    plt.tight_layout()
    plt.savefig(OUT / "fig3_metriques.png", dpi=130)
    plt.close()


def robustness_figure():
    sets = ["val", "test", "ood_theta_shift", "ood_theta_wide", "ood_noise_up", "ood_zoom_in",
            "ood_zoom_out", "ood_stage2"]
    labels = ["val", "test", "θ décalé", "θ large", "bruit ×1,5", "zoom 1 mV", "zoom 3 mV",
              "frames stage 2"]
    M = np.array([[R[m][s]["obj_f1"] for s in sets] for m in SHOW])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(sets)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(SHOW)), SHOW)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(len(sets) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(SHOW) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURF, lw=2)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, fraction=0.03, label="obj F1")
    ax.set_title("obj F1 avec le seuil val figé — jeux décalés (rapport seulement, jamais de sélection)",
                 fontsize=10, loc="left")
    plt.tight_layout()
    plt.savefig(OUT / "fig4_robustesse.png", dpi=130)
    plt.close()


def recall_figure():
    show = ["M2_matched", "M3_ridge", "M5_logreg", "M5_full", "M5_min"]
    bins = list(R[show[0]]["test"]["recall_by_amplitude"].keys())
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for m, col, mk in zip(show, SERIES, MARKERS):
        v = [R[m]["test"]["recall_by_amplitude"][b][0] for b in bins]
        ax.plot(x, v, color=col, lw=2, marker=mk, ms=8, mec=SURF, mew=1.5, label=m)
    n = [R[show[0]]["test"]["recall_by_amplitude"][b][1] for b in bins]
    ax.set_xticks(x, [f"{b}\n(n={k})" for b, k in zip(bins, n)])
    ax.set_xlabel("|amplitude| du stick (bruit σ_pix = 0,9)")
    ax.set_ylabel("rappel objet")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Limite de détection (test)", fontsize=10, loc="left")
    plt.tight_layout()
    plt.savefig(OUT / "fig5_rappel_amplitude.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    assert FEATS[0] == "z"
    masks_figure("test", 6, 3, "fig1_masques_test.png",
                 "Test : image, masque du générateur, puis masque binaire de chaque méthode (seuil val figé)")
    masks_figure("ood_stage2", 4, 3, "fig2_masques_stage2.png",
                 "Frames du challenge 2 (faible SNR) : même seuil, même affichage")
    metrics_figure()
    robustness_figure()
    recall_figure()
