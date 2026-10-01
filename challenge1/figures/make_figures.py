"""Figures du README du challenge 1 : régression logistique (M5_min) contre filtre adapté (M2).

    uv run python challenge1/figures/make_figures.py

Lit les résultats figés (challenge1/results/baselines.json, seuil choisi sur val) et les
jeux de data/eval_light. Les vérités terrain ne servent qu'à l'affichage et au scoring.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from detection.baselines import FEATS, N_FIT, build_methods, features, load_split, prep
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
EVAL = ROOT / "data/eval_light"
RES = json.loads((ROOT / "challenge1/results/baselines.json").read_text(encoding="utf-8"))["results"]
SWEEP = json.loads((ROOT / "challenge2/transfer_m5/results/sweep.json").read_text(encoding="utf-8"))
P5 = json.loads((ROOT / "challenge1/results/m5_min_params.json").read_text(encoding="utf-8"))

# les deux méthodes comparées : nom affiché, clé dans les résultats, couleur
REGLOG, FILTRE = "M5_min", "M2_matched"
NAMES = {REGLOG: "Régression logistique", FILTRE: "Filtre adapté"}
COL = {REGLOG: "#2a78d6", FILTRE: "#eb6834"}
MK = {REGLOG: "o", FILTRE: "s"}
SHOW = [FILTRE, REGLOG]

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dc"
OK, FP, FN = "#1baf7a", "#e0457b", "#eda100"  # bon pixel, fausse alarme, interdot manqué
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": SURF})

_METHODS = None


def methods():
    """Refit identique à detection.baselines (500 images de data/train, seed 0)."""
    global _METHODS
    if _METHODS is None:
        train = load_split(ROOT / "data/train", N_FIT, "train_fit")
        _METHODS = build_methods(features(prep(train["x"])), train["m"], (0,))[0]
    return _METHODS


def predict(F, m):
    fn, binz = methods()[m]
    return binz(fn(F), RES[m]["thr"])


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRID)


def overlay(z, pred, gt):
    """Image en gris + pixels prédits colorés : vert si un pixel vrai est à ±1 px, rose sinon ;
    pixels vrais non couverts en jaune."""
    g = np.clip((z + 3) / 13, 0, 1)
    rgb = np.repeat(g[..., None], 3, -1) * 0.75 + 0.25
    near_gt = ndimage.maximum_filter(gt, 3)
    near_pr = ndimage.maximum_filter(pred, 3)
    for mask, c in ((gt & ~near_pr, FN), (pred & near_gt, OK), (pred & ~near_gt, FP)):
        rgb[mask] = matplotlib.colors.to_rgb(c)
    return rgb


def legend_overlay(fig, y=0.005):
    h = [Patch(color=OK, label="pixel prédit correct (±1 px)"),
         Patch(color=FP, label="pixel prédit faux (fausse alarme)"),
         Patch(color=FN, label="pixel d'interdot manqué")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, y))


# ------------------------------------------------------------------ figures
def fig_pipeline():
    sp = load_split(EVAL / "test", 40)
    i = 7
    z = prep(sp["x"][i : i + 1])
    F = features(z)
    lg = methods()[REGLOG][0](F)[0]
    pred = lg > RES[REGLOG]["thr"]
    panels = [(z[0], "1. image normalisée\n(unités de σ du bruit)", "gray", (-3, 10)),
              (F[0, ..., FEATS.index("s2")], "2. feature s2\n(lissage gaussien σ=2 px)", "magma", None),
              (F[0, ..., FEATS.index("matched")], "3. feature matched\n(filtre adapté orienté)", "magma", None),
              (lg, "4. logit = w·x + b\n(régression logistique)", "RdBu_r", (-12, 12)),
              (pred, f"5. masque prédit\n(logit > {RES[REGLOG]['thr']:.2f})", "gray_r", None),
              (sp["m"][i], "6. masque vrai\n(générateur)", "gray_r", None)]
    fig, axs = plt.subplots(1, 6, figsize=(15, 3.2))
    for ax, (im, t, cm, lim) in zip(axs, panels):
        kw = {"vmin": lim[0], "vmax": lim[1]} if lim else {}
        ax.imshow(im, cmap=cm, origin="lower", **kw)
        ax.set_title(t, fontsize=9)
        clean(ax)
    fig.suptitle("Pipeline de la régression logistique sur une scène de test", x=0.01, ha="left",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_pipeline_reglog.png", dpi=120)
    plt.close()


def fig_masks(split, ids, fname, title):
    sp = load_split(EVAL / split)
    z = prep(sp["x"][ids])
    F = features(z)
    preds = {m: predict(F, m) for m in SHOW}
    cols = ["image", "masque vrai"] + [NAMES[m] for m in SHOW]
    fig, axs = plt.subplots(len(ids), 4, figsize=(9.6, 2.5 * len(ids) + 0.5))
    for r, i in enumerate(ids):
        gt = sp["m"][i]
        n_in = sum(0 <= s["row"] < 150 and 0 <= s["col"] < 150 for s in sp["sticks"][i])
        amp = np.median([abs(s["intensity"]) for s in sp["sticks"][i]]) if sp["sticks"][i] else 0
        axs[r, 0].imshow(z[r], cmap="gray", vmin=-3, vmax=10, origin="lower")
        axs[r, 0].set_ylabel(f"scène #{i}\n{n_in} interdots, |i| méd. {amp:.1f}", fontsize=8)
        axs[r, 1].imshow(gt, cmap="gray_r", origin="lower")
        for c, m in enumerate(SHOW):
            p = preds[m][r]
            axs[r, 2 + c].imshow(overlay(z[r], p, gt), origin="lower")
            dice = 2 * (p & gt).sum() / max(p.sum() + gt.sum(), 1)
            axs[r, 2 + c].set_xlabel(f"Dice {dice:.2f}", fontsize=8, color=INK2)
    for c, t in enumerate(cols):
        axs[0, c].set_title(t, fontsize=10)
    for a in axs.ravel():
        clean(a)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    plt.tight_layout(rect=(0, 0.035, 1, 0.98))
    legend_overlay(fig)
    plt.savefig(OUT / fname, dpi=110)
    plt.close()


def fig_boundary():
    """Espace des deux features : la régression logistique = filtre adapté + correction s2."""
    sp = load_split(EVAL / "val", 150)
    F = features(prep(sp["x"]))
    ci, cj = FEATS.index("matched"), FEATS.index("s2")
    X, Y, m = F[..., ci].ravel(), F[..., cj].ravel(), sp["m"].ravel()
    rng = np.random.default_rng(0)
    bg = rng.choice(np.flatnonzero(~m), 40_000, replace=False)
    fg = rng.choice(np.flatnonzero(m), min(8_000, m.sum()), replace=False)
    _, ax = plt.subplots(figsize=(6.6, 5))
    ax.scatter(X[bg], Y[bg], s=2, c="#a8a7a1", alpha=0.25, lw=0, label="fond (échantillon)")
    ax.scatter(X[fg], Y[fg], s=2, c="#1baf7a", alpha=0.35, lw=0, label="pixels d'interdot")
    (mu0, mu1), (sd0, sd1) = P5["mu"], P5["sd"]
    w0, w1, b = P5["w"]
    xs = np.linspace(-6, 40, 50)
    ys = mu1 + sd1 / w1 * (P5["thr"] - b - w0 * (xs - mu0) / sd0)  # w·x + b = thr
    ax.plot(xs, ys, color=COL[REGLOG], lw=2, label=f"{NAMES[REGLOG]} (frontière)")
    ax.axvline(RES[FILTRE]["thr"], color=COL[FILTRE], lw=2, ls="--",
               label=f"{NAMES[FILTRE]} (matched > {RES[FILTRE]['thr']:.2f})")
    ax.set_xlim(-6, 40)
    ax.set_ylim(-6, 40)
    ax.set_xlabel("matched (réponse du filtre adapté, en σ)")
    ax.set_ylabel("s2 (image lissée σ=2 px, en σ)")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=6)
    ax.set_title("Frontière de décision (val, 150 scènes)", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_frontiere_decision.png", dpi=130)
    plt.close()


def fig_metrics():
    keys = [("obj_precision", "précision objet"), ("obj_recall", "rappel objet"),
            ("obj_f1", "obj F1"), ("tol_f1", "tol F1 (±1 px)"), ("dice", "Dice global"),
            ("dice_img", "Dice / image")]
    _, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(keys))
    for k, m in enumerate(SHOW):
        t = RES[m]["test"]
        v = np.array([t[a] for a, _ in keys])
        lo = np.array([v[j] - t[a + "_ci"][0] if a + "_ci" in t else np.nan for j, (a, _) in enumerate(keys)])
        hi = np.array([t[a + "_ci"][1] - v[j] if a + "_ci" in t else np.nan for j, (a, _) in enumerate(keys)])
        off = (k - 0.5) * 0.36
        ax.bar(x + off, v, 0.34, color=COL[m], label=NAMES[m], edgecolor=SURF, lw=2)
        ax.errorbar(x + off, v, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for j, val in enumerate(v):
            ax.text(x[j] + off, val + 0.03, f"{val:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [lab for _, lab in keys])
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Test, 400 scènes, seuil choisi sur val (barres : IC 95 % bootstrap)",
                 loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig5_metriques.png", dpi=130)
    plt.close()


def fig_robustness():
    sets = ["val", "test", "ood_theta_shift", "ood_theta_wide", "ood_noise_up", "ood_zoom_in",
            "ood_zoom_out", "ood_stage2"]
    labels = ["val", "test", "θ décalé", "θ large", "bruit ×1,5", "zoom 1 mV", "zoom 3 mV",
              "frames\nchallenge 2"]
    _, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(sets))
    for m in SHOW:
        v = [RES[m][s]["obj_f1"] for s in sets]
        ax.plot(x, v, color=COL[m], marker=MK[m], ms=8, lw=2, mec=SURF, mew=1.5, label=NAMES[m])
    ax.axvspan(1.5, 7.5, color=GRID, alpha=0.35, lw=0)
    ax.text(4.5, 0.03, "décalages simulés : rapportés, jamais utilisés pour choisir",
            ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("obj F1")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="center left")
    ax.set_title("Robustesse : obj F1 avec le seuil val figé", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig6_robustesse.png", dpi=130)
    plt.close()


def fig_recall():
    bins = list(RES[REGLOG]["test"]["recall_by_amplitude"].keys())
    x = np.arange(len(bins))
    _, ax = plt.subplots(figsize=(6.5, 3.6))
    for m in SHOW:
        v = [RES[m]["test"]["recall_by_amplitude"][b][0] for b in bins]
        ax.plot(x, v, color=COL[m], lw=2, marker=MK[m], ms=8, mec=SURF, mew=1.5, label=NAMES[m])
    n = [RES[REGLOG]["test"]["recall_by_amplitude"][b][1] for b in bins]
    ax.set_xticks(x, [f"{b}\n(n={k})" for b, k in zip(bins, n)])
    ax.set_xlabel("|amplitude| de l'interdot (bruit σ_pix ≈ 0,9)")
    ax.set_ylabel("rappel objet")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Limite de détection (test)", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig7_rappel_amplitude.png", dpi=130)
    plt.close()


def fig_metric_explained():
    """Une scène zoomée : ce que mesurent Dice, tol F1 et obj F1."""
    sp = load_split(EVAL / "test", 40)
    i = 7
    z = prep(sp["x"][i : i + 1])
    pred = predict(features(z), REGLOG)[0]
    gt = sp["m"][i]
    st = [s for s in sp["sticks"][i] if 0 <= s["row"] < 150 and 0 <= s["col"] < 150]
    s0 = max(st, key=lambda s: abs(s["intensity"]))
    r0, c0 = int(s0["row"]), int(s0["col"])
    sl = (slice(max(r0 - 14, 0), r0 + 14), slice(max(c0 - 14, 0), c0 + 14))
    oy, ox = sl[0].start, sl[1].start
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
    # (a) Dice : pixel exact
    rgb = np.ones(gt[sl].shape + (3,))
    rgb[gt[sl] & pred[sl]] = matplotlib.colors.to_rgb(OK)
    rgb[pred[sl] & ~gt[sl]] = matplotlib.colors.to_rgb(FP)
    rgb[gt[sl] & ~pred[sl]] = matplotlib.colors.to_rgb(FN)
    axs[0].imshow(rgb, origin="lower")
    axs[0].set_title("Dice : 2·|P∩V| / (|P|+|V|)\npixel exact, sans tolérance", fontsize=9.5)
    # (b) tol F1 : ±1 px
    near = ndimage.maximum_filter(gt, 3)[sl]
    rgb = np.ones(gt[sl].shape + (3,))
    rgb[near] = (0.88, 0.88, 0.86)
    rgb[gt[sl]] = (0.55, 0.55, 0.53)
    rgb[pred[sl] & near] = matplotlib.colors.to_rgb(OK)
    rgb[pred[sl] & ~near] = matplotlib.colors.to_rgb(FP)
    axs[1].imshow(rgb, origin="lower")
    axs[1].legend(handles=[Patch(color=OK, label="P à ±1 px d'un pixel de V"),
                           Patch(color=(0.55, 0.55, 0.53), label="V"),
                           Patch(color=(0.88, 0.88, 0.86), label="V dilaté (tolérance)")],
                  frameon=False, fontsize=8, loc="upper left")
    axs[1].set_title("tol F1 : un pixel compte s'il est à ±1 px\n(gris clair = zone tolérée)",
                     fontsize=9.5)
    # (c) obj F1 : un interdot trouvé / un blob correct
    axs[2].imshow(np.clip((z[0][sl] + 3) / 13, 0, 1), cmap="gray", origin="lower")
    pr = np.ma.masked_where(~pred[sl], pred[sl])
    axs[2].imshow(pr, cmap=matplotlib.colors.ListedColormap([COL[REGLOG]]), origin="lower", alpha=0.8)
    for s in st:
        y, x = s["row"] - oy, s["col"] - ox
        if -3 < y < 31 and -3 < x < 31:
            axs[2].add_patch(Circle((x, y), s["len_px"] / 2 + 1, fill=False, ec=OK, lw=1.8))
            axs[2].plot(x, y, "+", color=OK, ms=8, mew=1.5)
    axs[2].set_xlim(-0.5, 27.5)
    axs[2].set_ylim(-0.5, 27.5)
    axs[2].set_title("obj F1 : interdot trouvé si un pixel prédit\ntombe dans le cercle len/2 + 1 px",
                     fontsize=9.5)
    for a in axs:
        clean(a)
    axs[2].legend(handles=[Line2D([], [], color=OK, marker="+", lw=1.8, label="centre + rayon de tolérance"),
                           Patch(color=COL[REGLOG], label="masque prédit")],
                  frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Trois façons de noter le même masque de la régression logistique (zoom 28×28 px, P = prédit, V = vrai)",
                 x=0.01, ha="left", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(OUT / "fig3_metriques_expliquees.png", dpi=120)
    plt.close()


def fig_transfer():
    """Balayage du seuil de la régression logistique : stage 1 contre frames du challenge 2."""
    t = np.array([r["thr"] for r in SWEEP])
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.7))
    for s, lab, c, mk in (("test", "test (challenge 1)", COL[REGLOG], "o"),
                          ("ood_stage2", "frames du challenge 2", "#1baf7a", "^")):
        axs[0].plot(t, [r[s]["obj_f1"] for r in SWEEP], color=c, marker=mk, ms=5, lw=2, label=lab)
    axs[1].plot(t, [r["null_blobs_per_img"] for r in SWEEP], color=INK2, marker="o", ms=5, lw=2)
    for a in axs:
        a.axvline(RES[REGLOG]["thr"], color=INK2, ls=":", lw=1.5)
        a.grid(color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.set_xlabel("seuil sur le logit")
    axs[0].text(RES[REGLOG]["thr"] + 0.08, 0.3, "seuil choisi\nsur val", fontsize=8, color=INK2)
    axs[0].set_ylabel("obj F1")
    axs[0].set_ylim(0, 1)
    axs[0].legend(frameon=False, loc="upper left")
    axs[0].set_title("Le seuil optimal n'est pas le même au challenge 2", loc="left", fontsize=10)
    axs[1].set_yscale("log")
    axs[1].set_ylabel("blobs / scène vide")
    axs[1].set_title("Fausses alarmes sur des scènes sans interdot", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig8_transfert_seuil.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    fig_pipeline()
    print("pipeline")
    fig_masks("test", [3, 7, 12, 21, 30], "fig2_masques_test.png",
              "Masques prédits sur le test (seuil choisi sur val, jamais retouché)")
    fig_masks("ood_stage2", [1, 5, 9, 14], "fig2b_masques_challenge2.png",
              "Masques prédits sur des frames du challenge 2 (faible contraste)")
    print("masks")
    fig_metric_explained()
    fig_boundary()
    fig_metrics()
    fig_robustness()
    fig_recall()
    fig_transfer()
    print("done")
