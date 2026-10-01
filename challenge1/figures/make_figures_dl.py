"""Figures de la partie 2 du README (U-Net, TransUNet) — même charte que make_figures.py.

    uv run python challenge1/figures/make_figures_dl.py

Entrées (toutes figées, aucune ne sert à sélectionner) :
  challenge1/results/transformers/logs/*.jsonl   courbes d'apprentissage (runs courts)
  challenge1/results/dl_p4/summary.json          DL : test + OOD (seuil choisi sur val)
  challenge1/results/p4_m5/p4_m5.json            régression logistique : mêmes jeux
  challenge1/results/robustness/*                suite de robustesse (data/robustness)
  data/preds_readme.npz                          masques prédits par les modèles exportés
                                                 (challenge1/models/*.pt, dump sur GPU)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from make_figures import (FN, FP, GRID, INK, INK2, OK, OUT, ROOT, SURF, clean, legend_overlay,  # noqa: E402
                          overlay, predict)

from detection.baselines import features, load_split, prep  # noqa: E402
from detection.robustness import LEVELS, set_name  # noqa: E402

RES = ROOT / "challenge1/results"
LOGS = RES / "transformers/logs"
DL = json.loads((RES / "dl_p4/summary.json").read_text(encoding="utf-8"))
M5 = json.loads((RES / "p4_m5/p4_m5.json").read_text(encoding="utf-8"))["M5_min[official]"]
ROB_M5 = json.loads((RES / "robustness/robustness_m5.json").read_text(encoding="utf-8"))["M5_min"]
ROB_DL = json.loads((RES / "robustness/robustness_dl_summary.json").read_text(encoding="utf-8"))
PRED = np.load(ROOT / "data/preds_readme.npz")
EVAL = ROOT / "data/eval_light"

# couleur = méthode (ordre catégoriel fixe) ; style de trait = variante d'entraînement
REGLOG, UNET, TUNET = "reglog", "unet", "transunet"
COL = {REGLOG: "#2a78d6", UNET: "#eb6834", TUNET: "#1baf7a"}
NAME = {REGLOG: "Régression logistique", UNET: "U-Net", TUNET: "TransUNet"}
LS = {"": "-", "_lowsnr": "--", "_shift": ":"}
VNAME = {"": "base", "_lowsnr": "lowsnr", "_shift": "shift"}


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ 1. données d'entraînement
def fig_synth():
    """Une géométrie du pool, retirée deux fois : intensité et bruit nouveaux à chaque batch."""
    t = np.load(ROOT / "data/pool/templates.npy", mmap_mode="r")[11].astype(np.float32)
    soft = np.load(ROOT / "data/pool/soft.npy", mmap_mode="r")[11] / 255.0
    rng = np.random.default_rng(3)

    def noise():
        n = 0.9 * rng.normal(size=t.shape) + 0.7 * rng.normal(size=(t.shape[0], 1))
        return ndimage.gaussian_filter(n, 0.5, mode="reflect")

    x1, x2 = -20 * t + noise(), -3 * t + noise()
    panels = [(t, "1. template (pool)\nrendu propre / i, sans bruit", "gray_r", None),
              (noise(), "2. bruit retiré sur GPU\nblanc + rayures, flouté", "gray", (-3, 3)),
              (x1, "3. image = i·template + bruit\ni = −20 (contraste fort)", "gray", None),
              (x2, "4. même géométrie\ni = −3 (régime du challenge 2)", "gray", None),
              (soft, "5. cible : label soft\n(pas le masque fragmenté)", "magma", (0, 1))]
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.3))
    for ax, (im, title, cm, lim) in zip(axs, panels):
        kw = {"vmin": lim[0], "vmax": lim[1]} if lim else {}
        ax.imshow(im, cmap=cm, origin="lower", **kw)
        ax.set_title(title, fontsize=9)
        clean(ax)
    fig.suptitle("Données d'entraînement : 30k géométries figées, intensité et bruit infinis",
                 x=0.01, ha="left", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    plt.savefig(OUT / "fig9_donnees_dl.png", dpi=120)
    plt.close()


# ------------------------------------------------------------------ 2. courbes d'apprentissage
def fig_curves():
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for arch in (UNET, TUNET):
        for v in ("", "_lowsnr", "_shift"):
            recs = [json.loads(line) for line in (LOGS / f"{arch}{v}.jsonl").open()]
            tr = [r for r in recs if "loss" in r]
            ev = [r for r in recs if r.get("eval")]
            x = np.array([r["step"] for r in tr]) * 32 / 1e3
            sm = np.convolve([r["loss"] for r in tr], np.ones(5) / 5, mode="valid")
            kw = dict(color=COL[arch], ls=LS[v], lw=2)
            axs[0].plot(x[4:], sm, **kw)
            xe = np.array([r["step"] for r in ev]) * 32 / 1e3
            axs[1].plot(xe, [r["val_obj_f1"] for r in ev], marker="o", ms=4, **kw)
            axs[2].plot(xe, [r["geom_gap"] for r in ev], marker="o", ms=4, **kw)
    axs[0].set(title="Perte d'entraînement (BCE + Dice, lissée)", ylabel="loss", ylim=(0.2, 0.8))
    axs[1].set(title="Val : obj F1 (poids EMA)", ylabel="obj F1", ylim=(0.93, 0.985))
    axs[2].set(title="geom_gap = F1(géométries vues) − F1(val)", ylabel="geom_gap", ylim=(-0.004, 0.025))
    axs[2].axhline(0.02, color=INK2, ls="--", lw=1)
    axs[2].text(10, 0.0205, "seuil d'alerte overfit", fontsize=8, color=INK2, va="bottom")
    axs[2].axhline(0, color=GRID, lw=1)
    for a in axs:
        a.set_xlabel("images vues (milliers)")
        style(a, "both")
    h = [Line2D([], [], color=COL[a], lw=2, label=NAME[a]) for a in (UNET, TUNET)]
    h += [Line2D([], [], color=INK2, ls=LS[v], lw=2, label=f"entraînement {VNAME[v]}") for v in LS]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Courbes d'apprentissage (runs courts, ~30 min de GPU chacun, seed 0)", x=0.01,
                 ha="left", fontsize=11)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(OUT / "fig10_courbes_dl.png", dpi=120)
    plt.close()


# ------------------------------------------------------------------ 3. masques prédits
def fig_masks_dl(split, fname, title):
    sp = load_split(EVAL / split)
    ids = list(PRED[f"{split}__ids"])
    z = prep(sp["x"][ids])
    reg = predict(features(z), "M5_min")
    cols = [("image", None), ("masque vrai\n(générateur)", None), (NAME[REGLOG], reg),
            (NAME[UNET] + " (lowsnr)", PRED[f"{split}__unet_lowsnr"]),
            (NAME[TUNET] + " (lowsnr)", PRED[f"{split}__transunet_lowsnr"])]
    fig, axs = plt.subplots(len(ids), 5, figsize=(12, 2.5 * len(ids) + 0.6))
    for r, i in enumerate(ids):
        gt = sp["m"][i]
        n_in = sum(0 <= s["row"] < 150 and 0 <= s["col"] < 150 for s in sp["sticks"][i])
        amp = np.median([abs(s["intensity"]) for s in sp["sticks"][i]]) if sp["sticks"][i] else 0
        axs[r, 0].imshow(z[r], cmap="gray", vmin=-3, vmax=10, origin="lower")
        axs[r, 0].set_ylabel(f"scène #{i}\n{n_in} interdots, |i| méd. {amp:.1f}", fontsize=8)
        axs[r, 1].imshow(gt, cmap="gray_r", origin="lower")
        for c, (_, pr) in enumerate(cols[2:]):
            p = pr[r]
            axs[r, 2 + c].imshow(overlay(z[r], p, gt), origin="lower")
            dice = 2 * (p & gt).sum() / max(p.sum() + gt.sum(), 1)
            axs[r, 2 + c].set_xlabel(f"Dice {dice:.2f}", fontsize=8, color=INK2)
    for c, (t, _) in enumerate(cols):
        axs[0, c].set_title(t, fontsize=10)
    for a in axs.ravel():
        clean(a)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    plt.tight_layout(rect=(0, 0.035, 1, 0.98))
    legend_overlay(fig)
    plt.savefig(OUT / fname, dpi=110)
    plt.close()


# ------------------------------------------------------------------ 4. la suite de robustesse
ROB_SHOW = [("clean", "référence"), ("white_x2", "bruit blanc ×2"), ("pink1f_x2", "1/f ×2"),
            ("drift_pp6", "dérive 6"), ("jumps_amp6", "sauts ×6"), ("stripes_x2", "rayures ×2"),
            ("lowpass_tau2px", "passe-bas τ=2 px"), ("spikes_0.5pct", "spikes 0,5 %"),
            ("saturate_c2.5", "saturation c=2,5"), ("polarity_flip", "polarité inversée")]


def fig_perturbations():
    k = 0  # scène #7 du test (premier indice dumpé)
    fig, axs = plt.subplots(3, len(ROB_SHOW), figsize=(19, 6.4))
    for c, (s, lab) in enumerate(ROB_SHOW):
        sp = load_split(ROOT / "data/robustness" / s, 8)
        i = int(PRED[f"rob_{s}__ids"][k])
        gt = sp["m"][i]
        x = PRED[f"rob_{s}__x"][k]
        g = (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6)
        axs[0, c].imshow(-g if s != "polarity_flip" else g, cmap="gray", vmin=-3, vmax=10, origin="lower")
        axs[0, c].set_title(lab, fontsize=9)
        z = prep(x[None])[0]
        reg = predict(features(z[None]), "M5_min")[0]
        axs[1, c].imshow(overlay(z, reg, gt), origin="lower")
        axs[2, c].imshow(overlay(z, PRED[f"rob_{s}__unet_lowsnr"][k], gt), origin="lower")
    axs[0, 0].set_ylabel("image brute", fontsize=9)
    axs[1, 0].set_ylabel(NAME[REGLOG], fontsize=9)
    axs[2, 0].set_ylabel(NAME[UNET] + " (lowsnr)", fontsize=9)
    for a in axs.ravel():
        clean(a)
    fig.suptitle("Suite de robustesse : la même scène de test (#7) sous chaque artefact, niveau le plus fort",
                 x=0.01, ha="left", fontsize=11)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    legend_overlay(fig)
    plt.savefig(OUT / "fig11_suite_robustesse.png", dpi=100)
    plt.close()


def fig_robustness_curves():
    series = [(REGLOG, "", lambda s: ROB_M5[s]),
              (UNET, "_lowsnr", lambda s: ROB_DL["fast_unet_lowsnr"]["sets"][s]["obj_f1"][0]),
              (UNET, "_shift", lambda s: ROB_DL["fast_unet_shift"]["sets"][s]["obj_f1"][0]),
              (TUNET, "_lowsnr", lambda s: ROB_DL["fast_transunet_lowsnr"]["sets"][s]["obj_f1"][0])]
    kinds = [k for k in LEVELS if k != "polarity"]
    labels = {"white": "bruit blanc (× σ_pix)", "pink": "bruit 1/f (× σ_pix)", "drift": "dérive (crête-crête)",
              "jumps": "sauts de charge (amplitude)", "stripes": "rayures (× σ_h)",
              "lowpass": "passe-bas τ (px)", "spikes": "spikes (% pixels)", "saturate": "saturation c (→ plus fort)"}
    fig, axs = plt.subplots(3, 3, figsize=(13, 10))
    for ax, k in zip(axs.ravel(), kinds + ["polarity"]):
        if k == "polarity":
            vals = [f("polarity_flip") for _, _, f in series]
            ax.bar(range(len(series)), vals, color=[COL[a] for a, _, _ in series], edgecolor=SURF, lw=2)
            for j, v in enumerate(vals):
                ax.text(j, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
            ax.set_xticks(range(len(series)), [f"{NAME[a]}\n{VNAME[v]}" if a != REGLOG else NAME[a]
                                               for a, v, _ in series], fontsize=7)
            ax.set_title("polarité inversée", fontsize=10)
            ax.set_ylim(0, 1.05)
            style(ax)
            continue
        lv = LEVELS[k]
        xs = [0] + [100 * v if k == "spikes" else v for v in lv]
        for a, v, f in series:
            ys = [f("clean")] + [f(set_name(k, x)) for x in lv]
            ax.plot(range(len(xs)), ys, color=COL[a], ls=LS[v], lw=2, marker="o", ms=5, mec=SURF)
        ax.set_xticks(range(len(xs)), ["0"] + [f"{x:g}" for x in xs[1:]])
        ax.set_title(labels[k], fontsize=10)
        ax.set_ylim(0, 1.0)
        style(ax)
    for a in axs[:, 0]:
        a.set_ylabel("obj F1")
    h = [Line2D([], [], color=COL[a], ls=LS[v], lw=2,
                label=NAME[a] if a == REGLOG else f"{NAME[a]} {VNAME[v]}") for a, v, _ in series]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("obj F1 selon la sévérité de chaque artefact (200 scènes de test, seuil figé sur val)",
                 x=0.01, ha="left", fontsize=11)
    plt.tight_layout(rect=(0, 0.04, 1, 0.98))
    plt.savefig(OUT / "fig12_robustesse_dl.png", dpi=110)
    plt.close()


# ------------------------------------------------------------------ 5. OOD + limite de détection
OOD = [("test", "test"), ("ood_theta_shift", "pente +0,35"), ("ood_zoom_out", "zoom 3 mV"),
       ("ood_noise_up", "bruit ×1,5"), ("ood_noise_1f", "1/f + dérive"),
       ("ood_charge_jump", "sauts de charge"), ("ood_stage2", "frames C2")]


def fig_ood():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(OOD))
    ax.plot(x, [M5[s]["obj_f1"] for s, _ in OOD], color=COL[REGLOG], lw=2, marker="o", ms=7,
            mec=SURF, label=NAME[REGLOG])
    for arch in (UNET, TUNET):
        for v in ("", "_lowsnr", "_shift"):
            g = DL[f"fast_{arch}{v}"]["sets"]
            ax.plot(x, [g[s]["obj_f1"][0] for s, _ in OOD], color=COL[arch], ls=LS[v], lw=2,
                    marker="o", ms=5, mec=SURF, label=f"{NAME[arch]} {VNAME[v]}")
    ax.set_xticks(x, [lab for _, lab in OOD])
    ax.axvspan(0.5, len(OOD) - 0.5, color=GRID, alpha=0.3, lw=0)
    ax.text(3.5, 0.6, "jeux décalés : rapport uniquement", ha="center", fontsize=8, color=INK2)
    ax.set_ylabel("obj F1 (seuil figé sur val)")
    ax.set_ylim(0.55, 1.0)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    ax.set_title("Test et jeux décalés : régression logistique contre U-Net / TransUNet", loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "fig13_ood_dl.png", dpi=120)
    plt.close()


def fig_recall_dl():
    bins = ["[0,2)", "[2,4)", "[4,8)", "[8,16)", "[16,inf)"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, s, t in ((axs[0], "test", "test (challenge 1)"), (axs[1], "ood_noise_1f", "test + bruit 1/f")):
        r5 = M5[s]["recall_by_amplitude"]
        ax.plot(range(5), r5, color=COL[REGLOG], lw=2, marker="o", ms=7, mec=SURF, label=NAME[REGLOG])
        for arch in (UNET, TUNET):
            rb = DL[f"fast_{arch}_lowsnr"]["sets"][s]["recall_by_amplitude"]
            ax.plot(range(5), [rb[b] for b in bins], color=COL[arch], ls="--", lw=2, marker="o", ms=6,
                    mec=SURF, label=f"{NAME[arch]} lowsnr")
        ax.set_xticks(range(5), ["0–2", "2–4", "4–8", "8–16", "> 16"])
        ax.set_xlabel("|amplitude du stick| (σ_pix = 0,9)")
        ax.set_title(t, fontsize=10, loc="left")
        style(ax)
    axs[0].set_ylabel("rappel objet")
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Limite de détection : le DL gagne surtout entre 2σ et 8σ", x=0.01, ha="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "fig14_rappel_amplitude_dl.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    fig_synth()
    fig_curves()
    fig_masks_dl("test", "fig15_masques_test_dl.png",
                 "Masques prédits sur le test : régression logistique, U-Net, TransUNet (seuils figés sur val)")
    fig_masks_dl("ood_stage2", "fig16_masques_challenge2_dl.png",
                 "Masques prédits sur des frames du challenge 2 (faible contraste)")
    fig_perturbations()
    fig_robustness_curves()
    fig_ood()
    fig_recall_dl()
    print("done")
