"""One official-generator scene, scored by method M5 (logistic regression).

Fit matches detection.baselines: first 500 images of data/train, seed 0.
Threshold 2.50 is the value frozen on data/val in that protocol.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from csd.config import GENERATOR
from csd.dataset import _stick_record
from csd.generator import build_interdots, generate_label, render_csd, scene_window
from detection.baselines import LogReg, load_split, pixel_sweep, prep
from detection.metrics import object_scores

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED_SCENE = 4242
THR = 2.50
N_FIT = 500


def stick_hits(pred: np.ndarray, sticks: list[dict]) -> list[dict]:
    h, w = pred.shape
    pix = np.argwhere(pred)
    rows = []
    for s in sticks:
        center = np.array([s["row"], s["col"]])
        inside = bool((0 <= center[0] < h) and (0 <= center[1] < w))
        if len(pix):
            d = float(np.sqrt(((center - pix) ** 2).sum(-1)).min())
        else:
            d = float("inf")
        hit = inside and d <= s["len_px"] / 2 + 1.0
        rows.append(
            {
                "row": round(float(s["row"]), 2),
                "col": round(float(s["col"]), 2),
                "len_px": round(float(s["len_px"]), 2),
                "intensity": round(float(s["intensity"]), 3),
                "inside": inside,
                "dist_px": None if d == float("inf") else round(d, 2),
                "found": bool(hit),
            }
        )
    return rows


def main() -> None:
    train = load_split(ROOT / "data" / "train", N_FIT)
    t0 = time.perf_counter()
    meth = LogReg().fit(prep(train["x"]), train["m"], seed=0)
    fit_s = time.perf_counter() - t0

    np.random.seed(SEED_SCENE)
    win = scene_window(GENERATOR)
    interdots, line_specs = build_interdots(GENERATOR, win)
    image = render_csd(
        interdots=interdots,
        line_specs=line_specs,
        scan_window=win,
        config=GENERATOR,
        normalize=False,
    ).astype(np.float32)
    label = generate_label(interdots, win, GENERATOR)
    mask = label > 0.5
    step = win.step_h
    sticks = []
    for s in interdots:
        rec = _stick_record(s)
        sticks.append(
            {
                "col": rec["x"] / step,
                "row": rec["y"] / step,
                "len_px": rec["length"] / step,
                "intensity": rec["intensity"],
                "theta": rec["theta"],
            }
        )

    z = prep(image[None])
    t1 = time.perf_counter()
    scores = meth.scores(z)
    dt_ms = (time.perf_counter() - t1) * 1e3
    pred = meth.binarize(scores, THR)[0]
    px = pixel_sweep(scores, mask[None], meth, [THR])[0]
    ob = object_scores(pred[None], [sticks], *pred.shape)
    hits = stick_hits(pred, sticks)

    gtd = ndimage.binary_dilation(mask, np.ones((3, 3)))
    pdl = ndimage.binary_dilation(pred, np.ones((3, 3)))
    overlay = np.ones(mask.shape + (3,))
    overlay[pred & gtd] = (0.10, 0.70, 0.20)
    overlay[pred & ~gtd] = (0.90, 0.15, 0.15)
    overlay[mask & ~pdl] = (0.15, 0.30, 0.95)

    s = scores[0]
    fig, axs = plt.subplots(2, 3, figsize=(12.2, 8.2))
    panels = [
        (axs[0, 0], image, "Image brute (générateur)", "gray", np.percentile(image, 1), np.percentile(image, 99)),
        (axs[0, 1], z[0], "Prétraitée (unités de σ)", "gray", -3, 10),
        (axs[0, 2], s, "Carte de score M5 (logit)", "magma", np.percentile(s, 1), np.percentile(s, 99.5)),
        (axs[1, 0], pred, f"Masque prédit (seuil {THR:g})", "gray_r", 0, 1),
        (axs[1, 1], mask, "Vérité (générateur)", "gray_r", 0, 1),
        (axs[1, 2], overlay, "Vert = bon, rouge = FP, bleu = manqué", None, None, None),
    ]
    for ax, img, title, cmap, vmin, vmax in panels:
        if cmap is None:
            ax.imshow(img, interpolation="nearest")
        else:
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    n_found = sum(h["found"] for h in hits)
    n_in = sum(h["inside"] for h in hits)
    fig.suptitle(
        f"M5 sur une scène du générateur (seed {SEED_SCENE}) — "
        f"{n_found}/{n_in} sticks au centre dans le cadre",
        fontsize=13,
    )
    fig.tight_layout()
    fig_path = OUT / "m5_scene.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    # standalone transformed outputs
    fig2, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(overlay, interpolation="nearest")
    ax.set_title("M5 — masque superposé à la vérité")
    ax.set_xticks([])
    ax.set_yticks([])
    fig2.tight_layout()
    fig2.savefig(OUT / "m5_overlay.png", dpi=140)
    plt.close(fig2)

    report = {
        "scene": {
            "generator": "csd.generate (GENERATOR officiel)",
            "seed": SEED_SCENE,
            "shape": list(image.shape),
            "span_V": win.span_h,
            "step_V": win.step_h,
            "n_sticks": len(sticks),
            "n_sticks_inside": n_in,
            "amplitude_abs_min": round(float(min(abs(s["intensity"]) for s in sticks)), 3),
            "amplitude_abs_max": round(float(max(abs(s["intensity"]) for s in sticks)), 3),
            "gt_positive_pixels": int(mask.sum()),
        },
        "method": {
            "name": "M5_logreg",
            "fit_images": N_FIT,
            "fit_source": "data/train (premières images, seed dataset 0)",
            "fit_seconds": round(fit_s, 2),
            "threshold": THR,
            "threshold_source": "figé sur data/val dans le protocole des baselines (max tol-F1)",
            "weights_standardized": [round(float(w), 4) for w in meth.w],
            "feature_names": ["z", "smooth_s1", "smooth_s2", "matched", "ridge", "local_std", "intercept"],
            "feature_mean": [round(float(v), 4) for v in meth.mu],
            "feature_std": [round(float(v), 4) for v in meth.sd],
            "ms": round(dt_ms, 1),
        },
        "pixel": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in px.items()},
        "object": {
            "precision": round(ob["obj_precision"], 4),
            "recall": round(ob["obj_recall"], 4),
            "f1": round(ob["obj_f1"], 4),
            "n_gt_sticks": ob["n_gt_sticks"],
            "n_pred_blobs": ob["n_pred_blobs"],
            "n_found": n_found,
            "recall_by_amplitude": {
                k: (None if v[0] is None else round(v[0], 3), v[1])
                for k, v in ob["recall_by_amplitude"].items()
            },
        },
        "sticks": hits,
        "pred_positive_pixels": int(pred.sum()),
    }
    (OUT / "compte_rendu.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "compte_rendu.md").write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"fig": str(fig_path), "fit_s": fit_s, "ms": dt_ms, "obj": report["object"], "pixel": report["pixel"]}, indent=2))


def render_md(r: dict) -> str:
    sc, m, px, ob = r["scene"], r["method"], r["pixel"], r["object"]
    wnames = m["feature_names"]
    weights = ", ".join(f"{n}={w:+.3f}" for n, w in zip(wnames, m["weights_standardized"]))
    lines = [
        "# Compte rendu : méthode 5 sur une image du générateur",
        "",
        "M5 est une régression logistique sur six cartes (image en unités de σ, deux lissages, filtre adapté, crête hessienne, écart-type local). Le seuil décide ensuite quels pixels sont des interdots.",
        "",
        "## Scène",
        "",
        f"- Générateur officiel `csd`, graine {sc['seed']} (distincte de train=0, val=999, test=2025).",
        f"- Grille {sc['shape'][0]}×{sc['shape'][1]} px, fenêtre {sc['span_V']} V, pas {sc['step_V']} V.",
        f"- {sc['n_sticks']} sticks, dont {sc['n_sticks_inside']} ont leur centre dans le cadre.",
        f"- |amplitude| de {sc['amplitude_abs_min']} à {sc['amplitude_abs_max']}.",
        f"- Masque officiel : {sc['gt_positive_pixels']} pixels positifs (seuil 0,5 sur l'étiquette floutée).",
        "",
        "## Ajustement",
        "",
        f"- Fit sur les {m['fit_images']} premières images de `data/train` ({m['fit_seconds']} s).",
        "- Positifs rares (~0,35 %) : tous les pixels positifs, plus 20 négatifs tirés au hasard par positif. L'ordonnée à l'origine est donc mal calibrée. Le seuil, choisi sur la validation, absorbe ce biais.",
        f"- Seuil retenu : **{m['threshold']}** ({m['threshold_source']}).",
        f"- Poids sur variables standardisées : {weights}.",
        f"- Temps de score sur cette image : {m['ms']} ms.",
        "",
        "## Métriques sur cette image",
        "",
        "| critère | valeur |",
        "|---|---|",
        f"| F1 pixel strict | {px['f1']:.3f} |",
        f"| IoU strict | {px['iou']:.3f} |",
        f"| précision tolérante (1 px) | {px['tol_precision']:.3f} |",
        f"| rappel tolérant (1 px) | {px['tol_recall']:.3f} |",
        f"| F1 tolérant | {px['tol_f1']:.3f} |",
        f"| précision objet | {ob['precision']:.3f} |",
        f"| rappel objet | {ob['recall']:.3f} ({ob['n_found']}/{ob['n_gt_sticks']}) |",
        f"| F1 objet | {ob['f1']:.3f} |",
        f"| blobs prédits | {ob['n_pred_blobs']} |",
        f"| pixels prédits | {r['pred_positive_pixels']} |",
        "",
        "Rappel objet par |amplitude| (nombre de sticks du bin entre parenthèses) :",
        "",
        "| bin | rappel | n |",
        "|---|---|---|",
    ]
    for k, (rec, n) in ob["recall_by_amplitude"].items():
        lines.append(f"| {k} | {'n/a' if rec is None else f'{rec:.2f}'} | {n} |")
    lines += [
        "",
        "## Sticks",
        "",
        "Un stick dont le centre est dans le cadre est compté trouvé si un pixel prédit tombe à moins de longueur/2 + 1 px de ce centre.",
        "",
        "| # | ligne | colonne | longueur (px) | intensité | distance (px) | trouvé |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, h in enumerate(r["sticks"], 1):
        dist = "n/a" if h["dist_px"] is None else f"{h['dist_px']:.2f}"
        status = "oui" if h["found"] else ("hors cadre" if not h["inside"] else "non")
        lines.append(
            f"| {i} | {h['row']:.1f} | {h['col']:.1f} | {h['len_px']:.1f} | {h['intensity']:.2f} | {dist} | {status} |"
        )
    lines += [
        "",
        "## Lecture",
        "",
        "Le vert marque un pixel prédit à ≤1 px d'un pixel du masque officiel. Le rouge est un faux positif (souvent une ligne de charge, non étiquetée). Le bleu est un pixel de vérité qu'aucun pixel prédit ne couvre à 1 px.",
        "",
        "L'IoU strict reste sensible à la rastérisation du masque officiel (rectangle flou seuillé à 0,5, souvent fragmenté). Le F1 objet répond à la question physique : l'interdot est-il localisé ?",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
