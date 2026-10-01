"""Robustness suite for challenge 1: realistic acquisition artefacts the generator never makes.

    uv run python -m detection.robustness make                 # numpy only, ~1 min, ~470 MB
    uv run python -m detection.robustness eval-m5              # classical M5 / M2, CPU
    python -m detection.evaluate --runs "runs/fast_unet*" --eval-dir data/robustness ...   # DL

Every set = the same N test images (official generator, seeds 2e7+k) + ONE perturbation at
one severity level. Masks and stick metadata are unchanged, so any drop is caused by the
perturbation alone. Report-only: never fit, select a threshold or a checkpoint on these.
Each perturbation is a lab artefact absent from csd.generator (noise there = white + row
stripes, blurred):

  white      extra white noise                  std  x [0.5, 1, 2] sigma_pix  (shorter integration)
  pink       1/f noise along the raster time     std  x [0.5, 1, 2] sigma_pix  (charge noise, amplifier)
  drift      slow polynomial background in time  p-p  [1.5, 3, 6]               (sensor-dot drift)
  jumps      random telegraph switches           amp  [1.5, 3, 6], ~4 / frame   (charge trap near sensor)
  stripes    extra per-row offsets               std  x [0.5, 1, 2] sigma_h     (line-to-line jitter)
  lowpass    1-pole filter along the fast axis   tau  [0.5, 1, 2] px            (lock-in time constant)
  spikes     isolated outlier pixels (+-10)      frac [0.1, 0.5, 2] %           (glitches, pickup)
  saturate   sensor non-linearity c*tanh(x/c)    c    [10, 5, 2.5]              (finite Coulomb-peak flank)
  polarity   sign flip                           —                              (other flank of the peak)

sigma_pix = 0.9, sigma_h = 0.7 (generator defaults). Time = raster order (rows = fast axis).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SIG_PIX, SIG_H = 0.9, 0.7
LEVELS = {
    "white": [0.5, 1.0, 2.0],
    "pink": [0.5, 1.0, 2.0],
    "drift": [1.5, 3.0, 6.0],
    "jumps": [1.5, 3.0, 6.0],
    "stripes": [0.5, 1.0, 2.0],
    "lowpass": [0.5, 1.0, 2.0],
    "spikes": [0.001, 0.005, 0.02],
    "saturate": [10.0, 5.0, 2.5],
    "polarity": [1.0],
}


def _pink(rng, t):
    f = np.fft.rfftfreq(t)
    spec = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / np.sqrt(np.maximum(f, 1.0 / t))
    x = np.fft.irfft(spec, n=t)
    return x / x.std()


def perturb(x: np.ndarray, kind: str, level: float, rng: np.random.Generator) -> np.ndarray:
    """x: one raw (H, W) image. Returns a new perturbed image."""
    h, w = x.shape
    t = h * w
    if kind == "white":
        return x + level * SIG_PIX * rng.normal(size=x.shape)
    if kind == "pink":
        return x + level * SIG_PIX * _pink(rng, t).reshape(h, w)
    if kind == "drift":
        tt = np.linspace(-1, 1, t)
        c = rng.uniform(-1, 1, 3)
        d = c[0] * tt + c[1] * (tt**2 - 1 / 3) + c[2] * (tt**3 - 0.6 * tt)
        d = d - d.min()
        return x + level * (d / max(d.max(), 1e-9) - 0.5).reshape(h, w)
    if kind == "jumps":
        cuts = np.sort(rng.integers(0, t, size=rng.poisson(4)))
        lev, state, prev = np.zeros(t), rng.integers(0, 2), 0
        for c in list(cuts) + [t]:
            lev[prev:c] = state * level
            state, prev = 1 - state, c
        return x + (lev - lev.mean()).reshape(h, w)
    if kind == "stripes":
        return x + level * SIG_H * rng.normal(size=(h, 1))
    if kind == "lowpass":  # causal 1-pole filter along the fast axis (scan direction = +col)
        a = np.exp(-1.0 / level)
        y = np.empty_like(x)
        y[:, 0] = x[:, 0]
        for j in range(1, w):
            y[:, j] = a * y[:, j - 1] + (1 - a) * x[:, j]
        return y
    if kind == "spikes":
        y = x.copy()
        k = rng.random(x.shape) < level
        y[k] += rng.choice([-10.0, 10.0], size=int(k.sum()))
        return y
    if kind == "saturate":
        med = np.median(x)
        return med + level * np.tanh((x - med) / level)
    if kind == "polarity":
        return 2 * np.median(x) - x
    raise ValueError(kind)


FMT = {  # set folder name = perturbation + its physical parameter value
    "white": "white_x{:g}", "pink": "pink1f_x{:g}", "drift": "drift_pp{:g}",
    "jumps": "jumps_amp{:g}", "stripes": "stripes_x{:g}", "lowpass": "lowpass_tau{:g}px",
    "spikes": "spikes_{:g}pct", "saturate": "saturate_c{:g}", "polarity": "polarity_flip",
}


def set_name(kind: str, level: float) -> str:
    return FMT[kind].format(100 * level if kind == "spikes" else level)


def set_names() -> list[str]:
    return [set_name(k, lv) for k, v in LEVELS.items() for lv in v]


README = """# Suite de robustesse — challenge 1 (détection d'interdots)

Généré par `uv run python -m detection.robustness make` (code : challenge1/detection/robustness.py).
**Rapport uniquement** : ne jamais ajuster, choisir un seuil ou un checkpoint sur ces jeux.

Chaque dossier = les mêmes {n} images de test (générateur officiel, seeds 2e7+k) + UNE perturbation
à un niveau. Masques (`masks.npy`) et sticks (`sticks.jsonl`) inchangés : toute baisse vient de
la perturbation seule. `clean/` = ces mêmes images sans perturbation (la référence).
Format identique aux jeux d'éval : images.npy (N,150,150) float32 brut, masks.npy, sticks.jsonl, meta.json.

| dossier | artefact de mesure | paramètre |
|---|---|---|
| white_x* | bruit blanc ajouté (intégration plus courte) | écart-type × σ_pix (0,9) |
| pink1f_x* | bruit 1/f le long du temps de scan (bruit de charge, ampli) | écart-type × σ_pix |
| drift_pp* | dérive lente du fond (point capteur) | amplitude crête à crête |
| jumps_amp* | sauts télégraphiques ~4/image (piège de charge) | amplitude du saut |
| stripes_x* | décalages ligne à ligne en plus | écart-type × σ_h (0,7) |
| lowpass_tau*px | filtre passe-bas 1 pôle le long de l'axe rapide (constante de temps lock-in) | τ en pixels |
| spikes_*pct | pixels aberrants ±10 (glitchs, parasites) | % de pixels touchés |
| saturate_c* | non-linéarité capteur c·tanh(x/c) (flanc fini du pic de Coulomb) | c (plus petit = plus saturé) |
| polarity_flip | signe inversé (autre flanc du pic) | — |

Temps = ordre raster (lignes = axe de scan rapide). Pour les modèles DL, `val/` et `test/` sont
des liens vers les jeux d'éval habituels (requis par detection.evaluate, seuil choisi sur val).
"""


def make(src: Path, out: Path, n: int) -> None:
    img = np.load(src / "images.npy", mmap_mode="r")[:n].astype(np.float64)
    masks = np.load(src / "masks.npy", mmap_mode="r")[:n]
    sticks = (src / "sticks.jsonl").read_text().splitlines()[:n]
    meta0 = json.loads((src / "meta.json").read_text())
    for si, (kind, levels) in enumerate(LEVELS.items()):
        for li, level in enumerate(levels):
            name = set_name(kind, level)
            rng = np.random.default_rng(50_000_000 + 1000 * si + li)
            x = np.stack([perturb(im, kind, level, rng) for im in img]).astype(np.float32)
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "images.npy", x)
            np.save(d / "masks.npy", np.asarray(masks))
            (d / "sticks.jsonl").write_text("\n".join(sticks) + "\n")
            meta = {**meta0, "name": name, "n": n, "kind": "robustness", "perturbation": kind,
                    "level": level, "source": str(src)}
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"  {name:12s} added std {float((x - img).std()):.2f}", flush=True)
    # the unperturbed reference (same n images) so drops are read against the same subset
    d = out / "clean"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "images.npy", img.astype(np.float32))
    np.save(d / "masks.npy", np.asarray(masks))
    (d / "sticks.jsonl").write_text("\n".join(sticks) + "\n")
    (d / "meta.json").write_text(json.dumps({**meta0, "name": "clean", "n": n}, indent=2))
    (out / "README.md").write_text(README.format(n=n), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(
        {"source": str(src), "n": n, "sets": {set_name(k, lv): {"perturbation": k, "level": lv}
                                               for k, v in LEVELS.items() for lv in v}}, indent=1))
    print(f"wrote {len(set_names()) + 1} sets x {n} images -> {out}")


def table(res: dict[str, dict[str, float]], metric_name: str) -> str:
    """res: method -> set name -> metric. Rows = methods, columns = perturbation levels."""
    lines = [f"### {metric_name} (clean = mêmes images sans perturbation)", "",
             "| méthode | clean | " + " | ".join(
                 f"{k} " + "/".join(f"{v:g}" for v in lv) for k, lv in LEVELS.items()) + " | **robustesse** |",
             "|---|---|" + "---|" * (len(LEVELS) + 1)]
    for m, r in res.items():
        cells, ratios = [], []
        for k, lv in LEVELS.items():
            names = [set_name(k, v) for v in lv]
            vals = [r[s] for s in names]
            ratios += [v / max(r["clean"], 1e-9) for v in vals]
            cells.append(" / ".join(f"{v:.2f}" for v in vals))
        lines.append(f"| {m} | {r['clean']:.3f} | " + " | ".join(cells) + f" | {np.mean(ratios):.3f} |")
    lines += ["", "robustesse = moyenne sur toutes les perturbations et tous les niveaux de "
              "(score perturbé / score clean) ; 1 = insensible."]
    return "\n".join(lines)


def eval_m5(robust: Path, val: Path, train: Path, out: Path) -> None:
    from detection.baselines import (N_FIT, build_methods, counts, features, load_split, prep,
                                     scores_from, select_threshold, stick_table)

    tr = load_split(train, N_FIT, "train_fit")
    methods, _ = build_methods(features(prep(tr["x"])), tr["m"], (0,))
    v = load_split(val)
    v["st"] = stick_table(v["sticks"])
    Fv = features(prep(v["x"]))
    keep = ("M2_matched", "M5_logreg", "M5_min")
    thr = {m: select_threshold(methods[m][0](Fv), methods[m][1], v)[0] for m in keep}
    res = {m: {} for m in keep}
    for s in ["clean"] + set_names():
        sp = load_split(robust / s)
        sp["st"] = stick_table(sp["sticks"])
        F = features(prep(sp["x"]))
        for m in keep:
            fn, binz = methods[m]
            res[m][s] = scores_from(counts(binz(fn(F), thr[m]), sp["m"], sp["st"])[0])["obj_f1"]
        print(f"  {s:12s} " + " ".join(f"{m}={res[m][s]:.3f}" for m in keep), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robustness_m5.json").write_text(json.dumps(res, indent=1))
    md = table(res, "obj-F1")
    (out / "robustness_m5.md").write_text(md + "\n", encoding="utf-8")
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("make")
    a.add_argument("--src", default="data/eval_light/test")
    a.add_argument("--out", default="data/robustness")
    a.add_argument("--n", type=int, default=200)
    b = sub.add_parser("eval-m5")
    b.add_argument("--robust", default="data/robustness")
    b.add_argument("--val", default="data/eval_light/val")
    b.add_argument("--train", default="data/train")
    b.add_argument("--out", default="challenge1/results/robustness")
    args = ap.parse_args()
    if args.cmd == "make":
        make(Path(args.src), Path(args.out), args.n)
    else:
        eval_m5(Path(args.robust), Path(args.val), Path(args.train), Path(args.out))


if __name__ == "__main__":
    main()
