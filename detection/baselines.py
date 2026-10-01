"""Challenge 1 — five cheap, classical detectors + a safeguarded protocol (v2).

    uv run python -m detection.baselines            # ~2 min on a laptop CPU, no torch needed
    uv run python -m detection.baselines --quick    # ~20 s smoke run (subsets, noisier numbers)

The first run renders the frozen evaluation sets once (~4 min on 8 cores, cached in
``data/eval_light``). They are *prefixes* of the sets of ``detection/data_gen.py`` (same
seed scheme), so these numbers are directly comparable with the transformer runs.

Purpose: honest first numbers on CPU, and a floor the neural detectors must beat.

Methods (each outputs a score map; M1-M3 are in units of noise sigma):

  M0 empty    predicts nothing                                         (metric floor)
  M1 smooth   destripe + robust z-score + Gaussian smoothing           (no shape prior)
  M2 matched  bank of oriented elongated templates (theta ~ pi/4)      (shape prior)
  M3 ridge    Hessian ridge filter (most negative curvature)           (local geometry)
  M4 hyst     M1 + hysteresis + small-blob removal                     (spatial coherence)
  M5 logreg   logistic regression on multi-scale features, *without* the raw pixel
              (decided a priori: with it, the fit builds a z - s1 high-pass that copies
              the rasterisation of the official mask and the simulator PSF — see v1).
              Ablations M5_full (with z) and M5_min (matched + s2) are reported.

Safeguards (each one answers a trap; cf. PROTOCOL.md section 1 and the README rules):

  S1  data       fit = data/train (official generate_dataset, seed 0); val / test / OOD =
                 data_gen seeds (1e7+k, 2e7+k, 3e7+...). Disjointness checked by hashing
                 every image. The v1 test set (seed 2025) is *burned* (we looked at it) and
                 is not used any more.
  S2  selection  one free parameter per method — the threshold — chosen on val only, by
                 sel = (obj_F1 + tol_F1) / 2. Neither alone is safe: obj_F1 can be gamed by
                 fat blobs, pixel metrics by matching the mask's arbitrary width. Every other
                 hyper-parameter is fixed a priori from the physics (stick ~5 px x 1 px,
                 theta ~ pi/4). A warning is raised if the threshold lands on the grid edge
                 (the v1 bug).
  S3  no leakage test is scored once with the frozen val threshold; OOD sets are report-only.
                 Methods only ever see the image; stage-2 labels come from simulator
                 internals and are used for *evaluation only*.
  S4  statistics 95 % bootstrap CIs over test images, paired bootstrap of every method vs M2,
                 and 3 fitting seeds for M5 (std reported).
  S5  gaming     width ratio (predicted px / mask px), blobs per found stick, and false
                 blobs per image on stick-free scenes (p_appear = 0): a lab's false-alarm rate.
  S6  metric     our fast per-image object counts are checked against
                 detection.metrics.object_scores (parity assertion).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from detection.metrics import INTENSITY_BINS, object_scores

SETS = ("val", "test", "ood_theta_shift", "ood_theta_wide", "ood_noise_up", "ood_zoom_in",
        "ood_zoom_out", "ood_stage2")
SEED_NULL = 39_000_000  # stick-free scenes; outside every data_gen seed range
N_FIT, N_VAL, N_TEST, N_OOD, N_NULL = 500, 300, 400, 150, 100
N_BOOT = 1000


# =================================================================== data (S1)
def load_split(path: Path, n: int | None = None, name: str | None = None) -> dict:
    meta = json.loads((path / "meta.json").read_text())
    x = np.asarray(np.load(path / "images.npy", mmap_mode="r")[:n], np.float32)
    m = np.asarray(np.load(path / "masks.npy", mmap_mode="r")[:n]).astype(bool)
    step = meta["scan_window"]["step_h"]
    sticks = []
    with (path / "sticks.jsonl").open() as f:
        for line, _ in zip(f, range(len(x))):
            ss = json.loads(line)["sticks"]
            if ss and "len_px" not in ss[0]:  # csd.generate_dataset stores volts
                ss = [{"col": s["x"] / step, "row": s["y"] / step, "len_px": s["length"] / step,
                       "intensity": s["intensity"]} for s in ss]
            sticks.append(ss)
    return {"name": name or path.name, "x": x, "m": m, "sticks": sticks}


def _null_sample(seed):
    from csd import GENERATOR
    from csd.generator import scene_window
    from detection.data_gen import _generator_sample

    cfg = replace(GENERATOR, p_appear=0.0)
    return _generator_sample((seed, asdict(cfg), asdict(scene_window(cfg))))


def _stage2_or_none(seed):
    """csd.new_experiment crashes on devices drawn with no stick at all (optimum() over an
    empty region list). Such seeds are skipped deterministically and logged in meta.json."""
    from detection.data_gen import _stage2_sample

    try:
        return _stage2_sample(seed)
    except ValueError:
        return None


def _write_set(d: Path, samples: list[dict], meta: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "images.npy", np.stack([s["image"] for s in samples]))
    np.save(d / "masks.npy", np.stack([s["mask"] for s in samples]))
    with (d / "sticks.jsonl").open("w") as f:
        for k, s in enumerate(samples):
            f.write(json.dumps({"image_id": k, "sticks": s["sticks"]}) + "\n")
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  {d.name}: {len(samples)} samples")


def ensure_eval_sets(root: Path, workers: int) -> None:
    from detection.data_gen import eval_set_specs, make_sets

    gen_sets = [s for s in SETS if s != "ood_stage2"]
    missing = [s for s in gen_sets if not (root / s / "images.npy").exists()]
    if missing:
        print(f"rendering frozen sets {missing} -> {root} (once)", flush=True)
        make_sets(root, workers, N_VAL, N_TEST, N_OOD, only=set(missing))
    d = root / "ood_stage2"
    if not (d / "images.npy").exists():
        base = eval_set_specs(N_VAL, N_TEST, N_OOD)["ood_stage2"]["seed"]
        seeds = list(range(base, base + 2 * N_OOD))
        with Pool(workers) as p:
            smp = p.map(_stage2_or_none, seeds, chunksize=8)
        skipped = [sd for sd, x in zip(seeds, smp) if x is None]
        kept = [x for x in smp if x is not None][:N_OOD]
        _write_set(d, kept, {"name": "ood_stage2", "kind": "stage2", "seed_base": base,
                             "skipped_seeds_no_sticks": skipped, "scan_window": {"step_h": 2e-3},
                             "labels": "simulator internals - evaluation only"})
    d = root / "null"
    if not (d / "images.npy").exists():
        with Pool(workers) as p:
            smp = p.map(_null_sample, range(SEED_NULL, SEED_NULL + N_NULL))
        _write_set(d, smp, {"name": "null", "seed_base": SEED_NULL, "p_appear": 0.0,
                            "scan_window": {"step_h": 2e-3}})


def check_disjoint(splits: list[dict]) -> None:
    seen: dict[str, str] = {}
    for sp in splits:
        for k, img in enumerate(sp["x"]):
            h = hashlib.sha1(img.tobytes()).hexdigest()
            assert h not in seen, f"image {sp['name']}[{k}] duplicates one in {seen[h]}"
            seen[h] = sp["name"]
    print(f"S1 ok: {len(seen)} images, all distinct across {[s['name'] for s in splits]}")


# ============================================================== preprocessing
def prep(x: np.ndarray) -> np.ndarray:
    """(N,H,W) raw -> stick-positive image in units of noise sigma.

    Row-median subtraction removes the horizontal stripes (fast-scan axis; sticks are
    sparse so the median is robust); sign flip (sticks are negative-going); MAD scale,
    because a z-score would be inflated by bright sticks and erase the absolute SNR.
    """
    x = x - np.median(x, axis=2, keepdims=True)
    mad = np.median(np.abs(x - np.median(x, axis=(1, 2), keepdims=True)), axis=(1, 2), keepdims=True)
    return (-x / (1.4826 * mad + 1e-6)).astype(np.float32)


def _whiten(r):
    """Re-express a filtered map in units of its own robust background std."""
    med = np.median(r, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(r - med), axis=(1, 2), keepdims=True)
    return ((r - med) / (1.4826 * mad + 1e-6)).astype(np.float32)


def _gauss(z, s):
    return ndimage.gaussian_filter(z, (0, s, s))


def _templates(thetas, length=5.0, width=1.0, size=11):
    """Zero-mean, unit-L2 elongated Gaussian ridges, long axis along theta (point-symmetric,
    so convolution == correlation)."""
    r = np.arange(size) - size // 2
    yy, xx = np.meshgrid(r, r, indexing="ij")
    bank = []
    for th in thetas:
        u = xx * np.cos(th) + yy * np.sin(th)
        v = -xx * np.sin(th) + yy * np.cos(th)
        t = np.exp(-0.5 * (u / (length / 2.5)) ** 2 - 0.5 * (v / width) ** 2)
        t -= t.mean()
        bank.append(t / np.sqrt((t**2).sum()))
    return bank


BANK = _templates((np.pi / 4 - 0.2, np.pi / 4, np.pi / 4 + 0.2))


def f_matched(z):
    out = np.max([fftconvolve(z, t[None], mode="same", axes=(1, 2)) for t in BANK], axis=0)
    return _whiten(out)  # max over the bank biases the null: re-whiten


def f_ridge(z, s=1.2):
    zs = _gauss(z, s)
    gy, gx = np.gradient(zs, axis=1), np.gradient(zs, axis=2)
    gyy, gxx, gxy = np.gradient(gy, axis=1), np.gradient(gx, axis=2), np.gradient(gy, axis=2)
    tr, det = gyy + gxx, gyy * gxx - gxy**2
    lam_min = tr / 2 - np.sqrt(np.maximum(tr**2 / 4 - det, 0))
    return _whiten(np.maximum(-lam_min, 0) * s**2)  # s^2: scale-normalised derivative


FEATS = ("z", "s1", "s2", "matched", "ridge", "loc_rms")


def features(z, chunk=100):
    out = np.empty(z.shape + (len(FEATS),), np.float32)
    for i in range(0, len(z), chunk):
        c = z[i : i + chunk]
        out[i : i + chunk] = np.stack(
            [c, _whiten(_gauss(c, 1.0)), _whiten(_gauss(c, 2.0)), f_matched(c), f_ridge(c),
             np.sqrt(ndimage.uniform_filter(c**2, (1, 7, 7)))], -1)
    return out


# ==================================================================== methods
class LogReg:
    """IRLS logistic regression on standardised features. Fit on all positives + 20x random
    negatives: this only shifts the intercept (prior correction), absorbed by the val
    threshold; probabilities are therefore NOT calibrated."""

    def __init__(self, cols, seed=0, lam=1e-3, neg_ratio=20):
        self.cols, self.seed, self.lam, self.neg_ratio = list(cols), seed, lam, neg_ratio

    def fit(self, F, m):
        X = F[..., self.cols].reshape(-1, len(self.cols))
        y = m.reshape(-1)
        rng = np.random.default_rng(self.seed)
        pos = np.flatnonzero(y)
        neg = rng.choice(np.flatnonzero(~y), self.neg_ratio * len(pos), replace=False)
        idx = np.concatenate([pos, neg])
        X, t = X[idx].astype(np.float64), y[idx].astype(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        w = np.zeros(X.shape[1])
        for _ in range(30):
            p = 1 / (1 + np.exp(-X @ w))
            H = (X * (p * (1 - p))[:, None]).T @ X + self.lam * np.eye(len(w))
            step = np.linalg.solve(H, X.T @ (t - p))
            w += step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def __call__(self, F):
        X = (F[..., self.cols] - self.mu) / self.sd
        return (X @ self.w[:-1] + self.w[-1]).astype(np.float32)  # logit


def hysteresis(s, t, low_frac=0.5, min_px=3):
    """Keep low-threshold components that contain a high pixel and have >= min_px pixels."""
    out = np.zeros(s.shape, bool)
    for k in range(len(s)):
        lab, n = ndimage.label(s[k] > low_frac * t, np.ones((3, 3)))
        keep = np.zeros(n + 1, bool)
        keep[np.unique(lab[s[k] > t])] = True
        keep &= np.bincount(lab.ravel(), minlength=n + 1) >= min_px
        keep[0] = False
        out[k] = keep[lab]
    return out


def thresh(s, t):
    return s > t


def build_methods(F_train, m_train, fit_seeds):
    c = {f: i for i, f in enumerate(FEATS)}
    methods = {
        "M0_empty": (lambda F: np.full(F.shape[:3], -np.inf, np.float32), thresh),
        "M1_smooth": (lambda F: F[..., c["s1"]], thresh),
        "M2_matched": (lambda F: F[..., c["matched"]], thresh),
        "M3_ridge": (lambda F: F[..., c["ridge"]], thresh),
        "M4_hyst": (lambda F: F[..., c["s1"]], hysteresis),
    }
    variants = {"M5_logreg": [c[f] for f in FEATS if f != "z"],
                "M5_full": list(range(len(FEATS))),
                "M5_min": [c["matched"], c["s2"]]}
    weights = {}
    for name, cols in variants.items():
        for sd in fit_seeds:
            lr = LogReg(cols, seed=sd).fit(F_train, m_train)
            key = name if sd == fit_seeds[0] else f"{name}@seed{sd}"
            methods[key] = (lr, thresh)
            weights[key] = dict(zip([FEATS[i] for i in cols] + ["bias"], np.round(lr.w, 3).tolist()))
    return methods, weights


# =================================================================== counting
CNT = ("tp", "n_pred", "n_gt", "tol_tp_p", "tol_tp_r", "o_tp_gt", "o_n_gt", "o_tp_pred", "o_n_pred")
NB = len(INTENSITY_BINS) - 1


def stick_table(sticks) -> dict:
    """Flatten per-image stick lists once per split (reused by every counts() call)."""
    img = np.array([k for k, ss in enumerate(sticks) for _ in ss], int)
    cen = np.array([[s["row"], s["col"]] for ss in sticks for s in ss], float).reshape(-1, 2)
    half = np.array([s["len_px"] / 2 for ss in sticks for s in ss])
    amp = np.abs([s["intensity"] for ss in sticks for s in ss])
    b = np.clip(np.digitize(amp, INTENSITY_BINS) - 1, 0, NB - 1)
    return {"img": img, "cen": cen, "half": half, "bin": b}


_OFF = 1e4  # image index -> far-away coordinate offset, so one KD-tree serves the batch
_ST3 = np.zeros((3, 3, 3), bool)
_ST3[1] = True  # 3x3 dilation inside each image, never across images
_LAB3 = np.zeros((3, 3, 3), bool)
_LAB3[1] = ndimage.generate_binary_structure(2, 1)  # label's default 4-connectivity, per image


def counts(pred: np.ndarray, m: np.ndarray, st: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-image counts (N, len(CNT)) and amplitude-bin (hit, n) (N, 2, NB), vectorised
    over the batch. Same definitions as detection.metrics (pixel_scores / object_scores);
    parity is asserted in run() (S6)."""
    n_img, h, w = pred.shape
    gt_d = ndimage.maximum_filter(m, size=(1, 3, 3))
    pr_d = ndimage.maximum_filter(pred, size=(1, 3, 3))
    C = np.zeros((n_img, len(CNT)))
    C[:, 0] = (pred & m).sum((1, 2))
    C[:, 1] = pred.sum((1, 2))
    C[:, 2] = m.sum((1, 2))
    C[:, 3] = (pred & gt_d).sum((1, 2))
    C[:, 4] = (m & pr_d).sum((1, 2))
    A = np.zeros((n_img, 2, NB))
    img, cen, half = st["img"], st["cen"], st["half"]
    inside = (cen[:, 0] >= 0) & (cen[:, 0] < h) & (cen[:, 1] >= 0) & (cen[:, 1] < w)
    pix = np.argwhere(pred).astype(float)
    # GT recall: a stick is found if a predicted pixel lies within len/2 + 1 px of its centre
    if len(img):
        if len(pix):
            tree = cKDTree(np.c_[pix[:, 0] * _OFF, pix[:, 1:]])
            d = tree.query(np.c_[img * _OFF, cen])[0]
        else:
            d = np.full(len(img), np.inf)
        hit = (d <= half + 1.0) & inside
        C[:, 5] = np.bincount(img, hit, n_img)
        C[:, 6] = np.bincount(img, inside, n_img)
        np.add.at(A, (img[inside], 0, st["bin"][inside]), hit[inside])
        np.add.at(A, (img[inside], 1, st["bin"][inside]), 1)
    # Precision: blobs = 3x3-dilation components, centroid on the predicted pixels only
    if len(pix):
        lab, n = ndimage.label(ndimage.binary_dilation(pred, _ST3), _LAB3)
        li = lab[pred]  # every predicted pixel belongs to exactly one dilated component
        cnt = np.bincount(li, minlength=n + 1)[1:]
        cms = np.stack([np.bincount(li, pix[:, a], n + 1)[1:] for a in range(3)], 1)
        cms = cms[cnt > 0] / cnt[cnt > 0, None]  # centroid over the predicted pixels only
        b_img = np.rint(cms[:, 0]).astype(int)
        C[:, 8] = np.bincount(b_img, minlength=n_img)
        if len(img) and len(cms):
            k = min(8, len(img))
            dd, j = cKDTree(np.c_[img * _OFF, cen]).query(np.c_[cms[:, 0] * _OFF, cms[:, 1:]], k=k)
            dd, j = dd.reshape(len(cms), k), j.reshape(len(cms), k)
            ok = (dd <= half[np.minimum(j, len(img) - 1)] + 2.0) & (j < len(img))
            C[:, 7] = np.bincount(b_img, ok.any(1), n_img)
    return C, A


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def scores_from(C: np.ndarray) -> dict:
    s = dict(zip(CNT, C.sum(0)))
    P, R = s["tp"] / max(s["n_pred"], 1), s["tp"] / max(s["n_gt"], 1)
    tP, tR = s["tol_tp_p"] / max(s["n_pred"], 1), s["tol_tp_r"] / max(s["n_gt"], 1)
    oP, oR = s["o_tp_pred"] / max(s["o_n_pred"], 1), s["o_tp_gt"] / max(s["o_n_gt"], 1)
    out = {"f1": _f1(P, R), "iou": s["tp"] / max(s["n_pred"] + s["n_gt"] - s["tp"], 1),
           "tol_precision": tP, "tol_recall": tR, "tol_f1": _f1(tP, tR),
           "obj_precision": oP, "obj_recall": oR, "obj_f1": _f1(oP, oR),
           "width_ratio": s["n_pred"] / max(s["n_gt"], 1),
           "blobs_per_found_stick": s["o_n_pred"] / max(s["o_tp_gt"], 1),
           "blobs_per_image": s["o_n_pred"] / len(C)}
    # Dice = 2TP / (|pred| + |gt|): "global" pools pixels over the set (== strict pixel F1),
    # "img" averages per image (empty pred & empty gt -> 1), the usual segmentation convention.
    den = C[:, 1] + C[:, 2]
    out["dice"] = out["f1"]
    out["dice_img"] = float(np.mean(np.where(den > 0, 2 * C[:, 0] / np.maximum(den, 1), 1.0)))
    out["sel"] = (out["obj_f1"] + out["tol_f1"]) / 2
    return out


# ================================================================ selection (S2)
def select_threshold(s, binarize, val, crit="sel", n_grid=24):
    """Grid = score quantiles giving 0.03 %..3 % predicted pixels (scale-free, so every
    method gets the same search budget), refined once around the best point."""
    finite = s[np.isfinite(s)]
    if finite.size == 0:  # M0
        return 0.0, {"edge": False, "grid": []}
    fr = np.geomspace(3e-4, 3e-2, n_grid)
    grid = np.unique(np.quantile(finite, 1 - fr))
    evals = {}

    def ev(t):
        if t not in evals:
            evals[t] = scores_from(counts(binarize(s, t), val["m"], val["st"])[0])[crit]
        return evals[t]

    best = max(grid, key=ev)
    i = int(np.searchsorted(grid, best))
    edge = i in (0, len(grid) - 1)
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    best = max(np.r_[np.linspace(lo, hi, 7), best], key=ev)
    return float(best), {"edge": bool(edge), "grid": [float(grid[0]), float(grid[-1])]}


# =============================================================== statistics (S4)
def bootstrap(C_by_method: dict[str, np.ndarray], ref: str, rng, keys=("obj_f1", "tol_f1", "dice", "dice_img")):
    n = len(next(iter(C_by_method.values())))
    idx = rng.integers(0, n, (N_BOOT, n))
    out = {}
    for name, C in C_by_method.items():
        bs = [scores_from(C[i]) for i in idx]
        bref = [scores_from(C_by_method[ref][i]) for i in idx] if name != ref else None
        out[name] = {}
        for k in keys:
            v = np.array([b[k] for b in bs])
            out[name][k + "_ci"] = np.percentile(v, [2.5, 97.5]).tolist()
            if bref is not None:
                d = v - np.array([b[k] for b in bref])
                out[name][k + f"_minus_{ref}_ci"] = np.percentile(d, [2.5, 97.5]).tolist()
    return out


# ======================================================================== run
def run(args) -> None:
    t_start = time.perf_counter()
    q = args.quick
    root = Path(args.eval)
    ensure_eval_sets(root, args.workers)
    lim = (lambda n: max(n // 3, 50)) if q else (lambda n: n)
    train = load_split(Path(args.train), 150 if q else N_FIT, "train_fit")
    splits = {s: load_split(root / s, lim(n)) for s, n in
              [("val", N_VAL), ("test", N_TEST)] + [(o, N_OOD) for o in SETS[2:]] + [("null", N_NULL)]}
    check_disjoint([train] + list(splits.values()))
    for sp in splits.values():
        sp["st"] = stick_table(sp["sticks"])

    t0 = time.perf_counter()
    methods, weights = build_methods(features(prep(train["x"])), train["m"], (0,) if q else (0, 1, 2))
    t_fit = time.perf_counter() - t0
    print(f"fit M5 variants: {t_fit:.1f}s")

    def score_maps(sp):
        t0 = time.perf_counter()
        F = features(prep(sp["x"]))
        t_feat = (time.perf_counter() - t0) / len(F)
        out = {}
        for mname, (fn, _) in methods.items():
            t1 = time.perf_counter()
            out[mname] = fn(F)
            ms_img.setdefault(mname, []).append(1e3 * (t_feat + (time.perf_counter() - t1) / len(F)))
        return out

    # S2: thresholds on val only
    ms_img: dict[str, list] = {}
    val_maps = score_maps(splits["val"])
    thr, sel_info = {}, {}
    for mname, (_, binz) in methods.items():
        thr[mname], sel_info[mname] = select_threshold(val_maps[mname], binz, splits["val"],
                                                          args.select)
        if sel_info[mname]["edge"]:
            print(f"WARNING {mname}: val-optimal threshold on the grid edge {sel_info[mname]['grid']}")
    print(f"thresholds selected on val: {time.perf_counter() - t_start:.0f}s")

    # S3: frozen thresholds on every split; test / OOD / null scored once (streamed: one
    # split's score maps in memory at a time)
    res = {m: {"thr": thr[m], "selection": sel_info[m]} for m in methods}
    C_test, test_maps = {}, None
    for sname, sp in splits.items():
        maps = val_maps if sname == "val" else score_maps(sp)
        for mname, (_, binz) in methods.items():
            C, A = counts(binz(maps[mname], thr[mname]), sp["m"], sp["st"])
            r = scores_from(C)
            if sname == "test":
                C_test[mname] = C
                a = A.sum(0)
                r["recall_by_amplitude"] = {
                    f"[{INTENSITY_BINS[i]:g},{INTENSITY_BINS[i + 1]:g})":
                        (float(a[0, i] / a[1, i]) if a[1, i] else None, int(a[1, i])) for i in range(NB)}
            res[mname][sname] = r
        if sname == "test":
            test_maps = {k: maps[k] for k in ("M2_matched", "M5_logreg")}
        del maps
    for m in methods:
        res[m]["ms_per_image"] = float(np.mean(ms_img[m]))

    # S6: metric parity with detection.metrics.object_scores
    for mname in ("M2_matched", "M5_logreg"):
        pred = methods[mname][1](test_maps[mname], thr[mname])
        ref = object_scores(pred, splits["test"]["sticks"], *pred.shape[1:])
        for k in ("obj_precision", "obj_recall"):
            assert abs(ref[k] - res[mname]["test"][k]) < 1e-9, (mname, k, ref[k], res[mname]["test"][k])
    print("S6 ok: per-image object counts match detection.metrics.object_scores")

    # S4: bootstrap CIs (main methods) + M5 seed spread
    main = [m for m in methods if "@seed" not in m]
    boot = bootstrap({m: C_test[m] for m in main if m != "M0_empty"}, "M2_matched",
                     np.random.default_rng(0))
    for m in boot:
        res[m]["test"].update(boot[m])
    for base in ("M5_logreg", "M5_full", "M5_min"):
        v = [res[k]["test"]["obj_f1"] for k in methods if k.split("@")[0] == base]
        res[base]["test"]["obj_f1_seed_std"] = float(np.std(v))
        res[base]["test"]["n_fit_seeds"] = len(v)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = run_info(args, time.perf_counter() - t_start, t_fit, {k: len(v["x"]) for k, v in
                                                                  {"train_fit": train, **splits}.items()})
    (out / "baselines.json").write_text(json.dumps({"info": info, "weights": weights, "results": res},
                                                   indent=1, default=float), encoding="utf-8")
    (out / "baselines.md").write_text(report(res, weights, info), encoding="utf-8")
    for m in (k for k in res if "@seed" not in k):
        t = res[m]["test"]
        print(f"  {m:11s} test obj_F1={t['obj_f1']:.3f} tol_F1={t['tol_f1']:.3f} "
              f"Dice={t['dice']:.3f} Dice_img={t['dice_img']:.3f}")
    print(f"wrote {out}/baselines.md  ({info['elapsed_s']:.0f}s)")


def run_info(args, elapsed, t_fit, sizes) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "?"
    return {"version": "v2", "args": vars(args), "sizes": sizes, "elapsed_s": elapsed, "fit_s": t_fit,
            "git_commit": commit, "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "machine": platform.processor() or platform.machine()}


# ===================================================================== report
def _ci(r, k):
    c = r.get(k + "_ci")
    return f"{r[k]:.3f} [{c[0]:.3f}, {c[1]:.3f}]" if c else f"{r[k]:.3f}"


def report(res, weights, info) -> str:
    main = [m for m in res if "@seed" not in m]
    crit = {"sel": "sel = (obj_F1 + tol_F1)/2", "dice": "global Dice"}[info["args"]["select"]]
    L = [f"# Challenge 1 — classical baselines ({info['version']})", "",
         f"Sizes: {info['sizes']}. Run time {info['elapsed_s']:.0f}s on CPU "
         f"(commit {info['git_commit']}). Threshold chosen on **val** by {crit}, "
         "frozen everywhere else. Brackets: 95 % bootstrap CI over test images.", "",
         "## Test (in-distribution, scored once)", "",
         "| method | obj F1 | Δ obj F1 vs M2 | tol F1 | IoU (strict) | obj P / R | width ratio | blobs / found stick | false blobs / img (null) | ms/img † |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m in main:
        t, n = res[m]["test"], res[m]["null"]
        d = t.get("obj_f1_minus_M2_matched_ci")
        ds = f"[{d[0]:+.3f}, {d[1]:+.3f}]" if d else "—"
        seed = f" ± {t['obj_f1_seed_std']:.3f} (seeds)" if "obj_f1_seed_std" in t else ""
        L.append(f"| {m} | {_ci(t, 'obj_f1')}{seed} | {ds} | {_ci(t, 'tol_f1')} | {t['iou']:.3f} | "
                 f"{t['obj_precision']:.2f} / {t['obj_recall']:.2f} | {t['width_ratio']:.2f} | "
                 f"{t['blobs_per_found_stick']:.2f} | {n['blobs_per_image']:.2f} | {res[m]['ms_per_image']:.1f} |")
    L += ["", "width ratio = predicted px / mask px (1 = the mask's arbitrary width is copied); "
          "null = stick-free scenes (p_appear = 0), i.e. the false-alarm rate per 0.3 V x 0.3 V scan. "
          "† the 6-feature stack is computed once and shared, so its cost is charged to every "
          "method (upper bound; M1 alone needs one Gaussian filter).", "",
          "## Robustness — obj F1 (val threshold frozen, report-only sets)", "",
          "| method | val | test | θ shift | θ wide | noise ×1.5 | zoom in (1 mV) | zoom out (3 mV) | stage-2 frames |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in main:
        L.append(f"| {m} | " + " | ".join(f"{res[m][s]['obj_f1']:.3f}" for s in SETS) + " |")
    L += ["", "## Object recall vs stick amplitude |i| (test)", "",
          "| method | " + " | ".join(f"[{INTENSITY_BINS[i]:g},{INTENSITY_BINS[i + 1]:g})" for i in range(NB)) + " |",
          "|---|" + "---|" * NB]
    for m in main:
        rb = res[m]["test"]["recall_by_amplitude"]
        L.append(f"| {m} | " + " | ".join("—" if v[0] is None else f"{v[0]:.2f}" for v in rb.values()) + " |")
    n_amp = list(res[main[0]]["test"]["recall_by_amplitude"].values())
    L += ["", "sticks per bin: " + ", ".join(str(v[1]) for v in n_amp), "",
          "## Selection diagnostics", "",
          "| method | thr | val sel | test sel | grid edge? |", "|---|---|---|---|---|"]
    for m in main:
        L.append(f"| {m} | {res[m]['thr']:.3g} | {res[m]['val']['sel']:.3f} | {res[m]['test']['sel']:.3f} | "
                 f"{'**yes**' if res[m]['selection']['edge'] else 'no'} |")
    L += ["", "## M5 weights (standardised features, fit seed 0)", ""]
    for m in ("M5_logreg", "M5_full", "M5_min"):
        L.append(f"- **{m}**: " + ", ".join(f"{k} {v:+.2f}" for k, v in weights[m].items()))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--eval", default="data/eval_light")
    ap.add_argument("--out", default="results/detection_baselines_v2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="1/3 subsets, 1 fit seed: smoke test")
    ap.add_argument("--select", choices=("sel", "dice"), default="sel",
                    help="val criterion for the threshold: (obj_F1+tol_F1)/2 (default) or global Dice")
    run(ap.parse_args())
