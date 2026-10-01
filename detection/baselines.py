"""Challenge 1 — five cheap, classical detectors + the protocol that scores them.

    uv run python -m detection.baselines            # ~1.5 min on a laptop CPU, no torch needed
    uv run python -m detection.baselines --quick    # ~10 s smoke run
(the first run also generates the 400-scene test set, seed 2025: ~5 min, once)

Purpose: get *first honest numbers* before spending GPU time on the transformers
(PROTOCOL.md), and give the neural models a floor they must beat to be justified.

The five methods (all numpy/scipy/skimage; "score map" = evidence that a pixel is a stick,
in units of noise sigma, so one threshold transfers across scenes):

  M1 smooth   destripe + robust z-score + Gaussian smoothing           (no shape prior)
  M2 matched  bank of oriented elongated templates (theta ~ pi/4)      (shape prior, optimal for white noise)
  M3 ridge    Hessian ridge filter (most negative curvature eigenvalue) (local-geometry prior)
  M4 hyst     M1 + hysteresis threshold + small-blob removal            (spatial-coherence prior)
  M5 logreg   logistic regression on the multi-scale features of M1-M3 (learned, ~7 weights)
  M0 empty    predicts nothing — the control that sets the metric floor.

Protocol (the traps this guards against, cf. PROTOCOL.md section 1):
  * fit (M5 only)  -> first ``--n-fit`` images of data/train
  * threshold      -> chosen on data/val ONLY (max tolerant-pixel F1), then frozen
  * test           -> fresh seed (2025), never used to choose anything, scored once
  * OOD            -> test + extra noise, to see which prior breaks first
  * metrics        -> strict pixel (what the challenge asks, but the official mask is
                      fragmented, so noisy), tolerant pixel (1 px), object level, and recall
                      vs |amplitude| (the physical detection limit)
  * a method that wins on strict IoU but loses on object F1 is exploiting rasterisation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from detection.metrics import INTENSITY_BINS, object_scores

SEED_TEST = 2025
THRS = np.unique(np.round(np.r_[np.arange(0.5, 8, 0.5), np.arange(8, 40.1, 2)], 2))  # noise-sigma units


# ----------------------------------------------------------------------------- data
def load_sticks(path: Path, step: float, n: int) -> list[list[dict]]:
    """Stick metadata in pixel units (dataset stores volts; col = x / step)."""
    out = []
    with (path / "sticks.jsonl").open() as f:
        for line, _ in zip(f, range(n)):
            out.append(
                [
                    {
                        "col": s["x"] / step,
                        "row": s["y"] / step,
                        "len_px": s["length"] / step,
                        "intensity": s["intensity"],
                    }
                    for s in json.loads(line)["sticks"]
                ]
            )
    return out


def load_split(path: Path, n: int | None = None) -> dict:
    meta = json.loads((path / "meta.json").read_text())
    x = np.load(path / "images.npy", mmap_mode="r")[:n]
    m = np.load(path / "masks.npy", mmap_mode="r")[:n]
    step = meta["scan_window"]["step_h"]
    return {"name": path.name, "x": np.asarray(x, np.float32), "m": np.asarray(m).astype(bool),
            "sticks": load_sticks(path, step, len(x))}


def ensure_test(root: Path, n: int) -> Path:
    path = root / "test_baseline"
    if not (path / "images.npy").exists():
        from csd import generate_dataset

        print(f"generating {n} test scenes (seed {SEED_TEST}) -> {path}")
        generate_dataset(n, path, seed=SEED_TEST)
    return path


def add_noise(x: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """OOD: extra white noise, blurred like the instrument (sigma_blur = 0.5 px)."""
    rng = np.random.default_rng(seed)
    n = ndimage.gaussian_filter(rng.normal(0, sigma, x.shape), (0, 0.5, 0.5))
    return (x + n).astype(np.float32)


# ------------------------------------------------------------------- preprocessing
def prep(x: np.ndarray) -> np.ndarray:
    """(N,H,W) raw -> stick-positive image in units of noise sigma.

    * row-median subtraction removes the horizontal-stripe noise (fast-scan axis; sticks
      are sparse so the row median is robust),
    * sign flip: sticks are negative-going,
    * MAD scale: the background is ~all of the image, so MAD ~ noise sigma; a z-score
      would be inflated by bright sticks and erase the absolute SNR (trap #7).
    """
    x = x - np.median(x, axis=2, keepdims=True)
    mad = np.median(np.abs(x - np.median(x, axis=(1, 2), keepdims=True)), axis=(1, 2), keepdims=True)
    return -x / (1.4826 * mad + 1e-6)


def _gauss(z, s):
    return ndimage.gaussian_filter(z, (0, s, s))


def _whiten(r):
    """Rescale a filtered map by its own robust background std -> sigma units again."""
    med = np.median(r, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(r - med), axis=(1, 2), keepdims=True)
    return (r - med) / (1.4826 * mad + 1e-6)


# ----------------------------------------------------------------------- the methods
def f_smooth(z, s=1.0):
    return _whiten(_gauss(z, s))


def _templates(thetas, length=5.0, width=1.0, size=11):
    """Zero-mean, unit-L2 elongated Gaussian ridges (row, col), long axis along theta."""
    r = np.arange(size) - size // 2
    yy, xx = np.meshgrid(r, r, indexing="ij")
    bank = []
    for th in thetas:
        u = xx * np.cos(th) + yy * np.sin(th)  # along the stick
        v = -xx * np.sin(th) + yy * np.cos(th)  # across
        t = np.exp(-0.5 * (u / (length / 2.5)) ** 2 - 0.5 * (v / width) ** 2)
        t -= t.mean()
        bank.append(t / np.sqrt((t**2).sum()))
    return bank


def f_matched(z, thetas=(np.pi / 4 - 0.2, np.pi / 4, np.pi / 4 + 0.2)):
    out = None
    for t in _templates(thetas):
        r = np.stack([ndimage.correlate(zi, t, mode="nearest") for zi in z])
        out = r if out is None else np.maximum(out, r)
    return _whiten(out)  # max over the bank biases the null: re-whiten


def f_ridge(z, s=1.2):
    zs = _gauss(z, s)
    gyy = np.gradient(np.gradient(zs, axis=1), axis=1)
    gxx = np.gradient(np.gradient(zs, axis=2), axis=2)
    gxy = np.gradient(np.gradient(zs, axis=1), axis=2)
    tr, det = gyy + gxx, gyy * gxx - gxy**2
    lam_min = tr / 2 - np.sqrt(np.maximum(tr**2 / 4 - det, 0))  # most negative curvature
    return _whiten(np.maximum(-lam_min, 0) * s**2)


def features(z):
    s1, s2 = f_smooth(z, 1.0), f_smooth(z, 2.0)
    loc_std = np.sqrt(np.maximum(ndimage.uniform_filter(z**2, (1, 7, 7)), 0))
    F = [z, s1, s2, f_matched(z), f_ridge(z), loc_std]
    return np.stack(F, -1)  # (N,H,W,6)


class Method:
    name = "?"

    def fit(self, z, m):  # most methods are fit-free
        return self

    def scores(self, z):
        raise NotImplementedError

    def binarize(self, s, t):
        return s > t


class Empty(Method):
    name = "M0_empty"

    def scores(self, z):
        return np.full(z.shape, -1e9, np.float32)


class Smooth(Method):
    name = "M1_smooth"

    def scores(self, z):
        return f_smooth(z)


class Matched(Method):
    name = "M2_matched"

    def scores(self, z):
        return f_matched(z)


class Ridge(Method):
    name = "M3_ridge"

    def scores(self, z):
        return f_ridge(z)


class Hysteresis(Method):
    name = "M4_hyst"

    def __init__(self, low_frac=0.5, min_px=3):
        self.low_frac, self.min_px = low_frac, min_px

    def scores(self, z):
        return f_smooth(z, 1.0)

    def binarize(self, s, t):
        out = np.zeros(s.shape, bool)
        for k in range(len(s)):
            hi, lo = s[k] > t, s[k] > self.low_frac * t
            lab, n = ndimage.label(lo, np.ones((3, 3)))
            keep = np.zeros(n + 1, bool)
            keep[np.unique(lab[hi])] = True
            keep[0] = False
            sizes = np.bincount(lab.ravel(), minlength=n + 1)
            keep &= sizes >= self.min_px
            out[k] = keep[lab]
        return out


class LogReg(Method):
    """Logistic regression (IRLS) on multi-scale features. Positives are rare (0.35 %):
    fit on all positives + a random 20x negatives, so the intercept is wrong by design —
    the decision threshold is re-tuned on val, which absorbs it (trap: calibration)."""

    name = "M5_logreg"

    def fit(self, z, m, seed=0, neg_ratio=20):
        F = features(z).reshape(-1, 6)
        y = m.reshape(-1)
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(y)
        neg = rng.choice(np.flatnonzero(~y), neg_ratio * len(pos), replace=False)
        idx = np.concatenate([pos, neg])
        X, t = F[idx], y[idx].astype(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        w = np.zeros(X.shape[1])
        for _ in range(25):
            p = 1 / (1 + np.exp(-X @ w))
            H = (X * (p * (1 - p))[:, None]).T @ X + 1e-3 * np.eye(len(w))
            w += np.linalg.solve(H, X.T @ (t - p))
        self.w = w
        return self

    def scores(self, z):
        F = (features(z) - self.mu) / self.sd
        return (F @ self.w[:-1] + self.w[-1]).astype(np.float32)  # logit


METHODS = [Empty, Smooth, Matched, Ridge, Hysteresis, LogReg]


# ----------------------------------------------------------------------- scoring
def _f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def pixel_sweep(s, m, method, thrs):
    """Strict + 1px-tolerant pixel metrics for every threshold."""
    gt_d = ndimage.maximum_filter(m, size=(1, 3, 3))
    rows = []
    for t in thrs:
        p = method.binarize(s, t)
        pd = ndimage.maximum_filter(p, size=(1, 3, 3))
        tp, npred, ngt = (p & m).sum(), p.sum(), m.sum()
        P, R = tp / max(npred, 1), tp / max(ngt, 1)
        tP, tR = (p & gt_d).sum() / max(npred, 1), (m & pd).sum() / max(ngt, 1)
        rows.append({"thr": float(t), "f1": _f1(P, R), "iou": tp / max(npred + ngt - tp, 1),
                     "tol_precision": tP, "tol_recall": tR, "tol_f1": _f1(tP, tR)})
    return rows


def evaluate(method, split, thr=None, z=None):
    z = prep(split["x"]) if z is None else z
    t0 = time.perf_counter()
    s = method.scores(z)
    dt = (time.perf_counter() - t0) / len(z) * 1e3
    if thr is None:  # allowed on val only
        rows = pixel_sweep(s, split["m"], method, THRS)
        thr = max(rows, key=lambda r: r["tol_f1"])["thr"]
    row = pixel_sweep(s, split["m"], method, [thr])[0]
    pred = method.binarize(s, thr)
    h, w = pred.shape[1:]
    obj = object_scores(pred, split["sticks"], h, w)
    return {**row, **obj, "ms_per_image": dt, "set": split["name"]}


def run(args) -> None:
    root = Path(args.data)
    n_ev = 100 if args.quick else None  # --quick: ~10 s smoke run, noisier numbers
    train = load_split(root / "train", 150 if args.quick else args.n_fit)
    val = load_split(root / "val", n_ev)
    test = load_split(ensure_test(root, args.n_test), n_ev)
    ood = {"name": "test+noise", "x": add_noise(test["x"], 0.9, 7), "m": test["m"],
           "sticks": test["sticks"]}
    z_train = prep(train["x"])
    results = {}
    for cls in METHODS:
        t0 = time.perf_counter()
        meth = cls().fit(z_train, train["m"]) if cls is LogReg else cls()
        t_fit = time.perf_counter() - t0
        v = evaluate(meth, val)  # threshold selected here, and only here
        thr = v["thr"]
        results[meth.name] = {"thr": thr, "fit_s": t_fit, "val": v,
                              "test": evaluate(meth, test, thr), "ood_noise": evaluate(meth, ood, thr)}
        r = results[meth.name]["test"]
        print(f"{meth.name:12s} thr={thr:5.2f}  test tol_f1={r['tol_f1']:.3f} iou={r['iou']:.3f} "
              f"obj_P/R/F1={r['obj_precision']:.2f}/{r['obj_recall']:.2f}/{r['obj_f1']:.3f} "
              f"{r['ms_per_image']:.1f} ms/img")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(results, indent=1, default=float))
    (out / "baselines.md").write_text(table(results))
    print(f"\nwrote {out}/baselines.md")


def table(res) -> str:
    L = ["| method | thr | val tol-F1 | test tol-F1 | test IoU | obj P | obj R | obj F1 | OOD obj F1 | ms/img |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        t, o = r["test"], r["ood_noise"]
        L.append(f"| {k} | {r['thr']:.2f} | {r['val']['tol_f1']:.3f} | {t['tol_f1']:.3f} | {t['iou']:.3f} | "
                 f"{t['obj_precision']:.2f} | {t['obj_recall']:.2f} | {t['obj_f1']:.3f} | {o['obj_f1']:.3f} | "
                 f"{t['ms_per_image']:.1f} |")
    L += ["", "Object recall vs stick amplitude |i| (test):", "",
          "| method | " + " | ".join(f"[{INTENSITY_BINS[i]:g},{INTENSITY_BINS[i+1]:g})" for i in range(5)) + " |",
          "|---|" + "---|" * 5]
    for k, r in res.items():
        rb = r["test"]["recall_by_amplitude"]
        L.append(f"| {k} | " + " | ".join("-" if v[0] is None else f"{v[0]:.2f}" for v in rb.values()) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/detection_baselines")
    ap.add_argument("--n-fit", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--quick", action="store_true", help="150 fit / 100 val / 100 test images")
    run(ap.parse_args())
