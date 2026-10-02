"""Detection metrics — pixel, tolerant-pixel and object (interdot) level.

Why three levels:
* the official mask is a 0.5-threshold of a blurred 1-2 px wide rectangle: it is
  often *fragmented* (~2.5 components per stick, ~2.4 px each), so strict pixel IoU
  mostly measures sub-pixel pixelization luck (how the generator snaps a blurred rectangle onto the pixel grid);
* ``tol`` metrics accept a 1-px offset (a prediction pixel counts if a GT pixel is
  within its 3x3 neighbourhood and vice versa);
* object metrics answer the experimentalist's question: *did I find this interdot?*
  A GT stick is found if a predicted pixel lies within len/2 + 1 px of its centre; a
  predicted blob (fragments merged by a 3x3 dilation) is correct if its centroid lies
  within len/2 + 2 px of some GT stick.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

INTENSITY_BINS = (0.0, 2.0, 4.0, 8.0, 16.0, np.inf)  # |stick amplitude|, noise sigma_pix = 0.9


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def pixel_counts(prob: torch.Tensor, gt: torch.Tensor, thresholds) -> dict[str, torch.Tensor]:
    """Accumulable counts for several thresholds. prob/gt: (B, 1, H, W) on any device."""
    import torch.nn.functional as F  # lazy: object_scores must work without torch
    import torch

    with torch.no_grad():
        return _pixel_counts(prob, gt, thresholds, F)


def _pixel_counts(prob, gt, thresholds, F):
    import torch

    gt = gt.bool()
    gt_dil = F.max_pool2d(gt.float(), 3, 1, 1).bool()
    out = {k: [] for k in ("tp", "fp", "fn", "tol_tp_p", "n_pred", "tol_tp_r", "n_gt")}
    for t in thresholds:
        pred = prob > t
        pred_dil = F.max_pool2d(pred.float(), 3, 1, 1).bool()
        out["tp"].append((pred & gt).sum())
        out["fp"].append((pred & ~gt).sum())
        out["fn"].append((~pred & gt).sum())
        out["tol_tp_p"].append((pred & gt_dil).sum())
        out["n_pred"].append(pred.sum())
        out["tol_tp_r"].append((gt & pred_dil).sum())
        out["n_gt"].append(gt.sum())
    return {k: torch.stack(v).double().cpu() for k, v in out.items()}


def pixel_scores(c: dict[str, torch.Tensor], thresholds) -> list[dict]:
    rows = []
    for i, t in enumerate(thresholds):
        tp, fp, fn = (float(c[k][i]) for k in ("tp", "fp", "fn"))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        tp_ = float(c["tol_tp_p"][i]) / max(float(c["n_pred"][i]), 1.0)
        tr_ = float(c["tol_tp_r"][i]) / max(float(c["n_gt"][i]), 1.0)
        rows.append(
            {
                "thr": float(t),
                "precision": p,
                "recall": r,
                "f1": _f1(p, r),
                "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
                "tol_precision": tp_,
                "tol_recall": tr_,
                "tol_f1": _f1(tp_, tr_),
            }
        )
    return rows


def object_scores(pred: np.ndarray, sticks: list[list[dict]], h: int, w: int) -> dict:
    """pred: (N, H, W) bool. sticks: per image, list of {row, col, len_px, intensity}."""
    st = np.ones((3, 3), bool)
    tp_gt = n_gt = tp_pred = n_pred = 0
    nb = len(INTENSITY_BINS) - 1
    bin_hit, bin_n = np.zeros(nb), np.zeros(nb)
    for k in range(len(pred)):
        ss = sticks[k]
        centers = np.array([[s["row"], s["col"]] for s in ss]).reshape(-1, 2)
        half = np.array([s["len_px"] / 2 for s in ss])
        inside = (
            (centers[:, 0] >= 0) & (centers[:, 0] < h) & (centers[:, 1] >= 0) & (centers[:, 1] < w)
        )
        pix = np.argwhere(pred[k])
        # GT recall (only sticks whose centre is inside the frame)
        if len(ss):
            if len(pix):
                d = np.sqrt(((centers[:, None, :] - pix[None, :, :]) ** 2).sum(-1)).min(1)
            else:
                d = np.full(len(ss), np.inf)
            hit = d <= half + 1.0
            amp = np.abs([s["intensity"] for s in ss])
            b = np.clip(np.digitize(amp, INTENSITY_BINS) - 1, 0, nb - 1)
            for bi, hi, ins in zip(b, hit, inside):
                if ins:
                    bin_n[bi] += 1
                    bin_hit[bi] += hi
            tp_gt += int((hit & inside).sum())
            n_gt += int(inside.sum())
        # Predicted blobs precision (matched against all sticks, incl. border ones)
        if len(pix):
            lab, n = ndimage.label(ndimage.binary_dilation(pred[k], st))
            lab = lab * pred[k]
            cms = ndimage.center_of_mass(pred[k], lab, range(1, n + 1))
            cms = np.array([c for c in cms if not np.isnan(c[0])]).reshape(-1, 2)
            n_pred += len(cms)
            if len(ss) and len(cms):
                d = np.sqrt(((cms[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
                tp_pred += int((d <= half[None, :] + 2.0).any(1).sum())
    p = tp_pred / n_pred if n_pred else 0.0
    r = tp_gt / n_gt if n_gt else 0.0
    return {
        "obj_precision": p,
        "obj_recall": r,
        "obj_f1": _f1(p, r),
        "n_gt_sticks": n_gt,
        "n_pred_blobs": n_pred,
        "recall_by_amplitude": {
            f"[{INTENSITY_BINS[i]:g},{INTENSITY_BINS[i + 1]:g})": (
                float(bin_hit[i] / bin_n[i]) if bin_n[i] else None,
                int(bin_n[i]),
            )
            for i in range(nb)
        },
    }
