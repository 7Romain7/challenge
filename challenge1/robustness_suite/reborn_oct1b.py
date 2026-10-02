"""Détecteur de RebornFlamme, branche autoresearch/oct1b (1er oct. 2026).

Copie figée de ``autoresearch/detect.py`` : médiane par ligne, filtre adapté
le long du stick à π/4, remplissage à une fraction du pic, filtre d'angle.
Paramètres tels que publiés. Aucun réglage sur la suite de robustesse.
La sortie est un masque 0/1 : le harnais ne peut pas rechoisir un seuil.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, correlate
from scipy.ndimage import label as ndlabel
from scipy.ndimage import maximum as ndmaximum
from skimage.measure import label, regionprops

STICK_THETA = np.pi / 4

PARAMS = {
    "k_line": 3.5,
    "line_len": 5,
    "dilate": 2,
    "k_low": 2.0,
    "half": 0.6,
    "k_peak": 3.0,
}
ANGLE_TOL = 0.6
ANGLE_MIN_LEN = 2.0


def robust_depth(a):
    med = np.median(a)
    return (med - a) / (1.4826 * np.median(np.abs(a - med)))


def dip_mask(image, k_line=3.5, line_len=5, dilate=2, k_low=2.0, half=0.5, k_peak=3.5):
    image = np.asarray(image, dtype=float)
    image = image - np.median(image, axis=1, keepdims=True)
    depth = robust_depth(image)
    along = robust_depth(correlate(image, np.eye(line_len) / line_len, mode="nearest"))
    seeds = along > k_line
    if dilate > 0:
        seeds = binary_dilation(seeds, iterations=dilate)
    cand = seeds & (depth > k_low)
    labels, n = ndlabel(cand, structure=np.ones((3, 3)))
    smooth_depth = correlate(depth, np.eye(2) / 2, mode="nearest")
    amp = np.concatenate([[0.0], ndmaximum(smooth_depth, labels, np.arange(1, n + 1))])[labels]
    return cand & (depth > half * amp) & (amp > k_peak)


def filter_by_angle(mask, stick_theta=STICK_THETA, angle_tol=ANGLE_TOL, min_len=ANGLE_MIN_LEN):
    labels = label(mask, connectivity=2)
    out = mask.copy()
    for region in regionprops(labels):
        if region.axis_major_length < min_len:
            continue
        phi = np.arctan2(np.cos(region.orientation), np.sin(region.orientation)) % np.pi
        deviation = abs((phi - stick_theta + np.pi / 2) % np.pi - np.pi / 2)
        if deviation > angle_tol:
            out[labels == region.label] = False
    return out


def predict(image):
    mask = dip_mask(image, **PARAMS)
    mask = filter_by_angle(mask)
    return mask.astype(np.uint8)


def detect(images: np.ndarray) -> np.ndarray:
    """Images brutes (N, 150, 150) -> masque 0/1."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 2:
        return predict(images).astype(np.float32)
    out = np.empty(images.shape, dtype=np.float32)
    for i in range(len(images)):
        out[i] = predict(images[i])
    return out
