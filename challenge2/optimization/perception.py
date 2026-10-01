"""Training-free perception: stripe removal + oriented line-filter bank.

Everything is estimated from the image itself (no instrument constants): the noise
level is a robust MAD, the polarity of the interdots is inferred on the first frame,
and orientation is not assumed (filter bank over angles).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x).ravel()
    return float(1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12)


def _kernel(theta: float, length: float, sw: float) -> np.ndarray:
    h = int(np.ceil(length / 2)) + 4
    y, x = np.mgrid[-h : h + 1, -h : h + 1].astype(float)
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    along = (np.abs(u) <= length / 2).astype(float)
    centre = along * np.exp(-0.5 * (v / sw) ** 2)
    flank = along * ((np.abs(v) >= 2 * sw + 0.5) & (np.abs(v) <= 2 * sw + 2.5))
    return centre / centre.sum() - flank / flank.sum()


@dataclass
class Frame:
    """Features of one image. Pixel (row, col) <-> volts: origin + (col, row) * step."""

    resp: np.ndarray  # filter response, background-subtracted, in image units
    readout: np.ndarray  # max over a 5x5 box of resp, noise-bias removed
    sigma: float  # robust noise std of ``readout`` (image units)
    peaks: np.ndarray  # (n, 3): row, col, response of detected interdots


class Perception:
    def __init__(self, length: float = 7.0, sigma_w: float = 0.8, n_orient: int = 8,
                 k_det: float = 5.0, detector=None) -> None:
        self.detector = detector  # optional learned mask (e.g. M5Min); replaces the k*sigma test
        self.kernels = [_kernel(t, length, sigma_w)
                        for t in np.linspace(0, np.pi, n_orient, endpoint=False)]
        self.k_det = k_det
        self.polarity: float | None = None

    def _response(self, img: np.ndarray, s: float) -> np.ndarray:
        d = s * (img - np.median(img, axis=1, keepdims=True))  # stripe removal
        return np.max([ndi.correlate(d, k, mode="nearest") for k in self.kernels], axis=0)

    def _calibrate_polarity(self, img: np.ndarray) -> None:
        best = None
        for s in (+1.0, -1.0):
            r = self._response(img, s)
            score = (np.percentile(r, 99.9) - np.median(r)) / robust_sigma(r)
            if best is None or score > best[0]:
                best = (score, s)
        self.polarity = best[1]

    def process(self, img: np.ndarray) -> Frame:
        if self.polarity is None:
            self._calibrate_polarity(img)
        r = self._response(img, self.polarity)
        r = r - np.median(r)
        readout_raw = ndi.maximum_filter(r, size=5)
        m_bg = float(np.median(readout_raw))  # noise-induced bias of the max readout
        readout = readout_raw - m_bg
        sigma = robust_sigma(readout_raw)
        if self.detector is None:
            on_stick = r > self.k_det * robust_sigma(r)
        else:  # 3x3 tolerance: the filter peak and the mask may be 1 px apart
            on_stick = ndi.maximum_filter(self.detector.logit(img), size=3) > self.detector.thr
        is_peak = (ndi.maximum_filter(r, size=7) == r) & on_stick
        rows, cols = np.nonzero(is_peak)
        peaks = np.c_[rows, cols, r[rows, cols]] if len(rows) else np.zeros((0, 3))
        return Frame(resp=r, readout=readout, sigma=sigma, peaks=peaks)

    @staticmethod
    def read_amplitudes(frame: Frame, rc: np.ndarray, margin: int = 3) -> np.ndarray:
        """Amplitude at expected pixel positions (NaN if close to / outside the border)."""
        n_v, n_h = frame.readout.shape
        out = np.full(len(rc), np.nan)
        for i, (r, c) in enumerate(rc):
            ri, ci = int(round(r)), int(round(c))
            if margin <= ri < n_v - margin and margin <= ci < n_h - margin:
                out[i] = frame.readout[ri, ci]
        return out
