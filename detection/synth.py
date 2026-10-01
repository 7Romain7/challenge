"""On-GPU batch synthesis from the template pool.

image = i * T(template) + blur(white * s_pix + row_stripe * s_h)

* ``i ~ U(intensity_range)``            — redrawn every time (exact generator law).
* noise                                 — redrawn every time (exact generator law,
                                          checked by ``data_gen verify``).
* ``T`` in the stick-preserving symmetry group {id, transpose, rot180, anti-transpose}:
  the generator's distribution is *exactly* invariant under it (stick theta ~ pi/4 is
  mapped to itself, the two charging-line families are swapped). Applied to the
  template **before** the noise, so the row-stripe noise keeps its true (fast-scan) axis.
  A horizontal/vertical flip alone would map theta -> -theta: **not** a symmetry.

Optional *sandbox* shifts (off for the baseline, see PROTOCOL.md):
* ``intensity_law="loguniform"`` — |i| log-uniform on the same support: ~40 % of scenes
                 below |i| = 4 instead of ~9 %. Challenge-2 frames live there (contrast
                 floor ``base`` = 3 away from the optimum).
* ``affine``   — random anisotropic scale + shear of template & label (other scan
                 resolution, other lever arms -> other stick slope/size in pixels).
* ``polarity`` — random sign flip (sensor biased on the other flank of a Coulomb peak).
* ``noise_jitter`` — noise sigmas scaled by U(1-j, 1+j).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, truncate: float = 4.0) -> torch.Tensor:
    """Same taps as ``scipy.ndimage.gaussian_filter``."""
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _pad_symmetric(x: torch.Tensor, r: int, dim: int) -> torch.Tensor:
    """scipy 'reflect' padding (edge sample repeated: d c b a | a b c d)."""
    n = x.shape[dim]
    lo = x.narrow(dim, 0, r).flip(dim)
    hi = x.narrow(dim, n - r, r).flip(dim)
    return torch.cat([lo, x, hi], dim=dim)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable blur of (B, 1, H, W) with scipy's taps and boundary mode."""
    if sigma <= 0:
        return x
    k = gaussian_kernel1d(sigma).to(x)
    r = (k.numel() - 1) // 2
    x = F.conv2d(_pad_symmetric(x, r, 3), k.view(1, 1, 1, -1))
    x = F.conv2d(_pad_symmetric(x, r, 2), k.view(1, 1, -1, 1))
    return x


def synth_noise(
    b: int, h: int, w: int, s_pix, s_h, sigma_blur: float,
    device: torch.device, gen: torch.Generator,
) -> torch.Tensor:
    """Blurred acquisition noise, (B, 1, H, W). ``s_pix``/``s_h`` scalars or (B,) tensors."""
    s_pix = torch.as_tensor(s_pix, dtype=torch.float32, device=device).reshape(-1, 1, 1, 1)
    s_h = torch.as_tensor(s_h, dtype=torch.float32, device=device).reshape(-1, 1, 1, 1)
    white = torch.randn(b, 1, h, w, device=device, generator=gen) * s_pix
    rows = torch.randn(b, 1, h, 1, device=device, generator=gen) * s_h
    return gaussian_blur(white + rows, sigma_blur)


def robust_normalize(x: torch.Tensor) -> torch.Tensor:
    """(x - median) / (1.4826 MAD), per image.

    Sticks are sparse (<1% of pixels), so median/MAD measure the *background* and
    the noise: the network sees amplitudes in units of noise sigma (an SNR), which
    is what transfers to a real device with another gain. A plain z-score would be
    inflated by bright scenes and hide the absolute SNR.
    """
    flat = x.flatten(1)
    med = flat.median(dim=1, keepdim=True).values
    mad = (flat - med).abs().median(dim=1, keepdim=True).values
    out = (flat - med) / (1.4826 * mad + 1e-6)
    return out.view_as(x)


@dataclass
class SynthConfig:
    intensity_range: tuple[float, float] = (-33.0, -1.0)
    noise_sigma_pixel: float = 0.9
    noise_sigma_h: float = 0.7
    sigma_blur: float = 0.5
    symmetry: bool = True
    intensity_law: str = "uniform"  # generator law; "loguniform" = sandbox low-SNR emphasis
    affine: bool = False
    affine_scale: float = 0.35  # log-uniform scale in [1/(1+s), 1+s] per axis
    affine_shear: float = 0.25
    polarity: bool = False
    noise_jitter: float = 0.0


class PoolSampler:
    """Draws training batches (normalized image, soft target) directly on ``device``."""

    def __init__(
        self,
        pool_dir: str | Path,
        device: torch.device,
        cfg: SynthConfig,
        pool_size: int | None = None,
        seed: int = 0,
        in_memory: bool = False,  # mmap: the OS page cache is shared by parallel runs
    ) -> None:
        pool_dir = Path(pool_dir)
        meta = json.loads((pool_dir / "meta.json").read_text())
        g = meta["generator_config"]
        cfg.intensity_range = tuple(g["intensity_range"])
        cfg.noise_sigma_pixel = g["noise_sigma_pixel"]
        cfg.noise_sigma_h = g["noise_sigma_h"]
        cfg.sigma_blur = g["sigma_blur"]
        self.cfg = cfg
        mode = None if in_memory else "r"
        self.templates = np.load(pool_dir / "templates.npy", mmap_mode="r")
        self.soft = np.load(pool_dir / "soft.npy", mmap_mode="r")
        n = len(self.templates) if pool_size is None else min(pool_size, len(self.templates))
        if mode is None:
            self.templates = np.ascontiguousarray(self.templates[:n])
            self.soft = np.ascontiguousarray(self.soft[:n])
        self.n = n
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, device=self.device, generator=self.gen)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = np.sort(self.rng.integers(0, self.n, size=batch_size))
        t = torch.from_numpy(self.templates[idx].astype(np.float32)).to(self.device, non_blocking=True)
        y = torch.from_numpy(self.soft[idx].astype(np.float32) / 255.0).to(self.device, non_blocking=True)
        t, y = t[:, None], y[:, None]
        b, _, h, w = t.shape
        c = self.cfg

        if c.symmetry and h == w:
            tr = self._rand(b, 1, 1, 1) < 0.5
            t = torch.where(tr, t.transpose(-1, -2), t)
            y = torch.where(tr, y.transpose(-1, -2), y)
            r180 = self._rand(b, 1, 1, 1) < 0.5
            t = torch.where(r180, t.flip(-1, -2), t)
            y = torch.where(r180, y.flip(-1, -2), y)

        if c.affine:
            t, y = self._affine(t, y)

        lo, hi = c.intensity_range
        u = self._rand(b, 1, 1, 1)
        if c.intensity_law == "loguniform":  # same support, more mass at low SNR
            i = -torch.exp(math.log(-hi) + u * (math.log(-lo) - math.log(-hi)))
        else:
            i = lo + (hi - lo) * u
        if c.polarity:
            i = torch.where(self._rand(b, 1, 1, 1) < 0.5, -i, i)
        js = 1.0 + c.noise_jitter * (2 * self._rand(b) - 1)
        noise = synth_noise(
            b, h, w, c.noise_sigma_pixel * js, c.noise_sigma_h * js, c.sigma_blur, self.device, self.gen
        )
        x = i * t + noise
        return robust_normalize(x), y

    def _affine(self, t: torch.Tensor, y: torch.Tensor):
        """Random anisotropic scale + shear around the image centre (sandbox)."""
        b = t.shape[0]
        c = self.cfg
        ls = math.log1p(c.affine_scale)
        sx = torch.exp((2 * self._rand(b) - 1) * ls)
        sy = torch.exp((2 * self._rand(b) - 1) * ls)
        sh = (2 * self._rand(b) - 1) * c.affine_shear
        theta = torch.zeros(b, 2, 3, device=self.device)
        # grid_sample maps output coords -> input coords, hence the inverse scales.
        theta[:, 0, 0] = 1 / sx
        theta[:, 0, 1] = sh
        theta[:, 1, 1] = 1 / sy
        grid = F.affine_grid(theta, list(t.shape), align_corners=False)
        t = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        y = F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return t, y
