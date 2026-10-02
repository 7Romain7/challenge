"""OOD sets with noise the generator never produces (report-only, never used to select).

Built from the frozen *test* set (official images, official masks, same sticks), so any
drop is due to the added noise alone. Time axis = raster order (fast scan along rows).

* ood_noise_1f    — 1/f (pink) noise along the acquisition time series, std = 0.9
                    (same as the white pixel noise), plus a slow quadratic drift of the
                    sensor background over the frame (peak-to-peak ~3).
* ood_charge_jump — random telegraph noise: the sensor level switches between two values
                    (amplitude ~ U(1, 3)) at Poisson times (~4 switches per frame), the
                    signature of a charge trap near the sensor dot.

    uv run python -m detection.make_real_noise data/eval_light/test data/eval_light
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
img = np.load(src / "images.npy").astype(np.float32)
n, h, w = img.shape
T = h * w
rng = np.random.default_rng(40_000_000)


def pink(t: int) -> np.ndarray:
    f = np.fft.rfftfreq(t)
    spec = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / np.sqrt(np.maximum(f, 1.0 / t))
    x = np.fft.irfft(spec, n=t)
    return x / x.std()


def drift() -> np.ndarray:
    tt = np.linspace(-1, 1, T)
    a, b = rng.uniform(-1.5, 1.5, 2)
    return a * tt + b * (tt**2 - 1 / 3)


def telegraph() -> np.ndarray:
    k = rng.poisson(4)
    cuts = np.sort(rng.integers(0, T, size=k))
    level = np.zeros(T)
    state = rng.integers(0, 2)
    amp = rng.uniform(1.0, 3.0)
    prev = 0
    for c in list(cuts) + [T]:
        level[prev:c] = state * amp
        state, prev = 1 - state, c
    return level - level.mean()


sets = {
    "ood_noise_1f": lambda: 0.9 * pink(T) + drift(),
    "ood_charge_jump": telegraph,
}
for name, fn in sets.items():
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    x = img + np.stack([fn().reshape(h, w) for _ in range(n)]).astype(np.float32)
    np.save(d / "images.npy", x)
    shutil.copy(src / "masks.npy", d / "masks.npy")
    shutil.copy(src / "sticks.jsonl", d / "sticks.jsonl")
    meta = json.loads((src / "meta.json").read_text())
    meta.update(name=name, kind="test+extra_noise", extra_noise=sets[name].__doc__ or name)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    print(name, x.shape, "added-noise std", float((x - img).std()))
