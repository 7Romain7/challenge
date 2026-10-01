"""P4 for M5: fitting sets drawn from the SAME pool geometries, only the synth law changes.

    uv run --extra train python -m detection.make_fit_sets data/pool data/fit_p4 500
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from detection.synth import PoolSampler, SynthConfig

pool, out, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
dev = torch.device("cpu")
variants = {
    "base": SynthConfig(),  # generator law (checked by data_gen verify)
    "lowsnr": SynthConfig(intensity_law="loguniform"),
    # M5's prep() hard-codes the sign (sticks negative-going): polarity is a physics prior
    # an experimentalist sets from the sensor flank, so it is NOT randomised here.
    "shift": SynthConfig(affine=True, noise_jitter=0.3),
}
for name, cfg in variants.items():
    s = PoolSampler(pool, dev, cfg, seed=0)  # same seed -> same geometry indices
    xs, ys = [], []
    for _ in range(n // 50):
        x, y = s.sample(50)
        xs.append(x[:, 0].numpy())
        ys.append((y[:, 0] > 0.5).numpy())
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "images.npy", np.concatenate(xs).astype(np.float32))
    np.save(d / "masks.npy", np.concatenate(ys).astype(np.uint8))
    (d / "sticks.jsonl").write_text("".join(json.dumps({"sticks": []}) + "\n" for _ in range(n)))
    meta = {"name": f"fit_{name}", "n": n, "synth": {k: str(v) for k, v in vars(cfg).items()},
            "scan_window": {"step_h": 0.002}}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    print(name, np.concatenate(ys).mean())
