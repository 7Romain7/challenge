"""Generate the training *pool* and the frozen evaluation sets.

Two very different products :

* ``pool``  — noise-free **templates** + soft labels. A template is the clean,
  blurred render divided by the scene intensity, so at train time we can redraw
  the scene intensity and the acquisition noise on the GPU, exactly as the
  generator would (blur is linear: blur(noise + sticks) = blur(noise) + blur(sticks)).
  Geometry (the slow part, ~0.2 s/scene) is drawn once; noise/intensity are infinite.
* ``sets``  — frozen val / test / OOD sets rendered by the **official** generator
  (official noise, official binary mask, stick metadata in pixels). These are the
  only images metrics are ever computed on.

Seeds are disjoint by construction: pool = [0, 1e7), val = 1e7 + k, test = 2e7 + k,
OOD set #s = 3e7 + 1e6 * s + k. Every sample is reproducible from its seed alone.

    uv run python -m detection.data_gen pool --n 30000 --out data/pool --workers 32
    uv run python -m detection.data_gen sets --out data/eval --workers 32
    uv run python -m detection.data_gen verify --pool data/pool --eval data/eval
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from csd import GENERATOR, GeneratorConfig, ScanWindow
from csd.challenge import new_experiment
from csd.generator import build_interdots, generate_label, render_csd, scene_window

SEED_VAL = 10_000_000
SEED_TEST = 20_000_000
SEED_OOD = 30_000_000


def _noiseless(cfg: GeneratorConfig) -> GeneratorConfig:
    return replace(cfg, noise_sigma_pixel=0.0, noise_sigma_h=0.0)


def _scene_scale(interdots, line_specs, cfg: GeneratorConfig) -> float:
    """Scene-wide mean intensity ``i_mean`` (negative).

    Exact when the scene has a connecting line (line intensity = i_mean * frac),
    otherwise the mean stick intensity (unbiased, within ~3%).
    """
    if line_specs:
        return line_specs[0].intensity / cfg.line_intensity_frac
    if interdots:
        return float(np.mean([s.intensity for s in interdots]))
    return -1.0  # empty scene: template is all zeros anyway


def _stick_records(interdots, window: ScanWindow) -> list[dict]:
    """Stick metadata in **pixel** units (col = x/step_h, row = y/step_v)."""
    out = []
    for s in interdots:
        out.append(
            {
                "col": float(s.middle[0] / window.step_h),
                "row": float(s.middle[1] / window.step_v),
                "len_px": float(s.length / window.step_h),
                "wid_px": float(s.width / window.step_h),
                "theta": float(s.theta),
                "intensity": float(s.intensity),
            }
        )
    return out


# --------------------------------------------------------------------------- workers
def _pool_sample(seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    cfg = GENERATOR
    window = scene_window(cfg)
    np.random.seed(seed)
    interdots, line_specs = build_interdots(cfg, window)
    clean = render_csd(interdots, line_specs, window, _noiseless(cfg), normalize=False)
    soft = generate_label(interdots, window, cfg)
    i_mean = _scene_scale(interdots, line_specs, cfg)
    template = (clean / i_mean).astype(np.float16)  # sticks ~ +1, lines ~ +0.5
    soft_u8 = np.clip(np.round(soft * 255), 0, 255).astype(np.uint8)
    return template, soft_u8, i_mean


def _generator_sample(args) -> dict:
    """One official sample from a (possibly shifted) GeneratorConfig / window."""
    seed, cfg_dict, win_dict = args
    cfg = GeneratorConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg_dict.items()})
    window = ScanWindow(**win_dict)
    np.random.seed(seed)
    interdots, line_specs = build_interdots(cfg, window)
    image = render_csd(interdots, line_specs, window, cfg, normalize=False)
    clean = render_csd(interdots, line_specs, window, _noiseless(cfg), normalize=False)
    label = generate_label(interdots, window, cfg)
    return {
        "image": image.astype(np.float32),
        "mask": (label > 0.5).astype(np.uint8),
        "clean": clean.astype(np.float32),
        "sticks": _stick_records(interdots, window),
    }


def _stage2_sample(seed: int) -> dict:
    """A challenge-2 frame at a random barrier point, plungers re-centred on the drift.

    Labels come from the simulator internals — **evaluation only**, never used by a
    detector or an optimizer.
    """
    exp = new_experiment(seed=seed)
    sim = exp._sim
    rng = np.random.default_rng(seed + 1)
    b = rng.uniform(-0.5, 0.5, size=3)
    drift = sim.drift_matrix.drift(b)
    wp = {
        "g1": float(b[0]),
        "g3": float(b[1]),
        "g5": float(b[2]),
        "g2": float(sim.start["g2"] + drift[0] + rng.uniform(-0.03, 0.03)),
        "g4": float(sim.start["g4"] + drift[1] + rng.uniform(-0.03, 0.03)),
    }
    window = sim.scan_window
    np.random.seed(seed + 2)
    image = sim.render(working_point=wp, scan_window=window)
    shift = sim._stick_shift(wp, window)
    factors = sim._stick_factors(wp)
    interdots = [
        replace(s, middle=s.middle + shift, intensity=s.intensity * f)
        for s, f in zip(sim.base_interdots, factors)
    ]
    label = generate_label(interdots, window, sim.generator_config)
    margin = 0.02
    visible = [
        s
        for s in interdots
        if -margin <= s.middle[0] <= window.span_h + margin
        and -margin <= s.middle[1] <= window.span_v + margin
    ]
    return {
        "image": image.astype(np.float32),
        "mask": (label > 0.5).astype(np.uint8),
        "clean": None,
        "sticks": _stick_records(visible, window),
    }


# --------------------------------------------------------------------------- pool
def make_pool(n: int, out: Path, workers: int, seed_offset: int = 0) -> None:
    assert seed_offset + n <= SEED_VAL, "pool seeds would collide with val seeds"
    out.mkdir(parents=True, exist_ok=True)
    window = scene_window(GENERATOR)
    h, w = window.n_v, window.n_h
    templates = np.lib.format.open_memmap(out / "templates.npy", "w+", np.float16, (n, h, w))
    soft = np.lib.format.open_memmap(out / "soft.npy", "w+", np.uint8, (n, h, w))
    i_mean = np.zeros(n, np.float32)

    t0 = time.time()
    seeds = range(seed_offset, seed_offset + n)
    with Pool(workers) as p:
        for k, (t, s, im) in enumerate(p.imap(_pool_sample, seeds, chunksize=16)):
            templates[k], soft[k], i_mean[k] = t, s, im
            if (k + 1) % 2000 == 0:
                rate = (k + 1) / (time.time() - t0)
                print(f"  pool {k + 1}/{n}  {rate:.0f}/s  eta {(n - k - 1) / rate / 60:.1f} min")
    templates.flush()
    soft.flush()
    np.save(out / "i_mean.npy", i_mean)
    meta = {
        "n": n,
        "seed_offset": seed_offset,
        "image_shape": [h, w],
        "generator_config": asdict(GENERATOR),
        "scan_window": asdict(window),
        "templates": "clean blurred render / i_mean (float16); image = i * template + blur(noise)",
        "soft": "soft label * 255 (uint8); official binary mask = soft > 0.5",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"pool: {n} templates in {(time.time() - t0) / 60:.1f} min -> {out}")


# --------------------------------------------------------------------------- eval sets
def eval_set_specs(n_val: int, n_test: int, n_ood: int) -> dict[str, dict]:
    """Every frozen set: name -> (seed base, size, config, window, kind).

    OOD sets are **report-only**: never used for checkpoint selection or thresholds.
    """
    g = GENERATOR
    w = scene_window(g)
    specs = {
        "val": dict(seed=SEED_VAL, n=n_val, cfg=g, win=w, kind="gen"),
        "test": dict(seed=SEED_TEST, n=n_test, cfg=g, win=w, kind="gen"),
        # Device-to-device spread of the interdot slope (lever arms / cross-capacitance).
        "ood_theta_shift": dict(cfg=replace(g, stick_theta=math.pi / 4 + 0.35), win=w),
        "ood_theta_wide": dict(cfg=replace(g, stick_theta_jitter=0.3), win=w),
        # A noisier fridge / shorter integration time.
        "ood_noise_up": dict(cfg=replace(g, noise_sigma_pixel=1.35, noise_sigma_h=1.05), win=w),
        # Same device, other scan resolution (sticks 2x bigger / 1.5x smaller in pixels).
        "ood_zoom_in": dict(cfg=g, win=ScanWindow(0.15, 0.15, 1e-3, 1e-3)),
        "ood_zoom_out": dict(cfg=g, win=ScanWindow(0.45, 0.45, 3e-3, 3e-3)),
        # The downstream distribution: challenge-2 frames (drift + region contrast).
        "ood_stage2": dict(cfg=g, win=w, kind="stage2"),
    }
    for s, name in enumerate(n for n in specs if n.startswith("ood_")):
        specs[name].update(seed=SEED_OOD + 1_000_000 * s, n=n_ood)
        specs[name].setdefault("kind", "gen")
    return specs


def make_sets(out: Path, workers: int, n_val: int, n_test: int, n_ood: int, only=None) -> None:
    specs = eval_set_specs(n_val, n_test, n_ood)
    for name, sp in specs.items():
        if only and name not in only:
            continue
        t0 = time.time()
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        seeds = range(sp["seed"], sp["seed"] + sp["n"])
        with Pool(workers) as p:
            if sp["kind"] == "stage2":
                samples = p.map(_stage2_sample, seeds, chunksize=8)
            else:
                cfg_d, win_d = asdict(sp["cfg"]), asdict(sp["win"])
                samples = p.map(_generator_sample, [(s, cfg_d, win_d) for s in seeds], chunksize=8)
        np.save(d / "images.npy", np.stack([s["image"] for s in samples]))
        np.save(d / "masks.npy", np.stack([s["mask"] for s in samples]))
        if samples[0]["clean"] is not None:
            np.save(d / "clean.npy", np.stack([s["clean"] for s in samples]).astype(np.float16))
        with (d / "sticks.jsonl").open("w") as f:
            for k, s in enumerate(samples):
                f.write(json.dumps({"image_id": k, "sticks": s["sticks"]}) + "\n")
        meta = {
            "name": name,
            "n": sp["n"],
            "seed_base": sp["seed"],
            "kind": sp["kind"],
            "generator_config": asdict(sp["cfg"]),
            "scan_window": asdict(sp["win"]),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  {name}: {sp['n']} samples in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- verify
def verify(pool: Path, eval_dir: Path) -> None:
    """Check the GPU noise model against the official one, and pool/val consistency.

    Official residual = image - clean (both official renders, same geometry) is
    exactly blur(acquisition noise). We compare its pixel std, row-mean std (the
    stripe component) and lag-1 autocorrelations with the torch re-implementation.
    """
    import torch

    from detection.synth import synth_noise

    val = eval_dir / "val"
    img = np.load(val / "images.npy")
    clean = np.load(val / "clean.npy").astype(np.float32)
    res = img - clean

    syn = synth_noise(
        res.shape[0], *res.shape[1:], GENERATOR.noise_sigma_pixel, GENERATOR.noise_sigma_h,
        GENERATOR.sigma_blur, torch.device("cpu"), torch.Generator().manual_seed(0),
    )[:, 0].numpy()

    def stats(x):
        xc = x - x.mean(axis=(1, 2), keepdims=True)
        lag_h = (xc[:, :, 1:] * xc[:, :, :-1]).mean() / xc.var()
        lag_v = (xc[:, 1:, :] * xc[:, :-1, :]).mean() / xc.var()
        return {
            "pixel_std": float(x.std()),
            "row_mean_std": float(x.mean(axis=2).std()),
            "lag1_h": float(lag_h),
            "lag1_v": float(lag_v),
        }

    print("official noise :", stats(res))
    print("synthetic noise:", stats(syn))

    t = np.load(pool / "templates.npy", mmap_mode="r")
    s = np.load(pool / "soft.npy", mmap_mode="r")
    m = np.load(val / "masks.npy")
    print(f"pool  positive fraction (soft>0.5): {(s[:2000] > 127).mean():.5f}")
    print(f"val   positive fraction           : {m.mean():.5f}")
    print(f"pool  template p99.9 over stick pixels (~1 expected): "
          f"{float(np.percentile(t[:2000][s[:2000] > 127], 99.9)):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pool")
    a.add_argument("--n", type=int, default=30_000)
    a.add_argument("--out", default="data/pool")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--seed-offset", type=int, default=0)
    b = sub.add_parser("sets")
    b.add_argument("--out", default="data/eval")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--n-val", type=int, default=1000)
    b.add_argument("--n-test", type=int, default=2000)
    b.add_argument("--n-ood", type=int, default=500)
    b.add_argument("--only", nargs="*", help="subset of set names")
    c = sub.add_parser("verify")
    c.add_argument("--pool", default="data/pool")
    c.add_argument("--eval", default="data/eval")
    args = ap.parse_args()

    if args.cmd == "pool":
        make_pool(args.n, Path(args.out), args.workers, args.seed_offset)
    elif args.cmd == "sets":
        make_sets(Path(args.out), args.workers, args.n_val, args.n_test, args.n_ood, args.only)
    else:
        verify(Path(args.pool), Path(args.eval))


if __name__ == "__main__":
    main()
