"""Train one segmentation model on the infinite synthetic stream.

    uv run python -m detection.train --arch segformer --seed 0 --out runs/segformer_s0

One run = one (arch, seed, data/augmentation setting). Checkpoint selection and the
decision threshold use **val only**; test and OOD sets are touched by evaluate.py.
Resumable: re-running the same command continues from ``<out>/last.pt``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from detection.evalsets import THRESHOLDS, EvalSet, predict, score
from detection.metrics import pixel_counts, pixel_scores
from detection.models import build_model, n_params
from detection.synth import PoolSampler, SynthConfig


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True, choices=["unet", "transunet", "segformer", "vit"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="data/pool")
    ap.add_argument("--val", default="data/eval/val")
    ap.add_argument("--seed", type=int, default=0)
    # optimisation
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-3 unet, 5e-4 transformers")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    ap.add_argument("--compile", action="store_true")
    # data
    ap.add_argument("--pool-size", type=int, default=None, help="use only the first K geometries")
    ap.add_argument("--no-symmetry", action="store_true")
    ap.add_argument("--intensity-law", choices=["uniform", "loguniform"], default="uniform",
                    help="sandbox: loguniform over-samples low-SNR scenes (stage-2 regime)")
    ap.add_argument("--affine", action="store_true", help="sandbox: random scale+shear")
    ap.add_argument("--polarity", action="store_true", help="sandbox: random sign flip")
    ap.add_argument("--noise-jitter", type=float, default=0.0, help="sandbox: noise sigma x U(1-j,1+j)")
    # monitoring
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--sanity", action="store_true", help="overfit ONE fixed batch (pipeline check)")
    return ap.parse_args(argv)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_fn(logits, target, dice_weight):
    """Soft-target BCE + batch-global soft Dice (positives are ~0.35 % of pixels)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)
    return bce + dice_weight * dice, bce.detach(), dice.detach()


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def ema_update(ema, model, decay):
    for e, m in zip(ema.parameters(), model.parameters()):
        e.lerp_(m, 1 - decay)
    for e, m in zip(ema.buffers(), model.buffers()):
        e.copy_(m)


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    if dev.type == "cpu":
        amp_dtype = None
    if args.lr is None:
        args.lr = 1e-3 if args.arch == "unet" else 5e-4

    scfg = SynthConfig(
        symmetry=not args.no_symmetry,
        intensity_law=args.intensity_law,
        affine=args.affine,
        polarity=args.polarity,
        noise_jitter=args.noise_jitter,
    )
    sampler = PoolSampler(args.pool, dev, scfg, pool_size=args.pool_size, seed=args.seed)
    val = EvalSet(args.val, limit=args.n_val)

    # Fixed probe of *training* geometries (fresh-but-frozen noise) to measure the
    # geometry generalisation gap: tol-F1(train geometries) - tol-F1(val geometries).
    probe_sampler = PoolSampler(
        args.pool, dev, SynthConfig(symmetry=False), pool_size=args.pool_size, seed=12345
    )
    probe = [probe_sampler.sample(64) for _ in range(8)]
    del probe_sampler

    model = build_model(args.arch).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.999))

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0))) * 0.99 + 0.01

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    fwd = torch.compile(model) if args.compile else model

    start, best = 0, -1.0
    if (out / "last.pt").exists() and not args.sanity:
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        sampler.rng = ck["np_rng"]
        sampler.gen.set_state(ck["torch_gen"].cpu())
        start, best = ck["step"], ck["best"]
        print(f"resumed at step {start} (best val tol_f1 {best:.4f})")

    cfg = {**vars(args), "synth": asdict(scfg), "n_params": n_params(model), "device": str(dev),
           "pool_geometries": sampler.n}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"{args.arch}: {n_params(model) / 1e6:.2f} M params, pool {sampler.n} geometries, {dev}")

    log = (out / "log.jsonl").open("a")
    fixed = sampler.sample(args.bs) if args.sanity else None
    t0, seen = time.time(), 0
    model.train()
    for step in range(start, args.steps):
        x, y = fixed if fixed is not None else sampler.sample(args.bs)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = fwd(x)
        loss, bce, dice = loss_fn(logits.float(), y, args.dice_weight)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema_update(ema, model, args.ema if step > args.warmup else 0.0)
        seen += x.shape[0]

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at step {step}")

        if (step + 1) % args.log_every == 0:
            rec = {
                "step": step + 1, "loss": loss.item(), "bce": bce.item(), "dice": dice.item(),
                "gnorm": float(gnorm), "lr": sched.get_last_lr()[0],
                "img_per_s": seen / (time.time() - t0),
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            if args.sanity:
                continue
            probs = predict(ema, val.images, dev, amp_dtype=amp_dtype)
            v = score(probs, val, thr=None, device=dev)
            # geometry gap at the val-selected threshold
            ema.eval()
            with torch.no_grad(), torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                c = None
                for px, py in probe:
                    pc = pixel_counts(torch.sigmoid(ema(px).float()), py > 0.5, [v["thr"]])
                    c = pc if c is None else {k: c[k] + pc[k] for k in pc}
            train_geom = pixel_scores(c, [v["thr"]])[0]["tol_f1"]
            rec = {
                "step": step + 1, "eval": True, "val_sel": v["sel"], "val_tol_f1": v["tol_f1"], "val_f1": v["f1"],
                "val_iou": v["iou"], "val_obj_f1": v["obj_f1"], "val_thr": v["thr"],
                "train_geom_tol_f1": train_geom, "geom_gap": train_geom - v["tol_f1"],
                "elapsed_min": (time.time() - t0) / 60,
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print("EVAL", json.dumps(rec))
            if v["sel"] > best:  # (obj_f1 + tol_f1) / 2, not tol_f1 alone (dilation loophole)
                best = v["sel"]
                torch.save({"model": ema.state_dict(), "arch": args.arch, "step": step + 1,
                            "val": {k: v[k] for k in v if k != "pr_curve"}}, out / "best.pt")
            torch.save(
                {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "np_rng": sampler.rng,
                 "torch_gen": sampler.gen.get_state(), "step": step + 1, "best": best},
                out / "last.pt",
            )
            model.train()

    if args.sanity:
        # Soft targets put a non-zero floor under BCE and Dice: judge on tol-F1 instead.
        model.eval()
        with torch.no_grad():
            c = pixel_counts(torch.sigmoid(model(fixed[0]).float()), fixed[1] > 0.5, [0.5])
        f1 = pixel_scores(c, [0.5])[0]["tol_f1"]
        print(f"sanity: tol_f1 on the memorised batch = {f1:.3f} (expect > 0.95; else the pipeline is broken)")
    print(f"done. best val tol_f1 = {best:.4f}")


if __name__ == "__main__":
    main()
