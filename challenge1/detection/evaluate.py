"""Evaluate trained runs on test + OOD sets, aggregate over seeds, draw figures.

    uv run python -m detection.evaluate --runs "runs/*" --out challenge1/results/transformers

Per run: the decision threshold is re-derived on **val** (tol-F1 optimal) and then
frozen for test and every OOD set. Runs named ``<group>_s<seed>`` are aggregated
per group as mean +- std over seeds.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from detection.evalsets import EvalSet, predict, score
from detection.models import build_model, n_params

KEYS = ("tol_f1", "f1", "iou", "obj_f1", "obj_precision", "obj_recall")


def evaluate_run(run: Path, sets: dict[str, EvalSet], val: EvalSet, dev, force=False) -> dict:
    out_file = run / "eval.json"
    if out_file.exists() and not force:
        return json.loads(out_file.read_text())
    ck = torch.load(run / "best.pt", map_location=dev, weights_only=False)
    model = build_model(ck["arch"]).to(dev)
    model.load_state_dict(ck["model"])
    amp = torch.bfloat16 if dev.type == "cuda" else None

    v = score(predict(model, val.images, dev, amp_dtype=amp), val, thr=None, device=dev)
    thr = v["thr"]
    res = {"run": run.name, "arch": ck["arch"], "step": ck["step"], "n_params": n_params(model),
           "thr": thr, "sets": {"val": v}}
    for name, es in sets.items():
        t = time.time()
        probs = predict(model, es.images, dev, amp_dtype=amp)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        r = score(probs, es, thr=thr, device=dev)
        r["ms_per_image"] = 1000 * (time.time() - t) / len(es)
        res["sets"][name] = r
        print(f"  {run.name:28s} {name:16s} tol_f1={r['tol_f1']:.3f} obj_f1={r['obj_f1']:.3f}")
    out_file.write_text(json.dumps(res, indent=1))
    return res


def group_of(run_name: str) -> str:
    return re.sub(r"_s\d+$", "", run_name)


def aggregate(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[group_of(r["run"])].append(r)
    agg = {}
    for g, rs in sorted(groups.items()):
        agg[g] = {"n_seeds": len(rs), "n_params": rs[0]["n_params"], "sets": {}}
        for s in rs[0]["sets"]:
            agg[g]["sets"][s] = {
                k: (float(np.mean([r["sets"][s][k] for r in rs])),
                    float(np.std([r["sets"][s][k] for r in rs])))
                for k in KEYS
            }
            bins = rs[0]["sets"][s]["recall_by_amplitude"].keys()
            agg[g]["sets"][s]["recall_by_amplitude"] = {
                b: float(np.mean([r["sets"][s]["recall_by_amplitude"][b][0] or 0 for r in rs]))
                for b in bins
            }
    return agg


def write_tables(agg: dict, out: Path) -> None:
    sets = list(next(iter(agg.values()))["sets"].keys())
    lines = []
    for key in ("tol_f1", "obj_f1", "iou"):
        lines += [f"\n### {key} (mean ± std over seeds)\n",
                  "| group | params | " + " | ".join(sets) + " |",
                  "|---|---|" + "---|" * len(sets)]
        for g, a in agg.items():
            cells = [f"{a['sets'][s][key][0]:.3f} ± {a['sets'][s][key][1]:.3f}" for s in sets]
            lines.append(f"| {g} (n={a['n_seeds']}) | {a['n_params'] / 1e6:.1f}M | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def figures(agg: dict, results: list[dict], sets: dict[str, EvalSet], out: Path, dev) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. recall vs stick amplitude on test (the detection limit)
    fig, ax = plt.subplots(figsize=(6, 4))
    for g, a in agg.items():
        rb = a["sets"]["test"]["recall_by_amplitude"]
        ax.plot(range(len(rb)), list(rb.values()), "o-", label=g)
        ax.set_xticks(range(len(rb)), list(rb.keys()))
    ax.set_xlabel("|stick amplitude| (noise σ_pixel = 0.9)")
    ax.set_ylabel("object recall")
    ax.set_title("Detection limit (test)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "recall_vs_amplitude.png", dpi=150)

    # 2. robustness: obj-F1 per set per group
    names = [s for s in next(iter(agg.values()))["sets"] if s != "val"]
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 3, 4))
    wbar = 0.8 / len(agg)
    for i, (g, a) in enumerate(agg.items()):
        m = [a["sets"][s]["obj_f1"][0] for s in names]
        e = [a["sets"][s]["obj_f1"][1] for s in names]
        ax.bar(np.arange(len(names)) + i * wbar, m, wbar, yerr=e, label=g, capsize=2)
    ax.set_xticks(np.arange(len(names)) + 0.4 - wbar / 2, names, rotation=30, ha="right")
    ax.set_ylabel("object F1")
    ax.set_title("In-distribution vs OOD")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)

    # 3. qualitative: same test images, every group's first run
    es = sets["test"]
    idx = np.linspace(0, len(es) - 1, 4).astype(int)
    firsts = {}
    for r in results:
        firsts.setdefault(group_of(r["run"]), r)
    fig, axes = plt.subplots(len(idx), 2 + len(firsts), figsize=(2.2 * (2 + len(firsts)), 2.2 * len(idx)))
    for row, k in enumerate(idx):
        axes[row, 0].imshow(es.images[k], origin="lower", cmap="plasma")
        axes[row, 1].imshow(es.masks[k], origin="lower", cmap="gray")
        for col, (g, r) in enumerate(firsts.items()):
            ck = torch.load(Path(r["_dir"]) / "best.pt", map_location=dev, weights_only=False)
            m = build_model(ck["arch"]).to(dev)
            m.load_state_dict(ck["model"])
            p = predict(m, es.images[k : k + 1], dev)[0]
            axes[row, 2 + col].imshow(p > r["thr"], origin="lower", cmap="gray")
            if row == 0:
                axes[row, 2 + col].set_title(g, fontsize=7)
    axes[0, 0].set_title("image")
    axes[0, 1].set_title("GT mask")
    for a in axes.flat:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--eval-dir", default="data/eval")
    ap.add_argument("--out", default="challenge1/results/transformers")
    ap.add_argument("--force", action="store_true", help="recompute eval.json")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = Path(args.eval_dir)
    val = EvalSet(ev / "val")
    sets = {p.name: EvalSet(p) for p in sorted(ev.iterdir()) if p.is_dir() and p.name != "val"}
    sets = {"test": sets.pop("test"), **sets}

    runs = sorted({Path(r) for pat in args.runs for r in glob.glob(pat)} )
    runs = [r for r in runs if (r / "best.pt").exists()]
    results = []
    for r in runs:
        res = evaluate_run(r, sets, val, dev, args.force)
        res["_dir"] = str(r)
        results.append(res)

    agg = aggregate(results)
    (out / "summary.json").write_text(json.dumps(agg, indent=1))
    write_tables(agg, out)
    figures(agg, results, sets, out, dev)
    print(f"\nwrote {out}/summary.md, summary.json, recall_vs_amplitude.png, robustness.png, examples.png")


if __name__ == "__main__":
    main()
