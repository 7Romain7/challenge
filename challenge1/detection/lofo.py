"""Leave-one-family-out (LOFO): does randomising K artefact families protect against an unseen one?

For each artefact family f of the robustness suite, a U-Net is trained on the randomised
generator with every family EXCEPT f, then scored on the robustness sets of f (never seen in
training). Two references bracket it, same recipe otherwise:
  * ``none`` — no artefact at all (only the physical input conditioning + polarity);
  * ``all``  — every family, f included (in-distribution upper bound).
Read: transfer = LOFO - none (what the other families buy on an unseen one),
      gap      = all - LOFO  (what is lost by not having seen f).
Two capacities: unet16_robust (~0.5 M params) and unet_robust (~1.9 M).

Thresholds/checkpoints are chosen on val only (train.py / evaluate.py); the robustness sets
are report-only.

    python -m detection.lofo jobs                 # one "name|train args" line per run
    python -m detection.lofo report --out challenge1/results/lofo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FAMILIES = ["white", "pink", "drift", "jumps", "stripes", "lowpass", "saturate", "spikes", "polarity"]
ARTIFACTS = FAMILIES[:-1]  # polarity is its own flag in train.py
SET_PREFIX = {"white": "white_x", "pink": "pink1f_x", "drift": "drift_pp", "jumps": "jumps_amp",
              "stripes": "stripes_x", "lowpass": "lowpass_tau", "saturate": "saturate_c",
              "spikes": "spikes_", "polarity": "polarity_flip"}
ARCHS = {"u16": "unet16_robust", "u32": "unet_robust"}
COMMON = "--seed 0 --steps 8000 --warmup 300 --eval-every 1000 --intensity-law loguniform --p-artifact 0.2"


def train_args(arch: str, held: str | None) -> str:
    """held = family left out; 'all' = nothing left out; 'none' = no artefact."""
    arts = [] if held == "none" else [a for a in ARTIFACTS if a != held]
    pol = "" if held == "polarity" else "--polarity"
    art = f"--artifacts {','.join(arts)}" if arts else ""
    return f"--arch {ARCHS[arch]} {COMMON} {pol} {art}".strip()


def jobs() -> list[tuple[str, str]]:
    out = []
    for a in ARCHS:
        for held in ["all", "none", *FAMILIES]:
            name = f"lofo_{a}_{held}" if held in ("all", "none") else f"lofo_{a}_no-{held}"
            out.append((name, train_args(a, held)))
    return out


def family_score(ev: dict, fam: str) -> tuple[float, float]:
    """Mean and worst obj-F1 over the severity levels of one family."""
    v = [s["obj_f1"] for k, s in ev["sets"].items() if k.startswith(SET_PREFIX[fam])]
    return float(np.mean(v)), float(np.min(v))


def report(runs: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ev = {p.parent.name: json.loads(p.read_text()) for p in runs.glob("lofo_*/eval.json")}
    rows, table = [], {}
    for a, arch in ARCHS.items():
        ref_all, ref_none = ev.get(f"lofo_{a}_all"), ev.get(f"lofo_{a}_none")
        for fam in FAMILIES:
            lofo = ev.get(f"lofo_{a}_no-{fam}")
            cell = {k: (family_score(e, fam) if e else None)
                    for k, e in (("none", ref_none), ("lofo", lofo), ("all", ref_all))}
            cell["clean_lofo"] = lofo["sets"]["test"]["obj_f1"] if lofo else None
            table.setdefault(fam, {})[a] = cell
        clean = {k: (e["sets"]["test"]["obj_f1"] if e else None) for k, e in (("none", ref_none), ("all", ref_all))}
        table.setdefault("_clean_test", {})[a] = clean
    (out / "lofo.json").write_text(json.dumps({"table": table, "n_runs": len(ev)}, indent=1))

    f = lambda x: "—" if x is None else f"{x[0]:.3f} ({x[1]:.2f})"
    g = lambda x: "—" if x is None else f"{x:.3f}"
    rows += ["obj F1 on the held-out family: mean over its severity levels (worst level).",
             "none = no artefact in training; LOFO = all families but this one; all = every family.", ""]
    for a, arch in ARCHS.items():
        n = next((e["n_params"] for k, e in ev.items() if k.startswith(f"lofo_{a}_")), None)
        rows += [f"### {arch} ({n / 1e6:.2f} M params)" if n else f"### {arch}", "",
                 "| held-out family | none | **LOFO** | all | transfer (LOFO − none) | gap (all − LOFO) | clean test (LOFO run) |",
                 "|---|---|---|---|---|---|---|"]
        for fam in FAMILIES:
            c = table[fam][a]
            tr = c["lofo"][0] - c["none"][0] if c["lofo"] and c["none"] else None
            gp = c["all"][0] - c["lofo"][0] if c["lofo"] and c["all"] else None
            sg = lambda x: "—" if x is None else f"{x:+.3f}"
            rows.append(f"| {fam} | {f(c['none'])} | **{f(c['lofo'])}** | {f(c['all'])} | {sg(tr)} | {sg(gp)} | {g(c['clean_lofo'])} |")
        cl = table["_clean_test"][a]
        rows += ["", f"clean test obj F1: none {g(cl['none'])}, all {g(cl['all'])}", ""]
    (out / "lofo.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("jobs")
    r = sub.add_parser("report")
    r.add_argument("--runs", default="runs")
    r.add_argument("--out", default="challenge1/results/lofo")
    args = ap.parse_args()
    if args.cmd == "jobs":
        for name, a in jobs():
            print(f"{name}|{a}")
    else:
        report(Path(args.runs), Path(args.out))


if __name__ == "__main__":
    main()
