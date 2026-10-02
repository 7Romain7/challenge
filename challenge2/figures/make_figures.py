"""Figures of the challenge-2 README, rebuilt from the result files.

    PYTHONPATH=challenge2:hackathon:challenge1 python challenge2/figures/make_figures.py

Reads ``challenge2/results/*.jsonl`` (produced by ``evaluation.run``); figure 6 also runs
one device locally (matched-filter front-end, CPU) and uses the hidden truth *for the
figure only* (region centres), as ``evaluation/`` does.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "figures"

# reference categorical palette (validated: adjacent CVD dE >= 9, see dataviz skill)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def load(name, method=None):
    p = RES / f"{name}.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in open(p)]
    return [r for r in rows if "error" not in r and (method is None or r["method"] == method)]


def per_device(rows, key="R_final"):
    d = {}
    for r in rows:
        d.setdefault(r["seed"], []).append(r[key])
    return np.array([np.mean(v) for v in d.values()])


def runs_R(rows):
    return np.array([r["R_final"] for r in rows])


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. the metric
def fig_metric():
    x = np.linspace(-0.6, 0.6, 600)
    base = 3.0
    peaks = [(-0.32, 0.07, 26.0), (0.05, 0.09, 22.0), (0.38, 0.06, 24.5)]
    f = np.full_like(x, base)
    for c, g, a in peaks:
        f = np.maximum(f, base + a * g * g / (g * g + (x - c) ** 2))
    fstar = base + max(a for *_, a in peaks)
    xb = 0.05
    fb = base + 22.0
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.plot(x, f, color=BLUE, lw=2)
    ax.axhline(fstar, color=INK2, lw=1, ls="--")
    ax.axhline(base, color=INK2, lw=1, ls=":")
    ax.text(0.6, fstar + 0.6, "f*: best achievable contrast", ha="right", color=INK2)
    ax.text(0.6, base + 0.6, "floor (contrast off resonance)", ha="right", color=INK2)
    ax.plot([xb], [fb], "o", ms=9, color=ORANGE, mec=SURF, mew=2, zorder=5)
    ax.annotate("", xy=(xb + 0.02, fb), xytext=(xb + 0.02, fstar),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5))
    ax.annotate("", xy=(xb - 0.02, base), xytext=(xb - 0.02, fstar),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text(xb + 0.04, (fb + fstar) / 2, "f* − f(b̂)\n(what is missing)", color=ORANGE, va="center")
    ax.text(xb - 0.04, (base + fstar) / 2 - 3, "f* − floor\n(achievable gain)", color=INK2, ha="right", va="center")
    ax.text(xb + 0.16, fb - 5, "b̂: returned point\n(peak of another region)", ha="left", color=ORANGE)
    ax.set_xlabel("barrier setting (one of three dimensions, schematic)")
    ax.set_ylabel("true contrast f")
    ax.set_title("Normalised regret  R = (f* − f(b̂)) / (f* − floor)   ·   here R ≈ 0.12: 88 % of the gain recovered")
    ax.set_ylim(0, fstar + 3)
    save(fig, "fig1_metrique.png")


# ---------------------------------------------------------------- 2. ablation ladder
LADDER = [
    ("Coordinate ascent\n(official baseline)", "classic_dev100", "coord_official"),
    ("Random search", "random_dev100", "random_dlf"),
    ("Plain BO\n(GP + EI, full frames)", "diag2_dev100", "bo_dlf"),
    ("+ ROI zooms", "diag3_dev100", "bo_roi_dlf"),
    ("+ computed\nbudget split", "auto_dev100", "bo_roi_auto_dlf"),
    ("+ UCB instead of EI", "ucb_dev100", "bo_roi_auto_ucb_dlf"),
    ("+ race between\ntwo regions", "race_roi_dev100", "bo_roi_race_dlf"),
]


def fig_ladder():
    rows = [(lab, runs_R(load(f, m))) for lab, f, m in LADDER]
    rows = [(lab, R) for lab, R in rows if len(R)]
    labels = [lab for lab, _ in rows]
    gain = [100 * (1 - np.median(R)) for _, R in rows]
    succ = [100 * np.mean(R <= 0.05) for _, R in rows]
    fail = [100 * np.mean(R > 0.3) for _, R in rows]
    y = np.arange(len(rows))[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, vals, col, title in ((axs[0], gain, BLUE, "Median gain recovered (1 − R)"),
                                 (axs[1], succ, AQUA, "Success: R ≤ 0.05"),
                                 (axs[2], fail, ORANGE, "Failures: R > 0.3  (lower is better)")):
        ax.barh(y, vals, height=0.62, color=col, edgecolor=SURF, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + 1.5, yi, f"{v:.0f} %", va="center", color=INK, fontsize=9)
        ax.set_title(title)
        ax.set_xlim(0, 112)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("% of the 100 devices (dev 0–99)" if ax is not axs[0] else "% (median over 100 devices)")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(labels)
    fig.suptitle("Each added component, same budget (1 M px, 300 measurements), same 100 devices", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_progression.png")


# ---------------------------------------------------------------- 3. val distribution
VAL = [("bo_roi_dlf (before)", "night_val200x3", "bo_roi_dlf", INK2),
       ("bo_roi_auto_ucb_dlf", "val_ucb", "bo_roi_auto_ucb_dlf", BLUE),
       ("bo_coarse_ucb_dlf", "val_coarseucb", "bo_coarse_ucb_dlf", ORANGE),
       ("bo_roi_race_dlf", "val_race", "bo_roi_race_dlf", AQUA)]


def fig_val():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for lab, f, m, col in VAL:
        rows = load(f, m) if f != "val_race" else load("val_race_a", m) + load("val_race_b", m)
        if not rows:
            continue
        R = np.sort(runs_R(rows))
        ax.step(R, np.arange(1, len(R) + 1) / len(R) * 100, where="post", color=col, lw=2,
                label=f"{lab}  (median {np.median(R):.3f})")
    ax.axvline(0.05, color=INK2, lw=1, ls="--")
    ax.axvline(0.3, color=INK2, lw=1, ls=":")
    ax.text(0.052, 101, "success ≤ 0.05", color=INK2, fontsize=9, va="bottom")
    ax.text(0.305, 101, "failure > 0.3", color=INK2, fontsize=9, va="bottom")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1])
    ax.set_xticklabels(["0", "0.02", "0.05", "0.1", "0.2", "0.3", "0.5", "1"])
    ax.set_xlabel("regret R of the returned point (0 = optimum)")
    ax.set_ylabel("% of runs with regret ≤ R")
    ax.set_title("Val: 200 unseen devices × 3 seeds (600 runs per method)")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig3_val_distribution.png")


# ---------------------------------------------------------------- 4. budget curve
def fig_budget():
    budgets = [250_000, 500_000, 1_000_000, 2_000_000, 4_000_000]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for tag, ref1, m, col, lab in (("ucb", "ucb_dev100", "bo_roi_auto_ucb_dlf", BLUE, "BO + zooms + UCB"),
                                   ("coarse", "coarseucb_dev100", "bo_coarse_ucb_dlf", ORANGE, "coarse exploration + UCB")):
        g, s = [], []
        for b in budgets:
            R = runs_R(load(ref1 if b == 1_000_000 else f"curve_{tag}_{b}", m))
            g.append(100 * (1 - np.median(R)))
            s.append(100 * np.mean(R <= 0.05))
        x = np.array(budgets) / 1e6
        axs[0].plot(x, g, "o-", color=col, lw=2, ms=8, mec=SURF, mew=2, label=lab)
        axs[1].plot(x, s, "o-", color=col, lw=2, ms=8, mec=SURF, mew=2, label=lab)
    for ax, t, yl in ((axs[0], "Median gain recovered", "median 1 − R (%)"),
                      (axs[1], "Success (R ≤ 0.05)", "% of the 100 devices")):
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(np.array(budgets) / 1e6)
        ax.set_xticklabels(["0.25 M\n75 meas.", "0.5 M\n150", "1 M\n300", "2 M\n600", "4 M\n1,200"])
        ax.axvline(1, color=INK2, lw=1, ls="--")
        ax.set_title(t)
        ax.set_ylabel(yl)
        ax.set_xlabel("budget: pixels and measurements scaled together")
        ax.set_ylim(0, 100)
    axs[0].text(1.05, 8, "reference budget", color=INK2, fontsize=9)
    axs[1].legend(loc="upper left")
    fig.suptitle("More budget helps little beyond 1 M px (dev 0–99)", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_budget.png")


# ---------------------------------------------------------------- 5. robustness
SHIFTS = [("S1", "noise ×2"), ("S2", "stripes ×3"), ("S3", "optimum further away"), ("S4a", "2 regions"),
          ("S4b", "8 regions"), ("S5a", "narrow peaks ×0.5"), ("S5b", "wide peaks ×2"),
          ("S6", "strongly non-linear drift"), ("S7", "weak contrast")]


def fig_robust():
    meths = [("bo_roi_dlf\n(before)", "rob", "bo_roi_dlf"), ("bo_roi_auto_ucb_dlf", "rob", "bo_roi_auto_ucb_dlf"),
             ("bo_coarse_ucb_dlf", "robc", "bo_coarse_ucb_dlf")]
    M = np.array([[np.median(runs_R(load(f"{p}_{s}", m))) for _, p, m in meths] for s, _ in SHIFTS])
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", ["#eef5fd", "#86b6ef", "#2a78d6", "#104281"])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            best = M[i, j] == M[i].min()
            ax.text(j, i, f"{M[i, j]:.2f}" + (" ★" if best else ""), ha="center", va="center",
                    color=SURF if M[i, j] > 0.45 else INK, fontsize=10, fontweight="bold" if best else None)
    ax.set_xticks(range(len(meths)))
    ax.set_xticklabels([m for m, *_ in meths])
    ax.set_yticks(range(len(SHIFTS)))
    ax.set_yticklabels([f"{s} · {d}" for s, d in SHIFTS])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("median regret R (lighter = better)")
    ax.set_title("Robustness: modified simulator, 50 devices per cell\n(★ = best method of the row)")
    save(fig, "fig5_robustesse.png")


# ---------------------------------------------------------------- 6. one device, end to end
def fig_example(seed: int = 3):
    from csd import new_experiment
    from evaluation import truth
    from optimization import BlindExperiment
    from optimization.methods.bo_roi import ROIBayesOpt
    from optimization.session import SessionConfig

    exp = new_experiment(seed=seed)
    base, fs = truth.base_and_best(exp)
    m = ROIBayesOpt(run_seed=0, switch="auto", acq="ucb", session_cfg=SessionConfig(keep_frames=True))
    keep = {}
    orig = m.search

    def search(s):
        keep["s"] = s
        orig(s)

    m.search = search
    bx = BlindExperiment(exp, pixel_cap=1_000_000)
    res = m.run(bx)
    s = keep["s"]
    sim = exp._sim
    fr = [f for f in s.frames if f["span"] >= 0.29]
    B = np.array([f["b"] for f in fr])
    fvals = [truth.true_factor(exp, {"g1": b[0], "g2": 0, "g3": b[1], "g4": 0, "g5": b[2]}) for b in B]
    j_lit = int(np.argmax(fvals))
    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    from matplotlib.patches import Rectangle
    frame0 = fr[0]
    img0 = frame0["img"].astype(float)
    peaks0 = s.percep.process(frame0["img"].astype(float)).peaks
    flit = fr[j_lit]
    imgl = flit["img"].astype(float)
    origin = flit["centre"] - flit["span"] / 2
    pos = (s.ref[m.focus] + s.drift.predict(flit["b"] - s.b0) - origin) / s.step  # (x=col, y=row)
    n = round(m.patch / s.step)
    for k, (img, title) in enumerate(((img0, "1. Reference frame (barriers at 0)\ninterdots found by the perception"),
                                       (imgl, f"2. Best frame found: a region lights up\n(contrast ×{fvals[j_lit] / base:.0f}); squares = ROI zooms"))):
        ax = fig.add_subplot(gs[0, k])
        img = img - np.median(img, axis=1, keepdims=True)
        ax.imshow(img, cmap="gray_r" if s.percep.polarity < 0 else "gray", origin="lower")
        if k == 0:
            ax.scatter(peaks0[:, 1], peaks0[:, 0], s=70, facecolors="none", edgecolors=ORANGE, linewidths=1.5)
        else:
            for (cx, cy) in pos:
                ax.add_patch(Rectangle((cx - n / 2, cy - n / 2), n, n, fill=False, ec=AQUA, lw=2))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    sub = gs[0, 2].subgridspec(2, 3, height_ratios=[1, 0.9])
    vmin, vmax = np.percentile(imgl - np.median(imgl, axis=1, keepdims=True), [1, 99.9])
    for i, (cx, cy) in enumerate(pos[:3]):
        ax = fig.add_subplot(sub[0, i])
        r0, c0 = int(round(cy - n / 2)), int(round(cx - n / 2))
        crop = (imgl - np.median(imgl, axis=1, keepdims=True))[max(r0, 0):r0 + n, max(c0, 0):c0 + n]
        ax.imshow(crop, cmap="gray_r" if s.percep.polarity < 0 else "gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(AQUA)
            sp.set_linewidth(2)
        if i == 1:
            ax.set_title("3. The 3 ROI zooms (25×25 px)", fontsize=10)
    ax = fig.add_subplot(sub[1, :])
    ax.axis("off")
    ax.text(0.0, 0.95,
            f"{len(fr)} full frames (150×150 px), then\n{len(m.Y2)} settings tested with zooms\n"
            f"(3 patches = 12× cheaper than a frame)\n"
            f"budget: {res['n_pixels'] / 1e6:.2f} M px, {res['n_meas']} measurements\n"
            f"final regret R = {(fs - truth.true_factor(exp, s.reco_log[-1]['wp'])) / (fs - base):.3f}",
            fontsize=9.5, transform=ax.transAxes, va="top")
    # trajectory in barrier space (g1, g3) with region centres
    ax = fig.add_subplot(gs[1, :2])
    live = sorted(set(int(k) for k in sim._region_of))
    best = max(live, key=lambda k: sim.contrast_model.regions[k].amplitude)
    ax.scatter(B[:, 0], B[:, 1], c=np.arange(len(B)), cmap="Blues", s=45, edgecolors=INK2, linewidths=0.5,
               vmin=-5, label="full frames (light → dark: order)", zorder=3)
    roi = np.array(m.X2) if len(m.X2) else np.zeros((0, 3))
    if len(roi):
        ax.scatter(roi[:, 0], roi[:, 1], s=10, color=ORANGE, alpha=0.6, label="settings tested with zooms", zorder=4)
    for k in live:
        c = sim.contrast_model.regions[k].center
        ax.scatter([c[0]], [c[1]], marker="*", s=380 if k == best else 200,
                   color=AQUA if k == best else INK2, edgecolors=SURF, linewidths=1.5, zorder=5)
    ax.scatter([], [], marker="*", s=200, color=AQUA, label="peak of the best region (truth, figure only)")
    ax.scatter([], [], marker="*", s=120, color=INK2, label="peaks of the other regions")
    ax.set_xlabel("g1 (V)")
    ax.set_ylabel("g3 (V)")
    ax.set_title("4. Trajectory in barrier space (g1–g3 projection)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    ax.set_aspect("equal")
    # anytime regret
    ax = fig.add_subplot(gs[1, 2])
    px = [r["pixels"] / 1e6 for r in s.reco_log]
    R = [(fs - truth.true_factor(exp, r["wp"])) / (fs - base) for r in s.reco_log]
    ax.step(px, R, where="post", color=BLUE, lw=2)
    ax.set_xlabel("pixels spent (M)")
    ax.set_ylabel("regret R of the recommended point")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("5. Regret along the run")
    fig.suptitle(f"One complete run on device dev {seed}", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_exemple_run.png")


# ---------------------------------------------------------------- 7. diagnosis of failures
def fig_diag():
    rows = load("diag3_dev100", "bo_roi_dlf")
    R = runs_R(rows)
    Rr = np.array([r["R_region"] for r in rows])
    lit = np.array([r["best_lit_max"] for r in rows])
    bad = Rr > 0.05
    cats = [("success\nR ≤ 0.05", np.sum(R <= 0.05), AQUA),
            ("lost by the region choice\nbest region never lit (< 10 %)", np.sum(bad & (lit < 0.1)), ORANGE),
            ("lost by the region choice\nlit 10–20 %", np.sum(bad & (lit >= 0.1) & (lit < 0.2)), YELLOW),
            ("lost by the region choice\nlit ≥ 20 % but not followed", np.sum(bad & (lit >= 0.2)), MAGENTA),
            ("right region,\nincomplete refinement", np.sum(~bad & (R > 0.05)), BLUE)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    y = np.arange(len(cats))[::-1]
    ax.barh(y, [c[1] for c in cats], color=[c[2] for c in cats], height=0.6, edgecolor=SURF, linewidth=2)
    for yi, (_, v, _) in zip(y, cats):
        ax.text(v + 0.8, yi, f"{v}", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in cats], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("number of devices out of 100 (bo_roi_dlf, dev 0–99)")
    ax.set_title("Where failures come from: mostly exploration")
    save(fig, "fig7_diagnostic.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_metric()
    fig_ladder()
    fig_val()
    fig_budget()
    fig_robust()
    fig_diag()
    fig_example()
