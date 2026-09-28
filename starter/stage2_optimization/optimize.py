"""Challenge 2 (starter) — find the working point with maximum interdot contrast.

    python starter/stage2_optimization/optimize.py

``new_experiment()`` gives you a fresh device with a *hidden* contrast sweet-spot
(randomised every run). You tune the five gate voltages and measure images:

    exp = new_experiment()
    img = exp.measure(g1=..., g2=..., g3=..., g4=..., g5=...)   # a raw CSD image
    score = img.std()                                           # a simple objective

Gates g2, g4 pan the measurement window; g1, g3, g5 are the barriers that set the
contrast (moving them also drifts the interdots, so you may need to re-centre g2,
g4 to keep them in view).

The contrast landscape is **multi-region**: different patches of the diagram light
up at different barrier settings, and only one region hides the global optimum.
So expect **local optima** — and note that a full-frame image std is a weak proxy
(one bright region gets diluted by the rest of the window). You will likely do
better by *scanning* to find live regions (e.g. a wide-span overview ``measure``),
*panning/zooming* into the best one (small ``span_h/span_v``), and scoring contrast
there — ideally with your stage-1 detector (a CNR on interdot pixels, see
``detector_score``) rather than raw std. ``exp.reveal()`` gives the true optimum
**barriers and plunger centre** plus your budget; grading uses the true hidden
contrast, so any route there is valid.

This baseline is a plain coordinate ascent on full-frame std — it will typically
**stall in a local optimum**, on purpose: it is a starting point to beat.
"""

from __future__ import annotations

import numpy as np

from csd import BARRIERS, GATES, new_experiment


def score(exp, point: dict[str, float], n_frames: int = 5) -> float:
    """Objective = image std, averaged over a few noisy frames for stability."""
    return float(np.mean([exp.measure(**point).std() for _ in range(n_frames)]))


# --- Optional: replace the objective with a detector-based one --------------
# If you trained a detector in challenge 1, you can score contrast only on the
# interdot pixels it finds (a contrast-to-noise ratio), which is far less noisy
# than the global std. Sketch:
#
# def detector_score(exp, point, detector, n_frames=5):
#     vals = []
#     for _ in range(n_frames):
#         img = exp.measure(**point)
#         mask = detector.predict(img).astype(bool)     # your model, your preprocessing
#         if mask.any():
#             cnr = abs(img[mask].mean() - img[~mask].mean()) / (img[~mask].std() + 1e-9)
#             vals.append(cnr)
#     return float(np.mean(vals)) if vals else 0.0


def optimize(exp) -> dict[str, float]:
    """A simple coordinate-ascent baseline over all five gates."""
    point = exp.start
    best = score(exp, point)

    step = 0.04
    for _ in range(6):  # a few refinement passes
        for gate in GATES:
            for direction in (+1, -1):
                trial = dict(point)
                trial[gate] += direction * step
                s = score(exp, trial)
                if s > best:
                    best, point = s, trial
        step *= 0.6  # home in
    return point


def main() -> None:
    exp = new_experiment()  # fresh, hidden optimum; call again for a new run

    start_score = score(exp, exp.start)
    best_point = optimize(exp)
    best_score = score(exp, best_point)

    truth = exp.reveal()
    print(f"start contrast metric : {start_score:.3f}")
    print(f"final contrast metric : {best_score:.3f}")
    print("found barriers        :", {g: round(best_point[g], 3) for g in BARRIERS})
    print("hidden optimum        :", {g: round(v, 3) for g, v in truth["optimum_barriers"].items()})
    print("measurements used     :", truth["n_measurements"])
    print("pixels integrated     :", f"{truth['n_pixels']:,}  (~ acquisition time)")


if __name__ == "__main__":
    main()
