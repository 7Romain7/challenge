"""Ground truth for *evaluation only* (reads the simulator's hidden state)."""

from __future__ import annotations

import numpy as np

from csd import ScanWindow


def base_and_best(exp) -> tuple[float, float]:
    """(floor contrast factor, best achievable factor f*) of this device."""
    return float(exp._sim.contrast_model.base), float(exp.reveal()["max_contrast_factor"])


def true_factor(exp, wp: dict) -> float:
    """Contrast factor the grader would assign to barriers wp (max over sticks)."""
    return float(exp._sim.contrast(wp))


def visible_factor(exp, wp: dict, span: float) -> float:
    """Best factor among sticks actually inside the measurement window at ``wp``."""
    sim = exp._sim
    win = ScanWindow(span_h=span, span_v=span, step_h=sim.scan_window.step_h,
                     step_v=sim.scan_window.step_v)
    shift = sim._stick_shift(wp, win)
    pos = np.array([s.middle for s in sim.base_interdots]) + shift
    inside = np.all((pos >= 0) & (pos <= span), axis=1)
    fac = sim._stick_factors(wp)
    return float(fac[inside].max()) if inside.any() else float(sim.contrast_model.base)


def device_descriptors(exp) -> dict:
    """Difficulty descriptors used to stratify results."""
    sim = exp._sim
    m = sim.contrast_model
    cnt = np.bincount(sim._region_of, minlength=len(m.regions))
    live = [k for k in range(len(m.regions)) if cnt[k] > 0]
    A = sorted((m.regions[k].amplitude for k in live), reverse=True)
    best = max(live, key=lambda k: m.regions[k].amplitude)
    return {"n_sticks": int(len(sim.base_interdots)),
            "sticks_in_best_region": int(cnt[best]),
            "gap_top2": float((A[0] - A[1]) / A[0]) if len(A) > 1 else 1.0}
