"""Hackathon interface for the CSD simulator.

Goal for participants
---------------------
Tune the gate voltages to **maximise the interdot contrast** in the measured
charge-stability diagram. The barriers (g1, g3, g5) set the contrast; the
plungers (g2, g4) pan the measurement window. Because moving the barriers also
*drifts* the interdots across the window, you typically need to re-centre the
plungers to keep them in view while you climb toward the contrast sweet-spot.

The location of maximum contrast is **hidden** and **randomised on every run**,
so a general optimiser wins — not a memorised answer.

Quick start
-----------
    from csd import new_experiment

    exp = new_experiment()                      # fresh run, hidden optimum
    img = exp.measure(g1=0.0, g3=0.0, g5=0.0)   # a measurement -> 2D image
    score = img.std()                           # a simple objective (higher = better)

    # ... your optimisation loop: propose gates, measure, keep what improves ...

    print(exp.reveal())     # after you're done: the true sweet-spot + your budget

Re-running: call ``new_experiment()`` again. Each call builds a brand-new
simulator with a fresh scene and a fresh hidden optimum — nothing carries over
between runs. Do **not** cache the object at module scope if you want a new
challenge each time; just call the factory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .config import CHALLENGE, ChallengeConfig
from .simulator import (
    BARRIERS,
    CSDSimulator,
    DriftMatrix,
    Region,
    RegionContrastModel,
    ScanWindow,
)

Float64Array = npt.NDArray[np.float64]


@dataclass
class Experiment:
    """A single hackathon run. Create one with :func:`new_experiment`.

    Use :meth:`measure` to acquire an image at a chosen working point, score it
    with an objective of your choice (the image std is a simple noise-stable
    start), and maximise that score. The optimum is hidden until :meth:`reveal`.
    """

    _sim: CSDSimulator
    _optimum: dict[str, object]  # {barriers, plungers, max_contrast_factor}
    n_measurements: int = field(default=0)
    n_pixels: int = field(default=0)  # cumulative pixels integrated ~ acquisition time

    def _working_point(self, requested: dict[str, float | None]) -> dict[str, float]:
        wp = dict(self._sim.start)
        for gate, value in requested.items():
            if value is not None:
                wp[gate] = float(value)
        return wp

    def _measurement_window(
        self,
        span_h: float | None,
        span_v: float | None,
        step_h: float | None,
        step_v: float | None,
    ) -> ScanWindow:
        """Build the 2-D measurement window, defaulting to the scene window."""
        base = self._sim.scan_window
        return ScanWindow(
            span_h=base.span_h if span_h is None else float(span_h),
            span_v=base.span_v if span_v is None else float(span_v),
            step_h=base.step_h if step_h is None else float(step_h),
            step_v=base.step_v if step_v is None else float(step_v),
        )

    def measure(
        self,
        g1: float | None = None,
        g2: float | None = None,
        g3: float | None = None,
        g4: float | None = None,
        g5: float | None = None,
        *,
        span_h: float | None = None,
        span_v: float | None = None,
        step_h: float | None = None,
        step_v: float | None = None,
    ) -> Float64Array:
        """Acquire one 2-D CSD image at the given gate voltages (volts).

        Any gate left as ``None`` keeps its starting value. ``g2``/``g4`` set the
        **centre** of the measurement window; the barriers set the contrast.

        The window span/step default to the values fixed at ``new_experiment``
        time, but you may override any of ``span_h, span_v, step_h, step_v`` to
        measure the **same** sticks through a window of any size/resolution
        (e.g. a wide low-res overview, then a zoomed fine-step scan). Measuring
        beyond the generated scene shows only noise there — nothing new appears.

        Each call is a fresh acquisition (new noise), so repeated measurements at
        the same point differ slightly — average them if you need a stabler score.
        """
        wp = self._working_point({"g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5})
        window = self._measurement_window(span_h, span_v, step_h, step_v)
        self.n_measurements += 1
        # A frame integrates the whole window; pixels are the real proxy for
        # acquisition time (each pixel has a fixed integration time).
        self.n_pixels += window.n_h * window.n_v
        return self._sim.render(working_point=wp, scan_window=window)

    def scan_1d(
        self,
        direction: tuple[float, float],
        span: float,
        step: float,
        *,
        g1: float | None = None,
        g2: float | None = None,
        g3: float | None = None,
        g4: float | None = None,
        g5: float | None = None,
        normalize: bool = False,
    ) -> Float64Array:
        """Acquire a 1-D cut through the CSD along an arbitrary (g2, g4) direction.

        The cut is a straight line in the plunger plane, **centred on (g2, g4)**
        (defaults to the start plungers) and running along ``direction`` — a
        ``(dg2, dg4)`` vector whose orientation is all that matters, e.g.
        ``(1, 0)`` pure-g2, ``(0, 1)`` pure-g4, ``(1, 1)`` a 45° detuning cut.
        ``span`` is the line length in volts and ``step`` the sample spacing,
        giving ``round(span/step) + 1`` samples. Barriers drift the sticks and
        set contrast exactly as in :meth:`measure`.

        Returns a 1-D array. Its cost is the number of samples on the line (not a
        full frame), which is what gets added to the pixel budget.
        """
        wp = self._working_point({"g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5})
        self.n_measurements += 1
        self.n_pixels += int(round(span / step)) + 1
        return self._sim.scan_1d(
            direction=direction, span=span, step=step, working_point=wp, normalize=normalize
        )

    @property
    def start(self) -> dict[str, float]:
        """The starting working point (barriers at 0 V, plungers centred)."""
        return dict(self._sim.start)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(g2_min, g2_max, g4_min, g4_max) in volts, for ``imshow`` extents."""
        return self._sim.extent

    def reveal(self) -> dict[str, object]:
        """Return the hidden optimum and your measurement budget (for self-check).

        ``optimum_barriers`` are the barriers of the best (highest-contrast)
        region; ``optimum_plungers`` are the (g2, g4) that centre that region's
        sticks in the window so you can actually see it; ``max_contrast_factor``
        is the peak contrast there.
        """
        return {
            "optimum_barriers": dict(self._optimum["barriers"]),
            "optimum_plungers": dict(self._optimum["plungers"]),
            "max_contrast_factor": self._optimum["max_contrast_factor"],
            "n_measurements": self.n_measurements,
            "n_pixels": self.n_pixels,
        }


def new_experiment(
    seed: int | None = None,
    *,
    config: ChallengeConfig = CHALLENGE,
) -> Experiment:
    """Build a fresh experiment with a hidden, randomised contrast sweet-spot.

    Participants call this with no arguments — or with a fixed ``seed`` to make a
    run reproducible. All challenge hyperparameters (scene span/step, contrast
    amplitude/width, how far the optimum can hide) are **fixed** in
    :data:`csd.config.CHALLENGE`, so every run is comparable and you never have to
    choose them here. What you *do* control is on the other side of the API: the
    gate voltages and the measurement span/step of each ``measure``/``scan_1d``.

    Parameters
    ----------
    seed:
        Seed for the whole run (scene layout *and* hidden optimum). ``None`` (the
        default) draws fresh randomness every call, so each run is different.
        Pass an int to make a run reproducible.
    config:
        Organiser-only escape hatch. The fixed challenge hyperparameters; defaults
        to the shared :data:`~csd.config.CHALLENGE`. Participants should leave this
        untouched.
    """
    rng = np.random.default_rng(seed)

    # Draw the regions: each has a spatial seed in the scene plane (for the
    # Voronoi partition of sticks) plus a barrier-space Lorentzian (centre, width,
    # amplitude). Amplitudes vary so exactly one region is the true global optimum;
    # the base floor keeps every stick faintly visible everywhere.
    regions = [
        Region(
            seed=rng.uniform(0.0, config.generator.scene_span, size=2),
            center=tuple(
                float(rng.uniform(-config.optimum_range, config.optimum_range))
                for _ in BARRIERS
            ),
            gamma=tuple(
                float(
                    rng.uniform(1 - config.gamma_jitter, 1 + config.gamma_jitter)
                    * config.gamma
                )
                for _ in BARRIERS
            ),
            amplitude=float(
                rng.uniform(1 - config.amplitude_jitter, 1 + config.amplitude_jitter)
                * config.amplitude
            ),
        )
        for _ in range(config.n_regions)
    ]
    contrast_model = RegionContrastModel(base=config.base, regions=regions)

    # Drift is randomised per run: linear lever arms near the defaults, plus small
    # quadratic terms so the interdots drift non-linearly over long barrier moves.
    drift_matrix = DriftMatrix.random(
        rng,
        lever_arms=config.drift_lever_arms,
        jitter=config.drift_jitter,
        curvature=config.drift_curvature,
    )

    # Independent seed for the (global-RNG) scene build, derived from `rng` so the
    # whole run is reproducible from `seed` yet decoupled from the region draws.
    scene_seed = int(rng.integers(0, 2**31 - 1))
    sim = CSDSimulator(
        generator_config=config.generator,
        drift_matrix=drift_matrix,
        contrast_model=contrast_model,
        seed=scene_seed,
    )

    return Experiment(_sim=sim, _optimum=sim.optimum())
