"""Charge Stability Diagram (CSD) simulator for a 5-gate DQD platform.

Gates: g1, g2, g3, g4, g5.

- The CSD is imaged in the (g2, g4) plunger plane. g2/g4 set the *centre* of the
  measurement window (panning). g1, g3, g5 are the barriers and act as parameters.
- The interdot geometry ("sticks") is generated **once** at construction via
  :func:`csd.generator.build_interdots`. We never create new sticks afterwards —
  moving the barriers only shifts each stick's centre.
- A **drift matrix** maps a change in the barriers (g1, g3, g5) to a displacement
  of the interdots in the (g2, g4) plane. Different lever arms per barrier mean a
  step on g1 moves g2 a lot and g4 a little; small quadratic terms make the drift
  bend over long barrier excursions (see :meth:`DriftMatrix.random`).
- A **region contrast model** (:class:`RegionContrastModel`) scales each stick's
  intensity by *its region's* factor: the (g2, g4) plane is partitioned into
  regions (Voronoi), each with its own Lorentzian peak in barrier space on a
  shared base floor. Different patches of the diagram therefore brighten and fade
  as the barriers move, and one region hides the global contrast optimum.
- Acquisition noise is redrawn on every frame (same statistics, fresh realisation),
  so each :meth:`CSDSimulator.render` call is a new measurement.

The scene is generated once over a fixed window; :meth:`CSDSimulator.render` and
:meth:`CSDSimulator.scan_1d` then view it through a measurement window of **any**
span/step centred on (g2, g4). Panning/drifting translates the fixed sticks under
that viewport; sticks that leave the window disappear and nothing new appears —
measuring beyond the generated scene shows only noise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates

from .config import GENERATOR, GeneratorConfig
from .generator import (
    Interdot,
    LineSpec,
    ScanWindow,
    build_interdots,
    render_csd,
    scene_window,
)

Float64Array = npt.NDArray[np.float64]

GATES = ("g1", "g2", "g3", "g4", "g5")
PLUNGERS = ("g2", "g4")
BARRIERS = ("g1", "g3", "g5")


@dataclass
class DriftMatrix:
    """Maps barrier deltas (g1, g3, g5) [V] to a plunger-plane drift (g2, g4) [V].

    ``data`` has shape (2, 3): rows are the plungers (g2, g4), columns are the
    barriers (g1, g3, g5). Entry ``[p, b]`` is the (linear) lever arm — how far
    plunger ``p``'s effective position shifts per volt applied to barrier ``b``.

    ``quad`` (optional, shape (2, 3, 3)) adds small **second-order** terms: for
    plunger ``p`` the drift gains ``dbᵀ · quad[p] · db``. It is symmetric per row
    and small, so drift is essentially linear near the start and only bends over
    long barrier excursions. ``None`` means pure-linear (the default).
    """

    data: Float64Array
    quad: Float64Array | None = None

    @classmethod
    def default(cls) -> "DriftMatrix":
        # g1 -> mostly g2, a bit g4;  g3 -> both;  g5 -> mostly g4, a bit g2.
        return cls(
            np.array(
                [
                    [-0.80, -0.35, -0.20],  # dg2 per volt on (g1, g3, g5)
                    [-0.20, -0.35, -0.80],  # dg4 per volt on (g1, g3, g5)
                ],
                dtype=float,
            )
        )

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        *,
        lever_arms: npt.ArrayLike | None = None,
        jitter: float = 0.15,
        curvature: float = 0.4,
    ) -> "DriftMatrix":
        """Random drift near the given lever arms, with small quadratic terms.

        ``lever_arms`` is the (2, 3) linear baseline (defaults to
        :meth:`default`). ``jitter`` is the fractional spread of the linear
        coefficients around it (``0.15`` -> ±15%, keeping the lever-arm
        signs/structure). ``curvature`` scales the order-2 terms — small, so the
        drift is ~linear for modest barrier moves and bends only over long
        excursions.
        """
        base = np.asarray(lever_arms, dtype=float) if lever_arms is not None else cls.default().data
        data = base * (1.0 + rng.uniform(-jitter, jitter, size=base.shape))
        quad = rng.uniform(-curvature, curvature, size=(2, 3, 3))
        quad = 0.5 * (quad + quad.transpose(0, 2, 1))  # symmetric per plunger row
        return cls(data=data, quad=quad)

    def drift(self, d_barriers: npt.ArrayLike) -> Float64Array:
        """Return the (dg2, dg4) displacement for barrier deltas (dg1, dg3, dg5)."""
        db = np.asarray(d_barriers, dtype=float)
        out = self.data @ db
        if self.quad is not None:
            out = out + np.einsum("pij,i,j->p", self.quad, db, db)
        return out


@dataclass
class Region:
    """One spatial region of the (g2, g4) plane with its own contrast peak.

    ``seed`` is the region's location in the scene plane; every stick is assigned
    to its nearest region seed (a Voronoi partition). ``center``/``gamma`` define
    a 1D-Lorentzian-product in barrier space peaked at ``center``, and
    ``amplitude`` is that region's peak contrast boost.
    """

    seed: Float64Array  # (2,) location in the (g2, g4) plane
    center: tuple[float, float, float]  # (g1, g3, g5) [V] barrier optimum
    gamma: tuple[float, float, float]  # HWHM per barrier [V]
    amplitude: float  # peak contrast boost


@dataclass
class RegionContrastModel:
    """Per-region contrast: a shared ``base`` plus one Lorentzian bump per region.

    A stick in region ``c`` has its intensity multiplied by
    ``base + A_c * L(b; center_c, gamma_c)``. The ``base`` floor keeps every stick
    faintly visible everywhere (the diagram is never globally blank), while each
    region's bump brightens its own patch of sticks near its barrier optimum and
    fades back toward ``base`` elsewhere — so different regions bloom and vanish as
    the barriers move. A single region reproduces a uniform global contrast (one
    Lorentzian for the whole scene). Amplitudes vary between regions, so exactly one
    region is the true global optimum and finding it takes a real search.
    """

    base: float
    regions: list[Region]

    def region_factor(self, region: Region, barriers: tuple[float, float, float]) -> float:
        lorentz = region.amplitude
        for x, c, g in zip(barriers, region.center, region.gamma):
            lorentz *= (g * g) / (g * g + (x - c) ** 2)
        return self.base + lorentz

    def factors(self, barriers: tuple[float, float, float]) -> Float64Array:
        """Per-region contrast factor, shape ``(len(regions),)``."""
        return np.array([self.region_factor(r, barriers) for r in self.regions])


def default_working_point(scan_window: ScanWindow) -> dict[str, float]:
    """Barriers at 0 V; plungers centred so the start window shows the sticks."""
    return {
        "g1": 0.0,
        "g2": scan_window.span_h / 2,
        "g3": 0.0,
        "g4": scan_window.span_v / 2,
        "g5": 0.0,
    }


class CSDSimulator:
    """Simulate CSD frames of a 5-gate DQD as the working point moves.

    The scene (interdots + connecting lines) is built once at construction. Each
    :meth:`render` translates the sticks according to barrier drift and window
    panning, scales their intensity by the contrast model, and renders a fresh
    frame (fresh noise).
    """

    def __init__(
        self,
        working_point: dict[str, float] | None = None,
        generator_config: GeneratorConfig | None = None,
        drift_matrix: DriftMatrix | None = None,
        contrast_model: RegionContrastModel | None = None,
        seed: int = 0,
    ) -> None:
        # All CSD appearance/physics (incl. the scene window) come from one config.
        self.generator_config = generator_config or GENERATOR
        self.scan_window = scene_window(self.generator_config)
        self.drift_matrix = drift_matrix or DriftMatrix.default()
        # Default to a single region (uniform global contrast: one Lorentzian for
        # the whole scene). new_experiment supplies the real multi-region model.
        self.contrast_model = contrast_model or RegionContrastModel(
            base=1.0,
            regions=[
                Region(
                    seed=np.array(
                        [self.scan_window.span_h / 2, self.scan_window.span_v / 2]
                    ),
                    center=(0.0, 0.0, 0.0),
                    gamma=(0.1, 0.1, 0.1),
                    amplitude=20.0,
                )
            ],
        )

        start = default_working_point(self.scan_window)
        if working_point:
            start.update(working_point)
        self.start: dict[str, float] = dict(start)

        # Origin (lower-left) of the window in absolute (g2, g4) coords at start.
        self._start_origin = np.array(
            [
                self.start["g2"] - self.scan_window.span_h / 2,
                self.start["g4"] - self.scan_window.span_v / 2,
            ]
        )

        # Build the scene ONCE. Sticks live in [0, span] window-local coords,
        # which coincide with absolute coords at the start working point.
        self.seed = seed
        np.random.seed(seed)
        self.base_interdots: list[Interdot]
        self.line_specs: list[LineSpec]
        self.base_interdots, self.line_specs = build_interdots(
            self.generator_config, self.scan_window
        )

        # Assign every stick to its nearest region seed (a Voronoi partition).
        # This uses only the stick positions the generator already produced —
        # generator.py stays untouched.
        self._region_of: Float64Array = self._assign_regions()

    def contrast(self, working_point: dict[str, float] | None = None) -> float:
        """Best contrast factor experienced by any stick at ``working_point``.

        This is the max over *populated* regions only (an empty Voronoi cell has no
        stick to experience its bump), so it never overreports.
        """
        wp = working_point or self.start
        factors = self._stick_factors(wp)  # per-stick -> only populated regions
        return float(factors.max()) if len(factors) else float(self.contrast_model.base)

    # -- regions -----------------------------------------------------------
    def _assign_regions(self) -> Float64Array:
        """Index of the nearest region seed for every base stick (shape (N,))."""
        pos = np.array([s.middle for s in self.base_interdots])  # (N, 2)
        if len(pos) == 0:
            return np.zeros(0, dtype=int)
        seeds = np.array([r.seed for r in self.contrast_model.regions])  # (K, 2)
        dists = np.linalg.norm(pos[:, None, :] - seeds[None, :, :], axis=2)  # (N, K)
        return dists.argmin(axis=1)

    def _stick_factors(self, wp: dict[str, float]) -> Float64Array:
        """Contrast factor for every base stick at ``wp`` (shape (N,))."""
        barriers = (wp["g1"], wp["g3"], wp["g5"])
        return self.contrast_model.factors(barriers)[self._region_of]

    def optimum(self) -> dict[str, object]:
        """The hidden global optimum of the region-based contrast model.

        Returns the barriers that maximise contrast (the amplitude-winning
        region's centre), the plunger centre that brings that region's sticks to
        the window centre, and the peak contrast factor there. The global max is
        exact: each stick peaks at ``base + A_c`` at its region's centre, so the
        best barriers are ``argmax_c A_c`` over regions that actually contain
        sticks.
        """
        model = self.contrast_model
        counts = np.bincount(self._region_of, minlength=len(model.regions))
        live = [c for c in range(len(model.regions)) if counts[c] > 0]
        best = max(live, key=lambda c: model.regions[c].amplitude)
        r = model.regions[best]

        pos = np.array([s.middle for s in self.base_interdots])[self._region_of == best]
        centroid = pos.mean(axis=0)
        # Plunger centre that puts this region's (drifted) sticks at window centre.
        drift = self.drift_matrix.drift(r.center)  # start barriers are 0 V
        plungers = centroid + drift + self._start_origin

        return {
            "barriers": {b: float(v) for b, v in zip(BARRIERS, r.center)},
            "plungers": {"g2": float(plungers[0]), "g4": float(plungers[1])},
            "max_contrast_factor": float(model.base + r.amplitude),
        }

    # -- geometry ----------------------------------------------------------
    def _stick_shift(
        self, wp: dict[str, float], scan_window: ScanWindow | None = None
    ) -> Float64Array:
        """Total translation (in window-local coords) applied to every stick.

        Two contributions add up in the same units:
          * barrier drift  = drift_matrix @ (barriers - start barriers)
          * panning        = -(window origin - start window origin)
        Panning the window right (g2 up) makes the sticks move left, hence the
        minus sign.

        ``scan_window`` is the *measurement* window (defaults to the scene
        window). Only its span enters here — it fixes the window origin around
        the (g2, g4) working point, so the same sticks can be viewed through a
        window of any span/step without rebuilding the scene.
        """
        scan_window = scan_window or self.scan_window
        d_barriers = np.array([wp[b] - self.start[b] for b in BARRIERS])
        drift = self.drift_matrix.drift(d_barriers)

        origin = np.array(
            [
                wp["g2"] - scan_window.span_h / 2,
                wp["g4"] - scan_window.span_v / 2,
            ]
        )
        pan = origin - self._start_origin
        return drift - pan

    def render(
        self,
        working_point: dict[str, float] | None = None,
        normalize: bool = False,
        scan_window: ScanWindow | None = None,
    ) -> Float64Array:
        """Render one CSD frame for ``working_point`` (defaults to the current one).

        ``normalize`` z-scores the image (as the generator does). Leave it False
        to keep raw amplitudes, so the contrast factor is visible against the
        fixed-statistics noise.

        ``scan_window`` is the *measurement* viewport (span + step). It defaults
        to the scene window, reproducing the previous behaviour; passing a
        different one measures the **same** sticks through a window of any
        span/step centred on the (g2, g4) working point. The scene itself is
        never rebuilt, so a window larger than the scene simply shows empty
        (noise-only) space beyond where sticks were generated.
        """
        scan_window = scan_window or self.scan_window
        wp = working_point or self.start
        shift = self._stick_shift(wp, scan_window)
        factors = self._stick_factors(wp)

        interdots = [
            replace(s, middle=s.middle + shift, intensity=s.intensity * f)
            for s, f in zip(self.base_interdots, factors)
        ]

        return render_csd(
            interdots=interdots,
            line_specs=self.line_specs,
            scan_window=scan_window,
            config=self.generator_config,
            normalize=normalize,
        )

    def scan_1d(
        self,
        direction: tuple[float, float],
        span: float,
        step: float,
        working_point: dict[str, float] | None = None,
        normalize: bool = False,
    ) -> Float64Array:
        """Acquire a 1-D cut through the CSD along an arbitrary (g2, g4) direction.

        The cut is a straight line in the plunger plane, centred on the (g2, g4)
        working point and running along ``direction`` — a ``(dg2, dg4)`` vector
        whose *orientation* is all that matters (it is normalised internally, so
        ``(1, 0)`` is a pure-g2 scan, ``(0, 1)`` pure-g4, ``(1, 1)`` a 45°
        detuning cut). ``span`` is the line length in volts and ``step`` the
        sample spacing, giving ``round(span/step) + 1`` samples.

        It is a zero-width cut. Drift, contrast, acquisition noise and blur are
        identical to :meth:`render`: the enclosing 2-D window is rendered and the
        line is sampled from it (bilinear), so on-line pixels see the same
        neighbourhood a full 2-D frame would.
        """
        wp = working_point or self.start
        d = np.asarray(direction, dtype=float)
        norm = float(np.hypot(d[0], d[1]))
        if norm == 0.0:
            raise ValueError("direction must be a non-zero (dg2, dg4) vector")
        d_hat = d / norm

        center = np.array([wp["g2"], wp["g4"]])
        n_points = int(round(span / step)) + 1
        t = np.linspace(-span / 2.0, span / 2.0, n_points)
        points = center[None, :] + t[:, None] * d_hat[None, :]  # absolute (g2, g4)

        # Enclosing 2-D window: line bbox padded for stick bodies + blur bleed,
        # so the sampled line sees the same neighbourhood a 2-D frame would (and
        # a near-axis-aligned cut still gets correct perpendicular blur).
        max_len = max((s.length for s in self.base_interdots), default=0.02)
        pad = max_len + 6.0 * self.generator_config.sigma_blur * step
        span_h = abs(span * d_hat[0]) + 2.0 * pad
        span_v = abs(span * d_hat[1]) + 2.0 * pad
        window = ScanWindow(span_h=span_h, span_v=span_v, step_h=step, step_v=step)

        image = self.render(working_point=wp, normalize=False, scan_window=window)

        origin = np.array([center[0] - span_h / 2.0, center[1] - span_v / 2.0])
        local = points - origin[None, :]
        cols = local[:, 0] / step
        rows = local[:, 1] / step
        trace = map_coordinates(image, [rows, cols], order=1, mode="nearest")

        if normalize:
            std = float(np.std(trace))
            if std > 0.0:
                trace = (trace - float(np.mean(trace))) / std
        return trace

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(left, right, bottom, top) in absolute (g2, g4) volts for imshow."""
        return (
            self.start["g2"] - self.scan_window.span_h / 2,
            self.start["g2"] + self.scan_window.span_h / 2,
            self.start["g4"] - self.scan_window.span_v / 2,
            self.start["g4"] + self.scan_window.span_v / 2,
        )


def main() -> None:
    """Render a single frame at the default working point and show it."""
    import matplotlib.pyplot as plt

    sim = CSDSimulator(seed=0)
    image = sim.render()

    plt.figure(figsize=(6, 6))
    plt.imshow(image, extent=sim.extent, origin="lower", cmap="plasma")
    plt.colorbar(label="Amplitude")
    plt.title(f"CSD @ start (contrast={sim.contrast():.2f})")
    plt.xlabel("$g_2$ [V]")
    plt.ylabel("$g_4$ [V]")
    plt.show()


if __name__ == "__main__":
    main()
