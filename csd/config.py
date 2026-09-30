"""Fixed engine hyperparameters — the settings every participant shares.

These are set **by the organiser** and are deliberately *not* exposed as arguments
to the participant-facing API. Two independent bundles:

* :class:`GeneratorConfig` — the CSD **appearance/physics** (charging-line lattice,
  stick shapes/intensities, noise, blur, and the scene window). Shared by *both*
  challenges: challenge-1 dataset generation and the challenge-2 scene.
* :class:`ChallengeConfig` — the challenge-2 **optimization landscape** (regions,
  drift, contrast). It *composes* a :class:`GeneratorConfig` (``.generator``) so a
  single object fully describes a challenge-2 run.

To retune, edit the defaults in this one place. Keep these at the defaults for
your **baseline** Challenge 1 & 2 results, so they stay reproducible and
comparable across participants. Changing them as a deliberate, documented
experiment is welcome — just keep the default run alongside it. See the
"How far can I go?" section of the README for the full sandbox contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class GeneratorConfig:
    """CSD appearance/physics — how a charge-stability diagram looks.

    Frozen and shared: :mod:`csd.dataset` (challenge 1) and the challenge-2 scene
    both build from a ``GeneratorConfig`` so their images are similarly
    distributed (a stage-1 detector transfers to stage-2).
    """

    # Scene-generation window (where the sticks are generated; also the challenge-1
    # dataset grid). NOT the challenge-2 measurement window — participants choose
    # that per ``measure`` / ``scan_1d`` call.
    scene_span: float = 0.3  # V, both axes
    scene_step: float = 2e-3  # V per pixel, both axes

    # Charging-line lattice.
    n_dots: int = 100  # max dots per side (n_l = n_r)
    avg_spacing: float = 0.08  # mean charging-line spacing [V]
    var_spacing: float = 70e-6  # variance of the spacing [V^2]
    var_interdot_middle: float = 1e-3  # jitter of an interdot centre on a CL crossing

    # Charging-line angle families (Beta concentration; left near pi/2, right near 0).
    angle_alpha: float = 1.0
    angle_beta: float = 5.0

    # Interdot ("stick") shape.
    stick_theta: float = math.pi / 4  # base orientation [rad]
    stick_theta_jitter: float = 0.1  # orientation noise (sigma) [rad]
    length_frac: tuple[float, float] = (0.1, 0.2)  # stick length as fraction of avg_spacing
    width_frac: tuple[float, float] = (0.1, 0.2)  # stick width as fraction of stick length

    # Interdot intensity (a scene-wide mean is drawn per scene, then per-stick jitter).
    intensity_range: tuple[float, float] = (-33.0, -1.0)  # i_mean ~ U(range)
    intensity_jitter: float = 0.1  # per-stick i ~ i_mean * [1-j, 1+j]
    line_intensity_frac: float = 0.5  # connecting-line intensity = i_mean * frac

    # Appearance probabilities, blur and acquisition noise.
    p_appear: float = 0.7  # probability a lattice site shows an interdot
    p_line: float = 0.2  # probability neighbouring interdots are joined by a line
    sigma_blur: float = 0.5  # gaussian blur (pixels)
    noise_sigma_pixel: float = 0.9  # 2D white noise
    noise_sigma_h: float = 0.7  # 1D horizontal-stripe noise


# The shared appearance/physics every participant runs against (challenge 1 uses
# this directly; challenge 2 reuses it via ChallengeConfig.generator).
GENERATOR = GeneratorConfig()


@dataclass(frozen=True)
class ChallengeConfig:
    """Challenge-2 optimization landscape — plus the scene it lives on.

    Frozen so every run shares one consistent configuration. ``generator`` is the
    shared CSD physics; the remaining fields define the hidden contrast landscape.
    """

    # Shared CSD physics. Defaults to the same GENERATOR the dataset uses, but with
    # a *fixed* stick intensity so challenge-2 contrast comes purely from the region
    # model (base + amplitude*L) rather than per-scene intensity swings.
    generator: GeneratorConfig = field(
        default_factory=lambda: replace(GENERATOR, intensity_range=(-1.0, -1.0))
    )

    # Contrast landscape: a base floor plus one Lorentzian bump per spatial region
    # (see csd.simulator.RegionContrastModel), so different patches of the diagram
    # light up at different barriers and finding the optimum takes a real search.
    n_regions: int = 4  # number of spatial regions, each with its own peak
    optimum_range: float = 0.5  # each region's barrier centre drawn in [-r, +r]
    base: float = 3.0  # floor contrast on every stick (never fully blank)

    amplitude: float = 25.0  # central per-region peak contrast boost
    amplitude_jitter: float = .2  # per-region A ~ [1-j, 1+j] x amplitude

    gamma: float = 0.10  # central Lorentzian half-width per barrier
    gamma_jitter: float = 0.5  # per-(region,barrier) width ~ [1-j, 1+j] x gamma

    # Drift: barrier->plunger lever arms, randomised near these defaults each run,
    # plus small quadratic terms so the interdots drift non-linearly over long
    # barrier excursions (see csd.simulator.DriftMatrix.random).
    drift_lever_arms: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.80, -0.35, -0.20),  # dg2 per volt on (g1, g3, g5)
        (-0.20, -0.35, -0.80),  # dg4 per volt on (g1, g3, g5)
    )
    drift_jitter: float = 0.15  # per-arm ~ [1-j, 1+j] x default lever arm
    drift_curvature: float = .5  # magnitude of the order-2 drift terms


# The single, shared configuration every participant runs against.
CHALLENGE = ChallengeConfig()
