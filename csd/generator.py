from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
from skimage.draw import line

from .config import GENERATOR, GeneratorConfig

Float64Array = npt.NDArray[np.float64]


def get_unit_vector(theta: float):
    """return the unit vector with angle theta.

    theta: angle
    """
    return np.array([np.cos(theta), np.sin(theta)])


def get_n_iid_normal(n: int, avg: float, var: float):
    return [np.random.normal(loc=avg, scale=np.sqrt(var)) for i in range(n)]


@dataclass
class ScanWindow:
    """Physical scan window, expressed in volts.

    span_h / span_v: horizontal / vertical span of the window in V.
    step_h / step_v: horizontal / vertical step (voltage per pixel) in V.

    The number of pixels follows naturally from span / step, so the pixel grid
    is derived rather than specified directly. Defaults reproduce the previous
    behaviour: 0.5 V span with a 1 mV step -> 500 x 500 pixels.
    """

    span_h: float = 0.5
    span_v: float = 0.5
    step_h: float = 1e-3
    step_v: float = 1e-3

    @property
    def n_h(self) -> int:
        """Number of horizontal pixels (columns)."""
        return round(self.span_h / self.step_h)

    @property
    def n_v(self) -> int:
        """Number of vertical pixels (rows)."""
        return round(self.span_v / self.step_v)


def scene_window(config: GeneratorConfig) -> ScanWindow:
    """The square scan window the scene is generated in, from ``config``."""
    return ScanWindow(
        span_h=config.scene_span,
        span_v=config.scene_span,
        step_h=config.scene_step,
        step_v=config.scene_step,
    )


@dataclass
class Interdot:
    middle: Float64Array  # shape (2,)
    theta: float
    length: float
    width: float
    intensity: float
    i: int
    j: int


def add_window_noise(image: Float64Array, config: GeneratorConfig) -> Float64Array:
    noisy_image = image.copy()
    n_v, n_h = noisy_image.shape

    # Centered Gaussian White Noise
    white_noise_2d = np.random.randn(n_v, n_h) * config.noise_sigma_pixel

    # 1D Horizontal Centered Gaussian White Noise
    horizontal_noise = np.random.randn(n_v, 1) * config.noise_sigma_h

    noisy_image += white_noise_2d + horizontal_noise

    return noisy_image


def random_interdots_positions(
    config: GeneratorConfig, scan_window: ScanWindow
) -> Float64Array:
    """Lay out the interdot lattice: two families of charging lines crossing."""
    n_l = n_r = config.n_dots
    avg_spacing = config.avg_spacing
    var_spacing = config.var_spacing
    var_interdot_middle = config.var_interdot_middle

    # Starting point scales with the window span so coverage stays consistent
    # across different spans (default span 0.5 V reproduces the old -0.5..0 range).
    random_starting_interdot = (
        np.random.uniform(0, -scan_window.span_h),
        np.random.uniform(0, -scan_window.span_v),
    )
    alpha, beta = config.angle_alpha, config.angle_beta

    # theta_l near pi/2 (left charging-line family)
    u_l = np.random.beta(alpha, beta)
    theta_l = np.pi / 2 + u_l * (np.pi / 2)

    # theta_r near 0 (right charging-line family)
    u_r = np.random.beta(alpha, beta)
    theta_r = -u_r * (np.pi / 2)

    e_l = get_unit_vector(theta_l)
    e_r = get_unit_vector(theta_r)

    spacings_l = np.array(get_n_iid_normal(n_l, avg_spacing, var_spacing))
    spacings_r = np.array(get_n_iid_normal(n_r, avg_spacing, var_spacing))

    interdots_pos = np.zeros((n_l, n_r, 2))
    for i in range(n_l):
        for j in range(n_r):
            interdots_pos[i, j] = (
                random_starting_interdot
                + np.sum(spacings_l[:i]) * e_l
                + np.sum(spacings_r[:j]) * e_r
                + (np.random.normal(0, var_interdot_middle), np.random.normal(0, var_interdot_middle))
            )
    return interdots_pos


def from_interdots_pos_to_interdots(
    interdots_pos: Float64Array,
    scan_window: ScanWindow,
    config: GeneratorConfig,
    intensity: tuple[float, float],
) -> list[Interdot]:
    """Turn lattice positions into rendered-ready sticks (shape, angle, intensity).

    ``intensity`` is the ``(i_min, i_max)`` band for this scene (drawn once in
    :func:`build_interdots` from the scene-wide mean).
    """
    i_min, i_max = intensity
    avg_spacing = config.avg_spacing
    l_min = avg_spacing * config.length_frac[0]
    l_max = avg_spacing * config.length_frac[1]
    w_min = l_min * config.width_frac[0]
    w_max = l_max * config.width_frac[1]

    interdots = []
    n_l, n_r, _ = interdots_pos.shape  # original grid size
    for i in range(n_l):
        for j in range(n_r):
            x, y = interdots_pos[i, j]

            # keep only points inside the window [0, span_h] x [0, span_v]
            if not (0 <= x <= scan_window.span_h and 0 <= y <= scan_window.span_v):
                continue

            # appearance probability
            if np.random.rand() > config.p_appear:
                continue

            interdot = Interdot(
                middle=np.array([x, y]),
                theta=config.stick_theta + np.random.normal(scale=config.stick_theta_jitter),
                length=np.random.uniform(l_min, l_max),
                width=np.random.uniform(w_min, w_max),
                intensity=np.random.uniform(i_min, i_max),
                i=i,  # row index
                j=j,  # column index
            )

            interdots.append(interdot)

    return interdots


def add_interdot(image: Float64Array, interdot: Interdot, scan_window: ScanWindow) -> Float64Array:
    image_with_interdot = image.copy()

    n_v, n_h = image_with_interdot.shape
    step_h = scan_window.step_h
    step_v = scan_window.step_v

    cx, cy = interdot.middle
    length = interdot.length
    width = interdot.width
    theta = interdot.theta
    intensity = interdot.intensity

    u_par = np.array([np.cos(theta), np.sin(theta)])
    u_perp = np.array([-np.sin(theta), np.cos(theta)])

    # bounding box in pixels
    margin = length
    xmin, xmax = cx - margin, cx + margin
    ymin, ymax = cy - margin, cy + margin

    ix_min = int(xmin / step_h)
    ix_max = int(xmax / step_h)
    iy_min = int(ymin / step_v)
    iy_max = int(ymax / step_v)

    for ix in range(ix_min, ix_max):
        for iy in range(iy_min, iy_max):
            if not (0 <= ix < n_h and 0 <= iy < n_v):
                continue

            x = ix * step_h
            y = iy * step_v
            r = np.array([x, y]) - np.array([cx, cy])

            if abs(r @ u_par) <= length / 2 and abs(r @ u_perp) <= width / 2:
                image_with_interdot[iy, ix] += intensity

    return image_with_interdot


def add_interdots(
    image: Float64Array, interdots: list[Interdot], scan_window: ScanWindow
) -> Float64Array:
    for interdot in interdots:
        image = add_interdot(image, interdot, scan_window)
    return image


def draw_line_h(image, interdot_top, interdot_bottom, scan_window: ScanWindow, intensity=5.0):
    """Connect the top of interdot_top to the bottom of interdot_bottom in the image."""
    n_v, n_h = image.shape
    step_h = scan_window.step_h
    step_v = scan_window.step_v

    # unit vector along each interdot
    u_par_top = np.array([np.cos(interdot_top.theta), np.sin(interdot_top.theta)])
    u_par_bottom = np.array([np.cos(interdot_bottom.theta), np.sin(interdot_bottom.theta)])

    # endpoints to connect
    top_point = interdot_top.middle + (interdot_top.length / 2) * u_par_top
    bottom_point = interdot_bottom.middle - (interdot_bottom.length / 2) * u_par_bottom

    # convert to pixel indices
    x0 = int(top_point[0] / step_h)
    y0 = int(top_point[1] / step_v)
    x1 = int(bottom_point[0] / step_h)
    y1 = int(bottom_point[1] / step_v)

    # line
    rr, cc = line(y0, x0, y1, x1)
    rr = np.clip(rr, 0, n_v - 1)
    cc = np.clip(cc, 0, n_h - 1)

    image[rr, cc] += intensity


def draw_line_v(image, interdot_left, interdot_right, scan_window: ScanWindow, intensity=5.0):
    """Connect interdot_left to interdot_right in the image (horizontal neighbour)."""
    n_v, n_h = image.shape
    step_h = scan_window.step_h
    step_v = scan_window.step_v

    # unit vector along each interdot
    u_par_left = np.array([np.cos(interdot_left.theta), np.sin(interdot_left.theta)])
    u_par_right = np.array([np.cos(interdot_right.theta), np.sin(interdot_right.theta)])

    # endpoints to connect
    left_point = interdot_left.middle + (interdot_left.length / 2) * u_par_left
    right_point = interdot_right.middle - (interdot_right.length / 2) * u_par_right

    # convert to pixel indices
    x0 = int(left_point[0] / step_h)
    y0 = int(left_point[1] / step_v)
    x1 = int(right_point[0] / step_h)
    y1 = int(right_point[1] / step_v)

    # line
    rr, cc = line(y0, x0, y1, x1)
    rr = np.clip(rr, 0, n_v - 1)
    cc = np.clip(cc, 0, n_h - 1)

    image[rr, cc] += intensity


@dataclass
class LineSpec:
    """One connecting charging line between two neighbouring interdots.

    ``kind`` is "h" (draw_line_h, vertical neighbour i -> i+1) or "v"
    (draw_line_v, horizontal neighbour j -> j+1). ``idx_a`` / ``idx_b`` index
    into the interdot list returned alongside these specs, so the line follows
    its endpoints when their positions are shifted (e.g. by the simulator).
    """

    kind: str  # "h" or "v"
    idx_a: int
    idx_b: int
    intensity: float


def build_interdots(
    config: GeneratorConfig = GENERATOR,
    scan_window: ScanWindow | None = None,
) -> tuple[list[Interdot], list[LineSpec]]:
    """Generate the *geometry* of a CSD once: the interdots and which
    neighbouring pairs are connected by a charging line.

    This performs all of the geometry randomness (positions, appearance, sizes,
    intensities, and the connecting-line coin flips) and returns plain data. It
    does no rendering, so the result can be rendered many times — possibly after
    moving the interdots — via :func:`render_csd`.
    """
    if scan_window is None:
        scan_window = scene_window(config)

    interdots_pos = random_interdots_positions(config, scan_window)

    # Scene-wide mean intensity, drawn once, then a per-stick band around it.
    i_mean = np.random.uniform(config.intensity_range[0], config.intensity_range[1])
    i_min = i_mean * (1 - config.intensity_jitter)
    i_max = i_mean * (1 + config.intensity_jitter)
    interdots = from_interdots_pos_to_interdots(
        interdots_pos=interdots_pos,
        scan_window=scan_window,
        config=config,
        intensity=(i_min, i_max),
    )

    idx_of = {(s.i, s.j): k for k, s in enumerate(interdots)}
    line_specs: list[LineSpec] = []
    line_intensity = i_mean * config.line_intensity_frac
    p_line = config.p_line

    # vertical lines (i -> i+1)
    for (i, j), k in idx_of.items():
        k_bottom = idx_of.get((i + 1, j))
        if k_bottom is not None and np.random.rand() > 1 - p_line:
            line_specs.append(LineSpec("h", k, k_bottom, line_intensity))

    # horizontal lines (j -> j+1)
    for (i, j), k in idx_of.items():
        k_right = idx_of.get((i, j + 1))
        if k_right is not None and np.random.rand() > 1 - p_line:
            line_specs.append(LineSpec("v", k, k_right, line_intensity))

    return interdots, line_specs


def render_csd(
    interdots: list[Interdot],
    line_specs: list[LineSpec],
    scan_window: ScanWindow,
    config: GeneratorConfig = GENERATOR,
    normalize: bool = True,
) -> Float64Array:
    """Render a CSD image from pre-built geometry.

    ``scan_window`` is the *measurement* grid (may differ from the scene window);
    ``config`` supplies the acquisition noise and blur. Deterministic except for
    the noise, which is drawn fresh on every call (same statistics, new
    realisation) — so calling this repeatedly on the same geometry models
    successive measurements.
    """
    image = np.zeros((scan_window.n_v, scan_window.n_h))
    image = add_window_noise(image, config)
    image = add_interdots(image, interdots, scan_window)

    for spec in line_specs:
        s_a, s_b = interdots[spec.idx_a], interdots[spec.idx_b]
        if spec.kind == "h":
            draw_line_h(image, s_a, s_b, scan_window, intensity=spec.intensity)
        else:
            draw_line_v(image, s_a, s_b, scan_window, intensity=spec.intensity)

    image = gaussian_filter(image, sigma=config.sigma_blur)

    if normalize:
        image = (image - np.mean(image)) / np.std(image)

    return image


def generate_csd_and_label(
    config: GeneratorConfig = GENERATOR,
    scan_window: ScanWindow | None = None,
    plot: bool = False,
    normalize: bool = False,
) -> tuple[Float64Array, Float64Array]:
    """Generate a CSD image and its interdot label mask.

    By default the image is returned **raw** (unnormalized), so it matches the
    distribution produced by :class:`csd.simulator.CSDSimulator`. Any per-image
    normalization is left to the consumer. Pass ``normalize=True`` for the legacy
    z-scored output.
    """
    if scan_window is None:
        scan_window = scene_window(config)

    interdots, line_specs = build_interdots(config, scan_window)

    image = render_csd(
        interdots=interdots,
        line_specs=line_specs,
        scan_window=scan_window,
        config=config,
        normalize=normalize,
    )

    if plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(
            image,
            extent=(0, scan_window.span_h, 0, scan_window.span_v),
            origin="lower",
            cmap="plasma",
        )
        plt.colorbar(label="Amplitude")
        plt.title("CSD")
        plt.show()

    image_label = generate_label(interdots, scan_window, config, plot)
    return image, image_label


def generate_label(
    interdots: list[Interdot],
    scan_window: ScanWindow,
    config: GeneratorConfig = GENERATOR,
    plot=False,
) -> Float64Array:
    image = np.zeros((scan_window.n_v, scan_window.n_h))
    for interdot in interdots:
        interdot2 = Interdot(
            middle=interdot.middle.copy(),
            theta=interdot.theta,
            length=interdot.length,
            width=interdot.width,
            intensity=1,  # label uses unit intensity for every stick
            i=interdot.i,
            j=interdot.j,
        )
        image = add_interdot(interdot=interdot2, image=image, scan_window=scan_window)

    image = gaussian_filter(image, sigma=config.sigma_blur)

    if plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(
            image,
            extent=(0, scan_window.span_h, 0, scan_window.span_v),
            origin="lower",
            cmap="plasma",
        )
        plt.colorbar(label="Amplitude")
        plt.title("CSD")
        plt.show()
    return image
