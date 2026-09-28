"""Interactive explorer for the stage-2 challenge — 1D and 2D measurements.

    python explore_simulator.py                       # 0.3 V scene, seed 0
    python explore_simulator.py --scene-span 0.6 -s 3

This drives the **participant-facing API** directly: a single
:class:`~csd.challenge.Experiment` from :func:`~csd.challenge.new_experiment`,
poked only through ``exp.measure(...)`` and ``exp.scan_1d(...)`` — exactly the two
calls students use in stage 2. Nothing here reaches into the simulator internals.

What you are looking at
-----------------------
Left panel  : a **2D measurement** (``exp.measure``) — an image of the CSD in the
              (g2, g4) plunger plane. The cyan dashed arrow drawn on top is the
              path of the 1D scan.
Right panel : the **1D measurement** (``exp.scan_1d``) — the amplitude sampled
              along that cyan line, plotted against position along the cut.

Both panels share one amplitude scale, so when the interdots brighten (higher
contrast) both panels brighten together. The optimum barrier point is **hidden
and randomised** each scene (just like the real challenge): the readout shows the
noisy image std (a simple objective you might optimise) and the running
measurement budget you have spent.

The sliders
-----------
Working point (left column):
  * g2, g4      -> centre of BOTH scans (pan the viewport / move the cut centre)
  * g1, g3, g5  -> barriers: drift the interdots AND set their (hidden) contrast

Scan settings (right column):
  * 2D span / 2D step -> field of view and resolution of the 2D image
  * 1D angle          -> orientation of the cut in the (g2, g4) plane
                         (0deg = pure g2, 90deg = pure g4, 45deg = diagonal detuning)
  * 1D span / 1D step -> length and sample spacing of the cut

Pedagogical points to try
-------------------------
  * Shrink "2D span" to zoom in; watch "2D pixels" (the cost) fall.
  * Rotate "1D angle" and watch the cyan arrow sweep and the trace change — a
    cut across an interdot shows a bump; a cut along a charging line stays flat.
  * Hunt the barriers for higher image std — the hidden sweet-spot brightens
    both panels; watch the measurement budget climb as you search.
  * The scene is generated once over a fixed window (the lime dotted box). Zoom
    "2D span" past it, or pan g2/g4 toward the extremes, and you fall off into
    noise-only space (nothing new is invented). Move a barrier and watch the box
    itself drift — the barriers translate the whole scene.

"New scene" starts a fresh ``new_experiment`` (new layout + new hidden optimum);
"Reset" restores the start working point.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from csd.challenge import new_experiment
from csd.config import CHALLENGE

BARRIERS = ("g1", "g3", "g5")
GATE_LABELS = {
    "g1": "g1 (barrier)",
    "g2": "g2 (plunger)",
    "g3": "g3 (barrier)",
    "g4": "g4 (plunger)",
    "g5": "g5 (barrier)",
}


class SimulatorExplorer:
    def __init__(self, scene_span: float = 0.3, step: float = 1e-3, seed: int = 0):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.plt = plt
        self.seed = seed
        self.scene_span = scene_span
        self.step = step
        # One Experiment, exactly as a participant would create it. The scene is
        # generated once on this window; every measure/scan_1d is just a viewport
        # over the same sticks. As an organiser tool the explorer overrides the
        # scene span/step (on the shared CHALLENGE config); students never do this.
        self._config = replace(
            CHALLENGE, generator=replace(CHALLENGE.generator, scene_span=scene_span, scene_step=step)
        )
        self.exp = new_experiment(seed=seed, config=self._config)
        # We hold the working point ourselves and pass it into each call — the
        # Experiment is otherwise stateless between measurements.
        self.wp = dict(self.exp.start)

        self.fig = plt.figure(figsize=(13, 9))
        self.fig.canvas.manager.set_window_title("CSD challenge explorer — 1D & 2D scans")

        # -- panels ---------------------------------------------------------
        self.ax_img = self.fig.add_axes([0.06, 0.46, 0.40, 0.48])
        self.ax_img.set_title("2D measurement  (exp.measure)")
        self.ax_img.set_xlabel("$g_2$ [V]")
        self.ax_img.set_ylabel("$g_4$ [V]")

        self.ax_1d = self.fig.add_axes([0.58, 0.46, 0.38, 0.40])
        self.ax_1d.set_title("1D measurement  (exp.scan_1d)")
        self.ax_1d.set_xlabel("position along cut [V]")
        self.ax_1d.set_ylabel("amplitude")
        self.ax_1d.grid(True, alpha=0.3)

        # Fixed amplitude scale from a frame at the (hidden) optimum, shared by
        # both panels so brightening/fading reads as real contrast change.
        self._clim = self._peak_clim()

        # Start a touch wider than the scene so the empty border is visible.
        self._zoom_max = 2.0 * scene_span
        span2d0 = min(1.4 * scene_span, self._zoom_max)
        image = self._measure_2d(span2d0, 2e-3)
        self.im = self.ax_img.imshow(
            image,
            extent=self._extent(span2d0),
            origin="lower",
            cmap="plasma",
            vmin=self._clim[0],
            vmax=self._clim[1],
            aspect="auto",
        )

        # Shared amplitude colourbar in the gap between the two panels.
        cax = self.fig.add_axes([0.475, 0.46, 0.013, 0.48])
        self.cbar = self.fig.colorbar(self.im, cax=cax)
        self.cbar.set_label("amplitude")

        # Boundary of the (finite) scene the sticks were generated in. It is a
        # fixed [0, scene_span]^2 box that the barriers drift bodily — outside it
        # there are no sticks, only noise. (Read-only peek at the sim geometry,
        # purely to draw this guide; students never need it.)
        from matplotlib.patches import Rectangle

        self.scene_box = Rectangle(
            (0.0, 0.0),
            scene_span,
            scene_span,
            fill=False,
            edgecolor="lime",
            lw=1.5,
            ls=":",
            label="scene extent (sticks live here)",
        )
        self.ax_img.add_patch(self.scene_box)
        self.ax_img.legend(loc="upper right", fontsize=7, framealpha=0.6)

        # Overlay: the 1D cut drawn on the 2D image.
        (self.cut_line,) = self.ax_img.plot([], [], color="cyan", lw=1.6, ls="--")
        (self.cut_dot,) = self.ax_img.plot([], [], "o", color="cyan", ms=5)
        self._cut_arrow = None

        # The 1D trace.
        (self.trace_line,) = self.ax_1d.plot([], [], color="C0", lw=1.5)
        self.ax_1d.axvline(0.0, color="cyan", ls="--", lw=1.0, alpha=0.7)
        self.ax_1d.set_ylim(*self._clim)

        # -- sliders --------------------------------------------------------
        self.fig.text(0.06, 0.40, "Working point", fontsize=11, fontweight="bold")
        self.fig.text(0.58, 0.40, "Scan settings", fontsize=11, fontweight="bold")

        self.sliders = {}
        span_pan = scene_span  # pan far enough to move the window fully off-scene
        bar = max(0.1, self._config.optimum_range * 1.1)  # reach the barrier optima
        s = self.exp.start

        left_specs = [
            ("g1", s["g1"] - bar, s["g1"] + bar, s["g1"]),
            ("g2", s["g2"] - span_pan, s["g2"] + span_pan, s["g2"]),
            ("g3", s["g3"] - bar, s["g3"] + bar, s["g3"]),
            ("g4", s["g4"] - span_pan, s["g4"] + span_pan, s["g4"]),
            ("g5", s["g5"] - bar, s["g5"] + bar, s["g5"]),
        ]
        for idx, (gate, vmin, vmax, vinit) in enumerate(left_specs):
            y = 0.35 - idx * 0.05
            ax = self.fig.add_axes([0.14, y, 0.26, 0.025])
            sl = Slider(ax, GATE_LABELS[gate], vmin, vmax, valinit=vinit)
            sl.on_changed(lambda _v, g=gate: self._on_gate(g))
            self.sliders[gate] = sl

        right_specs = [
            ("span2d", 0.05, self._zoom_max, span2d0, "2D span [V]"),
            ("step2d", 1e-3, 8e-3, 2e-3, "2D step [V]"),
            ("angle", 0.0, 180.0, 0.0, "1D angle [deg]"),
            ("span1d", 0.02, self._zoom_max, min(0.2, self._zoom_max), "1D span [V]"),
            ("step1d", 5e-4, 5e-3, 2e-3, "1D step [V]"),
        ]
        for idx, (key, vmin, vmax, vinit, label) in enumerate(right_specs):
            y = 0.35 - idx * 0.05
            ax = self.fig.add_axes([0.70, y, 0.24, 0.025])
            sl = Slider(ax, label, vmin, vmax, valinit=vinit)
            sl.on_changed(lambda _v: self._redraw())
            self.sliders[key] = sl

        # -- buttons + info -------------------------------------------------
        self.btn_new = Button(self.fig.add_axes([0.06, 0.05, 0.12, 0.04]), "New scene")
        self.btn_reset = Button(self.fig.add_axes([0.20, 0.05, 0.10, 0.04]), "Reset")
        self.btn_goto = Button(self.fig.add_axes([0.32, 0.05, 0.18, 0.04]), "Go to optimum")
        self.btn_new.on_clicked(self._on_new_scene)
        self.btn_reset.on_clicked(self._on_reset)
        self.btn_goto.on_clicked(self._on_goto_optimum)

        self._info = self.fig.text(0.58, 0.06, "", va="center", fontsize=10, family="monospace")

        self._redraw()

    # -- measurement helpers (participant API only) ------------------------
    def _measure_2d(self, span2d: float, step2d: float) -> np.ndarray:
        return self.exp.measure(
            **self.wp, span_h=span2d, span_v=span2d, step_h=step2d, step_v=step2d
        )

    def _extent(self, span2d: float) -> tuple[float, float, float, float]:
        return (
            self.wp["g2"] - span2d / 2,
            self.wp["g2"] + span2d / 2,
            self.wp["g4"] - span2d / 2,
            self.wp["g4"] + span2d / 2,
        )

    def _peak_clim(self) -> tuple[float, float]:
        """Colour limits from a frame at the hidden optimum (revealed here only to
        fix a stable amplitude scale — students never need this).

        The interdots are the *dark dips*; they cover well under 1% of the pixels.
        So the low colour limit must be the deepest dip itself (``frame.min()``),
        NOT ``percentile(1)`` — a 1% percentile never reaches the sticks and lands
        in the noise floor (~-2.5), which would clip every dip to the same darkest
        colour at every working point and hide the contrast change entirely. The
        high limit is the background ceiling (``percentile(99)`` ~ the noise top).
        """
        rev = self.exp.reveal()
        wp = dict(self.exp.start)
        wp.update(rev["optimum_barriers"])
        wp.update(rev["optimum_plungers"])  # centre the best region in the window
        frame = self.exp.measure(**wp)
        return float(frame.min()), float(np.percentile(frame, 99))

    # -- redraw ------------------------------------------------------------
    def _redraw(self) -> None:
        span2d = self.sliders["span2d"].val
        step2d = self.sliders["step2d"].val
        angle = np.deg2rad(self.sliders["angle"].val)
        span1d = self.sliders["span1d"].val
        step1d = self.sliders["step1d"].val
        direction = (float(np.cos(angle)), float(np.sin(angle)))

        # 2D panel
        image = self._measure_2d(span2d, step2d)
        self.im.set_data(image)
        self.im.set_extent(self._extent(span2d))
        self.im.set_clim(*self._clim)

        # Scene box drifts bodily with the barriers (drift = matrix @ d_barriers).
        sim = self.exp._sim
        drift = sim.drift_matrix.drift([self.wp[b] - sim.start[b] for b in BARRIERS])
        lower_left = sim._start_origin + drift
        self.scene_box.set_xy((lower_left[0], lower_left[1]))

        # 1D cut overlay on the 2D image
        c = np.array([self.wp["g2"], self.wp["g4"]])
        d = np.array(direction)
        p0 = c - (span1d / 2) * d
        p1 = c + (span1d / 2) * d
        self.cut_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        self.cut_dot.set_data([c[0]], [c[1]])
        if self._cut_arrow is not None:
            self._cut_arrow.remove()
        self._cut_arrow = self.ax_img.annotate(
            "",
            xy=(p1[0], p1[1]),
            xytext=(c[0], c[1]),
            arrowprops=dict(arrowstyle="->", color="cyan", lw=1.6),
        )

        # 1D panel
        trace = self.exp.scan_1d(direction, span1d, step1d, **self.wp)
        x = np.linspace(-span1d / 2, span1d / 2, len(trace))
        self.trace_line.set_data(x, trace)
        self.ax_1d.set_xlim(-span1d / 2, span1d / 2)
        self.ax_1d.set_ylim(*self._clim)

        # info: observable metric + running budget (pixels ~ acquisition time)
        rev = self.exp.reveal()
        self._info.set_text(
            f"image std = {image.std():5.2f}   seed = {self.seed}\n"
            f"2D: {image.shape[1]}x{image.shape[0]} = {image.size:,} px      "
            f"1D: {len(trace)} samples\n"
            f"budget: {rev['n_measurements']} measurements, {rev['n_pixels']:,} px"
        )
        self.fig.canvas.draw_idle()

    # -- callbacks ---------------------------------------------------------
    def _on_gate(self, gate: str) -> None:
        self.wp[gate] = self.sliders[gate].val
        self._redraw()

    def _set_gate_slider(self, gate: str, value: float) -> None:
        """Move a working-point slider to ``value``, widening its range if needed."""
        sl = self.sliders[gate]
        if value < sl.valmin or value > sl.valmax:
            margin = 0.1 * (abs(value) + 1e-3)
            sl.valmin = min(sl.valmin, value - margin)
            sl.valmax = max(sl.valmax, value + margin)
            sl.ax.set_xlim(sl.valmin, sl.valmax)
        sl.eventson = False
        sl.set_val(value)
        sl.eventson = True

    def _sync_gate_sliders(self) -> None:
        for gate in ("g1", "g2", "g3", "g4", "g5"):
            self._set_gate_slider(gate, self.wp[gate])

    def _on_new_scene(self, _event) -> None:
        self.seed += 1
        self.exp = new_experiment(seed=self.seed, config=self._config)
        self.wp = dict(self.exp.start)
        self._clim = self._peak_clim()
        self._sync_gate_sliders()
        self._redraw()

    def _on_reset(self, _event) -> None:
        self.wp = dict(self.exp.start)
        self._sync_gate_sliders()
        self._redraw()

    def _on_goto_optimum(self, _event) -> None:
        """Teleport the working point to the hidden highest-contrast region."""
        rev = self.exp.reveal()
        self.wp = dict(self.exp.start)
        self.wp.update(rev["optimum_barriers"])  # g1, g3, g5
        self.wp.update(rev["optimum_plungers"])  # g2, g4 -> centre that region
        self._sync_gate_sliders()
        self._redraw()

    def show(self) -> None:
        self.plt.show()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Interactive 1D/2D explorer for the stage-2 challenge.")
    p.add_argument(
        "--scene-span",
        type=float,
        default=0.3,
        help="span [V] of the fixed scene window the sticks are generated in (both axes)",
    )
    p.add_argument("--step", type=float, default=1e-3, help="scene generation step in V")
    p.add_argument("-s", "--seed", type=int, default=0, help="scene random seed")
    args = p.parse_args()

    print(f"scene window span: {args.scene_span} V (measurements are viewports over it)")
    SimulatorExplorer(scene_span=args.scene_span, step=args.step, seed=args.seed).show()


if __name__ == "__main__":
    main()
