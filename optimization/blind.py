"""Budget-enforcing, hidden-state-free view of an Experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BudgetExceeded(RuntimeError):
    """Raised *before* a measurement that would exceed the pixel / measurement cap."""


@dataclass
class Event:
    kind: str  # "measure" | "scan_1d" | "commit"
    n_pixels: int  # cumulative, after the event
    n_meas: int
    gates: dict
    span: tuple | None = None


class BlindExperiment:
    """What an algorithm may use of an :class:`csd.Experiment`.

    Exposes only ``measure``, ``scan_1d``, ``start``, ``extent`` and the budget
    counters. The pixel cap is a hard limit. All span/step arguments must be explicit
    except through :meth:`reference_frame`, which measures the default window once and
    thereby *discovers* the native step from the returned image (nothing is read
    from the simulator configuration).
    """

    def __init__(self, exp, pixel_cap: int, meas_cap: int = 300) -> None:
        self._exp = exp
        self.pixel_cap = int(pixel_cap)
        self.meas_cap = int(meas_cap)
        self.n_pixels = 0
        self.n_meas = 0
        self.events: list[Event] = []
        self.native_step: float | None = None
        self.native_span: float | None = None

    # -- read-only public facts ---------------------------------------------
    @property
    def start(self) -> dict:
        return self._exp.start

    @property
    def extent(self) -> tuple:
        return self._exp.extent

    @property
    def remaining(self) -> int:
        return self.pixel_cap - self.n_pixels

    # -- measurements ------------------------------------------------------
    def reference_frame(self) -> np.ndarray:
        """Measure the default window once; learn the native span/step from it."""
        img = self._exp.measure()
        left, right, bottom, top = self._exp.extent
        self.native_span = float(right - left)
        self.native_step = self.native_span / img.shape[1]
        self._log("measure", img.size, self.start, (self.native_span, top - bottom))
        return img

    def measure(
        self,
        *,
        g1=None, g2=None, g3=None, g4=None, g5=None,
        span_h: float, span_v: float, step_h: float, step_v: float,
    ) -> np.ndarray:
        cost = round(span_h / step_h) * round(span_v / step_v)
        if self.n_pixels + cost > self.pixel_cap or self.n_meas + 1 > self.meas_cap:
            raise BudgetExceeded(f"{self.n_pixels}+{cost} > {self.pixel_cap} px")
        img = self._exp.measure(
            g1=g1, g2=g2, g3=g3, g4=g4, g5=g5,
            span_h=span_h, span_v=span_v, step_h=step_h, step_v=step_v,
        )
        gates = dict(zip(("g1", "g2", "g3", "g4", "g5"), (g1, g2, g3, g4, g5)))
        self._log("measure", img.size, gates, (span_h, span_v))
        return img

    def scan_1d(self, direction, span: float, step: float, **gates) -> np.ndarray:
        cost = int(round(span / step)) + 1
        if self.n_pixels + cost > self.pixel_cap or self.n_meas + 1 > self.meas_cap:
            raise BudgetExceeded(f"{self.n_pixels}+{cost} > {self.pixel_cap} px")
        trace = self._exp.scan_1d(direction, span, step, **gates)
        self._log("scan_1d", cost, gates, None)
        return trace

    def commit(self, gates: dict) -> None:
        """Engage a recommendation (journalled; used by the evaluator)."""
        self.events.append(Event("commit", self.n_pixels, self.n_meas, dict(gates)))

    def _log(self, kind, cost, gates, span) -> None:
        self.n_pixels += int(cost)
        self.n_meas += 1
        self.events.append(
            Event(kind, self.n_pixels, self.n_meas,
                  {k: (None if v is None else float(v)) for k, v in dict(gates).items()}, span)
        )
