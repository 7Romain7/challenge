"""Shared measurement session: reference frame, drift calibration, tracked scoring.

Every method (random search, BO, learned policies...) talks to the device only through
``Session.evaluate(b)``: it moves the barriers ``b = (g1, g3, g5)``, re-centres the
plungers on the *predicted* drift of the tracked interdots, measures, registers the
image on the reference interdot set, refines the drift model, and records the
per-interdot amplitudes. Plungers are never decision variables (PROTOCOL P1).

Start-up = reference frame + 3 axis probes (lever arms). The 2nd-order drift model is
then refined by every registered frame. Because the drift is strongly non-linear, a
target is only measured where the model's predictive std is small enough for
registration to succeed (:meth:`feasible`): the explored region grows as the model
learns (a model-uncertainty trust region; also physically gentler on the device).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blind import BlindExperiment
from .perception import Perception
from .scoring import Scorer
from .tracking import DriftModel, register


@dataclass
class SessionConfig:
    box: float = 0.6  # barrier search box: |g1|,|g3|,|g5| <= box  [V] (hardware limit)
    span: float = 0.3  # window span [V]; the step is the native step
    probe: float = 0.08  # first axis probes [V]
    sigma_max: float = 0.025  # max drift-prediction std [V] for a point to be measurable
    k_det: float = 4.0
    shrink_z: float = 1.0
    floor_q: float = 0.3
    verify_frames: int = 2
    min_matches: int = 3
    max_radius: float = 0.12
    keep_frames: bool = False  # store raw frames (dataset generation for JEPA only)


class Session:
    def __init__(self, bx: BlindExperiment, cfg: SessionConfig | None = None) -> None:
        self.bx = bx
        self.cfg = cfg or SessionConfig()
        self.percep = Perception(k_det=self.cfg.k_det)
        self.drift = DriftModel()
        self.scorer = Scorer(floor_q=self.cfg.floor_q, z=self.cfg.shrink_z)
        self.b0 = np.array([bx.start["g1"], bx.start["g3"], bx.start["g5"]])
        self.ref = np.zeros((0, 2))  # reference interdots, absolute plunger volts @ b0
        self.centroid = np.zeros(2)
        self.step = 0.0
        self.roles: list[str] = []
        self.frames: list[dict] = []
        self.reco_log: list[dict] = []
        self.n_track_fail = 0
        self._started = False

    # ------------------------------------------------------------------ start
    def start(self) -> None:
        bx, cfg = self.bx, self.cfg
        img = bx.reference_frame()
        self.step = bx.native_step
        frame = self.percep.process(img)
        left, _, bottom, _ = bx.extent
        origin = np.array([left, bottom])
        self.ref = origin + frame.peaks[:, [1, 0]] * self.step
        self.centroid = (self.ref.mean(axis=0) if len(self.ref)
                         else origin + bx.native_span / 2)
        rc = (self.ref - origin) / self.step
        self.scorer.add(self.b0, Perception.read_amplitudes(frame, rc[:, ::-1]), frame.sigma, True)
        self.roles.append("ref")
        self.drift.add(np.zeros(3), np.zeros(2))
        self._started = True
        for a in range(3):  # lever-arm probes, one per barrier
            db = np.zeros(3)
            db[a] = cfg.probe
            self.evaluate(self.b0 + db, role="probe")

    # --------------------------------------------------------------- helpers
    def _focus_ref(self, span: float) -> np.ndarray:
        """Window centre (in reference coordinates) containing the most reference interdots."""
        if span >= 0.95 * self.bx.native_span or len(self.ref) < 3:
            return self.centroid
        half = span / 2 - 0.03
        best, best_n = self.centroid, -1
        for c in self.ref:  # candidate centres = shifted reference interdots
            for off in ((0, 0), (half / 2, half / 2), (-half / 2, -half / 2),
                        (half / 2, -half / 2), (-half / 2, half / 2)):
                cc = c + np.array(off)
                n = int(np.sum(np.all(np.abs(self.ref - cc) <= half, axis=1)))
                if n > best_n:
                    best, best_n = cc, n
        return best

    def centre_for(self, b, span: float | None = None) -> np.ndarray:
        span = self.cfg.span if span is None else span
        return self._focus_ref(span) + self.drift.predict(np.asarray(b, float) - self.b0)

    def wp_for(self, b, span: float | None = None) -> dict:
        b = np.asarray(b, float)
        c = self.centre_for(b, span)
        return {"g1": float(b[0]), "g2": float(c[0]), "g3": float(b[1]),
                "g4": float(c[1]), "g5": float(b[2])}

    def feasible(self, b) -> bool:
        """Is the drift at ``b`` predictable enough to be tracked (trust region)?"""
        return self.drift.predict_std(np.asarray(b, float) - self.b0) <= self.cfg.sigma_max

    def sample_feasible(self, rng, n: int = 1, n_try: int = 4000) -> np.ndarray:
        """Uniform random points of the box that are currently feasible."""
        cand = rng.uniform(-self.cfg.box, self.cfg.box, (n_try, 3))
        ok = np.array([self.feasible(c) for c in cand])
        cand = cand[ok]
        if len(cand) == 0:
            return self.b0[None, :].repeat(n, axis=0)
        return cand[rng.choice(len(cand), size=n)]

    def _radius(self, db) -> float:
        sd = self.drift.predict_std(db)
        return float(np.clip(0.02 + 4.0 * sd, 0.025, self.cfg.max_radius))

    # --------------------------------------------------------------- evaluate
    def evaluate(self, b, role: str = "eval", span: float | None = None) -> dict:
        assert self._started, "call start() first"
        cfg = self.cfg
        span = cfg.span if span is None else span
        b = np.clip(np.asarray(b, float), -cfg.box, cfg.box)
        db = b - self.b0
        d_pred = self.drift.predict(db)
        centre = self._focus_ref(span) + d_pred
        img = self.bx.measure(
            g1=b[0], g2=centre[0], g3=b[1], g4=centre[1], g5=b[2],
            span_h=span, span_v=span, step_h=self.step, step_v=self.step,
        )
        if cfg.keep_frames:
            self.frames.append({"img": img.astype(np.float16), "b": b.copy(),
                                "centre": centre.copy(), "span": span, "role": role})
        frame = self.percep.process(img)
        origin = centre - span / 2
        det = origin + frame.peaks[:, [1, 0]] * self.step
        radius = self._radius(db)
        inside = np.all((self.ref + d_pred > origin - radius)
                        & (self.ref + d_pred < origin + span + radius), axis=1)
        delta, n_match = register(self.ref[inside], det, d_pred, radius)
        ok = n_match >= cfg.min_matches
        if ok:
            self.drift.add(db, delta)
            self._grow_reference(det, delta, frame)
            rc = (self.ref + delta - origin) / self.step  # (x=col, y=row)
            amps = Perception.read_amplitudes(frame, rc[:, ::-1])
        else:
            self.n_track_fail += 1
            amps = np.full(len(self.ref), np.nan)
        self.scorer.add(b, amps, frame.sigma, ok)
        self.roles.append(role)
        return {"ok": ok, "matches": n_match, "role": role, "radius": radius}

    def _grow_reference(self, det, delta, frame) -> None:
        """Add interdots detected here but absent from the reference (position = det - drift).

        Earlier score rows are shorter and get NaN-padded by ``Scorer``; the current row
        is read *after* growth, so a newly seen interdot has an amplitude right now.
        """
        if len(det) == 0:
            return
        strong = frame.peaks[:, 2] > 6.0 * frame.sigma
        new = []
        for q, s in zip(det, strong):
            if not s:
                continue
            p = q - delta
            if len(self.ref) and np.min(np.linalg.norm(self.ref - p, axis=1)) < 6e-3:
                continue
            if new and np.min(np.linalg.norm(np.array(new) - p, axis=1)) < 6e-3:
                continue
            new.append(p)
        if new:
            self.ref = np.vstack([self.ref, np.array(new)])

    # ------------------------------------------------------------ bookkeeping
    def points(self, roles=("probe", "eval", "verify")):
        """(B, y, se) for points with a valid score, restricted to comparable frames."""
        y, se = self.scorer.scores()
        B = np.array(self.scorer.B)
        m = np.isfinite(y) & np.isin(np.array(self.roles), roles)
        return B[m], y[m], np.maximum(se[m], 1e-3)

    def verify(self, b, n: int | None = None) -> None:
        for _ in range(self.cfg.verify_frames if n is None else n):
            self.evaluate(b, role="verify")

    def set_reco(self, b) -> None:
        """Log the algorithm's current recommendation (anytime curve)."""
        self.reco_log.append({"pixels": self.bx.n_pixels, "wp": self.wp_for(b),
                              "span": self.cfg.span})
