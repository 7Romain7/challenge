"""T4 - the tracker follows the interdots through barrier drift (checked against truth)."""

import numpy as np

from csd import new_experiment
from optimization import BlindExperiment
from optimization.session import Session


def test_tracking_accuracy_on_dev_devices():
    errs, fails, n = [], 0, 0
    for seed in range(3):
        exp = new_experiment(seed=seed)
        s = Session(BlindExperiment(exp, 10**7))
        s.start()
        rng = np.random.default_rng(seed)
        for _ in range(8):
            b = s.sample_feasible(rng)[0]
            truth = exp._sim.drift_matrix.drift(b - s.b0)
            ok = s.evaluate(b)["ok"]
            n += 1
            if ok:
                errs.append(np.linalg.norm(s.drift.delta[-1] - truth))
            else:
                fails += 1
    assert fails / n < 0.15
    assert np.median(errs) < 4e-3  # registration error below 4 mV
