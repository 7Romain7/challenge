"""T1/T2 - fresh noise per measurement; hard budget; counters agree with Experiment."""

import numpy as np
import pytest

from csd import new_experiment
from optimization import BlindExperiment, BudgetExceeded


def test_measurements_have_independent_noise():
    exp = new_experiment(seed=0)
    bx = BlindExperiment(exp, pixel_cap=10**7)
    bx.reference_frame()
    kw = dict(span_h=0.1, span_v=0.1, step_h=bx.native_step, step_v=bx.native_step)
    a, b = bx.measure(**kw), bx.measure(**kw)
    assert not np.allclose(a, b)


def test_budget_is_hard_and_matches_experiment():
    exp = new_experiment(seed=1)
    bx = BlindExperiment(exp, pixel_cap=30_000)
    bx.reference_frame()  # 22 500 px
    kw = dict(span_h=0.3, span_v=0.3, step_h=bx.native_step, step_v=bx.native_step)
    with pytest.raises(BudgetExceeded):
        bx.measure(**kw)
    assert bx.n_pixels == exp.n_pixels == 22_500
    kw = dict(span_h=0.05, span_v=0.05, step_h=bx.native_step, step_v=bx.native_step)
    bx.measure(**kw)
    assert bx.n_pixels == exp.n_pixels
