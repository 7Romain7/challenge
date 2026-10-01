import numpy as np

from training.priors import sample_landscape


def test_landscapes_are_nonnegative_bounded_and_noisy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        L = sample_landscape(rng)
        B = rng.uniform(-0.6, 0.6, (50, 3))
        f = L.f(B)
        assert np.all(f >= 0) and np.all(f <= L.amps.max() + 1e-9)
        y, se = L.observe(B, rng)
        assert y.shape == se.shape == (50,) and np.all(se > 0)
