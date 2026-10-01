"""Methods 2-5 (GPU, need ``uv sync --extra train``). Interfaces fixed, bodies to implement.

Each class follows :class:`Method`: ``search(session)`` calls ``session.evaluate`` and keeps
``session.set_reco`` current. They are NOT in ``REGISTRY`` until implemented and covered by
the protocol tests (PROTOCOL_challenge2.md section 5). torch is imported lazily so that
the CPU-only parts of the repo keep working without it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import Method


def _need_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("install the GPU extra: uv sync --extra train") from e


@dataclass
class PFNConfig:
    """Method 2 - amortised BO. Transformer pre-trained on training.priors landscapes."""
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_obs: int = 100  # context length (observations per episode)
    n_pretrain_landscapes: int = 1_000_000  # 1k..50k for the data-scaling study
    lr: float = 3e-4


class PFNOptimizer(Method):
    """Input: tokens (b_j, y_j, se_j) of the episode so far. Output: posterior over y(b)
    at candidate points (discretised Riemann head); next point = argmax of EI under it.
    Candidates come from ``session.sample_feasible`` (same trust region as BO)."""
    name = "pfn"

    def __init__(self, *a, cfg: PFNConfig | None = None, ckpt: str | None = None, **kw) -> None:
        super().__init__(*a, **kw)
        _need_torch()
        self.cfg, self.ckpt = cfg or PFNConfig(), ckpt
        raise NotImplementedError("TODO: model + pretraining loop (training/pretrain_pfn.py)")


@dataclass
class RLConfig:
    """Method 3 - recurrent PPO policy trained on an abstract environment."""
    hidden: int = 128
    n_envs: int = 512
    horizon: int = 40
    pixel_penalty: float = 1e-6  # reward = verified score - lambda * pixels


class RLPolicy(Method):
    """Observation: history of (b_j, y_j, se_j) through a GRU. Actions: delta-barriers
    (3, bounded by the trust region), commit. Reward from OBSERVABLES only (verified y)."""
    name = "rl"

    def __init__(self, *a, cfg: RLConfig | None = None, ckpt: str | None = None, **kw) -> None:
        super().__init__(*a, **kw)
        _need_torch()
        raise NotImplementedError("TODO: abstract env (training.priors) + PPO")


@dataclass
class JEPAConfig:
    """Methods 4-5. ViT-tiny/small, patch 6-8 px, raw median-of-row-subtracted frames."""
    patch: int = 6
    d_model: int = 192
    depth: int = 6
    ema: float = 0.996
    mask_ratio: float = 0.6
    frames: int = 50_000  # data-scaling study: 1k, 5k, 10k, 20k, 50k (nested)


class JEPAEncoderBO(Method):
    """Method 4. Frozen JEPA embedding as an extra kernel input / state of the BO.
    Gate: the linear probe on y must beat the hand-crafted feature (PROTOCOL 5.4)."""
    name = "jepa_enc"

    def __init__(self, *a, cfg: JEPAConfig | None = None, ckpt: str | None = None, **kw) -> None:
        super().__init__(*a, **kw)
        _need_torch()
        raise NotImplementedError("TODO: training/pretrain_jepa.py + probe")


class JEPAWorldModel(Method):
    """Method 5. History-conditioned predictor z_{t+1} = f(z_<=t, a_<=t) + CEM planning in
    latent space, then local refinement with real measurements."""
    name = "jepa_world"

    def __init__(self, *a, cfg: JEPAConfig | None = None, ckpt: str | None = None, **kw) -> None:
        super().__init__(*a, **kw)
        _need_torch()
        raise NotImplementedError("TODO: causal transformer predictor + CEM")


PLANNED = {"pfn": PFNOptimizer, "rl": RLPolicy, "jepa_enc": JEPAEncoderBO,
           "jepa_world": JEPAWorldModel}
