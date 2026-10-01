from .challenge import Experiment, new_experiment
from .config import CHALLENGE, GENERATOR, ChallengeConfig, GeneratorConfig
from .dataset import DEFAULT_SCAN_WINDOW, generate_dataset, load_dataset
from .generator import (
    ScanWindow,
    build_interdots,
    generate_csd_and_label,
    render_csd,
)
from .simulator import (
    BARRIERS,
    GATES,
    PLUNGERS,
    CSDSimulator,
    DriftMatrix,
    Region,
    RegionContrastModel,
    default_working_point,
)

__all__ = [
    # challenge (hackathon-facing)
    "new_experiment",
    "Experiment",
    # fixed configuration (organiser-set)
    "ChallengeConfig",
    "CHALLENGE",
    "GeneratorConfig",
    "GENERATOR",
    # simulator
    "CSDSimulator",
    "DriftMatrix",
    "RegionContrastModel",
    "Region",
    "default_working_point",
    "GATES",
    "PLUNGERS",
    "BARRIERS",
    # dataset (challenge 1 data)
    "generate_dataset",
    "load_dataset",
    "DEFAULT_SCAN_WINDOW",
    # generator
    "ScanWindow",
    "build_interdots",
    "render_csd",
    "generate_csd_and_label",
]
