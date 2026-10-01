"""Challenge 2 - algorithm side.

FIREWALL: nothing in this package may import ``csd`` or touch ``exp._sim`` /
``exp._optimum`` / ``exp.reveal()``. Algorithms only see a :class:`BlindExperiment`
(measure / scan_1d / start / extent + budget counters). Enforced by
``tests/test_firewall.py``.
"""
from .blind import BlindExperiment, BudgetExceeded

__all__ = ["BlindExperiment", "BudgetExceeded"]
