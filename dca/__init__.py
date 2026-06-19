"""DCA (dollar-cost averaging) investment simulation toolkit."""

from dca.investment import (
    SimulationParams,
    simulate,
    summary,
    lump_sum_future_value,
)
from dca.monte_carlo import monte_carlo

__all__ = [
    "SimulationParams",
    "simulate",
    "summary",
    "lump_sum_future_value",
    "monte_carlo",
]
