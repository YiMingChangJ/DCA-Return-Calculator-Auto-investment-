"""Monte Carlo overlay for DCA simulations.

Each path samples annual returns from a Normal(mu, sigma) distribution and
applies them to year-end balances. Within each year, contributions are
distributed across ``times_per_year`` periods using the realised annual
return for that year (so timing matters).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dca.investment import SimulationParams, _periodic_rate


@dataclass(frozen=True)
class MonteCarloResult:
    """Container for Monte Carlo outputs.

    paths: ndarray of shape (n_paths, years + 1) with year-end balances.
    percentiles: DataFrame indexed by year with p10/p50/p90 columns.
    """

    paths: np.ndarray
    percentiles: pd.DataFrame


def monte_carlo(
    params: SimulationParams,
    *,
    mu: float | None = None,
    sigma: float = 0.15,
    n_paths: int = 1000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run a Monte Carlo simulation around the deterministic DCA model.

    Parameters
    ----------
    params : SimulationParams
        The base scenario. ``contribution_growth`` and ``initial_capital``
        are respected. ``inflation`` is *not* applied (caller can deflate
        the percentiles separately if desired).
    mu : float, optional
        Mean annual return. Defaults to ``params.annual_return``.
    sigma : float
        Standard deviation of annual returns.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int, optional
        Seed for the random generator.
    """
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    rng = np.random.default_rng(seed)
    mean_return = float(params.annual_return if mu is None else mu)
    annual_returns = rng.normal(mean_return, sigma, size=(n_paths, params.years))

    m = params.times_per_year
    paths = np.zeros((n_paths, params.years + 1), dtype=float)
    paths[:, 0] = params.initial_capital
    contribution = float(params.contribution)

    for year_idx in range(params.years):
        r_year = annual_returns[:, year_idx]
        # Per-period rate consistent with realised annual return for the path
        if params.compounding == "nominal":
            i = r_year / m
        else:
            # Effective: handle negative returns by clipping floor to -1
            base = np.maximum(1.0 + r_year, 1e-12)
            i = base ** (1.0 / m) - 1.0

        balance = paths[:, year_idx].copy()
        for _ in range(m):
            if params.timing == "begin":
                balance = (balance + contribution) * (1.0 + i)
            else:
                balance = balance * (1.0 + i) + contribution
        paths[:, year_idx + 1] = balance
        contribution *= 1.0 + params.contribution_growth

    pct = np.percentile(paths, [10, 50, 90], axis=0)
    percentiles = pd.DataFrame(
        {
            "year": np.arange(params.years + 1),
            "p10": pct[0],
            "p50": pct[1],
            "p90": pct[2],
        }
    )

    return MonteCarloResult(paths=paths, percentiles=percentiles)
