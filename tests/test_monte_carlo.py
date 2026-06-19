"""Tests for :mod:`dca.monte_carlo`."""
from __future__ import annotations

import numpy as np
import pytest

from dca.investment import SimulationParams, simulate
from dca.monte_carlo import monte_carlo


def test_monte_carlo_zero_sigma_matches_deterministic():
    params = SimulationParams(
        contribution=500,
        years=10,
        times_per_year=12,
        annual_return=0.07,
        compounding="effective",
        timing="end",
    )
    det = simulate(params)["balance"].to_numpy()
    mc = monte_carlo(params, sigma=0.0, n_paths=50, seed=0)
    # All paths identical and equal to deterministic
    np.testing.assert_allclose(mc.paths[0], det, rtol=1e-10)
    np.testing.assert_allclose(mc.paths.std(axis=0), 0.0, atol=1e-8)


def test_monte_carlo_seed_is_deterministic():
    params = SimulationParams(
        contribution=200, years=5, times_per_year=12, annual_return=0.06
    )
    a = monte_carlo(params, sigma=0.15, n_paths=50, seed=42)
    b = monte_carlo(params, sigma=0.15, n_paths=50, seed=42)
    np.testing.assert_array_equal(a.paths, b.paths)


def test_monte_carlo_percentile_ordering():
    params = SimulationParams(
        contribution=100, years=20, times_per_year=12, annual_return=0.07
    )
    mc = monte_carlo(params, sigma=0.18, n_paths=500, seed=1)
    # p10 <= p50 <= p90 at every year
    assert (mc.percentiles["p10"] <= mc.percentiles["p50"]).all()
    assert (mc.percentiles["p50"] <= mc.percentiles["p90"]).all()


def test_invalid_n_paths():
    params = SimulationParams(
        contribution=100, years=5, times_per_year=12, annual_return=0.05
    )
    with pytest.raises(ValueError):
        monte_carlo(params, sigma=0.1, n_paths=0)


def test_invalid_sigma():
    params = SimulationParams(
        contribution=100, years=5, times_per_year=12, annual_return=0.05
    )
    with pytest.raises(ValueError):
        monte_carlo(params, sigma=-0.1, n_paths=10)
