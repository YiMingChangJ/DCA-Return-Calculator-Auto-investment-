"""Tests for :mod:`dca.investment`.

Math is anchored on closed-form ordinary-annuity future-value formulas:

    FV = P * ((1+i)**n - 1) / i      (ordinary annuity, end of period)
    FV = P * ((1+i)**n - 1) / i * (1+i)   (annuity-due, begin of period)

with i = periodic rate, n = total number of periods.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from dca.investment import (
    SimulationParams,
    lump_sum_future_value,
    simulate,
    summary,
)


def _annuity_fv(P: float, i: float, n: int) -> float:
    if i == 0:
        return P * n
    return P * ((1 + i) ** n - 1) / i


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_years_raises():
    with pytest.raises(ValueError):
        SimulationParams(contribution=100, years=0, times_per_year=12, annual_return=0.05)


def test_invalid_frequency_raises():
    with pytest.raises(ValueError):
        SimulationParams(contribution=100, years=5, times_per_year=0, annual_return=0.05)


def test_invalid_compounding_raises():
    with pytest.raises(ValueError):
        SimulationParams(
            contribution=100,
            years=5,
            times_per_year=12,
            annual_return=0.05,
            compounding="weird",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Zero-return sanity checks
# ---------------------------------------------------------------------------


def test_zero_return_balance_equals_principal_monthly():
    params = SimulationParams(
        contribution=100,
        years=10,
        times_per_year=12,
        annual_return=0.0,
        timing="end",
    )
    df = simulate(params)
    assert df["balance"].iloc[-1] == pytest.approx(100 * 12 * 10)
    assert df["cumulative_principal"].iloc[-1] == pytest.approx(100 * 12 * 10)
    assert df["earnings"].iloc[-1] == pytest.approx(0.0)


def test_zero_return_annual_frequency():
    params = SimulationParams(
        contribution=1, years=10, times_per_year=1, annual_return=0.0
    )
    df = simulate(params)
    assert df["balance"].iloc[-1] == pytest.approx(10)
    # year 0 row is included
    assert len(df) == 11


# ---------------------------------------------------------------------------
# Closed-form annuity matching
# ---------------------------------------------------------------------------


def test_effective_annual_matches_closed_form():
    """With effective compounding, the annual realized return *is* r.

    So treating each year as a single annuity FV with periodic rate
    i = (1+r)^(1/m) - 1 should match the simulator exactly.
    """
    P, m, T, r = 1000.0, 12, 5, 0.06
    i = (1 + r) ** (1 / m) - 1
    n = m * T
    expected = _annuity_fv(P, i, n)

    params = SimulationParams(
        contribution=P,
        years=T,
        times_per_year=m,
        annual_return=r,
        compounding="effective",
        timing="end",
    )
    df = simulate(params)
    assert df["balance"].iloc[-1] == pytest.approx(expected, rel=1e-10)


def test_nominal_compounding_matches_closed_form():
    P, m, T, r = 500.0, 12, 20, 0.08
    i = r / m
    n = m * T
    expected = _annuity_fv(P, i, n)

    params = SimulationParams(
        contribution=P,
        years=T,
        times_per_year=m,
        annual_return=r,
        compounding="nominal",
        timing="end",
    )
    df = simulate(params)
    assert df["balance"].iloc[-1] == pytest.approx(expected, rel=1e-10)


def test_annuity_due_is_one_period_more_growth():
    """Annuity-due == ordinary annuity * (1+i)."""
    P, m, T, r = 200.0, 4, 10, 0.05
    i = (1 + r) ** (1 / m) - 1

    end = simulate(
        SimulationParams(
            contribution=P,
            years=T,
            times_per_year=m,
            annual_return=r,
            compounding="effective",
            timing="end",
        )
    )["balance"].iloc[-1]

    begin = simulate(
        SimulationParams(
            contribution=P,
            years=T,
            times_per_year=m,
            annual_return=r,
            compounding="effective",
            timing="begin",
        )
    )["balance"].iloc[-1]

    assert begin == pytest.approx(end * (1 + i), rel=1e-10)


# ---------------------------------------------------------------------------
# Initial capital
# ---------------------------------------------------------------------------


def test_initial_capital_compounds():
    """Initial capital alone (zero contribution) must grow as init*(1+r)^T."""
    init, T, r = 10_000.0, 7, 0.10
    params = SimulationParams(
        contribution=0,
        years=T,
        times_per_year=12,
        annual_return=r,
        initial_capital=init,
        compounding="effective",
    )
    df = simulate(params)
    assert df["balance"].iloc[-1] == pytest.approx(init * (1 + r) ** T, rel=1e-10)
    assert df["cumulative_principal"].iloc[-1] == pytest.approx(init)


# ---------------------------------------------------------------------------
# Inflation
# ---------------------------------------------------------------------------


def test_inflation_adjustment_columns_present():
    params = SimulationParams(
        contribution=100,
        years=5,
        times_per_year=12,
        annual_return=0.05,
        inflation=0.03,
    )
    df = simulate(params)
    assert "balance_real" in df.columns
    last = df.iloc[-1]
    assert last["balance_real"] == pytest.approx(last["balance"] / (1.03 ** 5))


# ---------------------------------------------------------------------------
# Contribution growth
# ---------------------------------------------------------------------------


def test_contribution_growth_increases_principal():
    base = SimulationParams(
        contribution=100, years=10, times_per_year=12, annual_return=0.05
    )
    grow = SimulationParams(
        contribution=100,
        years=10,
        times_per_year=12,
        annual_return=0.05,
        contribution_growth=0.03,
    )
    p_base = simulate(base)["cumulative_principal"].iloc[-1]
    p_grow = simulate(grow)["cumulative_principal"].iloc[-1]
    assert p_grow > p_base


def test_zero_contribution_growth_matches_constant():
    a = simulate(
        SimulationParams(
            contribution=100,
            years=10,
            times_per_year=12,
            annual_return=0.05,
            contribution_growth=0.0,
        )
    )
    b = simulate(
        SimulationParams(
            contribution=100, years=10, times_per_year=12, annual_return=0.05
        )
    )
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_lump_sum_future_value():
    assert lump_sum_future_value(1000, 10, 0.07) == pytest.approx(1000 * 1.07 ** 10)


def test_summary_keys_no_inflation():
    params = SimulationParams(
        contribution=100, years=3, times_per_year=12, annual_return=0.05
    )
    s = summary(simulate(params), params)
    assert {
        "years",
        "annual_return",
        "times_per_year",
        "per_period_contribution",
        "initial_capital",
        "total_principal",
        "terminal_balance",
        "terminal_earnings",
        "return_on_principal",
    } <= s.keys()
    assert "terminal_balance_real" not in s


def test_summary_keys_with_inflation():
    params = SimulationParams(
        contribution=100,
        years=3,
        times_per_year=12,
        annual_return=0.05,
        inflation=0.02,
    )
    s = summary(simulate(params), params)
    assert "terminal_balance_real" in s
