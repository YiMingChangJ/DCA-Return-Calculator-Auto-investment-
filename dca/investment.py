"""
Core DCA (dollar-cost averaging) investment simulation.

Pure functions only -- no plotting or printing. The Streamlit dashboard and
the CLI script both consume this module.

Conventions
-----------
- ``annual_return`` is a decimal (0.08 == 8%).
- ``compounding="effective"`` (default): per-period rate is
  ``(1+r)**(1/m) - 1``. The annual *effective* return matches the input.
- ``compounding="nominal"``: per-period rate is ``r/m`` (APR convention).
  In this mode the realized annual return is ``(1+r/m)**m - 1``, slightly
  above ``r``.
- ``timing="end"`` (default): contribution at end of each period (ordinary
  annuity). ``timing="begin"``: contribution at start of each period
  (annuity-due).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Compounding = Literal["nominal", "effective"]
Timing = Literal["end", "begin"]


@dataclass(frozen=True)
class SimulationParams:
    """Inputs for :func:`simulate`. All fields have safe defaults."""

    contribution: float = 1000.0
    years: int = 30
    times_per_year: int = 12
    annual_return: float = 0.08
    initial_capital: float = 0.0
    contribution_growth: float = 0.0
    inflation: float = 0.0
    compounding: Compounding = "effective"
    timing: Timing = "end"

    def __post_init__(self) -> None:
        if self.years < 1:
            raise ValueError("years must be >= 1")
        if self.times_per_year < 1:
            raise ValueError("times_per_year must be >= 1")
        if self.contribution < 0:
            raise ValueError("contribution must be >= 0")
        if self.initial_capital < 0:
            raise ValueError("initial_capital must be >= 0")
        if self.compounding not in ("nominal", "effective"):
            raise ValueError("compounding must be 'nominal' or 'effective'")
        if self.timing not in ("end", "begin"):
            raise ValueError("timing must be 'end' or 'begin'")


def _periodic_rate(annual_return: float, m: int, compounding: Compounding) -> float:
    if compounding == "nominal":
        return annual_return / m
    return (1.0 + annual_return) ** (1.0 / m) - 1.0


def simulate(params: SimulationParams) -> pd.DataFrame:
    """Run a deterministic DCA simulation.

    Returns a year-end DataFrame with columns:
        year, contributions_this_year, cumulative_principal, balance,
        earnings, return_on_principal
    plus inflation-adjusted columns when ``params.inflation > 0``.
    Year 0 is the starting state (only initial_capital invested).
    """
    m = params.times_per_year
    i = _periodic_rate(params.annual_return, m, params.compounding)

    balance = float(params.initial_capital)
    cumulative_principal = float(params.initial_capital)
    contribution = float(params.contribution)

    rows: list[dict[str, float]] = [
        {
            "year": 0,
            "contributions_this_year": 0.0,
            "cumulative_principal": cumulative_principal,
            "balance": balance,
        }
    ]

    for year in range(1, params.years + 1):
        contrib_this_year = 0.0
        for _ in range(m):
            if params.timing == "begin":
                balance += contribution
                balance *= 1.0 + i
            else:  # end of period
                balance *= 1.0 + i
                balance += contribution
            contrib_this_year += contribution
            cumulative_principal += contribution
        rows.append(
            {
                "year": year,
                "contributions_this_year": contrib_this_year,
                "cumulative_principal": cumulative_principal,
                "balance": balance,
            }
        )
        contribution *= 1.0 + params.contribution_growth

    df = pd.DataFrame(rows)
    df["earnings"] = df["balance"] - df["cumulative_principal"]
    df["return_on_principal"] = np.where(
        df["cumulative_principal"] > 0,
        df["earnings"] / df["cumulative_principal"],
        0.0,
    )

    if params.inflation > 0:
        deflator = (1.0 + params.inflation) ** df["year"].to_numpy()
        df["balance_real"] = df["balance"] / deflator
        df["cumulative_principal_real"] = df["cumulative_principal"] / deflator
        df["earnings_real"] = df["balance_real"] - df["cumulative_principal_real"]

    return df


def summary(df: pd.DataFrame, params: SimulationParams) -> dict[str, float]:
    """Compact summary metrics derived from a completed simulation."""
    last = df.iloc[-1]
    out: dict[str, float] = {
        "years": float(params.years),
        "annual_return": float(params.annual_return),
        "times_per_year": float(params.times_per_year),
        "per_period_contribution": float(params.contribution),
        "initial_capital": float(params.initial_capital),
        "total_principal": float(last["cumulative_principal"]),
        "terminal_balance": float(last["balance"]),
        "terminal_earnings": float(last["earnings"]),
        "return_on_principal": float(last["return_on_principal"]),
    }
    if params.inflation > 0:
        out["inflation"] = float(params.inflation)
        out["terminal_balance_real"] = float(last["balance_real"])
        out["terminal_earnings_real"] = float(last["earnings_real"])
    return out


def lump_sum_future_value(
    total_principal: float,
    years: int,
    annual_return: float,
) -> float:
    """FV if the full principal were invested as a single lump sum at t=0.

    Useful as a benchmark line on the DCA growth chart.
    """
    return float(total_principal) * (1.0 + float(annual_return)) ** int(years)
