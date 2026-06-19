"""Matplotlib helpers for the DCA simulator.

Kept separate from the core math so that headless / test environments do not
pull in matplotlib.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dca.investment import SimulationParams
from dca.monte_carlo import MonteCarloResult


def plot_growth(
    df: pd.DataFrame,
    params: SimulationParams,
    *,
    show_lump_sum: bool = True,
    show_principal: bool = True,
    mc: Optional[MonteCarloResult] = None,
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot the DCA growth trajectory with optional overlays."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        parent = ax.figure
        if not isinstance(parent, Figure):
            raise TypeError("ax must belong to a top-level Figure, not a SubFigure")
        fig = parent

    years = df["year"].to_numpy()
    balance_m = df["balance"].to_numpy() / 1e6

    ax.plot(
        years,
        balance_m,
        marker="o",
        markersize=4,
        markerfacecolor="r",
        markeredgecolor="k",
        linewidth=2,
        label="DCA portfolio",
    )

    if show_principal:
        ax.plot(
            years,
            df["cumulative_principal"].to_numpy() / 1e6,
            linestyle="--",
            color="gray",
            linewidth=1.5,
            label="Cumulative principal",
        )

    if show_lump_sum:
        total_principal = float(df["cumulative_principal"].iloc[-1])
        # Lump-sum benchmark: invest the *full* eventual principal at t=0 and
        # let it compound at the annual return.
        ls_curve = (
            total_principal * (1.0 + params.annual_return) ** years / 1e6
        )
        ax.plot(
            years,
            ls_curve,
            linestyle=":",
            color="tab:blue",
            linewidth=1.5,
            label="Lump-sum benchmark",
        )

    if mc is not None:
        p = mc.percentiles
        ax.fill_between(
            p["year"].to_numpy(),
            p["p10"].to_numpy() / 1e6,
            p["p90"].to_numpy() / 1e6,
            alpha=0.2,
            color="tab:orange",
            label="MC 10-90%",
        )
        ax.plot(
            p["year"].to_numpy(),
            p["p50"].to_numpy() / 1e6,
            linestyle="-",
            color="tab:orange",
            linewidth=1.2,
            label="MC median",
        )

    ax.set_xlabel("Years")
    ax.set_ylabel("Portfolio value ($M)")
    ax.set_title(
        f"DCA: ${params.contribution:,.0f} x {params.times_per_year}/yr "
        f"for {params.years}y @ {params.annual_return * 100:.1f}%/yr"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig
