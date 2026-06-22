"""Matplotlib helpers for the DCA simulator.

Kept separate from the core math so that headless / test environments do not
pull in matplotlib.

Style notes
-----------
Uses matplotlib's built-in *mathtext* (not full LaTeX) with a serif font
family, so labels render in a LaTeX-like style without requiring a TeX
install on the host machine. Set ``use_tex=True`` in :func:`plot_growth`
to switch to real LaTeX rendering when ``usetex`` is available locally.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dca.investment import SimulationParams
from dca.monte_carlo import MonteCarloResult


def _apply_style(use_tex: bool = False, base_size: int = 14) -> None:
    """Apply a serif / mathtext style consistent with the original notebook plots."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",  # Computer Modern -- LaTeX look
            "axes.titlesize": base_size + 2,
            "axes.labelsize": base_size + 2,
            "xtick.labelsize": base_size,
            "ytick.labelsize": base_size,
            "legend.fontsize": base_size - 2,
            "lines.linewidth": 2.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "text.usetex": bool(use_tex),
        }
    )


def plot_growth(
    df: pd.DataFrame,
    params: SimulationParams,
    *,
    show_lump_sum: bool = True,
    show_principal: bool = True,
    mc: Optional[MonteCarloResult] = None,
    ax: Optional[Axes] = None,
    use_tex: bool = False,
    annotate: bool = True,
) -> Figure:
    """Plot the DCA growth trajectory with optional overlays.

    Parameters
    ----------
    use_tex : bool
        If True, use a full LaTeX install via ``text.usetex``. Requires a
        working TeX distribution on the host. Defaults to False (mathtext).
    annotate : bool
        If True, draw the original-style inline annotations
        (``amount``, ``times``, ``r``) in the upper-left corner.
    """
    _apply_style(use_tex=use_tex)

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
        linestyle="-",
        marker="o",
        markersize=4,
        markerfacecolor="r",
        markeredgecolor="k",
        linewidth=2.5,
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
            label=r"MC $10\!-\!90\%$",
        )
        ax.plot(
            p["year"].to_numpy(),
            p["p50"].to_numpy() / 1e6,
            linestyle="-",
            color="tab:orange",
            linewidth=1.2,
            label="MC median",
        )

    ax.set_xlabel(r"Years")
    ax.set_ylabel(r"Principal and Earnings (\$M)")

    if annotate:
        # Original notebook-style inline annotations in upper-left of axes.
        txt_lines = [
            rf"amount $= \${params.contribution:,.0f}$",
            rf"times $= {params.times_per_year}$",
            rf"$r = {params.annual_return * 100:.1f}\,\%$",
        ]
        if params.contribution_growth:
            txt_lines.append(rf"$g = {params.contribution_growth * 100:.1f}\,\%$")
        if params.inflation:
            txt_lines.append(rf"$\pi = {params.inflation * 100:.1f}\,\%$")
        ax.text(
            0.03,
            0.97,
            "\n".join(txt_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=4),
        )

    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.78), frameon=False)
    fig.tight_layout()
    return fig

