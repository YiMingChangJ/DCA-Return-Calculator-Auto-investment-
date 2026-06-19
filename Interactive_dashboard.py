"""Streamlit dashboard for the DCA simulator.

Launch with::

    streamlit run Interactive_dashboard.py

All math lives in the :mod:`dca` package; this file only handles UI.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st

# Make ``dca`` importable when Streamlit runs the file as ``__main__``.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dca.investment import SimulationParams, lump_sum_future_value, simulate, summary  # noqa: E402
from dca.monte_carlo import monte_carlo  # noqa: E402
from dca.plotting import plot_growth  # noqa: E402


st.set_page_config(page_title="DCA Investment Calculator", layout="wide")
st.title("Automatic Investment Strategy Calculator")
st.caption(
    "Dollar-cost-averaging simulator with optional inflation adjustment, "
    "contribution growth, lump-sum benchmark, and Monte Carlo overlay."
)

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Plan inputs")
    contribution = st.number_input(
        "Investment per period ($)", min_value=0.0, value=4000.0, step=100.0
    )
    years = st.number_input(
        "Investment duration (years)", min_value=1, value=35, step=1
    )
    times = st.number_input(
        "Investments per year",
        min_value=1,
        value=12,
        step=1,
        help="12 = monthly, 26 = bi-weekly, 52 = weekly, 1 = annual",
    )
    annual_return = (
        st.slider("Expected annual return (%)", min_value=0.0,
                  max_value=30.0, value=8.0, step=0.5) / 100.0
    )
    initial_capital = st.number_input(
        "Initial capital ($)", min_value=0.0, value=0.0, step=1000.0
    )
    contribution_growth = (
        st.slider("Annual contribution growth (%)", min_value=0.0,
                  max_value=15.0, value=0.0, step=0.5,
                  help="e.g. raise contributions 3%/yr with salary") / 100.0
    )
    inflation = (
        st.slider("Inflation (%) for real values", min_value=0.0,
                  max_value=10.0, value=0.0, step=0.1) / 100.0
    )

    with st.expander("Advanced"):
        compounding = st.radio(
            "Periodic-rate convention",
            options=["effective", "nominal"],
            index=0,
            help=(
                "Effective: per-period rate = (1+r)^(1/m)-1 so the realised "
                "annual return matches the input. Nominal: per-period rate = "
                "r/m (APR convention; realised annual return is slightly higher)."
            ),
        )
        timing = st.radio(
            "Contribution timing",
            options=["end", "begin"],
            index=0,
            help="end = ordinary annuity (deposit then grow); "
                 "begin = annuity-due (deposit at start of period).",
        )

    st.header("Monte Carlo")
    run_mc = st.checkbox("Overlay Monte Carlo paths", value=False)
    mc_sigma = st.slider(
        "Annual return std-dev (%)",
        min_value=0.0, max_value=40.0, value=15.0, step=1.0,
        disabled=not run_mc,
    ) / 100.0
    mc_paths = st.number_input(
        "Number of paths", min_value=100, max_value=10_000, value=1_000, step=100,
        disabled=not run_mc,
    )
    mc_seed = st.number_input(
        "Seed (for reproducibility)", min_value=0, value=42, step=1,
        disabled=not run_mc,
    )


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
params = SimulationParams(
    contribution=float(contribution),
    years=int(years),
    times_per_year=int(times),
    annual_return=float(annual_return),
    initial_capital=float(initial_capital),
    contribution_growth=float(contribution_growth),
    inflation=float(inflation),
    compounding=compounding,
    timing=timing,
)

df = simulate(params)
s = summary(df, params)
ls_fv = lump_sum_future_value(
    s["total_principal"], int(years), float(annual_return)
)

mc = None
if run_mc:
    mc = monte_carlo(
        params,
        sigma=float(mc_sigma),
        n_paths=int(mc_paths),
        seed=int(mc_seed),
    )

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total principal", f"${s['total_principal']:,.0f}")
col2.metric("Terminal balance", f"${s['terminal_balance']:,.0f}")
col3.metric("Terminal earnings", f"${s['terminal_earnings']:,.0f}")
col4.metric("Return on principal", f"{s['return_on_principal'] * 100:.1f}%")

if "terminal_balance_real" in s:
    rcol1, rcol2 = st.columns(2)
    rcol1.metric(
        f"Real terminal balance (today's $, {inflation * 100:.1f}% inflation)",
        f"${s['terminal_balance_real']:,.0f}",
    )
    rcol2.metric(
        "Real terminal earnings",
        f"${s['terminal_earnings_real']:,.0f}",
    )

st.info(
    f"Lump-sum benchmark: if the full principal of "
    f"${s['total_principal']:,.0f} were invested in one shot at t=0 and "
    f"grew at {annual_return * 100:.1f}%/yr, terminal value would be "
    f"**${ls_fv:,.0f}**."
)

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
fig = plot_growth(df, params, mc=mc)
st.pyplot(fig)

# ---------------------------------------------------------------------------
# Year-by-year table + download
# ---------------------------------------------------------------------------
st.subheader("Year-by-year detail")
display_df = df.copy()
display_df = display_df.rename(
    columns={
        "year": "Year",
        "contributions_this_year": "Contributions",
        "cumulative_principal": "Cumulative principal",
        "balance": "Balance",
        "earnings": "Earnings",
        "return_on_principal": "Return on principal",
    }
)
fmt: dict[str, str] = {
    "Contributions": "${:,.0f}",
    "Cumulative principal": "${:,.0f}",
    "Balance": "${:,.0f}",
    "Earnings": "${:,.0f}",
    "Return on principal": "{:.1%}",
}
if "balance_real" in display_df.columns:
    display_df = display_df.rename(
        columns={
            "balance_real": "Balance (real)",
            "cumulative_principal_real": "Principal (real)",
            "earnings_real": "Earnings (real)",
        }
    )
    fmt.update(
        {
            "Balance (real)": "${:,.0f}",
            "Principal (real)": "${:,.0f}",
            "Earnings (real)": "${:,.0f}",
        }
    )
st.dataframe(display_df.style.format(fmt), use_container_width=True)  # type: ignore[arg-type]

csv_buf = io.StringIO()
df.to_csv(csv_buf, index=False)
st.download_button(
    "Download year-by-year CSV",
    data=csv_buf.getvalue(),
    file_name="dca_year_by_year.csv",
    mime="text/csv",
)
