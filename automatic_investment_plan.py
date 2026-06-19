"""Command-line DCA simulator.

Run from this folder:

    python automatic_investment_plan.py --contribution 4000 --years 35 \
        --times 12 --return 0.12 --plot

Use ``--help`` for the full list of options. For an interactive UI, see
``Interactive_dashboard.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the project folder without installing.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dca.investment import (  # noqa: E402  (after sys.path tweak)
    SimulationParams,
    lump_sum_future_value,
    simulate,
    summary,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate a dollar-cost-averaging investment plan."
    )
    p.add_argument("--contribution", type=float, default=4000.0,
                   help="Amount invested each period (default: 4000)")
    p.add_argument("--years", type=int, default=35,
                   help="Investment horizon in years (default: 35)")
    p.add_argument("--times", type=int, default=12,
                   help="Periods per year, e.g. 12 for monthly (default: 12)")
    p.add_argument("--return", dest="annual_return", type=float, default=0.08,
                   help="Annual return as a decimal, e.g. 0.08 for 8%% (default: 0.08)")
    p.add_argument("--initial", type=float, default=0.0,
                   help="Initial capital at t=0 (default: 0)")
    p.add_argument("--growth", type=float, default=0.0,
                   help="Annual contribution growth rate, e.g. 0.03 (default: 0)")
    p.add_argument("--inflation", type=float, default=0.0,
                   help="Annual inflation, e.g. 0.025 (default: 0)")
    p.add_argument("--compounding", choices=["effective", "nominal"],
                   default="effective",
                   help="Periodic-rate convention (default: effective)")
    p.add_argument("--timing", choices=["end", "begin"], default="end",
                   help="Contribution timing within each period (default: end)")
    p.add_argument("--plot", action="store_true",
                   help="Show a matplotlib growth chart")
    p.add_argument("--save-plot", type=str, default=None,
                   help="Path to save the plot (e.g. growth.png). Implies --plot.")
    p.add_argument("--csv", type=str, default=None,
                   help="Write the year-by-year table to this CSV path")
    return p.parse_args(argv)


def _print_summary(s: dict[str, float]) -> None:
    print(f"Investment horizon       : {int(s['years'])} years")
    print(f"Periods per year         : {int(s['times_per_year'])}")
    print(f"Per-period contribution  : ${s['per_period_contribution']:,.2f}")
    print(f"Initial capital          : ${s['initial_capital']:,.2f}")
    print(f"Annual return            : {s['annual_return'] * 100:.2f}%")
    print(f"Total principal invested : ${s['total_principal']:,.0f}")
    print(f"Terminal balance         : ${s['terminal_balance']:,.0f}")
    print(f"Terminal earnings        : ${s['terminal_earnings']:,.0f}")
    print(f"Return on principal      : {s['return_on_principal'] * 100:.1f}%")
    if "terminal_balance_real" in s:
        print(
            f"Terminal balance (real)  : ${s['terminal_balance_real']:,.0f}"
            f"   (deflated at {s['inflation'] * 100:.2f}%/yr)"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    params = SimulationParams(
        contribution=args.contribution,
        years=args.years,
        times_per_year=args.times,
        annual_return=args.annual_return,
        initial_capital=args.initial,
        contribution_growth=args.growth,
        inflation=args.inflation,
        compounding=args.compounding,
        timing=args.timing,
    )
    df = simulate(params)
    s = summary(df, params)
    _print_summary(s)

    ls = lump_sum_future_value(
        s["total_principal"], int(args.years), args.annual_return
    )
    print(
        f"Lump-sum benchmark FV    : ${ls:,.0f}"
        " (if all principal were invested at t=0)"
    )

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Year-by-year table -> {args.csv}")

    if args.plot or args.save_plot:
        # Import only when needed so headless runs do not require matplotlib.
        from dca.plotting import plot_growth

        fig = plot_growth(df, params)
        if args.save_plot:
            fig.savefig(args.save_plot, dpi=200, bbox_inches="tight")
            print(f"Plot saved -> {args.save_plot}")
        if args.plot:
            import matplotlib.pyplot as plt

            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
