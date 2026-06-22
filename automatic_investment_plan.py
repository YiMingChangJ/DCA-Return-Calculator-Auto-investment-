"""Command-line DCA simulator.

Run from this folder:

    python automatic_investment_plan.py --contribution 4000 --years 35 \
        --times 12 --return 0.12 --plot --save-plot

    python automatic_investment_plan.py --contribution 4000 --years 35 --times 12 --return 0.12 --growth 0.03 --inflation 0.025 --mc --mc-sigma 0.15 \
        --mc-paths 2000 --save-plot docs/dca_growth_mc.png  

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
                   help="Show a matplotlib growth chart on screen")
    p.add_argument("--save-plot", type=str, default=None,
                   help="Path to save the plot (e.g. docs/dca_growth.png). "
                        "If neither --plot nor --save-plot is given, the chart "
                        "is saved to docs/dca_growth.png by default.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plotting entirely (overrides the default save).")
    p.add_argument("--mc", action="store_true",
                   help="Overlay a Monte Carlo fan on the growth chart")
    p.add_argument("--mc-sigma", type=float, default=0.15,
                   help="Annual return std-dev for Monte Carlo (default: 0.15)")
    p.add_argument("--mc-paths", type=int, default=1000,
                   help="Number of Monte Carlo paths (default: 1000)")
    p.add_argument("--mc-seed", type=int, default=42,
                   help="RNG seed for Monte Carlo reproducibility (default: 42)")
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
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Year-by-year table -> {csv_path}")

    # Decide what to do about the plot.
    save_path: str | None = args.save_plot
    if not args.no_plot and not args.plot and save_path is None:
        # Default behaviour: save a figure into docs/ so it can be referenced
        # from the README without the user remembering --save-plot.
        default_dir = _THIS_DIR / "docs"
        default_dir.mkdir(exist_ok=True)
        save_path = str(default_dir / "dca_growth.png")

    if not args.no_plot and (args.plot or save_path is not None):
        # Import only when needed so headless runs do not require matplotlib.
        from dca.plotting import plot_growth

        mc_result = None
        if args.mc:
            from dca.monte_carlo import monte_carlo

            mc_result = monte_carlo(
                params,
                sigma=args.mc_sigma,
                n_paths=args.mc_paths,
                seed=args.mc_seed,
            )

        fig = plot_growth(df, params, mc=mc_result)
        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"Plot saved -> {out}")
        if args.plot:
            import matplotlib.pyplot as plt

            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
