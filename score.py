"""
score.py
────────────────────────────────────────────────────────────────────────────
Investment scoring CLI — run after pipeline.py has populated data/processed/.

Usage
─────
  # Default: balanced 50/50, both countries, top 10
  python score.py

  # Yield-focused investor (80% income, 20% growth)
  python score.py --objective yield --yield-weight 0.8 --growth-weight 0.2

  # Growth-focused investor, Norway only, minimum 4% yield
  python score.py --objective growth --country NO --min-yield 4.0

  # Show Pareto frontier (non-dominated options on yield + growth simultaneously)
  python score.py --pareto

  # Run all three scenarios (income / balanced / growth) and compare
  python score.py --scenarios

  # Export results to CSV
  python score.py --export scores_output.csv

  # Use mock data (no live API needed)
  python score.py --mock
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Rich console output ────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    from rich.panel import Panel
    from rich.text import Text
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None


def _print_table(df: pd.DataFrame, title: str, cols: list, headers: list,
                 highlight_col: str = None):
    """Print a formatted table — rich if available, plain otherwise."""
    if RICH:
        table = Table(title=title, show_header=True, header_style="bold cyan",
                      border_style="dim", show_lines=False)
        for h in headers:
            table.add_column(h, no_wrap=True)

        for _, row in df[cols].iterrows():
            values = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    values.append(f"{v:.1f}" if abs(v) < 1000 else f"{v:,.0f}")
                else:
                    values.append(str(v))

            # Colour code recommendations
            if "recommendation" in cols:
                rec = row.get("recommendation", "")
                style = "bold green" if rec == "BUY" else (
                        "yellow" if rec == "HOLD" else "dim red")
                table.add_row(*values, style=style)
            else:
                table.add_row(*values)

        console.print(table)
    else:
        print(f"\n{'─'*80}")
        print(f"  {title}")
        print(f"{'─'*80}")
        header = "  " + "  ".join(f"{h:<18}" for h in headers)
        print(header)
        print("  " + "─" * (len(header) - 2))
        for _, row in df[cols].iterrows():
            line = "  " + "  ".join(
                f"{str(row[c]):<18}" if not isinstance(row[c], float)
                else f"{row[c]:<18.1f}" for c in cols
            )
            print(line)


def _fmt_score(v) -> str:
    if pd.isna(v):
        return "N/A"
    return f"{v:.1f}"


def _fmt_pct(v) -> str:
    if pd.isna(v):
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def _fmt_currency(v, currency="") -> str:
    if pd.isna(v):
        return "N/A"
    return f"{v:,.0f} {currency}".strip()


# ── Main display functions ─────────────────────────────────────────────────────

def display_recommendations(recs: pd.DataFrame, objective: str, top_n: int):
    if recs.empty:
        print("No results — check that pipeline.py has been run first.")
        return

    buys  = recs[recs["recommendation"] == "BUY"].head(top_n)
    avoid = recs[recs["recommendation"] == "AVOID"].head(top_n // 2)

    title_map = {
        "yield":    "Income / Rental Yield",
        "growth":   "Capital Growth",
        "balanced": "Balanced (Yield + Growth)",
    }

    if RICH:
        console.print(Panel(
            f"[bold white]Objective: {title_map.get(objective, objective.title())}[/bold white]\n"
            f"[dim]Scores normalised within each country (0=worst, 100=best)[/dim]",
            style="blue"
        ))

    # BUY recommendations
    _print_table(
        buys,
        title=f"✅  TOP {len(buys)} BUY — {title_map.get(objective, objective)}",
        cols=[
            "recommendation", "country", "region", "dwelling_type",
            "composite_score", "yield_score", "growth_score",
            "gross_yield_pct", "growth_1Y_ann", "risk_score",
        ],
        headers=[
            "Signal", "Country", "Region", "Type",
            "Score", "Yield⬆", "Growth⬆",
            "Gross yield", "1Y growth", "Risk",
        ],
    )

    # Detail table for BUYs
    detail_cols = ["country", "region", "dwelling_type",
                   "avg_price_sqm", "avg_monthly_rent",
                   "permits_per_1000_stock", "volatility_qoq"]
    detail_cols = [c for c in detail_cols if c in buys.columns]
    if detail_cols:
        _print_table(
            buys,
            title="📊  BUY — Market Detail",
            cols=detail_cols,
            headers=["Country", "Region", "Type",
                     "Price/sqm", "Avg rent", "Permits/1k", "Volatility"],
        )

    # AVOID
    if len(avoid):
        _print_table(
            avoid,
            title="⛔  AVOID (bottom of ranking)",
            cols=[
                "country", "region", "dwelling_type",
                "composite_score", "yield_score", "growth_score",
                "gross_yield_pct", "growth_1Y_ann", "risk_score",
            ],
            headers=[
                "Country", "Region", "Type",
                "Score", "Yield⬆", "Growth⬆",
                "Gross yield", "1Y growth", "Risk",
            ],
        )


def display_pareto(frontier: pd.DataFrame):
    if frontier.empty:
        print("Pareto frontier empty.")
        return

    cols = [
        "country", "region", "dwelling_type",
        "yield_score", "growth_score", "composite_score",
        "gross_yield_pct", "growth_1Y_ann",
    ]
    cols = [c for c in cols if c in frontier.columns]

    _print_table(
        frontier,
        title="🔷  Pareto Frontier — Non-dominated options (no alternative beats on BOTH yield AND growth)",
        cols=cols,
        headers=["Country", "Region", "Type",
                 "Yield score", "Growth score", "Composite",
                 "Gross yield %", "1Y growth %"],
    )


def display_scenarios(scenario_df: pd.DataFrame):
    if scenario_df.empty:
        print("Scenario analysis empty.")
        return

    for scenario, grp in scenario_df.groupby("scenario"):
        buys = grp[grp["recommendation"] == "BUY"]
        yw   = grp["scenario_yield_weight"].iloc[0]
        gw   = grp["scenario_growth_weight"].iloc[0]

        _print_table(
            buys,
            title=f"📋  Scenario: {scenario}  (yield_weight={yw}, growth_weight={gw})",
            cols=["country", "region", "dwelling_type",
                  "composite_score", "gross_yield_pct", "growth_1Y_ann"],
            headers=["Country", "Region", "Type",
                     "Score", "Gross yield %", "1Y growth %"],
        )


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Nordic Real Estate — Investment Scorer")
    parser.add_argument("--objective",    choices=["yield", "growth", "balanced"],
                        default="balanced")
    parser.add_argument("--yield-weight",  type=float, default=0.5,
                        help="Weight on rental yield objective [0-1]")
    parser.add_argument("--growth-weight", type=float, default=0.5,
                        help="Weight on price growth objective [0-1]")
    parser.add_argument("--risk-penalty",  type=float, default=0.25,
                        help="How much to penalise risk [0-1]")
    parser.add_argument("--country",      choices=["NO", "SE"], default=None)
    parser.add_argument("--top-n",        type=int, default=10)
    parser.add_argument("--min-yield",    type=float, default=None,
                        help="Minimum gross yield %% to include")
    parser.add_argument("--min-growth",   type=float, default=None,
                        help="Minimum 1Y growth %% to include")
    parser.add_argument("--pareto",       action="store_true",
                        help="Show Pareto frontier instead of ranked list")
    parser.add_argument("--scenarios",    action="store_true",
                        help="Run income / balanced / growth scenario comparison")
    parser.add_argument("--export",       type=str, default=None,
                        help="Export full scores to CSV file")
    parser.add_argument("--mock",         action="store_true",
                        help="Run on mock data (no live API needed)")
    args = parser.parse_args()

    # Validate weights
    if abs(args.yield_weight + args.growth_weight - 1.0) > 0.01:
        print(f"ERROR: yield-weight ({args.yield_weight}) + growth-weight "
              f"({args.growth_weight}) must sum to 1.0")
        sys.exit(1)

    # Optionally seed with mock data
    if args.mock:
        logger.info("Loading mock data into processed/...")
        from tests.mock_data import get_mock_raw
        from pipeline import run_analysis
        raw = get_mock_raw()
        run_analysis(raw)

    from analysis.investment_model import InvestmentModel

    model = InvestmentModel(
        yield_weight  = args.yield_weight,
        growth_weight = args.growth_weight,
        risk_penalty  = args.risk_penalty,
    )

    logger.info("[score] Building feature table and scoring...")
    scores = model.score()

    if scores.empty:
        print("No scores generated. Run pipeline.py first (or use --mock).")
        sys.exit(1)

    if args.export:
        path = Path(args.export)
        scores.to_csv(path, index=False)
        logger.info(f"Full scores exported to {path}")

    if args.scenarios:
        logger.info("[score] Running scenario analysis...")
        scenario_df = model.scenario(scores)
        display_scenarios(scenario_df)
        return

    if args.pareto:
        logger.info("[score] Computing Pareto frontier...")
        frontier = model.pareto_frontier(scores, country=args.country)
        display_pareto(frontier)
        return

    logger.info(f"[score] Generating recommendations: objective={args.objective}, "
                f"country={args.country or 'ALL'}, top_n={args.top_n}")

    recs = model.recommend(
        scores,
        objective=args.objective,
        top_n=args.top_n,
        country=args.country,
        min_yield_pct=args.min_yield,
        min_growth_pct=args.min_growth,
    )

    display_recommendations(recs, args.objective, args.top_n)


if __name__ == "__main__":
    main()
