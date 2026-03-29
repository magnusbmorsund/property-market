#!/usr/bin/env python3
"""
run_pipeline.py — Norwegian & Swedish Property Investment Pipeline

Orchestrates:
  1. FRED macro signals (NIBOR/STIBOR, M3, bond yield)
  2. SSB/SCB region features (prices, rents, employment, debt)
  3. Finn.no / Hemnet.se listing scraper
  4. Property-level scoring
  5. Top-N ranked output

Usage:
  python run_pipeline.py                             # Norway default: top 50, balanced
  python run_pipeline.py --country sweden            # Sweden pipeline (Hemnet + SCB)
  python run_pipeline.py --country sweden --city stockholm
  python run_pipeline.py --top-n 100                # more results
  python run_pipeline.py --yield-weight 0.6         # income-focused
  python run_pipeline.py --max-listings 200         # limit scraping
  python run_pipeline.py --no-cache                 # force refresh SSB/SCB/FRED
  python run_pipeline.py --export results.csv       # custom export path
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CACHE_DIR, OUTPUT_DIR,
    DEFAULT_YIELD_WEIGHT, DEFAULT_GROWTH_WEIGHT,
    DEFAULT_RISK_PENALTY, DEFAULT_TOP_N,
    FINN_MAX_PAGES, FRED_API_KEY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Rich console output ──────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None


def _print_header(text: str):
    if RICH:
        console.print(Panel(text, style="bold blue"))
    else:
        print(f"\n{'═' * 80}")
        print(f"  {text}")
        print(f"{'═' * 80}")


def _print_results(df: pd.DataFrame, top_n: int):
    """Print top-N scored properties to console."""
    top = df.head(top_n)

    if RICH:
        table = Table(
            title=f"TOP {top_n} PROPERTY DEALS IN NORWAY",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            show_lines=False,
            expand=True,
        )

        table.add_column("#", style="dim", no_wrap=True)
        table.add_column("Score", no_wrap=True)
        table.add_column("Yield", no_wrap=True)
        table.add_column("Growth", no_wrap=True)
        table.add_column("Risk", no_wrap=True)
        table.add_column("Rec", no_wrap=True)
        table.add_column("Municipality", no_wrap=True)
        table.add_column("Type", no_wrap=True)
        table.add_column("Price (NOK)", no_wrap=True, justify="right")
        table.add_column("m²", no_wrap=True, justify="right")
        table.add_column("Yield%", no_wrap=True, justify="right")
        table.add_column("1Y Δ%", no_wrap=True, justify="right")

        for _, row in top.iterrows():
            rec = row.get("recommendation", "")
            style = "bold green" if rec == "BUY" else (
                "yellow" if rec == "HOLD" else "dim red")

            price = row.get("asking_price") or row.get("total_price_calc")
            price_str = f"{price:,.0f}" if price and not np.isnan(price) else "N/A"

            sqm = row.get("sqm")
            sqm_str = f"{sqm:.0f}" if sqm and not np.isnan(sqm) else "N/A"

            gy = row.get("gross_yield_pct")
            gy_str = f"{gy:.1f}" if gy and not np.isnan(gy) else "N/A"

            g1y = row.get("price_growth_1y")
            g1y_str = f"{g1y:+.1f}" if g1y is not None and not np.isnan(g1y) else "-"

            table.add_row(
                str(int(row.get("rank", 0))),
                f"{row.get('composite_score', 0):.1f}",
                f"{row.get('yield_score', 0):.1f}",
                f"{row.get('growth_score', 0):.1f}",
                str(row.get("risk_label", "N/A")),
                rec,
                str(row.get("municipality_name", row.get("address", "N/A")))[:18],
                str(row.get("property_type", "N/A"))[:12],
                price_str,
                sqm_str,
                gy_str,
                g1y_str,
                style=style,
            )

        console.print(table)
    else:
        print(f"\n{'═' * 120}")
        print(f"  TOP {top_n} PROPERTY DEALS IN NORWAY")
        print(f"{'═' * 120}")
        print(f"  {'#':<4} {'Score':<7} {'Yield':<7} {'Growth':<7} {'Risk':<7} "
              f"{'Municipality':<18} {'Price':>14} {'m²':>5} {'Yield%':>7} {'URL'}")
        print(f"  {'─' * 116}")

        for _, row in top.iterrows():
            price = row.get("asking_price") or row.get("total_price_calc")
            price_str = f"{price:>13,.0f}" if price and not np.isnan(price) else "          N/A"
            sqm = row.get("sqm")
            sqm_str = f"{sqm:>4.0f}" if sqm and not np.isnan(sqm) else " N/A"
            gy = row.get("gross_yield_pct")
            gy_str = f"{gy:>6.1f}" if gy and not np.isnan(gy) else "   N/A"

            print(f"  {row.get('rank', 0):<4} "
                  f"{row.get('composite_score', 0):<7.1f}"
                  f"{row.get('yield_score', 0):<7.1f}"
                  f"{row.get('growth_score', 0):<7.1f}"
                  f"{str(row.get('risk_label', 'N/A')):<7} "
                  f"{str(row.get('municipality_name', 'N/A')):<18} "
                  f"{price_str} {sqm_str} {gy_str}  "
                  f"{row.get('url', '')[:50]}")


def _print_macro_summary(signals: dict, country: str = "norway"):
    """Print macro environment summary."""
    rate_label = "STIBOR 3M" if country == "sweden" else "NIBOR 3M"
    if RICH:
        lines = []
        nibor = signals.get("nibor_current")
        if nibor is not None and not np.isnan(nibor):
            trend = signals.get("nibor_trend_6m", 0) or 0
            arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            lines.append(f"{rate_label}: {nibor:.2f}% {arrow} ({trend:+.2f}pp 6M)")

        m3 = signals.get("m3_yoy_growth")
        if m3 is not None and not np.isnan(m3):
            accel = signals.get("m3_acceleration", 0) or 0
            arrow = "↑" if accel > 0 else "↓" if accel < 0 else "→"
            lines.append(f"M3 Money Supply YoY: {m3:.1f}% {arrow}")

        bond = signals.get("bond_yield_10y")
        if bond is not None and not np.isnan(bond):
            lines.append(f"10Y Bond Yield: {bond:.2f}%")

        spread = signals.get("yield_curve_spread")
        if spread is not None and not np.isnan(spread):
            lines.append(f"Yield Curve Spread: {spread:+.2f}pp")

        console.print(Panel("\n".join(lines) if lines else "No FRED data available",
                            title="Macro Environment", style="green"))
    else:
        print(f"\n  Macro Environment ({country.title()}):")
        for k, v in signals.items():
            if v is not None and not np.isnan(float(v) if v is not None else float("nan")):
                print(f"    {k}: {v}")


# ── Pipeline steps ───────────────────────────────────────────────────────────

def step_fetch_macro(no_cache: bool, country: str = "norway") -> dict:
    """Step 1: Fetch FRED macro signals for the given country."""
    _print_header("Step 1/5: Fetching FRED macro signals...")

    if no_cache:
        prefix = "se_" if country == "sweden" else ""
        for f in CACHE_DIR.glob(f"fred_{prefix}*.json"):
            f.unlink()
        # Also clear the non-prefixed keys for Norway
        if country == "norway":
            for key in ("nibor_3m", "m3_money", "bond_10y"):
                p = CACHE_DIR / f"fred_{key}.json"
                if p.exists():
                    p.unlink()

    from data_sources.fred import get_macro_signals
    signals = get_macro_signals(country=country)
    _print_macro_summary(signals, country=country)
    return signals


def step_fetch_regions(no_cache: bool) -> tuple:
    """Step 2: Fetch SSB region features."""
    _print_header("Step 2/5: Fetching SSB region data...")

    if no_cache:
        for f in CACHE_DIR.glob("ssb_*.json"):
            f.unlink()

    from data_sources.ssb import build_region_features
    features, rent_data, index_signals = build_region_features()
    logger.info(f"[pipeline] Region features: {len(features)} municipalities")
    return features, rent_data, index_signals


def step_scrape_finn(max_pages: int, max_listings: int = None, location_params: str = "") -> pd.DataFrame:
    """Step 3: Scrape Finn.no listings."""
    _print_header("Step 3/5: Scraping Finn.no listings...")

    from scrapers.finn import scrape_listings
    listings = scrape_listings(max_pages=max_pages, max_listings=max_listings, location_params=location_params)
    logger.info(f"[pipeline] Scraped {len(listings)} listings from Finn.no")
    return listings


def step_enrich(listings: pd.DataFrame,
                region_features: pd.DataFrame,
                rent_data: pd.DataFrame,
                index_signals: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Map listings to municipalities and join region data."""
    _print_header("Step 4/5: Enriching listings with region data...")

    from mapping.region_mapper import enrich_listings_with_regions, estimate_rents_vectorized

    # Add municipality/region codes
    enriched = enrich_listings_with_regions(listings)

    # Join region features — use name-based matching since SSB codes
    # may differ from our mapper codes (pre/post 2020 municipality reform)
    if not region_features.empty and "municipality_name" in enriched.columns:
        enriched["municipality_clean"] = (
            enriched["municipality_name"]
            .str.strip()
            .str.lower()
        )

        if "municipality_clean" in region_features.columns:
            # Join on cleaned municipality name (most reliable across code systems)
            feat_deduped = region_features.drop_duplicates(subset=["municipality_clean"])
            join_cols = [c for c in feat_deduped.columns if c != "municipality_code"]
            enriched = enriched.merge(
                feat_deduped[join_cols],
                on="municipality_clean", how="left"
            )

    if not rent_data.empty:
        enriched["estimated_monthly_rent"] = estimate_rents_vectorized(enriched, rent_data)
        enriched["estimated_annual_rent"] = enriched["estimated_monthly_rent"] * 12
    else:
        enriched["estimated_monthly_rent"] = np.nan
        enriched["estimated_annual_rent"] = np.nan

    # Join index signals (SSB uses broad price-index regions, not municipalities)
    # Map municipality names to SSB index region codes
    if not index_signals.empty:
        _INDEX_REGION_MAP = {
            "oslo": "001", "bærum": "001",
            "stavanger": "002", "sandnes": "002", "sola": "002", "randaberg": "002",
            "bergen": "003",
            "trondheim": "004",
        }
        # Map fylke codes to index regions for remaining
        _FYLKE_TO_INDEX = {
            "32": "005",  # Akershus (uten Bærum)
            "31": "006", "33": "006", "38": "006", "39": "006",  # Østfold/Buskerud/Vestfold/Telemark
            "34": "007",  # Innlandet
            "40": "008", "11": "008",  # Agder + Rogaland (uten Stavanger)
            "15": "009", "42": "009",  # Møre og Romsdal + Vestland (uten Bergen)
            "50": "010",  # Trøndelag (uten Trondheim)
            "18": "011", "55": "011", "56": "011",  # Nord-Norge
        }

        def _map_to_index_region(row):
            name = str(row.get("municipality_clean", "")).lower()
            if name in _INDEX_REGION_MAP:
                return _INDEX_REGION_MAP[name]
            fylke = str(row.get("region_code", ""))
            return _FYLKE_TO_INDEX.get(fylke, "TOTAL")

        enriched["index_region_code"] = enriched.apply(_map_to_index_region, axis=1)
        idx_cols = ["region_code", "index_momentum_qoq", "volatility_qoq", "latest_index"]
        idx_cols = [c for c in idx_cols if c in index_signals.columns]
        idx_rename = {"region_code": "index_region_code"}
        idx_data = index_signals[idx_cols].rename(columns=idx_rename)
        enriched = enriched.merge(idx_data, on="index_region_code", how="left")

    # Compute gross yield — guard against zero/NaN total price to avoid silent NaN propagation
    if "total_price_calc" in enriched.columns and "estimated_annual_rent" in enriched.columns:
        valid_price = enriched["total_price_calc"].replace(0, np.nan)
        enriched["gross_yield_pct"] = (
            enriched["estimated_annual_rent"] / valid_price * 100
        ).replace([np.inf, -np.inf], np.nan).round(2)
        nan_yield = enriched["gross_yield_pct"].isna().sum()
        if nan_yield > 0:
            logger.warning(f"[pipeline] {nan_yield}/{len(enriched)} listings have NaN yield "
                           "(missing price or rent estimate) — they will score at dataset median")

    logger.info(f"[pipeline] Enriched {len(enriched)} listings with region data")
    return enriched


def step_fetch_regions_se(no_cache: bool) -> tuple:
    """Step 2 for Sweden: Fetch SCB region features."""
    _print_header("Step 2/5: Fetching SCB region data (Sweden)...")

    if no_cache:
        for f in CACHE_DIR.glob("scb_se_*.json"):
            f.unlink()

    from data_sources.scb import build_region_features
    features, rent_data, index_signals = build_region_features()
    logger.info(f"[pipeline] SE region features: {len(features)} municipalities")
    return features, rent_data, index_signals


def step_scrape_hemnet(max_pages: int,
                        max_listings: int = None,
                        location_params: str = "") -> pd.DataFrame:
    """Step 3 for Sweden: Scrape Hemnet.se listings."""
    _print_header("Step 3/5: Scraping Hemnet.se listings...")

    from scrapers.hemnet import scrape_listings
    listings = scrape_listings(
        max_pages=max_pages,
        max_listings=max_listings,
        location_params=location_params,
    )
    logger.info(f"[pipeline] Scraped {len(listings)} listings from Hemnet.se")
    return listings


def step_enrich_se(listings: pd.DataFrame,
                    region_features: pd.DataFrame,
                    rent_data: pd.DataFrame,
                    index_signals: pd.DataFrame) -> pd.DataFrame:
    """Step 4 for Sweden: Enrich listings with Swedish region data."""
    _print_header("Step 4/5: Enriching listings with Swedish region data...")

    from mapping.se_region_mapper import enrich_listings_with_regions, estimate_rents_vectorized

    # Add municipality/county codes via address matching
    enriched = enrich_listings_with_regions(listings)

    # Join region features on cleaned municipality name
    if not region_features.empty and "municipality_name" in enriched.columns:
        enriched["municipality_clean"] = (
            enriched["municipality_name"].str.strip().str.lower()
        )
        if "municipality_clean" in region_features.columns:
            feat_deduped = region_features.drop_duplicates(subset=["municipality_clean"])
            join_cols = [c for c in feat_deduped.columns if c != "municipality_code"]
            enriched = enriched.merge(
                feat_deduped[join_cols],
                on="municipality_clean",
                how="left",
            )

    # Estimate rents
    if not rent_data.empty:
        enriched["estimated_monthly_rent"] = estimate_rents_vectorized(enriched, rent_data)
        enriched["estimated_annual_rent"] = enriched["estimated_monthly_rent"] * 12
    else:
        enriched["estimated_monthly_rent"] = np.nan
        enriched["estimated_annual_rent"] = np.nan

    # Map to SCB price index regions (county-based)
    if not index_signals.empty:
        _SE_INDEX_REGION_MAP = {
            # Stockholm county municipalities
            "stockholm": "01", "solna": "01", "sundbyberg": "01", "nacka": "01",
            "huddinge": "01", "täby": "01", "järfälla": "01", "sollentuna": "01",
            "haninge": "01", "botkyrka": "01", "södertälje": "01",
            # Göteborg area
            "göteborg": "14", "mölndal": "14", "partille": "14", "kungälv": "14",
            "härryda": "14", "lerum": "14",
            # Malmö area
            "malmö": "12", "lund": "12", "vellinge": "12", "staffanstorp": "12",
            "burlöv": "12", "lomma": "12",
            # Uppsala
            "uppsala": "03", "knivsta": "03", "håbo": "03",
        }
        _COUNTY_TO_INDEX = {
            "01": "01",   # Stockholm
            "03": "03",   # Uppsala
            "04": "04", "05": "04", "18": "04", "19": "04",  # Mälardalen
            "06": "05", "07": "05", "08": "05",  # Småland
            "12": "12",  # Skåne
            "13": "13",  # Halland
            "14": "14",  # Västra Götaland
            "17": "17", "20": "17", "21": "17",  # Norrlands inland
            "22": "22", "23": "22", "24": "22", "25": "22",  # Norrland
        }

        def _map_to_se_index(row):
            name = str(row.get("municipality_clean", "")).lower()
            if name in _SE_INDEX_REGION_MAP:
                return _SE_INDEX_REGION_MAP[name]
            county = str(row.get("region_code", ""))
            return _COUNTY_TO_INDEX.get(county, "TOTAL")

        enriched["index_region_code"] = enriched.apply(_map_to_se_index, axis=1)
        idx_cols = [c for c in ["region_code", "index_momentum_qoq", "volatility_qoq",
                                 "latest_index"] if c in index_signals.columns]
        if idx_cols:
            idx_data = (
                index_signals[idx_cols]
                .rename(columns={"region_code": "index_region_code"})
            )
            enriched = enriched.merge(idx_data, on="index_region_code", how="left")

    # Gross yield
    if "total_price_calc" in enriched.columns and "estimated_annual_rent" in enriched.columns:
        valid_price = enriched["total_price_calc"].replace(0, np.nan)
        enriched["gross_yield_pct"] = (
            enriched["estimated_annual_rent"] / valid_price * 100
        ).replace([np.inf, -np.inf], np.nan).round(2)

    logger.info(f"[pipeline] Enriched {len(enriched)} SE listings")
    return enriched


def _print_results_se(df: pd.DataFrame, top_n: int, city: str = None):
    """Print top-N scored Swedish properties to console."""
    top = df.head(top_n)
    city_label = city.title() if city else "SWEDEN"

    if RICH:
        table = Table(
            title=f"TOP {top_n} PROPERTY DEALS IN {city_label.upper()}",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            show_lines=False,
            expand=True,
        )

        table.add_column("#", style="dim", no_wrap=True)
        table.add_column("Score", no_wrap=True)
        table.add_column("Yield", no_wrap=True)
        table.add_column("Growth", no_wrap=True)
        table.add_column("Risk", no_wrap=True)
        table.add_column("Rec", no_wrap=True)
        table.add_column("Municipality", no_wrap=True)
        table.add_column("Type", no_wrap=True)
        table.add_column("Price (SEK)", no_wrap=True, justify="right")
        table.add_column("m²", no_wrap=True, justify="right")
        table.add_column("Yield%", no_wrap=True, justify="right")
        table.add_column("1Y Δ%", no_wrap=True, justify="right")

        for _, row in top.iterrows():
            rec = row.get("recommendation", "")
            style = "bold green" if rec == "BUY" else (
                "yellow" if rec == "HOLD" else "dim red")

            price = row.get("asking_price") or row.get("total_price_calc")
            price_str = f"{price:,.0f}" if price and not np.isnan(price) else "N/A"
            sqm = row.get("sqm")
            sqm_str = f"{sqm:.0f}" if sqm and not np.isnan(sqm) else "N/A"
            gy = row.get("gross_yield_pct")
            gy_str = f"{gy:.1f}" if gy is not None and not np.isnan(gy) else "N/A"
            g1y = row.get("price_growth_1y")
            g1y_str = f"{g1y:+.1f}" if g1y is not None and not np.isnan(g1y) else "-"

            table.add_row(
                str(int(row.get("rank", 0))),
                f"{row.get('composite_score', 0):.1f}",
                f"{row.get('yield_score', 0):.1f}",
                f"{row.get('growth_score', 0):.1f}",
                str(row.get("risk_label", "N/A")),
                rec,
                str(row.get("municipality_name", row.get("address", "N/A")))[:18],
                str(row.get("property_type", "N/A"))[:12],
                price_str,
                sqm_str,
                gy_str,
                g1y_str,
                style=style,
            )

        console.print(table)
    else:
        print(f"\n{'═' * 120}")
        print(f"  TOP {top_n} PROPERTY DEALS IN {city_label.upper()}")
        print(f"{'═' * 120}")
        print(f"  {'#':<4} {'Score':<7} {'Yield':<7} {'Growth':<7} {'Risk':<7} "
              f"{'Municipality':<18} {'Price (SEK)':>14} {'m²':>5} {'Yield%':>7} {'URL'}")
        print(f"  {'─' * 116}")

        for _, row in top.iterrows():
            price = row.get("asking_price") or row.get("total_price_calc")
            price_str = f"{price:>13,.0f}" if price and not np.isnan(price) else "          N/A"
            sqm = row.get("sqm")
            sqm_str = f"{sqm:>4.0f}" if sqm and not np.isnan(sqm) else " N/A"
            gy = row.get("gross_yield_pct")
            gy_str = f"{gy:>6.1f}" if gy is not None and not np.isnan(gy) else "   N/A"

            print(f"  {row.get('rank', 0):<4} "
                  f"{row.get('composite_score', 0):<7.1f}"
                  f"{row.get('yield_score', 0):<7.1f}"
                  f"{row.get('growth_score', 0):<7.1f}"
                  f"{str(row.get('risk_label', 'N/A')):<7} "
                  f"{str(row.get('municipality_name', 'N/A')):<18} "
                  f"{price_str} {sqm_str} {gy_str}  "
                  f"{row.get('url', '')[:50]}")


def step_score(enriched: pd.DataFrame,
               macro_signals: dict,
               yield_weight: float,
               growth_weight: float,
               risk_penalty: float) -> pd.DataFrame:
    """Step 5: Score and rank properties."""
    _print_header("Step 5/5: Scoring properties...")

    from models.property_scorer import score_properties
    scored = score_properties(
        enriched,
        macro_signals=macro_signals,
        yield_weight=yield_weight,
        growth_weight=growth_weight,
        risk_penalty=risk_penalty,
    )
    return scored


# ── Main ─────────────────────────────────────────────────────────────────────

_NORWAY_CITIES = {"oslo", "bergen", "trondheim", "stavanger"}
_SWEDEN_CITIES = {"stockholm", "gothenburg", "malmo", "uppsala", "linkoping", "orebro"}


def main():
    parser = argparse.ArgumentParser(
        description="Norwegian & Swedish Property Investment Pipeline — "
                    "Scrape, enrich, score, and rank property deals"
    )
    parser.add_argument("--country", type=str, default="norway",
                        choices=["norway", "sweden"],
                        help="Country to run pipeline for (default: norway)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"Number of top properties to display (default: {DEFAULT_TOP_N})")
    parser.add_argument("--yield-weight", type=float, default=DEFAULT_YIELD_WEIGHT,
                        help=f"Weight on rental yield (default: {DEFAULT_YIELD_WEIGHT})")
    parser.add_argument("--growth-weight", type=float, default=DEFAULT_GROWTH_WEIGHT,
                        help=f"Weight on price growth (default: {DEFAULT_GROWTH_WEIGHT})")
    parser.add_argument("--risk-penalty", type=float, default=DEFAULT_RISK_PENALTY,
                        help=f"Risk penalty factor (default: {DEFAULT_RISK_PENALTY})")
    parser.add_argument("--max-pages", type=int, default=FINN_MAX_PAGES,
                        help=f"Max search pages to scrape (default: {FINN_MAX_PAGES})")
    parser.add_argument("--max-listings", type=int, default=None,
                        help="Cap on number of listings to fetch details for")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force refresh SSB/SCB/FRED cached data")
    parser.add_argument("--city", type=str, default=None,
                        help="Filter listings to a specific city "
                             "(Norway: oslo/bergen/trondheim/stavanger; "
                             "Sweden: stockholm/gothenburg/malmo/uppsala/linkoping/orebro)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export path for CSV (default: output/property_scores_{timestamp}.csv)")
    args = parser.parse_args()

    # Validate city for the selected country
    if args.city:
        valid_cities = _NORWAY_CITIES if args.country == "norway" else _SWEDEN_CITIES
        if args.city not in valid_cities:
            parser.error(
                f"City '{args.city}' is not valid for --country {args.country}. "
                f"Valid cities: {sorted(valid_cities)}"
            )

    start_time = time.time()

    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not set — macro signals will be unavailable. "
                       "Set via: export FRED_API_KEY=your_key")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    # ── Sweden pipeline ───────────────────────────────────────────────────────
    if args.country == "sweden":
        _HEMNET_LOCATION_PARAMS = {
            "stockholm":  "location_ids[]=898741",
            "gothenburg": "location_ids[]=898165",
            "malmo":      "location_ids[]=898273",
            "uppsala":    "location_ids[]=898319",
            "linkoping":  "location_ids[]=898354",
            "orebro":     "location_ids[]=898380",
        }
        location_params = _HEMNET_LOCATION_PARAMS.get(args.city, "") if args.city else ""

        _print_header("Swedish Property Investment Pipeline")

        macro_signals = step_fetch_macro(args.no_cache, country="sweden")
        region_features, rent_data, index_signals = step_fetch_regions_se(args.no_cache)
        listings = step_scrape_hemnet(args.max_pages, args.max_listings, location_params)

        if listings.empty:
            logger.error("No listings scraped from Hemnet.se — cannot proceed. "
                         "Check network connection, Hemnet availability, and location IDs.")
            sys.exit(1)

        enriched = step_enrich_se(listings, region_features, rent_data, index_signals)
        scored = step_score(
            enriched,
            macro_signals=macro_signals,
            yield_weight=args.yield_weight,
            growth_weight=args.growth_weight,
            risk_penalty=args.risk_penalty,
        )

        _print_results_se(scored, args.top_n, city=args.city)

        export_path = (
            Path(args.export) if args.export
            else OUTPUT_DIR / f"se_property_scores_{timestamp}.csv"
        )
        export_cols_se = [
            "rank", "recommendation", "composite_score", "yield_score", "growth_score",
            "risk_score", "risk_label",
            "hemnetkod", "url", "title", "address",
            "municipality_name", "region_name",
            "asking_price", "total_price_calc", "sqm", "price_per_sqm",
            "property_type", "ownership_type", "rooms", "year_built",
            "common_costs_monthly", "monthly_fee",
            "gross_yield_pct", "estimated_monthly_rent",
            "price_growth_1y", "price_growth_3y",
            "wage_growth_1y", "latest_wage",
            "permits_per_1000", "permit_momentum",
            "employment_hhi", "population",
        ]
        export_cols_se = [c for c in export_cols_se if c in scored.columns]
        scored[export_cols_se].to_csv(export_path, index=False)
        logger.info(f"[pipeline] SE results exported to {export_path}")

    # ── Norway pipeline (existing, unchanged behaviour) ───────────────────────
    else:
        _CITY_LOCATION_PARAMS = {
            "oslo":       "location=0.20061",
            "trondheim":  "lat=63.43049&lon=10.39506&radius=20000",
            "bergen":     "lat=60.39130&lon=5.32210&radius=20000",
            "stavanger":  "lat=58.97000&lon=5.73310&radius=15000",
        }
        location_params = _CITY_LOCATION_PARAMS.get(args.city, "") if args.city else ""

        _print_header("Norwegian Property Investment Pipeline")

        macro_signals = step_fetch_macro(args.no_cache, country="norway")
        region_features, rent_data, index_signals = step_fetch_regions(args.no_cache)
        listings = step_scrape_finn(args.max_pages, args.max_listings, location_params)

        if listings.empty:
            logger.error("No listings scraped — cannot proceed. "
                         "Check network connection and Finn.no availability.")
            sys.exit(1)

        enriched = step_enrich(listings, region_features, rent_data, index_signals)
        scored = step_score(
            enriched,
            macro_signals=macro_signals,
            yield_weight=args.yield_weight,
            growth_weight=args.growth_weight,
            risk_penalty=args.risk_penalty,
        )

        _print_results(scored, args.top_n)

        export_path = (
            Path(args.export) if args.export
            else OUTPUT_DIR / f"property_scores_{timestamp}.csv"
        )
        export_cols = [
            "rank", "recommendation", "composite_score", "yield_score", "growth_score",
            "risk_score", "risk_label",
            "finnkode", "url", "title", "address",
            "municipality_name", "region_name",
            "asking_price", "total_price_calc", "sqm", "price_per_sqm",
            "property_type", "ownership_type", "bedrooms", "year_built",
            "common_costs_monthly", "fellesgjeld",
            "gross_yield_pct", "estimated_monthly_rent",
            "price_growth_1y", "price_growth_3y",
            "wage_growth_1y", "latest_wage",
            "permits_per_1000", "permit_momentum",
            "debt_burden_pct", "employment_hhi",
            "population",
        ]
        export_cols = [c for c in export_cols if c in scored.columns]
        scored[export_cols].to_csv(export_path, index=False)
        logger.info(f"[pipeline] Results exported to {export_path}")

    elapsed = time.time() - start_time
    logger.info(f"[pipeline] Pipeline complete in {elapsed:.1f}s — "
                f"{len(scored)} properties scored, top {args.top_n} displayed")

    if RICH:
        console.print(f"\n[dim]Results saved to: {export_path}[/dim]")
        console.print(f"[dim]Total runtime: {elapsed:.1f}s[/dim]\n")


if __name__ == "__main__":
    main()
