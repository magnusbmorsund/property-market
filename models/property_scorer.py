"""
models/property_scorer.py — Property-level investment scoring engine.

Scores individual Finn.no listings using three dimensions:
  A. Rental Yield Score (default 40%)
  B. Price Growth Score (default 30%)
  C. Risk Score (default 30%, applied as penalty)

Each signal is min-max normalised to [0, 1] within the dataset.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import DEFAULT_YIELD_WEIGHT, DEFAULT_GROWTH_WEIGHT, DEFAULT_RISK_PENALTY

logger = logging.getLogger(__name__)


def _minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalise to [0, 1]. invert=True means lower raw value = higher score."""
    s = series.fillna(series.median())
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    norm = (s - mn) / (mx - mn)
    return 1 - norm if invert else norm


# ── Signal weights ───────────────────────────────────────────────────────────

YIELD_WEIGHTS = {
    "y1_gross_yield": 0.50,
    "y2_yield_premium": 0.25,
    "y3_affordability": 0.25,
}

GROWTH_WEIGHTS = {
    "g1_growth_1y": 0.30,
    "g2_growth_3y": 0.20,
    "g3_index_momentum": 0.20,
    "g4_wage_growth": 0.30,       # wage growth drives demand
}

RISK_WEIGHTS = {
    "r1_interest_sensitivity": 0.20,
    "r2_job_vulnerability": 0.20,
    "r3_price_volatility": 0.15,
    "r4_monetary_tightening": 0.10,
    "r5_debt_burden": 0.15,
    "r6_construction_pressure": 0.20,  # high building = oversupply risk
}


def score_properties(
    listings: pd.DataFrame,
    macro_signals: dict,
    yield_weight: float = DEFAULT_YIELD_WEIGHT,
    growth_weight: float = DEFAULT_GROWTH_WEIGHT,
    risk_penalty: float = DEFAULT_RISK_PENALTY,
) -> pd.DataFrame:
    """
    Score all properties and return ranked DataFrame.

    Parameters
    ----------
    listings : pd.DataFrame
        Enriched listings with region features joined in.
    macro_signals : dict
        FRED macro signals (nibor_current, m3_yoy_growth, etc.)
    yield_weight : float
        Weight on rental yield objective [0-1]
    growth_weight : float
        Weight on price growth objective [0-1]
    risk_penalty : float
        How much risk reduces the score [0-1]

    Returns
    -------
    pd.DataFrame with scores and ranking columns added
    """
    if listings.empty:
        logger.warning("[scorer] Empty listings — nothing to score")
        return listings

    df = listings.copy()
    n = len(df)
    logger.info(f"[scorer] Scoring {n} properties...")

    # ── A. Rental Yield Signals ──────────────────────────────────────────

    # y1: Gross rental yield (SSB rents are conservative, treat as approx net)
    if "gross_yield_pct" not in df.columns:
        if "estimated_annual_rent" in df.columns and "total_price_calc" in df.columns:
            df["gross_yield_pct"] = (
                df["estimated_annual_rent"] / df["total_price_calc"] * 100
            ).replace([np.inf, -np.inf], np.nan)
        else:
            df["gross_yield_pct"] = np.nan

    df["_y1"] = _minmax(df["gross_yield_pct"])

    # y2: Yield premium vs dataset median
    median_yield = df["gross_yield_pct"].median()
    df["yield_premium"] = df["gross_yield_pct"] - median_yield
    df["_y2"] = _minmax(df["yield_premium"])

    # y3: Affordability — lower price/sqm vs region = better entry
    if "price_per_sqm" in df.columns and "latest_price_sqm" in df.columns:
        df["price_ratio"] = df["price_per_sqm"] / df["latest_price_sqm"].replace(0, np.nan)
        df["_y3"] = _minmax(df["price_ratio"], invert=True)  # lower ratio = better
    else:
        df["_y3"] = _minmax(df.get("price_per_sqm", pd.Series(dtype=float)), invert=True)

    # Yield composite
    df["yield_score_raw"] = (
        YIELD_WEIGHTS["y1_gross_yield"] * df["_y1"] +
        YIELD_WEIGHTS["y2_yield_premium"] * df["_y2"] +
        YIELD_WEIGHTS["y3_affordability"] * df["_y3"]
    )

    # ── B. Price Growth Signals ──────────────────────────────────────────

    # g1: 1-year price growth in municipality
    if "price_growth_1y" in df.columns:
        df["_g1"] = _minmax(df["price_growth_1y"])
    else:
        df["_g1"] = 0.5

    # g2: 3-year annualized growth
    if "price_growth_3y" in df.columns:
        df["_g2"] = _minmax(df["price_growth_3y"])
    else:
        df["_g2"] = df["_g1"]  # fall back to 1Y

    # g3: Quarterly index momentum
    if "index_momentum_qoq" in df.columns:
        df["_g3"] = _minmax(df["index_momentum_qoq"])
    else:
        df["_g3"] = 0.5

    # g4: Wage growth in the municipality — higher wage growth = stronger demand
    if "wage_growth_1y" in df.columns:
        df["_g4"] = _minmax(df["wage_growth_1y"])
    else:
        df["_g4"] = 0.5

    # Growth composite
    df["growth_score_raw"] = (
        GROWTH_WEIGHTS["g1_growth_1y"] * df["_g1"] +
        GROWTH_WEIGHTS["g2_growth_3y"] * df["_g2"] +
        GROWTH_WEIGHTS["g3_index_momentum"] * df["_g3"] +
        GROWTH_WEIGHTS["g4_wage_growth"] * df["_g4"]
    )

    # ── C. Risk Signals ──────────────────────────────────────────────────

    # r1: Interest rate sensitivity
    #   = f(NIBOR level, NIBOR trend, local debt burden)
    nibor_pct = macro_signals.get("nibor_percentile_10y", 0.5)
    nibor_rising = 1.0 if macro_signals.get("nibor_trend_6m", 0) > 0 else 0.0
    debt_norm = _minmax(df.get("debt_burden_pct", pd.Series(dtype=float)))
    df["_r1"] = 0.4 * nibor_pct + 0.3 * nibor_rising + 0.3 * debt_norm

    # r2: Job market vulnerability (Herfindahl index)
    if "employment_hhi" in df.columns:
        df["_r2"] = _minmax(df["employment_hhi"])  # higher HHI = more concentrated = riskier
    else:
        df["_r2"] = 0.5

    # r3: Price volatility
    if "volatility_qoq" in df.columns:
        df["_r3"] = _minmax(df["volatility_qoq"])  # higher vol = riskier
    else:
        df["_r3"] = 0.5

    # r4: Monetary tightening (M3 deceleration)
    m3_accel = macro_signals.get("m3_acceleration", 0)
    # Negative acceleration = tightening = higher risk (applied uniformly)
    if m3_accel is not None and not np.isnan(m3_accel):
        # Scale: -5% acceleration → risk 1.0, +5% → risk 0.0
        r4_value = max(0, min(1, (-m3_accel + 5) / 10))
    else:
        r4_value = 0.5
    df["_r4"] = r4_value

    # r5: Mortgage debt burden (separate from r1 — structural leverage)
    if "debt_burden_pct" in df.columns:
        df["_r5"] = _minmax(df["debt_burden_pct"])  # higher = riskier
    else:
        df["_r5"] = 0.5

    # r6: Construction supply pressure — high permits per capita = oversupply risk
    if "permits_per_1000" in df.columns:
        df["_r6"] = _minmax(df["permits_per_1000"])  # more building = more supply = riskier
    else:
        df["_r6"] = 0.5

    # Risk composite
    df["risk_score"] = (
        RISK_WEIGHTS["r1_interest_sensitivity"] * df["_r1"] +
        RISK_WEIGHTS["r2_job_vulnerability"] * df["_r2"] +
        RISK_WEIGHTS["r3_price_volatility"] * df["_r3"] +
        RISK_WEIGHTS["r4_monetary_tightening"] * df["_r4"] +
        RISK_WEIGHTS["r5_debt_burden"] * df["_r5"] +
        RISK_WEIGHTS["r6_construction_pressure"] * df["_r6"]
    )

    # ── Composite Score ──────────────────────────────────────────────────

    # Normalise yield and growth weights to sum to 1.0
    total_w = yield_weight + growth_weight
    yw = yield_weight / total_w
    gw = growth_weight / total_w

    df["raw_score"] = yw * df["yield_score_raw"] + gw * df["growth_score_raw"]
    df["risk_adjustment"] = 1 - (risk_penalty * df["risk_score"])
    df["composite_raw"] = df["raw_score"] * df["risk_adjustment"]

    # Scale to 0-100
    df["yield_score"] = (df["yield_score_raw"] * 100).round(1)
    df["growth_score"] = (df["growth_score_raw"] * 100).round(1)
    df["composite_score"] = (df["composite_raw"] * 100).round(1)

    # ── Ranking and Recommendations ──────────────────────────────────────

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # Percentile rank
    df["percentile"] = (rankdata(df["composite_score"]) / len(df) * 100).round(1)

    # BUY / HOLD / AVOID tiers
    df["recommendation"] = "HOLD"
    top_n = max(1, len(df) // 5)  # top 20%
    bottom_n = max(1, len(df) // 5)  # bottom 20%
    df.loc[df.index[:top_n], "recommendation"] = "BUY"
    df.loc[df.index[-bottom_n:], "recommendation"] = "AVOID"

    # Risk label
    df["risk_label"] = df["risk_score"].apply(
        lambda x: "Low" if x < 0.33 else ("Medium" if x < 0.66 else "High")
    )

    # ── Clean up internal columns ────────────────────────────────────────
    internal_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=internal_cols + ["raw_score", "risk_adjustment", "composite_raw",
                                           "yield_score_raw", "growth_score_raw",
                                           "price_ratio", "yield_premium"],
                 errors="ignore")

    logger.info(f"[scorer] Scoring complete. Top score: {df['composite_score'].iloc[0]:.1f}, "
                f"Bottom: {df['composite_score'].iloc[-1]:.1f}")

    return df
