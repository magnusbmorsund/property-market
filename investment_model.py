"""
analysis/investment_model.py
──────────────────────────────────────────────────────────────────────────────
Housing stock investment scoring model.

Objective
─────────
Score every (country, region, dwelling_type) combination across two primary
investor objectives:

  A. MAX RENTAL YIELD   — buy for income; care about gross yield and yield
                          momentum (rising rents relative to price growth)
  B. MAX PRICE GROWTH   — buy for capital gain; care about price appreciation
                          velocity, supply constraint, and permit momentum

And an optional composite "balanced" score that lets you weight the two
objectives however you choose.

Methodology
───────────
Each signal is normalised to [0, 1] within its own country pool (min-max),
so regions are ranked within Norway (normalised to the same currency and
price level).

Signals used
────────────
  Yield signals (for objective A):
    y1  gross_yield_pct          — primary income signal
    y2  yield_vs_national_avg    — yield premium over country median
    y3  rent_growth_proxy        — building rate suppressed → upward rent pressure
    y4  price_affordability      — lower price/sqm → easier entry, better yield

  Growth signals (for objective B):
    g1  growth_1Y_ann            — recent momentum
    g2  growth_3Y_ann            — medium-term trend (if available)
    g3  supply_constraint        — low building rate → scarcity → price pressure
    g4  permit_momentum_reversal — permits falling sharply → supply tightening
    g5  index_level_gap          — region still below national avg → catch-up potential

  Risk signals (applied as penalty):
    r1  volatility               — std-dev of quarterly growth (high vol = risk)
    r2  completion_ratio         — ratio close to 1 = balanced; > 1.2 = oversupply risk

Output
──────
  investment_scores.parquet / .csv with columns:
    country, region, dwelling_type,
    yield_score [0-100], growth_score [0-100],
    composite_score [0-100],
    risk_penalty [0-1],
    recommendation [BUY / HOLD / AVOID],
    objective [YIELD / GROWTH / BALANCED],
    all raw signal values

Usage
─────
  from analysis.investment_model import InvestmentModel

  model = InvestmentModel(
      yield_weight=0.6,   # 60% income, 40% growth
      growth_weight=0.4,
  )
  scores = model.score()
  recs   = model.recommend(scores, top_n=10, objective='yield')
"""

import logging
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(name: str) -> pd.DataFrame:
    p = PROC_DIR / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    logger.warning(f"[model] {p} not found")
    return pd.DataFrame()


def _minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalise to [0, 1]. invert=True means lower raw = higher score."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    norm = (series - mn) / (mx - mn)
    return 1 - norm if invert else norm


def _normalise_within_country(df: pd.DataFrame,
                               col: str,
                               invert: bool = False) -> pd.Series:
    """Min-max normalise col within each country separately."""
    result = pd.Series(np.nan, index=df.index)
    for country, grp in df.groupby("country"):
        result.loc[grp.index] = _minmax(grp[col].fillna(grp[col].median()), invert)
    return result


# ── Data assembly ─────────────────────────────────────────────────────────────

def _build_feature_table() -> pd.DataFrame:
    """
    Join all relevant processed datasets into one wide feature table,
    one row per (country, region, dwelling_type).
    """

    # ── Price growth ──────────────────────────────────────────────────────────
    growth = _load("norway_price_growth")

    # ── Rental yield ──────────────────────────────────────────────────────────
    yield_df = _load("norway_rental_yield")
    # yield is region-level only (no dwelling_type dimension) — will merge on region
    yield_df = yield_df[["country", "region_code", "region",
                          "avg_monthly_rent", "avg_price_sqm",
                          "estimated_price", "gross_yield_pct"]]

    # ── Building rate ─────────────────────────────────────────────────────────
    br = _load("norway_building_rate")
    # Take most recent year
    br_latest = (
        br.sort_values("year")
          .groupby(["country", "region_code"])
          .last()
          .reset_index()
          [["country", "region_code", "permits_per_1000_stock", "permits_rolling3y"]]
    )

    # ── Completion ratio ──────────────────────────────────────────────────────
    cr = _load("norway_completion_ratio")
    cr_latest = (
        cr.sort_values("year")
          .groupby(["country", "region_code"])
          .last()
          .reset_index()
          [["country", "region_code", "completion_ratio"]]
    )

    # ── Permit momentum ───────────────────────────────────────────────────────
    mom = _load("norway_permit_momentum")
    # Latest 12m momentum
    mom_latest = (
        mom.sort_values("period")
           .groupby(["country", "region_code"])
           .last()
           .reset_index()
           [["country", "region_code", "momentum_yoy_pct"]]
          .rename(columns={"momentum_yoy_pct": "permit_momentum_yoy"})
    )

    # ── Price volatility (from index) ─────────────────────────────────────────
    idx = _load("norway_price_index")
    volatility = (
        idx.groupby(["country", "region_code", "dwelling_type"])["growth_qoq"]
           .std()
           .reset_index()
           .rename(columns={"growth_qoq": "volatility_qoq"})
    )

    # ── National average price/sqm (for relative affordability) ──────────────
    national_avg_psqm = (
        yield_df.groupby("country")["avg_price_sqm"]
                .median()
                .rename("national_avg_psqm")
    )

    # ── National average yield (for yield premium) ────────────────────────────
    national_avg_yield = (
        yield_df.groupby("country")["gross_yield_pct"]
                .median()
                .rename("national_avg_yield")
    )

    # ── Assemble ──────────────────────────────────────────────────────────────
    # Start from growth (has dwelling_type dimension)
    df = growth.copy()

    # Merge yield (region level)
    df = df.merge(yield_df, on=["country", "region_code", "region"], how="left")

    # Merge building rate
    df = df.merge(br_latest, on=["country", "region_code"], how="left")

    # Merge completion ratio
    df = df.merge(cr_latest, on=["country", "region_code"], how="left")

    # Merge permit momentum
    df = df.merge(mom_latest, on=["country", "region_code"], how="left")

    # Merge volatility (dwelling_type specific)
    df = df.merge(volatility, on=["country", "region_code", "dwelling_type"], how="left")

    # National averages
    df = df.merge(national_avg_psqm, on="country", how="left")
    df = df.merge(national_avg_yield, on="country", how="left")

    # Derived features
    df["yield_premium"]         = df["gross_yield_pct"] - df["national_avg_yield"]
    df["price_vs_national"]     = df["avg_price_sqm"] / df["national_avg_psqm"]  # <1 = cheap
    df["supply_tightening"]     = -df["permit_momentum_yoy"]  # negative momentum = tightening
    df["index_gap"]             = df["latest_index"].max() - df["latest_index"]  # catch-up room

    logger.info(f"[model] Feature table: {len(df)} rows, {len(df.columns)} features")
    return df


# ── Scoring model ─────────────────────────────────────────────────────────────

class InvestmentModel:
    """
    Multi-factor housing investment scorer.

    Parameters
    ──────────
    yield_weight   : float [0-1] — weight on income objective (default 0.5)
    growth_weight  : float [0-1] — weight on capital growth objective (default 0.5)
                     yield_weight + growth_weight must = 1.0
    risk_penalty   : float [0-1] — how much to penalise for risk (default 0.3)
                     final_score = raw_score × (1 - risk_penalty × risk_score)

    Signal weights (internal, override if needed):
    yield_signal_weights  : dict with keys y1..y4
    growth_signal_weights : dict with keys g1..g5
    """

    # Default signal weights — reflect relative reliability/importance
    DEFAULT_YIELD_WEIGHTS = {
        "y1_gross_yield":       0.45,   # direct income signal — most important
        "y2_yield_premium":     0.20,   # above-average yield region
        "y3_rent_pressure":     0.20,   # supply-driven rent upside
        "y4_affordability":     0.15,   # lower entry price = better yield base
    }

    DEFAULT_GROWTH_WEIGHTS = {
        "g1_growth_1y":         0.35,   # recent momentum
        "g2_growth_3y":         0.25,   # medium-term trend
        "g3_supply_constraint": 0.20,   # scarcity = price pressure
        "g4_supply_tightening": 0.10,   # permit deceleration
        "g5_index_gap":         0.10,   # catch-up to national average
    }

    RISK_SIGNAL_WEIGHTS = {
        "r1_volatility":        0.60,
        "r2_oversupply":        0.40,
    }

    def __init__(self,
                 yield_weight: float = 0.5,
                 growth_weight: float = 0.5,
                 risk_penalty: float = 0.25):
        assert abs(yield_weight + growth_weight - 1.0) < 1e-6, \
            "yield_weight + growth_weight must equal 1.0"
        self.yield_weight  = yield_weight
        self.growth_weight = growth_weight
        self.risk_penalty  = risk_penalty
        self._features: pd.DataFrame = pd.DataFrame()

    # ── Signal construction ───────────────────────────────────────────────────

    def _yield_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute normalised yield signals y1–y4, within-country."""
        out = df.copy()

        # y1: raw gross yield — higher = better
        out["_y1"] = _normalise_within_country(out, "gross_yield_pct", invert=False)

        # y2: yield premium over national median — higher premium = better
        out["_y2"] = _normalise_within_country(out, "yield_premium", invert=False)

        # y3: rent pressure — LOW building rate → upward rent pressure
        out["_y3"] = _normalise_within_country(out, "permits_per_1000_stock", invert=True)

        # y4: affordability — LOW price/sqm relative to national → better yield entry
        out["_y4"] = _normalise_within_country(out, "price_vs_national", invert=True)

        return out

    def _growth_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute normalised growth signals g1–g5, within-country."""
        out = df.copy()

        # g1: 1Y annualised growth — higher = better
        out["_g1"] = _normalise_within_country(out, "growth_1Y_ann", invert=False)

        # g2: 3Y annualised growth — higher = better (may be NaN for some regions)
        if "growth_3Y_ann" in out.columns and out["growth_3Y_ann"].notna().any():
            out["_g2"] = _normalise_within_country(out, "growth_3Y_ann", invert=False)
        else:
            out["_g2"] = out["_g1"]  # fall back to 1Y if 3Y not available

        # g3: supply constraint — LOW building rate → scarcity → price pressure
        out["_g3"] = _normalise_within_country(out, "permits_per_1000_stock", invert=True)

        # g4: supply tightening — permits falling fast → tightening supply
        out["_g4"] = _normalise_within_country(out, "supply_tightening", invert=False)

        # g5: index gap vs national max — room to catch up
        out["_g5"] = _normalise_within_country(out, "index_gap", invert=False)

        return out

    def _risk_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute risk score (0=low risk, 1=high risk), within-country."""
        out = df.copy()

        # r1: quarterly growth volatility — high = risky
        out["_r1"] = _normalise_within_country(out, "volatility_qoq", invert=False)

        # r2: completion ratio oversupply risk — > 1.1 = oversupply, penalise
        # completion_ratio: clamp at 0–2, high = risky
        cr = out["completion_ratio"].fillna(1.0).clip(0, 2.0)
        out["_r2_raw"] = (cr - 1.0).clip(lower=0)  # only penalise > 1.0
        out["_r2"] = _normalise_within_country(out, "_r2_raw", invert=False)

        return out

    # ── Composite scoring ─────────────────────────────────────────────────────

    def score(self) -> pd.DataFrame:
        """
        Run the full scoring model.
        Returns DataFrame with scores and all signal breakdowns.
        """
        df = _build_feature_table()
        self._features = df.copy()

        if df.empty:
            logger.error("[model] Empty feature table — cannot score")
            return pd.DataFrame()

        df = self._yield_signals(df)
        df = self._growth_signals(df)
        df = self._risk_signals(df)

        yw = self.DEFAULT_YIELD_WEIGHTS
        gw = self.DEFAULT_GROWTH_WEIGHTS
        rw = self.RISK_SIGNAL_WEIGHTS

        # Raw yield score [0,1]
        df["yield_score_raw"] = (
            yw["y1_gross_yield"]   * df["_y1"] +
            yw["y2_yield_premium"] * df["_y2"] +
            yw["y3_rent_pressure"] * df["_y3"] +
            yw["y4_affordability"] * df["_y4"]
        )

        # Raw growth score [0,1]
        df["growth_score_raw"] = (
            gw["g1_growth_1y"]         * df["_g1"] +
            gw["g2_growth_3y"]         * df["_g2"] +
            gw["g3_supply_constraint"] * df["_g3"] +
            gw["g4_supply_tightening"] * df["_g4"] +
            gw["g5_index_gap"]         * df["_g5"]
        )

        # Risk score [0,1] — higher = riskier
        df["risk_score"] = (
            rw["r1_volatility"] * df["_r1"] +
            rw["r2_oversupply"] * df["_r2"]
        )

        # Composite raw score
        df["composite_raw"] = (
            self.yield_weight  * df["yield_score_raw"] +
            self.growth_weight * df["growth_score_raw"]
        )

        # Risk-adjusted (penalty reduces score proportionally)
        df["risk_adjustment"] = 1 - (self.risk_penalty * df["risk_score"])
        df["composite_adj"]   = df["composite_raw"] * df["risk_adjustment"]

        # Scale to 0–100
        for col in ["yield_score_raw", "growth_score_raw", "composite_adj"]:
            df[col + "_100"] = (df[col] * 100).round(1)

        df = df.rename(columns={
            "yield_score_raw_100": "yield_score",
            "growth_score_raw_100": "growth_score",
            "composite_adj_100":   "composite_score",
        })

        # Percentile ranks within country
        for obj, col in [("yield", "yield_score"), ("growth", "growth_score"),
                         ("composite", "composite_score")]:
            for country, grp in df.groupby("country"):
                df.loc[grp.index, f"{obj}_pct_rank"] = (
                    rankdata(grp[col]) / len(grp) * 100
                ).round(1)

        logger.info(f"[model] Scoring complete: {len(df)} rows")
        return df

    # ── Recommendations ───────────────────────────────────────────────────────

    def recommend(self,
                  scores: pd.DataFrame,
                  objective: Literal["yield", "growth", "balanced"] = "balanced",
                  top_n: int = 10,
                  country: str = None,
                  min_yield_pct: float = None,
                  min_growth_pct: float = None) -> pd.DataFrame:
        """
        Generate ranked BUY / HOLD / AVOID recommendations.

        Parameters
        ──────────
        scores        : output of .score()
        objective     : 'yield', 'growth', or 'balanced'
        top_n         : number of top BUY recommendations to return
        country       : filter to specific country code (e.g. 'NO')
        min_yield_pct : minimum gross_yield_pct to include (e.g. 4.0)
        min_growth_pct: minimum growth_1Y_ann to include (e.g. 3.0)
        """
        df = scores.copy()

        if country:
            df = df[df["country"] == country]

        # Hard filters
        if min_yield_pct:
            df = df[df["gross_yield_pct"] >= min_yield_pct]
        if min_growth_pct:
            df = df[df["growth_1Y_ann"] >= min_growth_pct]

        # Choose ranking column
        rank_col = {
            "yield":    "yield_score",
            "growth":   "growth_score",
            "balanced": "composite_score",
        }[objective]

        df = df.sort_values(rank_col, ascending=False).reset_index(drop=True)

        # Assign recommendation tiers
        n = len(df)
        df["recommendation"] = "HOLD"
        df.loc[df.index[:max(1, top_n)],          "recommendation"] = "BUY"
        df.loc[df.index[max(1, n - top_n // 2):], "recommendation"] = "AVOID"
        df["objective"] = objective.upper()

        # Build clean output
        output_cols = [
            "recommendation", "country", "region", "dwelling_type",
            "composite_score", "yield_score", "growth_score",
            "gross_yield_pct", "growth_1Y_ann",
            "avg_price_sqm", "avg_monthly_rent",
            "permits_per_1000_stock", "volatility_qoq",
            "risk_score", "objective",
        ]
        output_cols = [c for c in output_cols if c in df.columns]
        result = df[output_cols].copy()

        # Format risk as text
        result["risk_score"] = result["risk_score"].apply(
            lambda x: "Low" if x < 0.33 else ("Medium" if x < 0.66 else "High")
        )

        return result

    # ── Frontier analysis ────────────────────────────────────────────────────

    def pareto_frontier(self, scores: pd.DataFrame,
                         country: str = None) -> pd.DataFrame:
        """
        Identify Pareto-optimal investments: no other option dominates on
        BOTH yield AND growth simultaneously.
        These are the true 'efficient frontier' picks.
        """
        df = scores.copy()
        if country:
            df = df[df["country"] == country]

        dominated = []
        for i, row_i in df.iterrows():
            for j, row_j in df.iterrows():
                if i == j:
                    continue
                # j dominates i if j >= i on both, strictly > on at least one
                if (row_j["yield_score"]  >= row_i["yield_score"] and
                    row_j["growth_score"] >= row_i["growth_score"] and
                    (row_j["yield_score"] > row_i["yield_score"] or
                     row_j["growth_score"] > row_i["growth_score"])):
                    dominated.append(i)
                    break

        frontier = df[~df.index.isin(dominated)].copy()
        frontier["pareto_optimal"] = True
        frontier = frontier.sort_values("composite_score", ascending=False)
        logger.info(f"[model] Pareto frontier: {len(frontier)} non-dominated options")
        return frontier

    # ── Scenario analysis ─────────────────────────────────────────────────────

    def scenario(self,
                 scores: pd.DataFrame,
                 scenarios: dict = None) -> pd.DataFrame:
        """
        Run multiple objective-weight scenarios and show how rankings shift.

        scenarios: dict of {name: (yield_weight, growth_weight)}
        Default: income-focused, balanced, growth-focused
        """
        if scenarios is None:
            scenarios = {
                "Income (80/20)":   (0.80, 0.20),
                "Balanced (50/50)": (0.50, 0.50),
                "Growth (20/80)":   (0.20, 0.80),
            }

        results = []
        for name, (yw, gw) in scenarios.items():
            m = InvestmentModel(yield_weight=yw, growth_weight=gw,
                                risk_penalty=self.risk_penalty)
            s = m.score()
            top5 = m.recommend(s, objective="balanced", top_n=5)
            top5["scenario"] = name
            top5["scenario_yield_weight"]  = yw
            top5["scenario_growth_weight"] = gw
            results.append(top5)

        return pd.concat(results, ignore_index=True)
