"""
data_sources/fred.py — FRED API client for Norwegian macro signals.

Fetches:
  - 3-month NIBOR (interbank rate)
  - M3 broad money supply
  - 10-year government bond yield

Returns a flat dict of computed signals used by the scoring model.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import FRED_API_KEY, FRED_SERIES, CACHE_DIR, CACHE_TTL_HOURS

logger = logging.getLogger(__name__)


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"fred_{key}.json"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_TTL_HOURS


def _fetch_series(series_id: str, key: str, periods: int = 120) -> pd.Series:
    """Fetch a FRED series, using cache if valid."""
    cache = _cache_path(key)

    if _is_cache_valid(cache):
        logger.info(f"[fred] Using cached {key}")
        data = json.loads(cache.read_text())
        return pd.Series(data["values"], index=pd.to_datetime(data["dates"]), name=key)

    if not FRED_API_KEY:
        logger.warning("[fred] No FRED_API_KEY set — returning empty series")
        return pd.Series(dtype=float, name=key)

    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

    logger.info(f"[fred] Fetching {series_id} ({key})...")
    raw = fred.get_series(series_id)
    # Keep last N observations
    raw = raw.dropna().tail(periods)

    # Cache
    cache.write_text(json.dumps({
        "dates": [d.isoformat() for d in raw.index],
        "values": [float(v) for v in raw.values],
        "fetched_at": datetime.now().isoformat(),
    }))

    return raw.rename(key)


def _compute_nibor_signals(series: pd.Series) -> dict:
    """Compute interest rate signals from NIBOR series."""
    if series.empty:
        return {
            "nibor_current": np.nan,
            "nibor_trend_6m": np.nan,
            "nibor_percentile_10y": np.nan,
        }

    current = float(series.iloc[-1])

    # 6-month trend: difference between current and 6 months ago
    six_months_ago = series.index[-1] - timedelta(days=180)
    past = series[series.index <= six_months_ago]
    trend_6m = float(current - past.iloc[-1]) if len(past) > 0 else 0.0

    # Percentile within 10-year history
    ten_years = series.tail(120)  # ~10 years of monthly data
    percentile = float((ten_years < current).sum() / len(ten_years))

    return {
        "nibor_current": round(current, 3),
        "nibor_trend_6m": round(trend_6m, 3),
        "nibor_percentile_10y": round(percentile, 3),
    }


def _compute_m3_signals(series: pd.Series) -> dict:
    """Compute monetary signals from M3 money supply."""
    if series.empty:
        return {
            "m3_yoy_growth": np.nan,
            "m3_acceleration": np.nan,
        }

    # YoY growth rate: requires 13 points (current + 12 months back)
    if len(series) >= 13:
        yoy = float((series.iloc[-1] / series.iloc[-13] - 1) * 100)
    else:
        yoy = np.nan

    # Acceleration: compare recent 6m growth rate vs prior 6m
    # requires 13 points: iloc[-1], iloc[-7], iloc[-13]
    if len(series) >= 13:
        recent_6m = float((series.iloc[-1] / series.iloc[-7] - 1) * 100)
        prior_6m = float((series.iloc[-7] / series.iloc[-13] - 1) * 100)
        acceleration = recent_6m - prior_6m
    else:
        acceleration = np.nan

    return {
        "m3_yoy_growth": round(yoy, 2) if not np.isnan(yoy) else np.nan,
        "m3_acceleration": round(acceleration, 2) if not np.isnan(acceleration) else np.nan,
    }


def _compute_bond_signals(series: pd.Series, nibor_current: float) -> dict:
    """Compute bond yield signals."""
    if series.empty:
        return {
            "bond_yield_10y": np.nan,
            "yield_curve_spread": np.nan,
        }

    current = float(series.iloc[-1])
    spread = current - nibor_current if not np.isnan(nibor_current) else np.nan

    return {
        "bond_yield_10y": round(current, 3),
        "yield_curve_spread": round(spread, 3) if spread is not None and not np.isnan(spread) else np.nan,
    }


def get_macro_signals() -> dict:
    """
    Fetch all FRED data and return a flat dict of computed macro signals.

    Returns dict with keys:
        nibor_current, nibor_trend_6m, nibor_percentile_10y,
        m3_yoy_growth, m3_acceleration,
        bond_yield_10y, yield_curve_spread
    """
    nibor = _fetch_series(FRED_SERIES["nibor_3m"], "nibor_3m")
    m3 = _fetch_series(FRED_SERIES["m3_money"], "m3_money")
    bond = _fetch_series(FRED_SERIES["bond_10y"], "bond_10y")

    nibor_signals = _compute_nibor_signals(nibor)
    m3_signals = _compute_m3_signals(m3)
    bond_signals = _compute_bond_signals(bond, nibor_signals["nibor_current"])

    signals = {**nibor_signals, **m3_signals, **bond_signals}
    logger.info(f"[fred] Macro signals: {signals}")
    return signals
