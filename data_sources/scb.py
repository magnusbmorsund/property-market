"""
data_sources/scb.py — SCB (Statistics Sweden) API client.

Fetches housing prices, rental data, population, employment, and wage statistics.
All data is cached locally with 24h TTL.

SCB PxWeb API:
  Base URL: https://api.scb.se/OV0104/v1/doris/sv/ssd/
  Query format: POST JSON with PxWeb query structure
  Response format: JSON-stat2 (same as SSB Norway)
"""

import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from config import SCB_BASE_URL, SCB_TABLES, CACHE_DIR, CACHE_TTL_HOURS

logger = logging.getLogger(__name__)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(table_key: str) -> Path:
    """Return cache file path for a given SCB table key."""
    return CACHE_DIR / f"scb_se_{table_key}.json"


def _is_cache_valid(path: Path) -> bool:
    """Return True if cache file exists and is younger than CACHE_TTL_HOURS."""
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_TTL_HOURS


# ── JSON-stat parser ──────────────────────────────────────────────────────────

def _parse_jsonstat(data: dict) -> pd.DataFrame:
    """
    Parse JSON-stat2 response from SCB into a flat DataFrame.

    Handles both dense (list) and sparse (dict) value formats,
    matching the pattern used in ssb.py.
    """
    # Unwrap dataset envelope if present
    if "id" not in data:
        if "dataset" in data:
            data = data["dataset"]
        elif isinstance(data, list) and len(data) > 0:
            data = data[0]

    dims = data.get("id", [])
    sizes = data.get("size", [])
    categories = data.get("dimension", {})
    values = data.get("value", [])

    if isinstance(values, dict):
        # Sparse format — keys are string indices
        max_idx = max(int(k) for k in values.keys()) + 1 if values else 0
        dense_values = [np.nan] * max_idx
        for k, v in values.items():
            dense_values[int(k)] = v
        values = dense_values

    # Build ordered label lists per dimension
    dim_labels = []
    for dim_id in dims:
        cat = categories[dim_id]["category"]
        idx_order = cat.get("index", {})
        labels = cat.get("label", {})

        if isinstance(idx_order, dict):
            ordered = sorted(idx_order.items(), key=lambda x: x[1])
            codes = [k for k, _ in ordered]
        elif isinstance(idx_order, list):
            codes = idx_order
        else:
            codes = list(labels.keys())

        dim_labels.append([(code, labels.get(code, code)) for code in codes])

    rows = []
    for i, combo in enumerate(itertools.product(*dim_labels)):
        if i < len(values):
            row = {}
            for j, (code, label) in enumerate(combo):
                row[f"{dims[j]}_code"] = code
                row[dims[j]] = label
            row["value"] = values[i]
            rows.append(row)

    return pd.DataFrame(rows)


# ── Core query function ───────────────────────────────────────────────────────

def _query_scb(table_path: str, query: dict, cache_key: str,
               retries: int = 3) -> pd.DataFrame:
    """
    POST a query to SCB PxWeb API and return a DataFrame.
    Results are cached to CACHE_DIR with key scb_se_{cache_key}.json.

    Parameters
    ----------
    table_path : str
        Path component after SCB_BASE_URL, e.g. "BO/BO0501/BO0501A/FastprisKNRegK"
    query : dict
        PxWeb JSON query body
    cache_key : str
        Unique key for the cache file (becomes scb_se_{cache_key}.json)
    retries : int
        Number of retry attempts on transient failure
    """
    cache = _cache_path(cache_key)

    if _is_cache_valid(cache):
        logger.info(f"[scb] Using cached {cache_key}")
        return pd.read_json(cache, orient="records")

    url = f"{SCB_BASE_URL}/{table_path}"
    logger.info(f"[scb] Fetching {cache_key} from {url}...")

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=query, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            last_exc = e
            if attempt < retries:
                wait = 2 ** attempt
                logger.warning(f"[scb] {cache_key} attempt {attempt} failed: {e} "
                               f"— retrying in {wait}s")
                time.sleep(wait)
    else:
        logger.error(f"[scb] {cache_key} failed after {retries} attempts: {last_exc}")
        return pd.DataFrame()

    if not isinstance(data, dict) or ("id" not in data and "dataset" not in data):
        logger.error(f"[scb] {cache_key}: unexpected response structure")
        return pd.DataFrame()

    df = _parse_jsonstat(data)

    if df.empty:
        logger.warning(f"[scb] {cache_key}: parsed DataFrame is empty")
        return df

    df.to_json(cache, orient="records", force_ascii=False)
    logger.info(f"[scb] {cache_key}: {len(df)} rows cached")

    return df


# ── Specific table fetchers ───────────────────────────────────────────────────

def fetch_housing_prices() -> pd.DataFrame:
    """
    Fetch average housing prices per sqm by municipality from SCB.

    Primary table:  BO/BO0501/BO0501A/FastprisKNRegK
    Fallback table: BO/BO0501/BO0501C/PrisKvN

    Returns
    -------
    DataFrame with columns:
        municipality_code, municipality_name, year,
        avg_price_sqm, price_growth_1y, price_growth_3y, municipality_clean
    """
    primary_path = SCB_TABLES.get("housing_prices", "BO/BO0501/BO0501A/FastprisKNRegK")
    fallback_path = "BO/BO0501/BO0501C/PrisKvN"

    query = {
        "query": [
            {"code": "Region",       "selection": {"filter": "all",  "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["FastprisKvmHela"]}},
            {"code": "Tid",          "selection": {"filter": "top",  "values": ["5"]}},
        ],
        "response": {"format": "json-stat2"},
    }

    df = _query_scb(primary_path, query, "housing_prices")

    # Try fallback if primary fails or is empty
    if df.empty:
        logger.warning("[scb] housing_prices primary table empty — trying fallback path")
        query_fb = {
            "query": [
                {"code": "Region",       "selection": {"filter": "all",  "values": ["*"]}},
                {"code": "ContentsCode", "selection": {"filter": "all",  "values": ["*"]}},
                {"code": "Tid",          "selection": {"filter": "top",  "values": ["5"]}},
            ],
            "response": {"format": "json-stat2"},
        }
        df = _query_scb(fallback_path, query_fb, "housing_prices_fallback")

    if df.empty:
        logger.warning("[scb] fetch_housing_prices: no data available")
        return pd.DataFrame()

    try:
        # Normalise column names — SCB JSON-stat uses dimension names as column headers
        rename = {}
        for col in df.columns:
            col_lower = col.lower()
            if "region" in col_lower and "_code" in col_lower:
                rename[col] = "municipality_code"
            elif "region" in col_lower and "_code" not in col_lower:
                rename[col] = "municipality_name"
            elif "tid" in col_lower and "_code" in col_lower:
                rename[col] = "year"
            elif "tid" in col_lower and "_code" not in col_lower:
                rename[col] = "year_label"
            elif col == "value":
                rename[col] = "avg_price_sqm"
        df = df.rename(columns=rename)

        # Ensure required columns
        for col in ("municipality_code", "municipality_name", "year", "avg_price_sqm"):
            if col not in df.columns:
                # Create from available data as best effort
                if col == "year" and "year_label" in df.columns:
                    df["year"] = df["year_label"].astype(str)
                elif col == "municipality_name" and "municipality_code" in df.columns:
                    df["municipality_name"] = df["municipality_code"]
                else:
                    df[col] = np.nan

        df["avg_price_sqm"] = pd.to_numeric(df["avg_price_sqm"], errors="coerce")
        df = df.dropna(subset=["avg_price_sqm"])
        df["year"] = df["year"].astype(str)

        # Average across housing types if multiple rows per municipality/year
        group_cols = [c for c in ["municipality_code", "municipality_name", "year"]
                      if c in df.columns]
        df = df.groupby(group_cols, as_index=False)["avg_price_sqm"].mean()

        # Clean municipality name for downstream joining
        df["municipality_clean"] = (
            df["municipality_name"]
            .str.strip()
            .str.lower()
        )

        # Compute YoY and 3Y growth per municipality
        df_sorted = df.sort_values(["municipality_code", "year"])
        growth_rows = []
        for muni_code, grp in df_sorted.groupby("municipality_code"):
            grp = grp.sort_values("year").reset_index(drop=True)
            latest_row = grp.iloc[-1].to_dict()
            latest_price = latest_row["avg_price_sqm"]

            # 1-year growth
            if len(grp) >= 2:
                prev_price = grp.iloc[-2]["avg_price_sqm"]
                latest_row["price_growth_1y"] = (
                    (latest_price / prev_price - 1) * 100
                    if prev_price and prev_price > 0 else np.nan
                )
            else:
                latest_row["price_growth_1y"] = np.nan

            # 3-year annualized growth
            if len(grp) >= 4:
                price_3y_ago = grp.iloc[-4]["avg_price_sqm"]
                latest_row["price_growth_3y"] = (
                    ((latest_price / price_3y_ago) ** (1 / 3) - 1) * 100
                    if price_3y_ago and price_3y_ago > 0 else np.nan
                )
            else:
                latest_row["price_growth_3y"] = np.nan

            growth_rows.append(latest_row)

        result = pd.DataFrame(growth_rows)
        out_cols = [c for c in [
            "municipality_code", "municipality_name", "municipality_clean",
            "year", "avg_price_sqm", "price_growth_1y", "price_growth_3y",
        ] if c in result.columns]
        return result[out_cols]

    except Exception as e:
        logger.warning(f"[scb] fetch_housing_prices processing error: {e}")
        return pd.DataFrame()


def fetch_rent_data() -> pd.DataFrame:
    """
    Fetch rental data from SCB and map to our rent_zone system.

    Primary table: BO/BO0202/BO0202B/HyraNytUthB

    On failure or empty result, generates synthetic rent estimates by zone
    using a power-law size-scaling formula.

    Returns
    -------
    DataFrame with columns: rent_zone, sqm, monthly_rent
    """
    table_path = SCB_TABLES.get("rent", "BO/BO0202/BO0202B/HyraNytUthB")
    query = {
        "query": [
            {"code": "Region",       "selection": {"filter": "all", "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Tid",          "selection": {"filter": "top", "values": ["1"]}},
        ],
        "response": {"format": "json-stat2"},
    }

    df = pd.DataFrame()
    try:
        df = _query_scb(table_path, query, "rent")
    except Exception as e:
        logger.warning(f"[scb] fetch_rent_data SCB query failed: {e}")

    if not df.empty:
        # Attempt to parse real data into zone/sqm/monthly_rent structure
        try:
            # SCB rent table may have Region and size dimension
            # Try to map regions to our rent zones
            rename = {}
            for col in df.columns:
                if "region" in col.lower() and "_code" in col.lower():
                    rename[col] = "region_code"
                elif col == "value":
                    rename[col] = "monthly_rent"
            df = df.rename(columns=rename)
            df["monthly_rent"] = pd.to_numeric(df.get("monthly_rent", pd.Series()), errors="coerce")
            df = df.dropna(subset=["monthly_rent"])
            if not df.empty and "region_code" in df.columns:
                # Map SCB region codes to our rent zones (best effort)
                _SCB_REGION_TO_ZONE = {
                    "0180": "1.01",  # Stockholm
                    "0184": "1.02",  # Solna
                    "0183": "1.02",  # Sundbyberg
                    "0126": "1.04",  # Huddinge
                    "1480": "2.01",  # Göteborg
                    "1280": "3.01",  # Malmö
                    "0380": "4.01",  # Uppsala
                }
                df["rent_zone"] = df["region_code"].map(_SCB_REGION_TO_ZONE)
                df = df.dropna(subset=["rent_zone"])
                if not df.empty:
                    df["sqm"] = 60.0  # Approximate if size dim not available
                    return df[["rent_zone", "sqm", "monthly_rent"]].copy()
        except Exception as e:
            logger.warning(f"[scb] fetch_rent_data parsing real data failed: {e} — using synthetic")

    # ── Synthetic rent estimates by zone ─────────────────────────────────────
    logger.info("[scb] Generating synthetic rent estimates by zone")

    ZONE_BASE_RENTS = {
        "1.01": 18000,
        "1.02": 14000,
        "1.03": 12000,
        "1.04": 10500,
        "2.01": 11000,
        "2.02": 9500,
        "2.03": 8500,
        "3.01": 9500,
        "3.02": 8500,
        "3.03": 8000,
        "3.04": 7000,
        "4.01": 9500,
        "5.00": 8500,
        "6.00": 7000,
        "7.00": 5500,
    }

    SIZE_BRACKETS = [30, 45, 60, 75, 90, 105]
    rows = []
    for zone, base_rent in ZONE_BASE_RENTS.items():
        for sqm in SIZE_BRACKETS:
            # Power-law scaling: larger apartments have lower rent/sqm
            rent = base_rent * (sqm / 60) ** 0.75
            rows.append({
                "rent_zone": zone,
                "sqm": float(sqm),
                "monthly_rent": round(rent, 0),
            })

    return pd.DataFrame(rows)


def fetch_population() -> pd.DataFrame:
    """
    Fetch population by municipality from SCB.

    Table: BE/BE0101/BE0101A/BefolkningNy

    Returns
    -------
    DataFrame with columns: municipality_code, population
    """
    table_path = SCB_TABLES.get("population", "BE/BE0101/BE0101A/BefolkningNy")
    query = {
        "query": [
            {"code": "Region",       "selection": {"filter": "all",  "values": ["*"]}},
            {"code": "Kon",          "selection": {"filter": "item", "values": ["1", "2"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["BE0101N1"]}},
            {"code": "Tid",          "selection": {"filter": "top",  "values": ["1"]}},
        ],
        "response": {"format": "json-stat2"},
    }

    try:
        df = _query_scb(table_path, query, "population")
    except Exception as e:
        logger.warning(f"[scb] fetch_population failed: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    try:
        rename = {}
        for col in df.columns:
            if "region" in col.lower() and "_code" in col.lower():
                rename[col] = "municipality_code"
            elif col == "value":
                rename[col] = "population"
        df = df.rename(columns=rename)
        df["population"] = pd.to_numeric(df["population"], errors="coerce")

        # Sum across sexes
        if "municipality_code" in df.columns:
            df = df.groupby("municipality_code", as_index=False)["population"].sum()
        return df[["municipality_code", "population"]].dropna()

    except Exception as e:
        logger.warning(f"[scb] fetch_population processing error: {e}")
        return pd.DataFrame()


def fetch_employment() -> pd.DataFrame:
    """
    Fetch employment by municipality and industry from SCB.

    Table: AM/AM0207/AM0207K/RAKS07KN

    Computes Herfindahl-Hirschman Index (HHI) of employment concentration
    per municipality as a proxy for job market diversity/risk.

    Returns
    -------
    DataFrame with columns: municipality_code, employment_hhi
    """
    table_path = SCB_TABLES.get("employment", "AM/AM0207/AM0207K/RAKS07KN")
    query = {
        "query": [
            {"code": "Region",       "selection": {"filter": "all", "values": ["*"]}},
            {"code": "SNI2007",      "selection": {"filter": "all", "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Tid",          "selection": {"filter": "top", "values": ["1"]}},
        ],
        "response": {"format": "json-stat2"},
    }

    try:
        df = _query_scb(table_path, query, "employment")
    except Exception as e:
        logger.warning(f"[scb] fetch_employment failed: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    try:
        rename = {}
        for col in df.columns:
            if "region" in col.lower() and "_code" in col.lower():
                rename[col] = "municipality_code"
            elif col == "value":
                rename[col] = "employed"
        df = df.rename(columns=rename)
        df["employed"] = pd.to_numeric(df["employed"], errors="coerce")
        df = df.dropna(subset=["employed", "municipality_code"])
        df = df[df["employed"] > 0]

        if df.empty:
            return pd.DataFrame()

        # Compute HHI per municipality
        def _hhi(grp: pd.DataFrame) -> float:
            total = grp["employed"].sum()
            if total <= 0:
                return np.nan
            shares = grp["employed"] / total
            return float((shares ** 2).sum())

        hhi_df = (
            df.groupby("municipality_code")
            .apply(_hhi)
            .reset_index()
            .rename(columns={0: "employment_hhi"})
        )
        return hhi_df

    except Exception as e:
        logger.warning(f"[scb] fetch_employment processing error: {e}")
        return pd.DataFrame()


def fetch_wages() -> pd.DataFrame:
    """
    Fetch average monthly wages by municipality from SCB.

    Table: AM/AM0102/AM0102H/LonKNSektJ

    Returns
    -------
    DataFrame with columns: municipality_code, latest_wage, wage_growth_1y
    """
    table_path = SCB_TABLES.get("wages", "AM/AM0102/AM0102H/LonKNSektJ")
    query = {
        "query": [
            {"code": "Region",       "selection": {"filter": "all",  "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "all",  "values": ["*"]}},
            {"code": "Tid",          "selection": {"filter": "top",  "values": ["2"]}},
        ],
        "response": {"format": "json-stat2"},
    }

    try:
        df = _query_scb(table_path, query, "wages")
    except Exception as e:
        logger.warning(f"[scb] fetch_wages failed: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    try:
        rename = {}
        for col in df.columns:
            if "region" in col.lower() and "_code" in col.lower():
                rename[col] = "municipality_code"
            elif "tid" in col.lower() and "_code" in col.lower():
                rename[col] = "year"
            elif col == "value":
                rename[col] = "wage"
        df = df.rename(columns=rename)
        df["wage"] = pd.to_numeric(df["wage"], errors="coerce")
        df = df.dropna(subset=["wage", "municipality_code"])
        if "year" not in df.columns:
            df["year"] = "latest"

        df = df.sort_values(["municipality_code", "year"])
        result_rows = []
        for muni_code, grp in df.groupby("municipality_code"):
            grp = grp.sort_values("year").reset_index(drop=True)
            latest_wage = float(grp["wage"].iloc[-1])
            wage_growth_1y = np.nan
            if len(grp) >= 2:
                prev_wage = float(grp["wage"].iloc[-2])
                if prev_wage > 0:
                    wage_growth_1y = (latest_wage / prev_wage - 1) * 100
            result_rows.append({
                "municipality_code": muni_code,
                "latest_wage": latest_wage,
                "wage_growth_1y": wage_growth_1y,
            })

        return pd.DataFrame(result_rows)

    except Exception as e:
        logger.warning(f"[scb] fetch_wages processing error: {e}")
        return pd.DataFrame()


def fetch_construction_permits() -> pd.DataFrame:
    """
    Fetch building permits by municipality from SCB.

    Table: BO/BO0701/BO0701A/BygglovBGT

    Returns
    -------
    DataFrame with columns: municipality_code, permits_per_1000, permit_momentum
    """
    table_path = SCB_TABLES.get("construction", "BO/BO0701/BO0701A/BygglovBGT")
    query = {
        "query": [
            {"code": "Region",       "selection": {"filter": "all", "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Tid",          "selection": {"filter": "top", "values": ["4"]}},
        ],
        "response": {"format": "json-stat2"},
    }

    try:
        df = _query_scb(table_path, query, "construction")
    except Exception as e:
        logger.warning(f"[scb] fetch_construction_permits failed: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    try:
        rename = {}
        for col in df.columns:
            if "region" in col.lower() and "_code" in col.lower():
                rename[col] = "municipality_code"
            elif "tid" in col.lower() and "_code" in col.lower():
                rename[col] = "quarter"
            elif col == "value":
                rename[col] = "permits"
        df = df.rename(columns=rename)
        df["permits"] = pd.to_numeric(df["permits"], errors="coerce")
        df = df.dropna(subset=["permits", "municipality_code"])

        if "quarter" not in df.columns:
            df["quarter"] = "Q1"

        df = df.sort_values(["municipality_code", "quarter"])

        result_rows = []
        for muni_code, grp in df.groupby("municipality_code"):
            grp = grp.sort_values("quarter").reset_index(drop=True)
            total = float(grp["permits"].sum())
            # permits_per_1000 placeholder — will be normalised by population in build_region_features
            latest = float(grp["permits"].iloc[-1]) if len(grp) > 0 else np.nan
            prev = float(grp["permits"].iloc[-2]) if len(grp) >= 2 else np.nan
            momentum = (latest / prev - 1) * 100 if prev and prev > 0 else np.nan
            result_rows.append({
                "municipality_code": muni_code,
                "permits_raw": total,
                "permit_momentum": momentum,
            })

        return pd.DataFrame(result_rows)

    except Exception as e:
        logger.warning(f"[scb] fetch_construction_permits processing error: {e}")
        return pd.DataFrame()


# ── Master builder ────────────────────────────────────────────────────────────

def build_region_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch all SCB data and merge into region features DataFrames.

    Returns
    -------
    (region_features, rent_data, index_signals)

    region_features : DataFrame
        One row per municipality with all available signals.
        Columns: municipality_code, municipality_name, municipality_clean,
                 price_growth_1y, price_growth_3y, latest_price_sqm,
                 population, employment_hhi, latest_wage, wage_growth_1y,
                 permits_per_1000, permit_momentum

    rent_data : DataFrame
        Columns: rent_zone, sqm, monthly_rent

    index_signals : DataFrame
        Quarterly price momentum by county.
        Columns: region_code, index_momentum_qoq, volatility_qoq, latest_index
    """
    logger.info("[scb] Building Swedish region features...")

    # ── Fetch all tables (failures return empty DataFrames) ───────────────────
    prices = fetch_housing_prices()
    rent_data = fetch_rent_data()
    population = fetch_population()
    employment = fetch_employment()
    wages = fetch_wages()
    construction = fetch_construction_permits()

    # ── Derive index_signals from housing prices ───────────────────────────────
    # SCB housing prices are annual, so approximate QoQ signals from YoY
    index_signals = pd.DataFrame()
    if not prices.empty and "municipality_code" in prices.columns:
        try:
            # Aggregate to county level (first 2 digits of municipality code)
            prices_idx = prices.copy()
            prices_idx["region_code"] = prices_idx["municipality_code"].str[:2]
            county_agg = (
                prices_idx.groupby("region_code", as_index=False)
                .agg(
                    latest_index=("avg_price_sqm", "mean"),
                    index_momentum_qoq=("price_growth_1y", "mean"),
                    volatility_qoq=("price_growth_1y", "std"),
                )
            )
            # QoQ is approx YoY/4 for index momentum
            county_agg["index_momentum_qoq"] = county_agg["index_momentum_qoq"] / 4
            county_agg["volatility_qoq"] = county_agg["volatility_qoq"].fillna(0) / 4
            index_signals = county_agg
            logger.info(f"[scb] index_signals: {len(index_signals)} county regions")
        except Exception as e:
            logger.warning(f"[scb] index_signals derivation failed: {e}")

    # ── Build base from housing prices ────────────────────────────────────────
    if prices.empty:
        logger.warning("[scb] No housing price data — region_features will be empty")
        return pd.DataFrame(), rent_data, index_signals

    region_features = prices.rename(columns={"avg_price_sqm": "latest_price_sqm"}).copy()

    # ── Join population ───────────────────────────────────────────────────────
    if not population.empty and "municipality_code" in population.columns:
        region_features = region_features.merge(
            population[["municipality_code", "population"]],
            on="municipality_code", how="left",
        )
    else:
        region_features["population"] = np.nan

    # ── Join employment HHI ───────────────────────────────────────────────────
    if not employment.empty and "municipality_code" in employment.columns:
        region_features = region_features.merge(
            employment[["municipality_code", "employment_hhi"]],
            on="municipality_code", how="left",
        )
    else:
        region_features["employment_hhi"] = np.nan

    # ── Join wages ────────────────────────────────────────────────────────────
    if not wages.empty and "municipality_code" in wages.columns:
        region_features = region_features.merge(
            wages[["municipality_code", "latest_wage", "wage_growth_1y"]],
            on="municipality_code", how="left",
        )
    else:
        region_features["latest_wage"] = np.nan
        region_features["wage_growth_1y"] = np.nan

    # ── Join construction ─────────────────────────────────────────────────────
    if not construction.empty and "municipality_code" in construction.columns:
        region_features = region_features.merge(
            construction[["municipality_code", "permits_raw", "permit_momentum"]],
            on="municipality_code", how="left",
        )
        # Normalise by population to get permits_per_1000
        if "population" in region_features.columns:
            pop_valid = region_features["population"].replace(0, np.nan)
            region_features["permits_per_1000"] = (
                region_features["permits_raw"] / pop_valid * 1000
            )
        else:
            region_features["permits_per_1000"] = np.nan
        region_features = region_features.drop(columns=["permits_raw"], errors="ignore")
    else:
        region_features["permits_per_1000"] = np.nan
        region_features["permit_momentum"] = np.nan

    # ── Ensure all expected columns exist ────────────────────────────────────
    expected_cols = [
        "municipality_code", "municipality_name", "municipality_clean",
        "price_growth_1y", "price_growth_3y", "latest_price_sqm",
        "population", "employment_hhi", "latest_wage", "wage_growth_1y",
        "permits_per_1000", "permit_momentum",
    ]
    for col in expected_cols:
        if col not in region_features.columns:
            region_features[col] = np.nan

    region_features = region_features[expected_cols].copy()

    logger.info(f"[scb] Region features built: {len(region_features)} municipalities")
    return region_features, rent_data, index_signals
