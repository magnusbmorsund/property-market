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
    Fetch housing purchase prices by municipality from SCB.

    Table: BO/BO0501/BO0501B/FastprisSHRegionAr
    Variable BO0501C2 = mean purchase price (tkr) for permanent residences (småhus).
    Note: this is total transaction price, not price/sqm — avg_price_sqm is left NaN.
    price_growth_1y and price_growth_3y are derived from the annual price time-series.

    Returns
    -------
    DataFrame with columns: municipality_code, municipality_name, municipality_clean,
                             year, avg_price_sqm (NaN), price_growth_1y, price_growth_3y
    """
    table_path = SCB_TABLES.get("housing_prices", "BO/BO0501/BO0501B/FastprisSHRegionAr")
    query = {
        "query": [
            {"code": "Region",        "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Fastighetstyp", "selection": {"filter": "item", "values": ["220"]}},
            {"code": "ContentsCode",  "selection": {"filter": "item", "values": ["BO0501C2"]}},
            {"code": "Tid",           "selection": {"filter": "top",  "values": ["5"]}},
        ],
        "response": {"format": "json-stat2"},
    }

    try:
        df = _query_scb(table_path, query, "housing_prices")
    except Exception as e:
        logger.warning(f"[scb] fetch_housing_prices failed: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    try:
        rename = {}
        for col in df.columns:
            col_lower = col.lower()
            if "region" in col_lower and "_code" in col_lower:
                rename[col] = "municipality_code"
            elif "region" in col_lower and "_code" not in col_lower:
                rename[col] = "municipality_name"
            elif "tid" in col_lower and "_code" in col_lower:
                rename[col] = "year"
            elif col == "value":
                rename[col] = "avg_price_tkr"
        df = df.rename(columns=rename)

        if "municipality_name" not in df.columns:
            df["municipality_name"] = df.get("municipality_code", "")
        if "year" not in df.columns and "Tid" in df.columns:
            df["year"] = df["Tid"].astype(str)

        df["avg_price_tkr"] = pd.to_numeric(df.get("avg_price_tkr", pd.Series(dtype=float)), errors="coerce")
        df = df.dropna(subset=["avg_price_tkr", "municipality_code"])
        df["year"] = df["year"].astype(str)

        group_cols = [c for c in ["municipality_code", "municipality_name", "year"] if c in df.columns]
        df = df.groupby(group_cols, as_index=False)["avg_price_tkr"].mean()
        df["municipality_clean"] = df["municipality_name"].str.strip().str.lower()

        df_sorted = df.sort_values(["municipality_code", "year"])
        growth_rows = []
        for muni_code, grp in df_sorted.groupby("municipality_code"):
            grp = grp.sort_values("year").reset_index(drop=True)
            latest_row = grp.iloc[-1].to_dict()
            latest_price = latest_row["avg_price_tkr"]

            latest_row["price_growth_1y"] = (
                (latest_price / grp.iloc[-2]["avg_price_tkr"] - 1) * 100
                if len(grp) >= 2 and grp.iloc[-2]["avg_price_tkr"] > 0 else np.nan
            )
            latest_row["price_growth_3y"] = (
                ((latest_price / grp.iloc[-4]["avg_price_tkr"]) ** (1 / 3) - 1) * 100
                if len(grp) >= 4 and grp.iloc[-4]["avg_price_tkr"] > 0 else np.nan
            )
            # avg_price_sqm not available from this table (total price, not per-sqm)
            latest_row["avg_price_sqm"] = np.nan
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
    Fetch median monthly rent per sqm by municipality from SCB.

    Table: BO/BO0406/BO0406E/BO0406Tab01
    Variable Hyresuppg=Mh_kvm: median monthly rent per sqm (SEK).
    Returns plain PxWeb JSON (not JSON-stat2) — parsed directly.

    Builds zone/sqm/monthly_rent table by mapping municipality codes to
    rent zones and expanding across standard sqm brackets.

    Falls back to synthetic estimates if the API call fails.

    Returns
    -------
    DataFrame with columns: rent_zone, sqm, monthly_rent
    """
    table_path = SCB_TABLES.get("rent", "BO/BO0406/BO0406E/BO0406Tab01")
    url = f"{SCB_BASE_URL}/{table_path}"
    query = {
        "query": [
            {"code": "Region",       "selection": {"filter": "all",  "values": ["*"]}},
            {"code": "Hyresuppg",    "selection": {"filter": "item", "values": ["Mh_kvm"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["000000J4"]}},
            {"code": "Tid",          "selection": {"filter": "top",  "values": ["1"]}},
        ],
        "response": {"format": "json"},
    }

    rent_per_sqm: dict[str, float] = {}
    cache = _cache_path("rent")

    if _is_cache_valid(cache):
        logger.info("[scb] Using cached rent")
        try:
            rent_per_sqm = json.load(open(cache))
        except Exception:
            pass
    else:
        try:
            resp = requests.post(url, json=query, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # Plain PxWeb JSON: {"columns": [...], "data": [{"key": [...], "values": [...]}]}
            for row in data.get("data", []):
                region_code = str(row["key"][0]).zfill(4)
                try:
                    val = float(row["values"][0])
                    if val > 0:
                        rent_per_sqm[region_code] = val
                except (ValueError, IndexError):
                    pass
            if rent_per_sqm:
                json.dump(rent_per_sqm, open(cache, "w"))
                logger.info(f"[scb] rent: {len(rent_per_sqm)} municipalities cached")
        except Exception as e:
            logger.warning(f"[scb] fetch_rent_data failed: {e} — using synthetic")

    if rent_per_sqm:
        # Map municipality codes to rent zones
        _MUNI_TO_ZONE = {
            "0180": "1.01", "0181": "1.01", "0182": "1.01", "0183": "1.02",
            "0184": "1.02", "0186": "1.02", "0187": "1.02", "0188": "1.02",
            "0191": "1.03", "0192": "1.03", "0127": "1.03", "0128": "1.03",
            "0126": "1.04", "0136": "1.04", "0138": "1.04", "0139": "1.04",
            "1480": "2.01", "1481": "2.01", "1482": "2.01", "1484": "2.02",
            "1485": "2.02", "1486": "2.02", "1487": "2.02", "1488": "2.03",
            "1280": "3.01", "1281": "3.01", "1282": "3.01", "1283": "3.02",
            "1285": "3.02", "1286": "3.03", "1287": "3.03", "1290": "3.04",
            "0380": "4.01", "0381": "4.01", "0382": "4.01",
        }
        # Average rent/sqm per zone
        zone_sqm_rents: dict[str, list[float]] = {}
        for muni_code, rent_sqm in rent_per_sqm.items():
            zone = _MUNI_TO_ZONE.get(muni_code, "7.00")
            zone_sqm_rents.setdefault(zone, []).append(rent_sqm)

        SIZE_BRACKETS = [30, 45, 60, 75, 90, 105]
        rows = []
        for zone, rents in zone_sqm_rents.items():
            avg_rent_sqm = sum(rents) / len(rents)
            for sqm in SIZE_BRACKETS:
                rows.append({
                    "rent_zone": zone,
                    "sqm": float(sqm),
                    "monthly_rent": round(avg_rent_sqm * sqm, 0),
                })
        if rows:
            return pd.DataFrame(rows)

    # ── Synthetic fallback ────────────────────────────────────────────────────
    logger.info("[scb] Generating synthetic rent estimates by zone")
    ZONE_BASE_RENTS = {
        "1.01": 18000, "1.02": 14000, "1.03": 12000, "1.04": 10500,
        "2.01": 11000, "2.02": 9500,  "2.03": 8500,
        "3.01": 9500,  "3.02": 8500,  "3.03": 8000,  "3.04": 7000,
        "4.01": 9500,  "5.00": 8500,  "6.00": 7000,  "7.00": 5500,
    }
    SIZE_BRACKETS = [30, 45, 60, 75, 90, 105]
    rows = []
    for zone, base_rent in ZONE_BASE_RENTS.items():
        for sqm in SIZE_BRACKETS:
            rows.append({
                "rent_zone": zone,
                "sqm": float(sqm),
                "monthly_rent": round(base_rent * (sqm / 60) ** 0.75, 0),
            })
    return pd.DataFrame(rows)


def fetch_population() -> pd.DataFrame:
    """
    Fetch population by municipality from SCB.

    Table: BE/BE0101/BE0101A/BefolkningNy
    Queried in batches (one sex at a time) to stay under SCB cell limits.

    Returns
    -------
    DataFrame with columns: municipality_code, population
    """
    table_path = "BE/BE0101/BE0101A/BefolkningNy"

    # SCB cell limit: query one sex at a time, one age bracket (0-9) × all municipalities
    # Simpler: request only "0" age (infants) to verify structure, then use sex=1+2
    # Best approach: filter to working-age population (20-64) as a proxy, or
    # use sex='1' only and double it.
    # Easiest: request ContentsCode + latest year, one sex, no age filter (all ages)
    # ~290 municipalities × 100 ages × 1 sex × 4 civil statuses = ~116,000 cells → OK
    # But we need to exclude Civilstand dimension by not specifying it → all 4 values
    # 290 × ~100 ages × 4 civil statuses = ~116,000 → borderline
    # Safe approach: request sex=1 only, all ages, all civil statuses, then double
    query = {
        "query": [
            {"code": "Region",       "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Civilstand",   "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Alder",        "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Kon",          "selection": {"filter": "item", "values": ["1"]}},
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
                rename[col] = "population_half"
        df = df.rename(columns=rename)
        df["population_half"] = pd.to_numeric(df["population_half"], errors="coerce")

        # Sum across ages and civil statuses, then double for both sexes
        if "municipality_code" in df.columns:
            df = df.groupby("municipality_code", as_index=False)["population_half"].sum()
            df["population"] = (df["population_half"] * 2).round(0)
            return df[["municipality_code", "population"]].dropna()
        return pd.DataFrame()

    except Exception as e:
        logger.warning(f"[scb] fetch_population processing error: {e}")
        return pd.DataFrame()


def fetch_employment() -> pd.DataFrame:
    """
    Fetch employment by municipality and industry from SCB.

    Table: AM/AM0207/AM0207Z/NattSni07KonKN
    (Förvärvsarbetande 16-74 år efter region och näring, 2019-2021)

    Computes Herfindahl-Hirschman Index (HHI) of employment concentration
    per municipality as a proxy for job market diversity/risk.

    Returns
    -------
    DataFrame with columns: municipality_code, employment_hhi
    """
    table_path = "AM/AM0207/AM0207Z/NattSni07KonKN"
    query = {
        "query": [
            {"code": "Region",       "selection": {"filter": "all", "values": ["*"]}},
            {"code": "SNI2007",      "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Kon",          "selection": {"filter": "item", "values": ["1", "2"]}},
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
    Fetch mean earned income by municipality from SCB as a wage proxy.

    Table: HE/HE0110/HE0110A/SamForvInk1
    Variable HE0110J7 = mean earned income (tkr/year) for ages 20+, both sexes.
    Converted to monthly SEK for latest_wage; wage_growth_1y from YoY change.

    Returns
    -------
    DataFrame with columns: municipality_code, latest_wage, wage_growth_1y
    """
    table_path = SCB_TABLES.get("wages", "HE/HE0110/HE0110A/SamForvInk1")
    query = {
        "query": [
            {"code": "Region",       "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Kon",          "selection": {"filter": "item",  "values": ["1", "2"]}},
            {"code": "Alder",        "selection": {"filter": "item",  "values": ["tot20+"]}},
            {"code": "Inkomstklass", "selection": {"filter": "item",  "values": ["TOT"]}},
            {"code": "ContentsCode", "selection": {"filter": "item",  "values": ["HE0110J7"]}},
            {"code": "Tid",          "selection": {"filter": "top",   "values": ["3"]}},
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
            col_lower = col.lower()
            if "region" in col_lower and "_code" in col_lower:
                rename[col] = "municipality_code"
            elif "tid" in col_lower and "_code" in col_lower:
                rename[col] = "year"
            elif col == "value":
                rename[col] = "income_tkr"
        df = df.rename(columns=rename)

        df["income_tkr"] = pd.to_numeric(df.get("income_tkr", pd.Series(dtype=float)), errors="coerce")
        df = df.dropna(subset=["income_tkr", "municipality_code"])
        df["year"] = df["year"].astype(str)

        # Average across genders per municipality/year
        df = df.groupby(["municipality_code", "year"], as_index=False)["income_tkr"].mean()

        wage_rows = []
        for muni_code, grp in df.sort_values(["municipality_code", "year"]).groupby("municipality_code"):
            grp = grp.sort_values("year").reset_index(drop=True)
            latest_tkr = grp.iloc[-1]["income_tkr"]
            wage_rows.append({
                "municipality_code": muni_code,
                "latest_wage": latest_tkr * 1000 / 12,  # annual tkr → monthly SEK
                "wage_growth_1y": (
                    (latest_tkr / grp.iloc[-2]["income_tkr"] - 1) * 100
                    if len(grp) >= 2 and grp.iloc[-2]["income_tkr"] > 0 else np.nan
                ),
            })

        return pd.DataFrame(wage_rows)

    except Exception as e:
        logger.warning(f"[scb] fetch_wages processing error: {e}")
        return pd.DataFrame()


def fetch_construction_permits() -> pd.DataFrame:
    """
    Fetch building permits (new apartments) by municipality from SCB.

    Table: BO/BO0101/BO0101C/LagenhetNyKv16
    (Lägenheter i nybyggda hus efter region och hustyp, quarterly)

    Returns
    -------
    DataFrame with columns: municipality_code, permits_raw, permit_momentum
    """
    table_path = "BO/BO0101/BO0101C/LagenhetNyKv16"
    query = {
        "query": [
            {"code": "Region",       "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Hustyp",       "selection": {"filter": "item", "values": ["FLERBO", "SMÅHUS"]}},
            {"code": "ContentsCode", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Tid",          "selection": {"filter": "top", "values": ["8"]}},
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


def fetch_debt_burden() -> pd.DataFrame:
    """
    Fetch household mortgage debt burden proxy by municipality from SCB.

    Table: HE/HE0110/HE0110B/Skatteutrakning
    Variable SREDKAP = tax reduction for capital deficit (mortgage interest deduction).
    Mean deduction value per taxpayer is a reliable proxy for mortgage debt load.
    Normalized to 0-1 scale (min-max across all municipalities).

    Returns
    -------
    DataFrame with columns: municipality_code, debt_burden_pct
    """
    table_path = SCB_TABLES.get("debt_burden", "HE/HE0110/HE0110B/Skatteutrakning")
    query = {
        "query": [
            {"code": "Region",          "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Skatteutrakning", "selection": {"filter": "item", "values": ["SREDKAP"]}},
            {"code": "Kon",             "selection": {"filter": "item", "values": ["1", "2"]}},
            {"code": "ContentsCode",    "selection": {"filter": "item", "values": ["000006VL"]}},
            {"code": "Tid",             "selection": {"filter": "top",  "values": ["1"]}},
        ],
        "response": {"format": "json-stat2"},
    }

    try:
        df = _query_scb(table_path, query, "debt_burden")
    except Exception as e:
        logger.warning(f"[scb] fetch_debt_burden failed: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    try:
        rename = {}
        for col in df.columns:
            if "region" in col.lower() and "_code" in col.lower():
                rename[col] = "municipality_code"
            elif col == "value":
                rename[col] = "deduction_tkr"
        df = df.rename(columns=rename)

        df["deduction_tkr"] = pd.to_numeric(df.get("deduction_tkr", pd.Series(dtype=float)), errors="coerce")
        df = df.dropna(subset=["deduction_tkr", "municipality_code"])

        # Average across genders
        df = df.groupby("municipality_code", as_index=False)["deduction_tkr"].mean()

        # Normalize to 0-1: higher deduction = higher debt burden
        lo, hi = df["deduction_tkr"].min(), df["deduction_tkr"].max()
        df["debt_burden_pct"] = (
            (df["deduction_tkr"] - lo) / (hi - lo) if hi > lo else 0.5
        )

        return df[["municipality_code", "debt_burden_pct"]].dropna()

    except Exception as e:
        logger.warning(f"[scb] fetch_debt_burden processing error: {e}")
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
    debt = fetch_debt_burden()

    # ── Normalise municipality_code to zero-padded 4-char string ─────────────
    # SCB returns codes as integers (180) but the municipality JSON uses "0180"
    def _norm_code(df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and "municipality_code" in df.columns:
            df["municipality_code"] = (
                df["municipality_code"].astype(str).str.zfill(4)
            )
        return df

    prices      = _norm_code(prices)
    population  = _norm_code(population)
    employment  = _norm_code(employment)
    wages       = _norm_code(wages)
    construction = _norm_code(construction)
    debt        = _norm_code(debt)

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

    # ── Build base from municipality list (JSON file) ─────────────────────────
    # Don't depend on housing prices — use the static municipality list as foundation
    import json as _json
    _muni_path = Path(__file__).parent.parent / "mapping" / "se_municipality_codes.json"
    with open(_muni_path, "r", encoding="utf-8") as f:
        _muni_data = _json.load(f)
    base_munis = pd.DataFrame([
        {
            "municipality_code": m["code"],
            "municipality_name": m["name"],
            "municipality_clean": m["name"].strip().lower(),
        }
        for m in _muni_data["municipalities"]
    ])

    # Join housing prices if available
    if not prices.empty and "municipality_code" in prices.columns:
        prices_sub = prices.rename(columns={"avg_price_sqm": "latest_price_sqm"})
        price_cols = [c for c in ["municipality_code", "latest_price_sqm",
                                   "price_growth_1y", "price_growth_3y"] if c in prices_sub.columns]
        base_munis = base_munis.merge(prices_sub[price_cols], on="municipality_code", how="left")
    else:
        base_munis["latest_price_sqm"] = np.nan
        base_munis["price_growth_1y"] = np.nan
        base_munis["price_growth_3y"] = np.nan

    region_features = base_munis

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

    # ── Join debt burden ──────────────────────────────────────────────────────
    if not debt.empty and "municipality_code" in debt.columns:
        region_features = region_features.merge(
            debt[["municipality_code", "debt_burden_pct"]],
            on="municipality_code", how="left",
        )
    else:
        region_features["debt_burden_pct"] = np.nan

    # ── Ensure all expected columns exist ────────────────────────────────────
    expected_cols = [
        "municipality_code", "municipality_name", "municipality_clean",
        "price_growth_1y", "price_growth_3y", "latest_price_sqm",
        "population", "employment_hhi", "latest_wage", "wage_growth_1y",
        "permits_per_1000", "permit_momentum", "debt_burden_pct",
    ]
    for col in expected_cols:
        if col not in region_features.columns:
            region_features[col] = np.nan

    region_features = region_features[expected_cols].copy()

    logger.info(f"[scb] Region features built: {len(region_features)} municipalities")
    return region_features, rent_data, index_signals
