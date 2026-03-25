"""
data_sources/ssb.py — SSB (Statistics Norway) API client.

Fetches housing prices, rental data, population, employment, and debt statistics.
All data is cached locally with a configurable TTL.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from config import SSB_BASE_URL, SSB_TABLES, CACHE_DIR, CACHE_TTL_HOURS

logger = logging.getLogger(__name__)


def _cache_path(table_id: str) -> Path:
    return CACHE_DIR / f"ssb_{table_id}.json"


def _is_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_TTL_HOURS


def _query_ssb(table_id: str, query: dict) -> pd.DataFrame:
    """POST a query to SSB API and return a DataFrame."""
    cache = _cache_path(table_id)

    if _is_cache_valid(cache):
        logger.info(f"[ssb] Using cached table {table_id}")
        return pd.read_json(cache, orient="records")

    url = f"{SSB_BASE_URL}/{table_id}"
    logger.info(f"[ssb] Fetching table {table_id}...")

    resp = requests.post(url, json=query, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    df = _parse_jsonstat(data)

    # Cache as records JSON
    df.to_json(cache, orient="records", force_ascii=False)
    logger.info(f"[ssb] Table {table_id}: {len(df)} rows cached")

    return df


def _parse_jsonstat(data: dict) -> pd.DataFrame:
    """Parse JSON-stat2 response from SSB into a flat DataFrame."""
    # SSB returns JSON-stat format
    if "id" not in data:
        # Might be JSON-stat collection
        if "dataset" in data:
            data = data["dataset"]
        elif isinstance(data, list) and len(data) > 0:
            data = data[0]

    dims = data.get("id", [])
    sizes = data.get("size", [])
    categories = data.get("dimension", {})
    values = data.get("value", [])

    if isinstance(values, dict):
        # Sparse format — fill missing with NaN
        max_idx = max(int(k) for k in values.keys()) + 1 if values else 0
        dense_values = [np.nan] * max_idx
        for k, v in values.items():
            dense_values[int(k)] = v
        values = dense_values

    # Build index combinations
    import itertools
    dim_labels = []
    for dim_id in dims:
        cat = categories[dim_id]["category"]
        idx_order = cat.get("index", {})
        labels = cat.get("label", {})

        if isinstance(idx_order, dict):
            # {code: position} — sort by position
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


# ── Specific table fetchers ───────────────────────────────────────────────────

def fetch_price_per_sqm() -> pd.DataFrame:
    """
    Table 06035: Average price per sqm by municipality and housing type.
    Returns last 5 years for growth calculations.
    """
    # Boligtype: 01=Eneboliger, 02=Småhus, 03=Blokkleiligheter (no "00" total)
    # Use all three types and average later, or pick blokkleiligheter for rental properties
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Boligtype", "selection": {"filter": "item", "values": ["01", "02", "03"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["KvPris"]}},
            {"code": "Tid", "selection": {"filter": "top", "values": ["5"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = _query_ssb(SSB_TABLES["price_sqm"], query)

    if df.empty:
        return df

    # Rename columns for clarity
    rename = {}
    for col in df.columns:
        if "Region" in col and "_code" in col:
            rename[col] = "municipality_code"
        elif "Region" in col:
            rename[col] = "municipality"
        elif "Tid" in col and "_code" in col:
            rename[col] = "year"
        elif col == "value":
            rename[col] = "price_sqm"
    df = df.rename(columns=rename)

    # Keep only needed columns, drop NaN prices
    cols = [c for c in ["municipality_code", "municipality", "year", "price_sqm"] if c in df.columns]
    df = df[cols].dropna(subset=["price_sqm"])
    df["price_sqm"] = pd.to_numeric(df["price_sqm"], errors="coerce")
    if "year" in df.columns:
        df["year"] = df["year"].astype(str)

    # Average across housing types per municipality per year
    df = df.groupby(["municipality_code", "municipality", "year"], as_index=False)["price_sqm"].mean()

    # Filter to current municipalities (remove historical entries with year suffixes)
    df = df[~df["municipality"].str.contains(r"\(-\d{4}\)|\(\d{4}-\d{4}\)", regex=True, na=False)]

    # Normalize municipality names for downstream joining
    df["municipality_clean"] = (
        df["municipality"]
        .str.replace(r"\s*\(.*\)$", "", regex=True)   # Remove parenthetical suffixes like "(-2019)"
        .str.replace(r"\s*-\s+\S+$", "", regex=True)  # Remove Sami names like "- Tråante"
        .str.strip()
        .str.lower()
    )

    return df


def fetch_price_index() -> pd.DataFrame:
    """
    Table 07221: Housing price index by region, quarterly.
    Returns last 20 quarters for volatility and growth calculations.
    """
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Boligtype", "selection": {"filter": "item", "values": ["00"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Boligindeks"]}},
            {"code": "Tid", "selection": {"filter": "top", "values": ["20"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = _query_ssb(SSB_TABLES["price_index"], query)

    if df.empty:
        return df

    rename = {}
    for col in df.columns:
        if "Region" in col and "_code" in col:
            rename[col] = "region_code"
        elif "Region" in col:
            rename[col] = "region"
        elif "Tid" in col and "_code" in col:
            rename[col] = "quarter"
        elif col == "value":
            rename[col] = "index_value"
    df = df.rename(columns=rename)

    cols = [c for c in ["region_code", "region", "quarter", "index_value"] if c in df.columns]
    df = df[cols].dropna(subset=["index_value"])
    df["index_value"] = pd.to_numeric(df["index_value"], errors="coerce")

    return df


def fetch_predicted_rent() -> pd.DataFrame:
    """
    Table 09897: Predicted monthly rent by price zone.
    """
    # Soner2: zone codes like "01.01", "01.02", etc.
    # AntRomBRA: room/area combos like "1.030" (1 room, 30sqm), pick a typical 2-room ~50sqm
    query = {
        "query": [
            {"code": "Soner2", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "AntRomBRA", "selection": {"filter": "item", "values": ["2.050"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Husleie"]}},
            {"code": "Tid", "selection": {"filter": "top", "values": ["1"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = _query_ssb(SSB_TABLES["rent"], query)

    if df.empty:
        return df

    rename = {}
    for col in df.columns:
        if "Soner2" in col and "_code" in col:
            rename[col] = "zone_code"
        elif "Soner2" in col:
            rename[col] = "zone"
        elif col == "value":
            rename[col] = "monthly_rent"
    df = df.rename(columns=rename)

    cols = [c for c in ["zone_code", "zone", "monthly_rent"] if c in df.columns]
    df = df[cols].dropna(subset=["monthly_rent"])
    df["monthly_rent"] = pd.to_numeric(df["monthly_rent"], errors="coerce")

    return df


def fetch_population() -> pd.DataFrame:
    """
    Table 07459: Population by municipality.
    Latest year, both sexes, all ages.
    """
    # Query in batches to avoid SSB cell limit (all regions x all ages too large).
    # Fetch one sex at a time, all ages, then sum.
    cache = _cache_path(SSB_TABLES["population"])
    if _is_cache_valid(cache):
        logger.info(f"[ssb] Using cached table {SSB_TABLES['population']}")
        df = pd.read_json(cache, orient="records")
        df["population"] = pd.to_numeric(df.get("population", pd.Series(dtype=float)), errors="coerce")
        return df.dropna(subset=["population"])

    all_dfs = []
    for kjonn in ["1", "2"]:
        query = {
            "query": [
                {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
                {"code": "Kjonn", "selection": {"filter": "item", "values": [kjonn]}},
                {"code": "Alder", "selection": {"filter": "all", "values": ["*"]}},
                {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Personer1"]}},
                {"code": "Tid", "selection": {"filter": "top", "values": ["1"]}},
            ],
            "response": {"format": "json-stat2"},
        }
        url = f"{SSB_BASE_URL}/{SSB_TABLES['population']}"
        logger.info(f"[ssb] Fetching population (Kjonn={kjonn})...")
        resp = requests.post(url, json=query, timeout=120)
        resp.raise_for_status()
        batch_df = _parse_jsonstat(resp.json())
        all_dfs.append(batch_df)

    df = pd.concat(all_dfs, ignore_index=True)

    if df.empty:
        return df

    rename = {}
    for col in df.columns:
        if "Region" in col and "_code" in col:
            rename[col] = "municipality_code"
        elif "Region" in col:
            rename[col] = "municipality"
        elif col == "value":
            rename[col] = "population"
    df = df.rename(columns=rename)

    df["population"] = pd.to_numeric(df.get("population", pd.Series(dtype=float)), errors="coerce")

    # Sum across all ages and both sexes per municipality
    cols = [c for c in ["municipality_code", "municipality"] if c in df.columns]
    df = df.groupby(cols, as_index=False)["population"].sum()
    df = df.dropna(subset=["population"])

    # Cache the summed result
    df.to_json(cache, orient="records", force_ascii=False)
    logger.info(f"[ssb] Population: {len(df)} municipalities cached")

    return df


def fetch_employment() -> pd.DataFrame:
    """
    Table 13472: Employment by municipality and industry (SIC2007).
    Used to compute Herfindahl index (industry concentration).
    """
    # NACE2007: "00-99"=Total, "01-03"=Agriculture, "05-43"=Industry,
    #           "45-82"=Services, "84"=Public admin, etc.
    # Sektor: "ALLE"=All sectors
    # Use industry breakdown (exclude total "00-99") for HHI
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "NACE2007", "selection": {"filter": "item",
                "values": ["01-03", "05-43", "45-82", "84", "85", "86-88", "90-99"]}},
            {"code": "Sektor", "selection": {"filter": "item", "values": ["ALLE"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["SysselEtterBoste"]}},
            {"code": "Tid", "selection": {"filter": "top", "values": ["1"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = _query_ssb(SSB_TABLES["employment"], query)

    if df.empty:
        return df

    rename = {}
    for col in df.columns:
        if "Region" in col and "_code" in col:
            rename[col] = "municipality_code"
        elif "Region" in col:
            rename[col] = "municipality"
        elif "NACE" in col and "_code" in col:
            rename[col] = "industry_code"
        elif "NACE" in col:
            rename[col] = "industry"
        elif col == "value":
            rename[col] = "employed"
    df = df.rename(columns=rename)

    cols = [c for c in ["municipality_code", "municipality", "industry_code", "industry", "employed"]
            if c in df.columns]
    df = df[cols]
    df["employed"] = pd.to_numeric(df.get("employed", pd.Series(dtype=float)), errors="coerce")

    return df


def fetch_debt_burden() -> pd.DataFrame:
    """
    Table 08781: Share of households with debt > 3x gross income, by municipality.
    """
    # GjeldStor: "06" = "Gjeld større enn 3 gonger samla inntekt" (debt > 3x income)
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "GjeldStor", "selection": {"filter": "item", "values": ["06"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["ProsentHush"]}},
            {"code": "Tid", "selection": {"filter": "top", "values": ["1"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = _query_ssb(SSB_TABLES["debt_burden"], query)

    if df.empty:
        return df

    rename = {}
    for col in df.columns:
        if "Region" in col and "_code" in col:
            rename[col] = "municipality_code"
        elif "Region" in col:
            rename[col] = "municipality"
        elif col == "value":
            rename[col] = "debt_burden_pct"
    df = df.rename(columns=rename)

    cols = [c for c in ["municipality_code", "municipality", "debt_burden_pct"] if c in df.columns]
    df = df[cols].dropna(subset=["debt_burden_pct"])
    df["debt_burden_pct"] = pd.to_numeric(df["debt_burden_pct"], errors="coerce")

    return df


def fetch_mortgage_rates() -> pd.DataFrame:
    """
    Table 10748: Mortgage interest rates, monthly.
    """
    # Utlanstype: "04"=Nedbetalingslån (repayment loans)
    # Rentebinding: "99"=Alle (all), "08"=Flytende rente (floating)
    # Sektor: "04b"=Husholdninger (households)
    query = {
        "query": [
            {"code": "Utlanstype", "selection": {"filter": "item", "values": ["04"]}},
            {"code": "Sektor", "selection": {"filter": "item", "values": ["04b"]}},
            {"code": "Rentebinding", "selection": {"filter": "item", "values": ["08"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["RenterNyeBolig"]}},
            {"code": "Tid", "selection": {"filter": "top", "values": ["24"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = _query_ssb(SSB_TABLES["mortgage_rate"], query)

    if df.empty:
        return df

    rename = {}
    for col in df.columns:
        if "Tid" in col and "_code" in col:
            rename[col] = "period"
        elif col == "value":
            rename[col] = "mortgage_rate"
    df = df.rename(columns=rename)

    cols = [c for c in ["period", "mortgage_rate"] if c in df.columns]
    df = df[cols].dropna(subset=["mortgage_rate"])
    df["mortgage_rate"] = pd.to_numeric(df["mortgage_rate"], errors="coerce")

    return df


def fetch_wages() -> pd.DataFrame:
    """
    Table 12852: Monthly earnings by municipality.
    Mean wages, both sexes, all ages, full-time, residence-based.
    Returns last 5 years for growth calculation.
    """
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "MaaleMetode", "selection": {"filter": "item", "values": ["02"]}},
            {"code": "ArbBostedRegion", "selection": {"filter": "item", "values": ["2"]}},
            {"code": "Kjonn", "selection": {"filter": "item", "values": ["0"]}},
            {"code": "Alder", "selection": {"filter": "item", "values": ["Ialt"]}},
            {"code": "AvtaltVanlig", "selection": {"filter": "item", "values": ["5"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Manedslonn"]}},
            {"code": "Tid", "selection": {"filter": "top", "values": ["5"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = _query_ssb(SSB_TABLES["wages"], query)

    if df.empty:
        return df

    # Rename — be specific to avoid matching ArbBostedRegion
    df = df.rename(columns={
        "Region_code": "municipality_code",
        "Region": "municipality",
        "Tid_code": "year",
        "value": "monthly_wage",
    })

    cols = [c for c in ["municipality_code", "municipality", "year", "monthly_wage"] if c in df.columns]
    df = df[cols].dropna(subset=["monthly_wage"])
    df["monthly_wage"] = pd.to_numeric(df["monthly_wage"], errors="coerce")

    # Filter to current municipalities
    df = df[~df["municipality"].str.contains(r"\(-\d{4}\)|\(\d{4}-\d{4}\)", regex=True, na=False)]
    df["municipality_clean"] = (
        df["municipality"]
        .str.replace(r"\s*\(.*\)$", "", regex=True)
        .str.replace(r"\s*-\s+\S+$", "", regex=True)
        .str.strip()
        .str.lower()
    )

    return df


def fetch_construction() -> pd.DataFrame:
    """
    Table 05889: Building permits (igangsatte boliger) by municipality, quarterly.
    Returns last 8 quarters for trend calculation.
    """
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Byggeareal", "selection": {"filter": "item", "values": [
                "111", "112", "131", "141", "142", "143"
            ]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Igangsatte"]}},
            {"code": "Tid", "selection": {"filter": "top", "values": ["8"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = _query_ssb(SSB_TABLES["construction"], query)

    if df.empty:
        return df

    rename = {}
    for col in df.columns:
        if "Region" in col and "_code" in col:
            rename[col] = "municipality_code"
        elif "Region" in col:
            rename[col] = "municipality"
        elif "Tid" in col and "_code" in col:
            rename[col] = "quarter"
        elif col == "value":
            rename[col] = "permits"
    df = df.rename(columns=rename)

    cols = [c for c in ["municipality_code", "municipality", "quarter", "permits"] if c in df.columns]
    df = df[cols]
    df["permits"] = pd.to_numeric(df.get("permits", pd.Series(dtype=float)), errors="coerce")

    # Sum across building types per municipality per quarter
    group_cols = [c for c in ["municipality_code", "municipality", "quarter"] if c in df.columns]
    df = df.groupby(group_cols, as_index=False)["permits"].sum()

    # Filter to current municipalities
    df = df[~df["municipality"].str.contains(r"\(-\d{4}\)|\(\d{4}-\d{4}\)", regex=True, na=False)]
    df["municipality_clean"] = (
        df["municipality"]
        .str.replace(r"\s*\(.*\)$", "", regex=True)
        .str.replace(r"\s*-\s+\S+$", "", regex=True)
        .str.strip()
        .str.lower()
    )

    return df


# ── Region feature assembly ──────────────────────────────────────────────────

def _compute_price_growth(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 1Y and 3Y price growth per municipality from price_sqm data."""
    if price_df.empty or "year" not in price_df.columns:
        return pd.DataFrame(columns=["municipality_code", "price_growth_1y", "price_growth_3y",
                                     "latest_price_sqm"])

    price_df = price_df.sort_values("year")
    years = sorted(price_df["year"].unique())

    results = []
    for muni, grp in price_df.groupby("municipality_code"):
        grp = grp.sort_values("year")
        if len(grp) < 2:
            continue

        latest_price = grp["price_sqm"].iloc[-1]
        latest_year = grp["year"].iloc[-1]

        # 1Y growth
        prev_year = grp[grp["year"] < latest_year]
        if len(prev_year) > 0:
            growth_1y = (latest_price / prev_year["price_sqm"].iloc[-1] - 1) * 100
        else:
            growth_1y = np.nan

        # 3Y growth (annualized)
        three_years_back = grp.head(1) if len(grp) >= 4 else None
        if three_years_back is not None and len(grp) >= 4:
            base_price = grp["price_sqm"].iloc[-4]
            if base_price > 0:
                growth_3y = ((latest_price / base_price) ** (1/3) - 1) * 100
            else:
                growth_3y = np.nan
        else:
            growth_3y = np.nan

        # Get clean name for name-based joining
        muni_clean = grp["municipality_clean"].iloc[-1] if "municipality_clean" in grp.columns else ""

        results.append({
            "municipality_code": muni,
            "municipality_clean": muni_clean,
            "price_growth_1y": round(growth_1y, 2) if not np.isnan(growth_1y) else np.nan,
            "price_growth_3y": round(growth_3y, 2) if not np.isnan(growth_3y) else np.nan,
            "latest_price_sqm": latest_price,
        })

    return pd.DataFrame(results)


def _compute_wage_growth(wage_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 1Y and 3Y wage growth per municipality."""
    if wage_df.empty or "year" not in wage_df.columns:
        return pd.DataFrame(columns=["municipality_clean", "wage_growth_1y", "wage_growth_3y",
                                     "latest_wage"])

    results = []
    for muni, grp in wage_df.groupby("municipality_clean"):
        grp = grp.sort_values("year")
        if len(grp) < 2:
            continue

        latest = grp["monthly_wage"].iloc[-1]
        if pd.isna(latest) or latest <= 0:
            continue

        # 1Y growth
        prev = grp["monthly_wage"].iloc[-2]
        growth_1y = (latest / prev - 1) * 100 if prev > 0 else np.nan

        # 3Y annualized
        if len(grp) >= 4:
            base = grp["monthly_wage"].iloc[-4]
            growth_3y = ((latest / base) ** (1/3) - 1) * 100 if base > 0 else np.nan
        else:
            growth_3y = np.nan

        results.append({
            "municipality_clean": muni,
            "wage_growth_1y": round(growth_1y, 2) if not np.isnan(growth_1y) else np.nan,
            "wage_growth_3y": round(growth_3y, 2) if not np.isnan(growth_3y) else np.nan,
            "latest_wage": latest,
        })

    return pd.DataFrame(results)


def _compute_construction_pressure(constr_df: pd.DataFrame,
                                    pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute construction supply pressure per municipality.
    Permits per 1000 inhabitants = supply pressure signal.
    Also compute permit momentum (recent vs prior period).
    """
    if constr_df.empty:
        return pd.DataFrame(columns=["municipality_clean", "permits_per_1000",
                                     "permit_momentum"])

    constr_df = constr_df.sort_values("quarter")
    quarters = sorted(constr_df["quarter"].unique())

    results = []
    for muni, grp in constr_df.groupby("municipality_clean"):
        grp = grp.sort_values("quarter")
        total_permits = grp["permits"].sum()

        # Recent 4Q vs prior 4Q momentum
        if len(quarters) >= 8:
            recent_qs = quarters[-4:]
            prior_qs = quarters[-8:-4]
            recent = grp[grp["quarter"].isin(recent_qs)]["permits"].sum()
            prior = grp[grp["quarter"].isin(prior_qs)]["permits"].sum()
            momentum = ((recent / prior - 1) * 100) if prior > 0 else np.nan
        else:
            recent = total_permits
            momentum = np.nan

        results.append({
            "municipality_clean": muni,
            "total_permits_2y": total_permits,
            "recent_permits_1y": recent if len(quarters) >= 4 else total_permits,
            "permit_momentum": round(momentum, 1) if momentum is not None and not np.isnan(momentum) else np.nan,
        })

    df = pd.DataFrame(results)

    # Join population to compute permits per 1000 inhabitants
    if not pop_df.empty:
        pop_clean = pop_df.copy()
        if "municipality_clean" not in pop_clean.columns and "municipality" in pop_clean.columns:
            pop_clean["municipality_clean"] = (
                pop_clean["municipality"]
                .str.replace(r"\s*\(.*\)$", "", regex=True)
                .str.replace(r"\s*-\s+\S+$", "", regex=True)
                .str.strip()
                .str.lower()
            )
        if "municipality_clean" in pop_clean.columns:
            pop_lookup = pop_clean.drop_duplicates("municipality_clean")[["municipality_clean", "population"]]
            df = df.merge(pop_lookup, on="municipality_clean", how="left")
            df["permits_per_1000"] = (
                df["recent_permits_1y"] / (df["population"] / 1000)
            ).replace([np.inf, -np.inf], np.nan).round(2)
        else:
            df["permits_per_1000"] = np.nan
    else:
        df["permits_per_1000"] = np.nan

    return df[["municipality_clean", "permits_per_1000", "permit_momentum"]]


def _compute_index_signals(index_df: pd.DataFrame) -> pd.DataFrame:
    """Compute price momentum and volatility from quarterly index data."""
    if index_df.empty:
        return pd.DataFrame(columns=["region_code", "index_momentum_qoq", "volatility_qoq"])

    results = []
    for region, grp in index_df.groupby("region_code"):
        grp = grp.sort_values("quarter")
        if len(grp) < 2:
            continue

        values = grp["index_value"].values
        # QoQ growth rates
        qoq = np.diff(values) / values[:-1] * 100

        results.append({
            "region_code": region,
            "region": grp["region"].iloc[-1],
            "index_momentum_qoq": round(float(qoq[-1]), 2) if len(qoq) > 0 else np.nan,
            "volatility_qoq": round(float(np.std(qoq)), 2) if len(qoq) > 1 else np.nan,
            "latest_index": float(values[-1]),
        })

    return pd.DataFrame(results)


def _compute_employment_hhi(emp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Herfindahl-Hirschman Index of industry concentration per municipality."""
    if emp_df.empty:
        return pd.DataFrame(columns=["municipality_code", "employment_hhi", "top3_industry_share"])

    # Filter out total/aggregate rows (industry_code "00-99" or similar)
    if "industry_code" in emp_df.columns:
        emp_df = emp_df[~emp_df["industry_code"].isin(["00-99", "00", "ALLE"])]

    results = []
    for muni, grp in emp_df.groupby("municipality_code"):
        grp = grp.dropna(subset=["employed"])
        total = grp["employed"].sum()
        if total <= 0:
            continue

        shares = (grp["employed"] / total)
        hhi = float((shares ** 2).sum())
        top3 = float(shares.nlargest(3).sum()) * 100

        results.append({
            "municipality_code": muni,
            "employment_hhi": round(hhi, 4),
            "top3_industry_share": round(top3, 1),
        })

    return pd.DataFrame(results)


def build_region_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch all SSB data and assemble into a single wide DataFrame
    keyed on municipality_code.

    Returns (features_df, rent_df, index_signals_df)
    """
    logger.info("[ssb] Building region features...")

    # Fetch all tables
    price_df = fetch_price_per_sqm()
    index_df = fetch_price_index()
    rent_df = fetch_predicted_rent()
    pop_df = fetch_population()
    emp_df = fetch_employment()
    debt_df = fetch_debt_burden()
    mortgage_df = fetch_mortgage_rates()
    wage_df = fetch_wages()
    constr_df = fetch_construction()

    # Compute derived signals
    price_growth = _compute_price_growth(price_df)
    index_signals = _compute_index_signals(index_df)
    emp_hhi = _compute_employment_hhi(emp_df)
    wage_growth = _compute_wage_growth(wage_df)
    construction = _compute_construction_pressure(constr_df, pop_df)

    # Start with price growth (municipality level)
    features = price_growth.copy()

    # Merge population
    if not pop_df.empty and "municipality_code" in pop_df.columns:
        features = features.merge(
            pop_df[["municipality_code", "population"]],
            on="municipality_code", how="left"
        )

    # Merge debt burden
    if not debt_df.empty and "municipality_code" in debt_df.columns:
        features = features.merge(
            debt_df[["municipality_code", "debt_burden_pct"]],
            on="municipality_code", how="left"
        )

    # Merge employment HHI
    if not emp_hhi.empty:
        features = features.merge(emp_hhi, on="municipality_code", how="left")

    # Mortgage rate — national level, add as a constant column
    if not mortgage_df.empty:
        latest_rate = mortgage_df["mortgage_rate"].iloc[-1]
        features["mortgage_rate_current"] = latest_rate
        if len(mortgage_df) >= 12:
            rate_12m_ago = mortgage_df["mortgage_rate"].iloc[-12]
            features["mortgage_rate_trend"] = latest_rate - rate_12m_ago
        else:
            features["mortgage_rate_trend"] = np.nan
    else:
        features["mortgage_rate_current"] = np.nan
        features["mortgage_rate_trend"] = np.nan

    # Ensure municipality_clean is present before merging wage/construction
    if "municipality_clean" not in features.columns:
        name_lookup = price_df.drop_duplicates("municipality_code")[["municipality_code", "municipality_clean"]]
        features = features.merge(name_lookup, on="municipality_code", how="left")

    # Merge wage growth (name-based — SSB codes may differ)
    if not wage_growth.empty:
        features = features.merge(wage_growth, on="municipality_clean", how="left")

    # Merge construction pressure (name-based)
    if not construction.empty:
        features = features.merge(construction, on="municipality_clean", how="left")

    # Ensure municipality_clean is present for name-based joining
    if "municipality_clean" not in features.columns:
        # Build from municipality names in price_df
        name_lookup = price_df.drop_duplicates("municipality_code")[["municipality_code", "municipality_clean"]]
        features = features.merge(name_lookup, on="municipality_code", how="left")

    logger.info(f"[ssb] Region features: {len(features)} municipalities, "
                f"{len(features.columns)} columns")
    return features, rent_df, index_signals
