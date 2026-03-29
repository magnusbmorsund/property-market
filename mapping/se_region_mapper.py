"""
mapping/se_region_mapper.py — Map Hemnet.se addresses to SCB municipality codes.

Provides:
  - Address string → municipality_code lookup (fuzzy matching)
  - Municipality → county (region) mapping
  - Municipality → rent_zone mapping
  - Rent estimation by sqm with interpolation

Swedish addresses typically follow one of these formats:
  "Gatan 1, 114 35 Stockholm"
  "Stockholmsvägen 5, Solna"
  "Södermalm, Stockholm"
"""

import difflib
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).parent / "se_municipality_codes.json"


def _load_municipalities() -> dict:
    """Load Swedish municipality lookup data."""
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Pre-build lookup structures on import
_MUNICIPALITIES = _load_municipalities()

# Code → name
_CODE_TO_NAME = {m["code"]: m["name"] for m in _MUNICIPALITIES["municipalities"]}

# Name → code (lowercase for matching)
_NAME_TO_CODE = {m["name"].lower(): m["code"] for m in _MUNICIPALITIES["municipalities"]}

# Code → county (region)
_CODE_TO_COUNTY = {m["code"]: m["county"] for m in _MUNICIPALITIES["municipalities"]}
_CODE_TO_COUNTY_NAME = {m["code"]: m["county_name"] for m in _MUNICIPALITIES["municipalities"]}

# Code → rent_zone
_CODE_TO_ZONE = {m["code"]: m.get("rent_zone", "7.00") for m in _MUNICIPALITIES["municipalities"]}

# Build broad alias dict (municipality names + cleaned variants)
_ALIASES: dict[str, str] = {}
for m in _MUNICIPALITIES["municipalities"]:
    _ALIASES[m["name"].lower()] = m["code"]
    # Also index without special Swedish characters replaced by ASCII
    clean = (
        m["name"].lower()
        .replace("å", "a").replace("ä", "a").replace("ö", "o")
        .replace("-", " ").strip()
    )
    _ALIASES[clean] = m["code"]

# ── Swedish neighbourhood / district → municipality aliases ──────────────────
# Stockholm city districts (all → 0180)
_STOCKHOLM_DISTRICTS = [
    "södermalm", "östermalm", "kungsholmen", "vasastan", "norrmalm",
    "gamla stan", "djurgården", "hammarby sjöstad", "liljeholmen",
    "skärholmen", "farsta", "enskede", "bromma", "spånga", "hässelby",
    "vällingby", "blackeberg", "rågsved", "bandhagen", "bagarmossen",
    "skarpnäck", "årsta", "johanneshov", "hammarby", "sundbyberg s",
    "danderyds sjukhus",
]
for d in _STOCKHOLM_DISTRICTS:
    _ALIASES[d] = "0180"

# Solna / Sundbyberg districts that might appear in addresses
_ALIASES["arenastaden"] = "0184"   # Solna (Friends Arena area)
_ALIASES["hagalund"] = "0184"      # Solna
_ALIASES["huvudsta"] = "0183"      # Sundbyberg

# Göteborg (Gothenburg) city districts (all → 1480)
_GOTHENBURG_DISTRICTS = [
    "hisingen", "majorna", "linnéstaden", "avenyn", "linné",
    "vasastaden", "johanneberg", "guldheden", "örgryte", "härlanda",
    "kortedala", "bergsjön", "angered", "lärjedalen", "tuve",
    "backa", "biskopsgården", "lundby", "kärra", "rödbo",
    "frölunda", "tynnered", "högsbo", "askim", "älvsborg",
    "västra frölunda",
]
for d in _GOTHENBURG_DISTRICTS:
    _ALIASES[d] = "1480"

# Malmö city districts (all → 1280)
_MALMO_DISTRICTS = [
    "husie", "hyllie", "limhamn", "limhamn-bunkeflo", "bunkeflo",
    "kirseberg", "centrum", "fosie", "oxie", "rosengård",
    "södra innerstaden", "västra innerstaden", "husietorp",
]
for d in _MALMO_DISTRICTS:
    _ALIASES[d] = "1280"

# Place names → municipality (suburbs, postal areas, smaller towns that aren't
# municipality names themselves)
_PLACE_ALIASES: dict[str, str] = {
    # Stockholm region
    "kista":          "0180",   # Stockholm (Kista district)
    "spånga":         "0180",   # Stockholm
    "älvsjö":         "0180",   # Stockholm
    "skärholmen":     "0180",   # Stockholm
    "midsommarkransen": "0180", # Stockholm
    "telefonplan":    "0180",   # Stockholm
    "hägersten":      "0180",   # Stockholm
    "bredäng":        "0180",   # Stockholm
    "sätra":          "0180",   # Stockholm
    "vårberg":        "0180",   # Stockholm
    "fittja":         "0127",   # Botkyrka
    "alby":           "0127",   # Botkyrka
    "norsborg":       "0127",   # Botkyrka
    "hallunda":       "0127",   # Botkyrka
    "tumba":          "0127",   # Botkyrka
    "tullinge":       "0127",   # Botkyrka
    "flemingsberg":   "0126",   # Huddinge
    "masmo":          "0126",   # Huddinge
    "skogås":         "0126",   # Huddinge
    "stuvsta":        "0126",   # Huddinge
    "trångsund":      "0126",   # Huddinge
    "jordbro":        "0136",   # Haninge
    "handen":         "0136",   # Haninge
    "brandbergen":    "0136",   # Haninge
    "vendelsö":       "0136",   # Haninge
    "tungelsta":      "0136",   # Haninge
    "tyresö":         "0138",   # Tyresö
    "boo":            "0182",   # Nacka
    "fisksätra":      "0182",   # Nacka
    "saltsjöbaden":   "0182",   # Nacka
    "gustavsberg":    "0120",   # Värmdö
    "ingarö":         "0120",   # Värmdö
    "djurö":          "0120",   # Värmdö
    "täby centrum":   "0160",   # Täby
    "arninge":        "0160",   # Täby
    "roslags näsby":  "0160",   # Täby
    "viggbyholm":     "0160",   # Täby
    "djursholm":      "0162",   # Danderyd
    "stocksund":      "0162",   # Danderyd
    "enebyberg":      "0162",   # Danderyd
    "mörby":          "0162",   # Danderyd
    "sollentuna":     "0163",   # Sollentuna
    "häggvik":        "0163",   # Sollentuna
    "rotebro":        "0163",   # Sollentuna
    "helenelund":     "0163",   # Sollentuna
    "tureberg":       "0163",   # Sollentuna
    "barkarby":       "0123",   # Järfälla
    "jakobsberg":     "0123",   # Järfälla
    "kallhäll":       "0123",   # Järfälla
    "bro":            "0139",   # Upplands-Bro
    "kungsängen":     "0139",   # Upplands-Bro
    "upplands väsby": "0114",   # Upplands Väsby (exact match)
    "väsby":          "0114",   # Upplands Väsby
    "märsta":         "0191",   # Sigtuna
    "arlanda":        "0191",   # Sigtuna

    # Uppsala region
    "knivsta":        "0319",   # Knivsta
    "bålsta":         "0305",   # Håbo

    # Göteborg region
    "mölnlycke":      "1401",   # Härryda
    "landvetter":     "1401",   # Härryda
    "sävedalen":      "1402",   # Partille
    "kållered":       "1481",   # Mölndal
    "kungsbacka":     "1384",   # Kungsbacka
    "lindome":        "1481",   # Mölndal
    "stenungsund":    "1415",   # Stenungsund
    "ytterby":        "1482",   # Kungälv
    "kode":           "1482",   # Kungälv
    "lerum":          "1441",   # Lerum
    "floda":          "1441",   # Lerum
    "nödinge":        "1440",   # Ale
    "skepplanda":     "1440",   # Ale
    "lödöse":         "1462",   # Lilla Edet
    "sjövik":         "1462",   # Lilla Edet

    # Malmö region
    "vellinge":       "1233",   # Vellinge
    "höllviken":      "1233",   # Vellinge
    "skurup":         "1264",   # Skurup
    "staffanstorp":   "1230",   # Staffanstorp
    "kävlinge":       "1261",   # Kävlinge
    "eslöv":          "1285",   # Eslöv
    "höör":           "1267",   # Höör
    "klippan":        "1276",   # Klippan
    "ängelholm":      "1292",   # Ängelholm
    "åstorp":         "1277",   # Åstorp
    "bjuv":           "1260",   # Bjuv
    "helsingborg":    "1283",   # Helsingborg (municipality name match)
    "höganäs":        "1284",   # Höganäs
    "mölle":          "1284",   # Höganäs
    "viken":          "1284",   # Höganäs
    "landskrona":     "1282",   # Landskrona
    "ystad":          "1286",   # Ystad
    "trelleborg":     "1287",   # Trelleborg
    "tomelilla":      "1270",   # Tomelilla
    "sjöbo":          "1265",   # Sjöbo
    "hörby":          "1266",   # Hörby

    # Other large cities
    "linköping":      "0580",   # Linköping
    "norrköping":     "0581",   # Norrköping
    "jönköping":      "0680",   # Jönköping
    "huskvarna":      "0680",   # Jönköping municipality
    "örebro":         "1880",   # Örebro
    "västerås":       "1980",   # Västerås
    "eskilstuna":     "0484",   # Eskilstuna
    "karlstad":       "1780",   # Karlstad
    "sundsvall":      "2281",   # Sundsvall
    "timrå":          "2262",   # Timrå
    "härnösand":      "2280",   # Härnösand
    "gävle":          "2180",   # Gävle
    "sandviken":      "2181",   # Sandviken
    "umeå":           "2480",   # Umeå
    "skellefteå":     "2482",   # Skellefteå
    "luleå":          "2580",   # Luleå
    "östersund":      "2380",   # Östersund
    "falun":          "2080",   # Falun
    "borlänge":       "2081",   # Borlänge
    "borås":          "1490",   # Borås
    "uddevalla":      "1485",   # Uddevalla
    "trollhättan":    "1488",   # Trollhättan
    "skövde":         "1496",   # Skövde
    "nyköping":       "0480",   # Nyköping
    "växjö":          "0780",   # Växjö
    "kalmar":         "0880",   # Kalmar
    "gotland":        "0980",   # Gotland
    "visby":          "0980",   # Gotland
    "karlskrona":     "1080",   # Karlskrona
    "kristianstad":   "1290",   # Kristianstad
    "halmstad":       "1380",   # Halmstad
    "varberg":        "1383",   # Varberg
}
_ALIASES.update(_PLACE_ALIASES)

# All place names available for fuzzy matching
_ALL_NAMES = list(set(list(_NAME_TO_CODE.keys()) + list(_PLACE_ALIASES.keys())))


def map_address_to_municipality(address: str) -> tuple[str, str]:
    """
    Map a Hemnet.se address string to (municipality_code, municipality_name).

    Swedish addresses typically look like:
      "Storgatan 1, 114 35 Stockholm"
      "Vasastan, Stockholm"
      "Lidingövägen 5, Lidingö"

    Strategy:
      1. Exact match on each comma-separated part (right to left)
      2. Postal code area extraction (5-digit Swedish postal codes)
      3. Whole-word substring match across all known names
      4. Fuzzy match on last meaningful address component
      5. Return ("0000", "Okänd") if no match found
    """
    if not address or not isinstance(address, str):
        return ("0000", "Okänd")

    address = address.strip()
    parts = [p.strip() for p in address.split(",")]

    # Strategy 1: exact match on each part (right to left)
    for part in reversed(parts):
        clean = part.strip().lower()
        # Remove Swedish postal codes (5 digits, optionally split "XXX XX")
        clean = re.sub(r"\b\d{3}\s?\d{2}\b", "", clean).strip()
        clean = re.sub(r"\s+", " ", clean).strip()

        if clean in _ALIASES:
            code = _ALIASES[clean]
            return (code, _CODE_TO_NAME.get(code, code))

    # Strategy 2: look for Swedish postal code and infer city from suffix
    addr_lower = address.lower()
    postal_match = re.search(r"\b(\d{3})\s?(\d{2})\b\s+(\w[\w\s-]*)", address)
    if postal_match:
        city_candidate = postal_match.group(3).strip().lower()
        # Remove extra words after the city name
        city_candidate = city_candidate.split(" ")[0]
        if city_candidate in _ALIASES:
            code = _ALIASES[city_candidate]
            return (code, _CODE_TO_NAME.get(code, code))

    # Strategy 3: whole-word substring match (longest match first to avoid false positives)
    for name in sorted(_ALL_NAMES, key=len, reverse=True):
        if len(name) >= 3 and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", addr_lower):
            code = _ALIASES[name]
            return (code, _CODE_TO_NAME.get(code, code))

    # Strategy 4: fuzzy match on the last meaningful part
    for part in reversed(parts):
        clean = re.sub(r"\b\d{3}\s?\d{2}\b", "", part.strip()).strip().lower()
        if len(clean) < 2:
            continue
        matches = difflib.get_close_matches(clean, _ALL_NAMES, n=1, cutoff=0.75)
        if matches:
            code = _ALIASES[matches[0]]
            return (code, _CODE_TO_NAME.get(code, code))

    logger.debug(f"[se_mapper] Could not map address: {address}")
    return ("0000", "Okänd")


def get_county_code(municipality_code: str) -> str:
    """Get the county (län) code for a municipality."""
    return _CODE_TO_COUNTY.get(municipality_code, "00")


def get_county_name(municipality_code: str) -> str:
    """Get the county (län) name for a municipality."""
    return _CODE_TO_COUNTY_NAME.get(municipality_code, "Okänd")


def get_rent_zone(municipality_code: str) -> str:
    """Get the rent zone for a municipality."""
    return _CODE_TO_ZONE.get(municipality_code, "7.00")


def _interpolate_rent_by_sqm(zone_rows: pd.DataFrame, sqm: float) -> float:
    """
    Interpolate monthly rent for a given sqm from zone data.

    zone_rows must have 'sqm' and 'monthly_rent' columns.
    Uses linear interpolation between the two nearest size brackets.
    """
    if zone_rows.empty:
        return np.nan
    if "sqm" not in zone_rows.columns:
        return float(zone_rows["monthly_rent"].iloc[0]) if not zone_rows.empty else np.nan

    zr = zone_rows.dropna(subset=["sqm", "monthly_rent"]).sort_values("sqm")
    if zr.empty:
        return np.nan

    sqm_vals = zr["sqm"].values
    rent_vals = zr["monthly_rent"].values

    if sqm <= sqm_vals[0]:
        return float(rent_vals[0])
    if sqm >= sqm_vals[-1]:
        return float(rent_vals[-1])

    return float(np.interp(sqm, sqm_vals, rent_vals))


def estimate_rent_for_municipality(municipality_code: str,
                                    rent_data: pd.DataFrame,
                                    sqm: float = 65.0) -> float:
    """
    Estimate monthly rent for a property in a given Swedish municipality,
    interpolated by property size (sqm).

    Looks up the municipality's rent_zone and interpolates from rent_data.
    Falls back to median if zone not found.

    Parameters
    ----------
    municipality_code : str
        4-digit Swedish municipality code.
    rent_data : DataFrame
        Must have columns: rent_zone, sqm, monthly_rent.
    sqm : float
        Property size in square metres.
    """
    if rent_data.empty:
        return np.nan

    zone = _CODE_TO_ZONE.get(municipality_code, "7.00")
    zone_rows = rent_data[rent_data["rent_zone"] == zone]

    if not zone_rows.empty:
        return _interpolate_rent_by_sqm(zone_rows, sqm)

    # Fallback: interpolate from median across all zones per size bracket
    if "sqm" in rent_data.columns:
        median_by_sqm = (
            rent_data.groupby("sqm")["monthly_rent"].median().reset_index()
        )
        return _interpolate_rent_by_sqm(median_by_sqm, sqm)

    return float(rent_data["monthly_rent"].median())


def estimate_rents_vectorized(df: pd.DataFrame, rent_data: pd.DataFrame) -> pd.Series:
    """
    Vectorized rent estimation for all listings in df.

    Uses municipality_code → rent_zone lookup, then interpolates by sqm.
    One interpolation call per unique (zone, sqm) pair for efficiency.

    Parameters
    ----------
    df : DataFrame
        Must have 'municipality_code' and 'sqm' columns.
    rent_data : DataFrame
        Must have 'rent_zone', 'sqm', 'monthly_rent' columns.

    Returns
    -------
    pd.Series of estimated monthly rents aligned to df.index.
    """
    if rent_data.empty:
        return pd.Series(np.nan, index=df.index)

    codes = df["municipality_code"].fillna("0000")
    zones = codes.map(_CODE_TO_ZONE).fillna("7.00")
    sqms = df["sqm"].fillna(65.0)

    # Compute rent for each unique (zone, sqm) pair
    unique_pairs = pd.DataFrame({"zone": zones, "sqm": sqms}).drop_duplicates()
    pair_rents: dict[tuple, float] = {}

    for _, pair in unique_pairs.iterrows():
        zone_rows = rent_data[rent_data["rent_zone"] == pair["zone"]]
        if zone_rows.empty:
            if "sqm" in rent_data.columns:
                median_by_sqm = (
                    rent_data.groupby("sqm")["monthly_rent"].median().reset_index()
                )
                rent_val = _interpolate_rent_by_sqm(median_by_sqm, pair["sqm"])
            else:
                rent_val = float(rent_data["monthly_rent"].median())
        else:
            rent_val = _interpolate_rent_by_sqm(zone_rows, pair["sqm"])
        pair_rents[(pair["zone"], pair["sqm"])] = rent_val

    return pd.Series(
        [pair_rents.get((z, s), np.nan) for z, s in zip(zones, sqms)],
        index=df.index,
    )


def enrich_listings_with_regions(listings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Swedish region fields to a DataFrame of Hemnet listings.

    Adds the following columns:
      municipality_code   — 4-digit SCB municipality code
      municipality_name   — Official municipality name (Swedish)
      municipality_clean  — Lowercase stripped name for joining
      region_code         — County (län) code, 2 digits
      region_name         — County name
      rent_zone           — Rent zone string, e.g. "1.01", "3.02", "7.00"

    Parameters
    ----------
    listings_df : DataFrame
        Must have an 'address' column.
    """
    df = listings_df.copy()

    if "address" not in df.columns:
        df["address"] = ""

    # Map each listing's address to municipality (fuzzy, per-row)
    mappings = df["address"].apply(map_address_to_municipality)
    df["municipality_code"] = [m[0] for m in mappings]
    df["municipality_name"] = [m[1] for m in mappings]

    # Vectorized county lookups
    df["region_code"] = df["municipality_code"].map(_CODE_TO_COUNTY).fillna("00")
    df["region_name"] = df["municipality_code"].map(_CODE_TO_COUNTY_NAME).fillna("Okänd")
    df["rent_zone"] = df["municipality_code"].map(_CODE_TO_ZONE).fillna("7.00")
    df["municipality_clean"] = df["municipality_name"].str.strip().str.lower()

    mapped = (df["municipality_code"] != "0000").sum()
    logger.info(f"[se_mapper] Mapped {mapped}/{len(df)} listings to municipalities")

    return df
