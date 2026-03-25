"""
mapping/region_mapper.py — Map Finn.no addresses to SSB municipality/region codes.

Provides:
  - Address string → municipality_code lookup (fuzzy matching)
  - Municipality → region (fylke) mapping
  - Municipality → SSB price zone mapping (for rent estimation)
"""

import difflib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).parent / "municipality_codes.json"


def _load_municipalities() -> dict:
    """Load municipality lookup data."""
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Pre-build lookup structures on import
_MUNICIPALITIES = _load_municipalities()

# Code → name
_CODE_TO_NAME = {m["code"]: m["name"] for m in _MUNICIPALITIES["municipalities"]}

# Name → code (lowercase for matching)
_NAME_TO_CODE = {m["name"].lower(): m["code"] for m in _MUNICIPALITIES["municipalities"]}

# Also index by common aliases
_ALIASES = {}
for m in _MUNICIPALITIES["municipalities"]:
    _ALIASES[m["name"].lower()] = m["code"]
    # Add without special chars
    clean = re.sub(r"[^a-zæøå0-9 ]", "", m["name"].lower()).strip()
    _ALIASES[clean] = m["code"]

# Place names that aren't municipality names but map to known municipalities
_PLACE_ALIASES = {
    "rørvik": "5059",      # Nærøysund
    "kolvereid": "5059",   # Nærøysund
    "jessheim": "3209",    # Ullensaker
    "ski": "3207",         # Nordre Follo
    "sandvika": "3201",    # Bærum
    "mjøndalen": "3301",   # Drammen
    "hønefoss": "3305",    # Ringerike
    "mo i rana": "1833",   # Rana
    "harstad": "5503",     # Harstad
    "sortland": "1870",    # Sortland
    "finnsnes": "5526",    # Senja
    "alta": "5601",        # Alta
    "hammerfest": "5603",  # Hammerfest
    "kirkenes": "5605",    # Sør-Varanger
    "bryne": "1121",       # Time
    "mandal": "4062",      # Lindesnes
    "porsgrunn": "3901",   # Porsgrunn
    "skien": "3903",       # Skien
    "moss": "3103",        # Moss
    "fredrikstad": "3107", # Fredrikstad
    "sarpsborg": "3105",   # Sarpsborg
    "halden": "3101",      # Halden
    "førde": "4237",       # Sunnfjord
    "sogndal": "4230",     # Sogndal
    "stord": "4203",       # Stord
    "voss": "4208",        # Voss herad — mapped to Ullensvang area
    "odda": "4208",        # Ullensvang
    # Smaller towns / neighbourhoods
    "hokksund": "3314",    # Øvre Eiker
    "mysen": "3118",       # Indre Østfold
    "askim": "3118",       # Indre Østfold
    "fosnavåg": "1515",    # Herøy (Møre og Romsdal)
    "søgne": "4001",       # Kristiansand
    "ellingsøy": "1507",   # Ålesund
    "fyllingsdalen": "4201", # Bergen
    "førresfjorden": "1146", # Tysvær
    "fosslandsosen": "5054", # Heim
    "flekkerøy": "4001",  # Kristiansand
    "randesund": "4001",   # Kristiansand
    "vågsbygd": "4001",   # Kristiansand
    "lund": "4001",        # Kristiansand (bydel)
    "ågotnes": "4631",     # Øygarden
    "straume": "4631",     # Øygarden
    "sotra": "4631",       # Øygarden
    "askøy": "4218",       # Askøy
    "kleppestø": "4218",   # Askøy
    "arna": "4201",        # Bergen
    "fana": "4201",        # Bergen
    "åsane": "4201",       # Bergen
    "ytrebygda": "4201",   # Bergen
    "laksevåg": "4201",    # Bergen
    "nesttun": "4201",     # Bergen
    "råholt": "3209",      # Ullensaker
    "kløfta": "3209",      # Ullensaker
    "ås": "3218",          # Ås
    "drøbak": "3216",      # Frogn
    "langhus": "3207",     # Nordre Follo
    "kolbotn": "3207",     # Nordre Follo
    "lørenskog": "3211",   # Lørenskog
    "fjellhamar": "3211",  # Lørenskog
    "rælingen": "3213",    # Rælingen
    "nittedal": "3233",    # Nittedal
    "nesoddtangen": "3214", # Nesodden
    "vinterbro": "3218",   # Ås
    "heimdal": "5001",     # Trondheim
    "byåsen": "5001",      # Trondheim
    "tiller": "5001",      # Trondheim
    "ranheim": "5001",     # Trondheim
    "lade": "5001",        # Trondheim
    "saupstad": "5001",    # Trondheim
    "moholt": "5001",      # Trondheim
    "lambertseter": "0301", # Oslo
    "grorud": "0301",      # Oslo
    "stovner": "0301",     # Oslo
    "furuset": "0301",     # Oslo
    "manglerud": "0301",   # Oslo
    "nordstrand": "0301",  # Oslo
    "oppsal": "0301",      # Oslo
    "bøler": "0301",       # Oslo
    "holmlia": "0301",     # Oslo
    "mortensrud": "0301",  # Oslo
    "tåsen": "0301",       # Oslo
    "majorstuen": "0301",  # Oslo
    "grünerløkka": "0301", # Oslo
    "frogner": "0301",     # Oslo
    "sagene": "0301",      # Oslo
    "ullern": "0301",      # Oslo
    "hundvåg": "1103",     # Stavanger
    "storhaug": "1103",    # Stavanger
    "hillevåg": "1103",    # Stavanger
    "tasta": "1103",       # Stavanger
    "madla": "1103",       # Stavanger
    "hinna": "1103",       # Stavanger
    "tananger": "1124",    # Sola
    "sola": "1124",        # Sola
    "randaberg": "1127",   # Randaberg
    "kopervik": "1149",    # Karmøy
    "haugesund": "1106",   # Haugesund
    "lyngdal": "4225",     # Lyngdal
    "farsund": "4206",     # Farsund
    "vennesla": "4223",    # Vennesla
    "songdalen": "4001",   # Kristiansand
    "lillesand": "4215",   # Lillesand
    "tvedestrand": "4213", # Tvedestrand
    "risør": "4211",       # Risør
    "notodden": "3808",    # Notodden
    "rjukan": "3812",      # Tinn
    "bø": "3817",          # Bø
    "horten": "3801",      # Horten
    "holmestrand": "3802", # Holmestrand
    "stavern": "3805",     # Larvik
    "stathelle": "3814",   # Bamble
    "brevik": "3901",      # Porsgrunn
    "langesund": "3814",   # Bamble
    "kongsvinger": "3401",  # Kongsvinger
    "moelv": "3412",       # Ringsaker
    "brumunddal": "3412",  # Ringsaker
    "otta": "3434",        # Sel
    "vinstra": "3435",     # Nord-Fron
    "fagernes": "3449",    # Nord-Aurdal
    "leira": "3441",       # Lom
    "dombås": "3432",      # Dovre
    "raufoss": "3415",     # Vestre Toten
    "gran": "3446",        # Gran
    "jevnaker": "3053",    # Jevnaker
    "lunner": "3054",      # Lunner
    "flå": "3313",         # Flå
    "gol": "3340",         # Gol
    "hemsedal": "3338",    # Hemsedal
    "ål": "3336",          # Ål
    "geilo": "3336",       # Ål
    "nesbyen": "3316",     # Nesbyen (Nes)
}
_ALIASES.update(_PLACE_ALIASES)

# Code → region (fylke)
_CODE_TO_REGION = {m["code"]: m["region_code"] for m in _MUNICIPALITIES["municipalities"]}
_CODE_TO_REGION_NAME = {m["code"]: m["region_name"] for m in _MUNICIPALITIES["municipalities"]}

# Code → SSB price zone (for rent estimation)
_CODE_TO_ZONE = {m["code"]: m.get("rent_zone", "99") for m in _MUNICIPALITIES["municipalities"]}

# All municipality and place names for fuzzy matching
_ALL_NAMES = list(set(list(_NAME_TO_CODE.keys()) + list(_PLACE_ALIASES.keys())))


def map_address_to_municipality(address: str) -> tuple[str, str]:
    """
    Map a Finn.no address string to (municipality_code, municipality_name).

    The address typically looks like:
      "Storgata 1, 0182 Oslo" or "Bergen, Hordaland" or just "Tromsø"

    Strategy:
      1. Try exact match on last component of comma-separated address
      2. Try matching postal code patterns (4-digit → municipality lookup)
      3. Fuzzy match against all municipality names
      4. Return ("0000", "Ukjent") if no match
    """
    if not address or not isinstance(address, str):
        return ("0000", "Ukjent")

    address = address.strip()

    # Strategy 1: Try each comma-separated part (right to left — municipality usually last)
    parts = [p.strip() for p in address.split(",")]
    for part in reversed(parts):
        # Clean the part
        clean = part.strip().lower()
        clean = re.sub(r"\d{4}\s*", "", clean).strip()  # remove postal codes
        clean = re.sub(r"\s+", " ", clean).strip()

        if clean in _ALIASES:
            code = _ALIASES[clean]
            return (code, _CODE_TO_NAME[code])

    # Strategy 2: Check if any known municipality name appears as a whole word
    addr_lower = address.lower()
    # Sort by name length descending to match longest first (e.g., "Oslo" before "Os")
    for name in sorted(_ALL_NAMES, key=len, reverse=True):
        if len(name) >= 3 and re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', addr_lower):
            code = _ALIASES[name]
            return (code, _CODE_TO_NAME[code])

    # Strategy 3: Fuzzy match on the last meaningful part
    for part in reversed(parts):
        clean = re.sub(r"\d{4}\s*", "", part.strip()).strip().lower()
        if len(clean) < 2:
            continue
        matches = difflib.get_close_matches(clean, _ALL_NAMES, n=1, cutoff=0.7)
        if matches:
            code = _ALIASES[matches[0]]
            return (code, _CODE_TO_NAME[code])

    logger.debug(f"[mapper] Could not map address: {address}")
    return ("0000", "Ukjent")


def get_region_code(municipality_code: str) -> str:
    """Get the fylke (region) code for a municipality."""
    return _CODE_TO_REGION.get(municipality_code, "00")


def get_region_name(municipality_code: str) -> str:
    """Get the fylke (region) name for a municipality."""
    return _CODE_TO_REGION_NAME.get(municipality_code, "Ukjent")


def get_rent_zone(municipality_code: str) -> str:
    """Get the SSB rent price zone for a municipality."""
    return _CODE_TO_ZONE.get(municipality_code, "99")


# SSB rent zone mapping: municipality_code → SSB zone_code
# Based on actual SSB table 09897 zone descriptions
_MUNICIPALITY_TO_SSB_ZONE = {
    # Oslo zones (1.01-1.04) — all map to Oslo municipality 0301
    "0301": "1.01",  # Default Oslo to central zone; individual listings can't be sub-zoned

    # Akershus zones
    "3201": "2.01",  # Bærum — near Oslo
    "3203": "2.01",  # Asker — near Oslo
    "3205": "2.01",  # Lillestrøm — near Oslo
    "3207": "2.01",  # Nordre Follo — near Oslo
    "3209": "2.01",  # Lørenskog — near Oslo
    "3211": "2.01",  # Rælingen — near Oslo
    "3213": "2.02",  # Nesodden — outer Akershus
    "3215": "2.02",  # Frogn
    "3217": "2.02",  # Vestby
    "3219": "2.02",  # Ås
    "3220": "2.02",  # Enebakk
    "3222": "2.02",  # Aurskog-Høland
    "3224": "2.02",  # Nes
    "3226": "2.02",  # Gjerdrum
    "3228": "2.02",  # Nannestad
    "3230": "2.02",  # Eidsvoll
    "3232": "2.02",  # Nittedal
    "3234": "2.02",  # Ullensaker
    "3236": "2.02",  # Hurdal

    # Bergen zones
    "4601": "3.01",  # Bergen — Bergenhus (central)

    # Trondheim zones
    "5001": "4.01",  # Trondheim — central

    # Stavanger
    "1103": "5.00",

    # Kristiansand
    "4001": "6.00",  # Kristiansand

    # Tromsø
    "5501": "7.00",
}


def estimate_rent_for_municipality(municipality_code: str,
                                    rent_data: pd.DataFrame,
                                    sqm: float = 70.0) -> float:
    """
    Estimate monthly rent for a property in a given municipality.

    Uses SSB table 09897 rent zones. Falls back to size-based zone
    (20.00 for 20k+ cities, 21.00 for 2k-20k, 22.00 for smaller).
    """
    if rent_data.empty:
        return np.nan

    # Ensure zone_code is string for consistent matching
    # (SSB JSON-stat parsing may produce float64 zone codes like 1.01)
    # Normalize: float 5.0 → "5.00", 20.0 → "20.00" to match our lookup strings
    rent_data = rent_data.copy()
    def _norm_zone(v):
        s = str(v)
        if "." in s:
            integer, decimal = s.split(".", 1)
            return f"{integer}.{decimal.ljust(2, '0')}"
        return s
    rent_data["zone_code"] = rent_data["zone_code"].apply(_norm_zone)

    # Try direct municipality-to-zone mapping
    zone = _MUNICIPALITY_TO_SSB_ZONE.get(municipality_code)

    if zone:
        zone_rows = rent_data[rent_data["zone_code"] == zone]
        if not zone_rows.empty:
            return float(zone_rows["monthly_rent"].iloc[0])

    # Fallback: use population-based generic zones
    # Zone 20.00 = cities 20k+, 21.00 = 2k-20k, 22.00 = <2k
    # Default to medium city zone
    fallback_zone = get_rent_zone(municipality_code)

    # Map old zone codes to SSB zone codes
    zone_map = {
        "01": "1.01", "02": "1.02", "03": "1.03", "04": "1.04",
        "05": "2.01", "06": "2.02",
        "07": "3.01", "08": "4.01", "09": "5.00", "10": "6.00",
        "11": "7.00", "12": "20.00", "13": "20.00", "14": "21.00",
        "15": "21.00", "16": "22.00", "99": "21.00",
    }
    ssb_zone = zone_map.get(fallback_zone, "21.00")
    zone_rows = rent_data[rent_data["zone_code"] == ssb_zone]
    if not zone_rows.empty:
        return float(zone_rows["monthly_rent"].iloc[0])

    # Final fallback: median of all zones
    avg = rent_data["monthly_rent"].median()
    return float(avg) if not np.isnan(avg) else np.nan


def enrich_listings_with_regions(listings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add municipality_code, region_code, region_name, and rent_zone
    to a DataFrame of Finn listings.
    """
    df = listings_df.copy()

    # Map each listing's address to municipality
    mappings = df["address"].apply(map_address_to_municipality)
    df["municipality_code"] = [m[0] for m in mappings]
    df["municipality_name"] = [m[1] for m in mappings]
    df["region_code"] = df["municipality_code"].apply(get_region_code)
    df["region_name"] = df["municipality_code"].apply(get_region_name)
    df["rent_zone"] = df["municipality_code"].apply(get_rent_zone)

    mapped = (df["municipality_code"] != "0000").sum()
    logger.info(f"[mapper] Mapped {mapped}/{len(df)} listings to municipalities")

    return df
