"""
config.py — Central configuration for the Norwegian property investment pipeline.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
CACHE_DIR = PROJECT_DIR / "cache"
OUTPUT_DIR = PROJECT_DIR / "output"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Cache TTL ─────────────────────────────────────────────────────────────────
CACHE_TTL_HOURS = 24  # SSB/FRED data cached for 24 hours

# ── FRED API ──────────────────────────────────────────────────────────────────
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

FRED_SERIES = {
    "nibor_3m": "IR3TIB01NOM156N",       # 3-month interbank rate (NIBOR proxy)
    "m3_money": "MABMM301NOM189S",       # M3 broad money supply Norway
    "bond_10y": "IRLTLT01NOM156N",       # 10-year government bond yield
}

# ── SSB API ───────────────────────────────────────────────────────────────────
SSB_BASE_URL = "https://data.ssb.no/api/v0/no/table"

SSB_TABLES = {
    "price_sqm": "06035",        # Avg price/sqm by municipality (annual)
    "price_index": "07221",      # Housing price index by region (quarterly)
    "rent": "09897",             # Predicted monthly rent by price zone
    "population": "07459",       # Population by municipality
    "employment": "13472",       # Employment by municipality + industry
    "debt_burden": "08781",      # Households with debt > 3x income
    "mortgage_rate": "10748",    # Mortgage interest rates (monthly)
    "wages": "12852",            # Monthly earnings by municipality (annual)
    "construction": "05889",     # Building permits by municipality (quarterly)
}

# ── Finn.no ───────────────────────────────────────────────────────────────────
FINN_SEARCH_URL = "https://www.finn.no/realestate/homes/search.html"
FINN_DEFAULT_SORT = "1"  # newest first
FINN_MAX_PAGES = 20      # pages to scrape (roughly 25 listings/page)
FINN_CONCURRENCY = 15    # concurrent httpx requests for listing details
FINN_DELAY_BETWEEN_PAGES = 1.5  # seconds between search page loads (Playwright)
FINN_DELAY_BETWEEN_BATCHES = 0.5  # seconds between httpx batches

# ── Sweden - FRED API (Swedish macro series) ──────────────────────────────────
FRED_SERIES_SE = {
    "stibor_3m": "IR3TIB01SEM156N",     # 3-month STIBOR (Swedish interbank rate)
    "m3_money":  "MABMM301SEM189S",     # M3 broad money supply Sweden
    "bond_10y":  "IRLTLT01SEM156N",     # 10-year Swedish government bond yield
}

# ── SCB (Statistics Sweden) API ───────────────────────────────────────────────
SCB_BASE_URL = "https://api.scb.se/OV0104/v1/doris/sv/ssd"

SCB_TABLES = {
    "housing_prices": "BO/BO0501/BO0501A/FastprisKNRegK",
    "rent":           "BO/BO0202/BO0202B/HyraNytUthB",
    "population":     "BE/BE0101/BE0101A/BefolkningNy",
    "employment":     "AM/AM0207/AM0207K/RAKS07KN",
    "wages":          "AM/AM0102/AM0102H/LonKNSektJ",
    "construction":   "BO/BO0701/BO0701A/BygglovBGT",
}

# ── Hemnet.se ─────────────────────────────────────────────────────────────────
HEMNET_SEARCH_URL = "https://www.hemnet.se/bostader"
HEMNET_MAX_PAGES = 20
HEMNET_CONCURRENCY = 15
HEMNET_DELAY_BETWEEN_PAGES = 2.0
HEMNET_DELAY_BETWEEN_BATCHES = 0.5

# Hemnet location IDs for Swedish cities (passed as location_ids[] param).
# NOTE: These are Hemnet's internal location IDs — verify via Hemnet search UI
# if scraping returns no results (inspect network request to /bostader).
HEMNET_CITY_LOCATION_IDS = {
    "stockholm":  "898051",   # Stockholm municipality
    "gothenburg": "898165",   # Göteborg municipality
    "malmo":      "898273",   # Malmö municipality
    "uppsala":    "898319",   # Uppsala municipality
    "linkoping":  "898354",   # Linköping municipality
    "orebro":     "898380",   # Örebro municipality
}

# ── Scoring defaults ─────────────────────────────────────────────────────────
DEFAULT_YIELD_WEIGHT = 0.40
DEFAULT_GROWTH_WEIGHT = 0.30
DEFAULT_RISK_PENALTY = 0.30
DEFAULT_TOP_N = 50
