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
}

# ── Finn.no ───────────────────────────────────────────────────────────────────
FINN_SEARCH_URL = "https://www.finn.no/realestate/homes/search.html"
FINN_DEFAULT_SORT = "1"  # newest first
FINN_MAX_PAGES = 20      # pages to scrape (roughly 25 listings/page)
FINN_CONCURRENCY = 15    # concurrent httpx requests for listing details
FINN_DELAY_BETWEEN_PAGES = 1.5  # seconds between search page loads (Playwright)
FINN_DELAY_BETWEEN_BATCHES = 0.5  # seconds between httpx batches

# ── Scoring defaults ─────────────────────────────────────────────────────────
DEFAULT_YIELD_WEIGHT = 0.40
DEFAULT_GROWTH_WEIGHT = 0.30
DEFAULT_RISK_PENALTY = 0.30
DEFAULT_TOP_N = 50
