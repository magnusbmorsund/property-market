# Norwegian Property Investment Pipeline

## What this project does

End-to-end pipeline that scrapes Norwegian property listings from Finn.no, enriches them with macroeconomic and regional statistics, scores each property on yield/growth/risk dimensions, and outputs ranked investment recommendations.

## Architecture

```
run_pipeline.py          — Main orchestrator (5-step pipeline)
config.py                — All configuration: API keys, URLs, weights, paths
data_sources/
  fred.py                — FRED API: NIBOR, M3 money supply, 10Y bond yield
  ssb.py                 — SSB (Statistics Norway): prices, rents, population, employment, wages, construction, debt
scrapers/
  finn.py                — Two-stage Finn.no scraper: Playwright (search pages) → httpx+BS4 (listing details)
mapping/
  region_mapper.py       — Address → municipality/region/rent-zone mapping (fuzzy matching)
  municipality_codes.json — Static lookup: all Norwegian municipalities with codes, regions, rent zones
models/
  property_scorer.py     — Scoring engine: yield (40%), growth (30%), risk penalty (30%)
investment_model.py      — Older region-level scoring model (reads from data/processed/ parquet files)
score.py                 — CLI for investment_model.py (requires parquet data from a separate pipeline)
```

## How to run

```bash
# Activate the venv first
source .venv/bin/activate

# Full pipeline (scrapes Finn.no, fetches SSB/FRED, scores)
python run_pipeline.py

# Options
python run_pipeline.py --top-n 100 --yield-weight 0.6 --max-listings 200 --no-cache --export results.csv
```

## Key dependencies

- **Playwright** (headless Chromium) for Finn.no search page scraping (JS-rendered)
- **httpx** (async) for concurrent listing detail fetching
- **BeautifulSoup + lxml** for HTML parsing
- **fredapi** for FRED macro data (requires `FRED_API_KEY` env var)
- **requests** for SSB API calls
- **pandas/numpy/scipy** for data processing and scoring
- **rich** (optional) for formatted console output

Install: `pip install -r requirements.txt && playwright install chromium`

## Environment variables

- `FRED_API_KEY` — Required for macro signals (NIBOR, M3, bond yields). Pipeline runs without it but macro signals will be NaN.

## Data flow

1. **FRED** → macro signals dict (nibor_current, m3_yoy_growth, bond_yield_10y, etc.)
2. **SSB** → region features DataFrame (197 municipalities: price growth, population, employment HHI, debt burden, wages, construction permits)
3. **SSB** → rent data DataFrame (predicted monthly rent by price zone)
4. **SSB** → index signals DataFrame (quarterly price index momentum/volatility by broad region)
5. **Finn.no** → listings DataFrame (address, price, sqm, property type, etc.)
6. **Enrichment** → listings joined with region features, rent estimates, index signals
7. **Scoring** → composite score = yield_weight * yield_score + growth_weight * growth_score, adjusted by risk penalty
8. **Output** → CSV to `output/property_scores.csv`, console table of top-N

## Scoring model (property_scorer.py)

**Yield signals** (default 40% weight):
- y1: Gross rental yield (net of felleskost) — 50%
- y2: Yield premium vs dataset median — 25%
- y3: Price affordability (price/sqm vs region avg, inverted) — 25%

**Growth signals** (default 30% weight):
- g1: 1Y municipality price growth — 30%
- g2: 3Y annualized price growth — 20%
- g3: Quarterly index momentum — 20%
- g4: Wage growth in municipality — 30%

**Risk signals** (default 30% penalty):
- r1: Interest rate sensitivity (NIBOR level + trend + local debt) — 20%
- r2: Job market vulnerability (employment HHI) — 20%
- r3: Price volatility (quarterly index std dev) — 15%
- r4: Monetary tightening (M3 deceleration) — 10%
- r5: Mortgage debt burden — 15%
- r6: Construction supply pressure (permits per capita) — 20%

Final: `composite = raw_score * (1 - risk_penalty * risk_score)`, scaled to 0-100.

## Caching

SSB and FRED data cached in `cache/` with 24h TTL. Use `--no-cache` to force refresh.

## Legacy files

- `investment_model.py` + `score.py` — Older region-level model that reads pre-processed parquet files from `data/processed/`. Not used by `run_pipeline.py`. Kept for reference/alternate analysis.

## Common issues

- **Finn.no scraping failures**: Site structure changes break selectors. Check `scrapers/finn.py` JS evaluation patterns and `FIELD_MAP` labels.
- **SSB query errors**: SSB API has cell limits. Population table is fetched in batches (by sex) to avoid hitting limits.
- **Municipality code mismatches**: SSB uses pre/post-2020 municipality reform codes. Joining is done on cleaned municipality names (lowercase, stripped) rather than codes to handle this.
- **Missing rent data**: Not all municipalities have SSB rent zones. Falls back to population-based generic zones, then median.
