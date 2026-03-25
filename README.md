# Norwegian Property Investment Pipeline

Automated pipeline that identifies the best property investment opportunities in Norway by combining live listing data from Finn.no with macroeconomic signals and regional statistics.

## How it works

1. **Macro signals** — Fetches NIBOR interest rates, M3 money supply, and bond yields from FRED
2. **Region data** — Pulls housing prices, rents, population, employment, wages, construction permits, and debt statistics from SSB (Statistics Norway) for ~200 municipalities
3. **Listing scraper** — Scrapes Finn.no property listings using Playwright (JS-rendered search pages) + async httpx (listing details)
4. **Enrichment** — Maps each listing to its municipality, joins region features, estimates rental income
5. **Scoring** — Scores every property on rental yield (40%), price growth potential (30%), and risk (30% penalty), then ranks them

## Quick start

```bash
# Set up
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# Optional: set FRED API key for macro signals
export FRED_API_KEY=your_key_here

# Run
python run_pipeline.py
```

## Usage

```bash
python run_pipeline.py                          # default: top 50, balanced weights
python run_pipeline.py --top-n 100              # show more results
python run_pipeline.py --yield-weight 0.6       # income-focused (60% yield, 40% growth)
python run_pipeline.py --max-listings 200       # limit Finn.no scraping
python run_pipeline.py --no-cache               # force refresh SSB/FRED data
python run_pipeline.py --export results.csv     # custom export path
```

## Output

- **Console** — Rich-formatted table of top-N ranked properties with scores, yield, growth, risk, price, and location
- **CSV** — Full scored dataset exported to `output/property_scores.csv`

Each property gets:
- **Composite score** (0-100) — weighted combination of yield and growth, adjusted for risk
- **Recommendation** — BUY (top 20%), HOLD, or AVOID (bottom 20%)
- **Risk label** — Low / Medium / High

## Project structure

```
run_pipeline.py              Main pipeline orchestrator
config.py                    Configuration (API keys, URLs, weights)
data_sources/
  fred.py                    FRED macro data client
  ssb.py                     SSB statistics client
scrapers/
  finn.py                    Finn.no two-stage scraper
mapping/
  region_mapper.py           Address → municipality mapper
  municipality_codes.json    Municipality lookup data
models/
  property_scorer.py         Multi-factor scoring engine
```

## Data sources

| Source | Data | Update frequency |
|--------|------|-----------------|
| [Finn.no](https://www.finn.no) | Property listings (price, sqm, type, address) | Live (scraped) |
| [SSB](https://data.ssb.no) | Housing prices, rents, population, employment, wages, construction, debt | Cached 24h |
| [FRED](https://fred.stlouisfed.org) | NIBOR 3M, M3 money supply, 10Y bond yield | Cached 24h |

## Requirements

- Python 3.12+
- Chromium (installed via Playwright)
- FRED API key (optional, for macro signals)
