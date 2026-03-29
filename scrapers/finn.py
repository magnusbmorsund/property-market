"""
scrapers/finn.py — Two-stage Finn.no property listing scraper.

Stage 1: Playwright (headless browser) to collect listing URLs from
         JavaScript-rendered search result pages.
Stage 2: httpx (async HTTP) + BeautifulSoup to fetch and parse
         individual listing detail pages.

Returns a DataFrame with one row per listing.
"""

import asyncio
import logging
import re
import time
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from config import (
    FINN_SEARCH_URL,
    FINN_DEFAULT_SORT,
    FINN_MAX_PAGES,
    FINN_CONCURRENCY,
    FINN_DELAY_BETWEEN_PAGES,
    FINN_DELAY_BETWEEN_BATCHES,
)

logger = logging.getLogger(__name__)

# User-Agent rotation for respectful scraping
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
]

# Field extraction mapping: Norwegian label → our field name
FIELD_MAP = {
    "prisantydning": "asking_price",
    "totalpris": "total_price",
    "fellesgjeld": "fellesgjeld",
    "felleskostnader/mnd.": "common_costs_monthly",
    "felleskost/mnd.": "common_costs_monthly",
    "felleskostnader": "common_costs_monthly",
    "eieform": "ownership_type",
    "boligtype": "property_type",
    "soverom": "bedrooms",
    "primærrom": "sqm_primary",
    "bruksareal": "sqm_usable",
    "bruttoareal": "sqm_gross",
    "byggeår": "year_built",
    "etasje": "floor",
    "tomteareal": "plot_size",
    "energimerking": "energy_rating",
}


# ── Stage 1: Collect listing URLs ────────────────────────────────────────────

async def _collect_listing_urls(max_pages: int = FINN_MAX_PAGES, location_params: str = "") -> list[dict]:
    """
    Use Playwright to navigate Finn.no search pages and extract listing URLs.
    Returns list of {url, finnkode} dicts.

    location_params: pre-built query string fragment, e.g.
      "location=0.20061"  for Oslo municipality
      "lat=63.43049&lon=10.39506&radius=20000"  for Trondheim
    """
    from playwright.async_api import async_playwright

    listings = []
    seen_codes = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=USER_AGENTS[0],
            viewport={"width": 1920, "height": 1080},
            locale="nb-NO",
        )
        page = await context.new_page()

        loc_suffix = f"&{location_params}" if location_params else ""
        consecutive_errors = 0
        _MAX_CONSECUTIVE_ERRORS = 3
        for page_num in range(1, max_pages + 1):
            url = f"{FINN_SEARCH_URL}?sort={FINN_DEFAULT_SORT}&page={page_num}{loc_suffix}"
            logger.info(f"[finn] Scraping search page {page_num}/{max_pages}...")

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(2000)  # let JS render

                # Extract listing links — look for ad URLs
                links = await page.evaluate("""
                    () => {
                        const results = [];
                        // Find all links that point to individual listings
                        const allLinks = document.querySelectorAll('a[href*="/realestate/homes/ad.html"]');
                        allLinks.forEach(a => {
                            const href = a.href;
                            const match = href.match(/finnkode=(\\d+)/);
                            if (match) {
                                results.push({url: href.split('?')[0] + '?finnkode=' + match[1], finnkode: match[1]});
                            }
                        });
                        // Also try data attributes or other patterns
                        const adLinks = document.querySelectorAll('a[id*="ad-"]');
                        adLinks.forEach(a => {
                            const href = a.href;
                            if (href && href.includes('finnkode')) {
                                const match = href.match(/finnkode=(\\d+)/);
                                if (match) {
                                    results.push({url: href.split('?')[0] + '?finnkode=' + match[1], finnkode: match[1]});
                                }
                            }
                        });
                        return results;
                    }
                """)

                if not links:
                    # Try alternative: look for any link with finnkode in href
                    links = await page.evaluate("""
                        () => {
                            const results = [];
                            document.querySelectorAll('a').forEach(a => {
                                const href = a.getAttribute('href') || '';
                                if (href.includes('finnkode=') && href.includes('realestate')) {
                                    const match = href.match(/finnkode=(\\d+)/);
                                    if (match) {
                                        const fullUrl = href.startsWith('http') ? href : 'https://www.finn.no' + href;
                                        results.push({url: fullUrl, finnkode: match[1]});
                                    }
                                }
                            });
                            return results;
                        }
                    """)

                new_count = 0
                for item in links:
                    code = item["finnkode"]
                    if code not in seen_codes:
                        seen_codes.add(code)
                        listings.append(item)
                        new_count += 1

                logger.info(f"[finn] Page {page_num}: found {new_count} new listings "
                           f"(total: {len(listings)})")

                consecutive_errors = 0  # reset on success
                if new_count == 0 and page_num > 1:
                    logger.info("[finn] No new listings on page — stopping pagination")
                    break

            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"[finn] Error on search page {page_num} "
                               f"(consecutive={consecutive_errors}): {e}")
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    logger.error(f"[finn] {_MAX_CONSECUTIVE_ERRORS} consecutive page errors — "
                                 "aborting Stage 1. Finn.no may have changed structure.")
                    break
                continue

            await asyncio.sleep(FINN_DELAY_BETWEEN_PAGES)

        await browser.close()

    logger.info(f"[finn] Stage 1 complete: {len(listings)} unique listing URLs collected")
    return listings


# ── Stage 2: Fetch and parse listing details ─────────────────────────────────

def _parse_number(text: str) -> Optional[float]:
    """Parse a Norwegian number string like '3 500 000 kr' → 3500000.0"""
    if not text:
        return None
    # Remove currency, spaces, non-breaking spaces
    cleaned = re.sub(r"[^\d,.-]", "", text.replace("\xa0", "").replace(" ", ""))
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _parse_listing_html(html: str, url: str, finnkode: str) -> dict:
    """Parse a single Finn.no listing page and extract property data."""
    soup = BeautifulSoup(html, "lxml")
    data = {"url": url, "finnkode": finnkode}

    # ── Extract title ────────────────────────────────────────────────────
    title_tag = soup.find("h1")
    if title_tag:
        data["title"] = title_tag.get_text(strip=True)

    # ── Parse all JSON-LD blocks once; use for address + price ──────────────
    _ld_blocks = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            ld = _json.loads(script.string or "")
            if isinstance(ld, list):
                ld = next((x for x in ld if isinstance(x, dict)), {})
            if isinstance(ld, dict):
                _ld_blocks.append(ld)
        except Exception:
            pass

    for ld in _ld_blocks:
        addr = ld.get("address") or {}
        if isinstance(addr, dict):
            parts = [addr.get("streetAddress"), addr.get("postalCode"), addr.get("addressLocality")]
            combined = " ".join(p for p in parts if p)
            if combined:
                data["address"] = combined
                break
        elif isinstance(addr, str) and addr:
            data["address"] = addr
            break

    # Pattern 1: HTML data-testid / CSS selectors (falls back if JSON-LD missing)
    if "address" not in data:
        for selector in ["[data-testid='object-address']", ".u-t3", ".ads__ad__content__keys"]:
            addr_el = soup.select_one(selector)
            if addr_el:
                data["address"] = addr_el.get_text(" ", strip=True)
                break

    # Pattern 2: postal-code heuristic in <p> tags
    if "address" not in data:
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if re.search(r"\d{4}\s+\w+", text) and len(text) < 200:
                data["address"] = text
                break

    # ── Extract key-value pairs from definition lists ────────────────────
    # Finn.no uses <dt>/<dd> pairs for property details
    for dt in soup.find_all("dt"):
        label = dt.get_text(strip=True).lower().rstrip(":")
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        value_text = dd.get_text(strip=True)

        if label in FIELD_MAP:
            field = FIELD_MAP[label]
            if field in ("asking_price", "total_price", "fellesgjeld",
                         "common_costs_monthly", "sqm_primary", "sqm_usable",
                         "sqm_gross", "plot_size"):
                data[field] = _parse_number(value_text)
            elif field == "bedrooms":
                num = _parse_number(value_text)
                data[field] = int(num) if num else None
            elif field == "year_built":
                num = _parse_number(value_text)
                data[field] = int(num) if num and 1800 < num < 2030 else None
            else:
                data[field] = value_text

    # ── Also try table-based layouts ─────────────────────────────────────
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True).lower().rstrip(":")
            value_text = cells[1].get_text(strip=True)
            if label in FIELD_MAP and FIELD_MAP[label] not in data:
                field = FIELD_MAP[label]
                if field in ("asking_price", "total_price", "fellesgjeld",
                             "common_costs_monthly", "sqm_primary", "sqm_usable",
                             "sqm_gross"):
                    data[field] = _parse_number(value_text)
                else:
                    data[field] = value_text

    # ── Use pre-parsed JSON-LD blocks for price + title fallback ────────────
    for ld in _ld_blocks:
        if ld.get("@type") in ("Product", "RealEstateListing", "Residence"):
            if "title" not in data and "name" in ld:
                data["title"] = ld["name"]
            if "address" not in data:
                addr = ld.get("address", {})
                if isinstance(addr, dict):
                    parts = [addr.get("streetAddress", ""),
                             addr.get("postalCode", ""),
                             addr.get("addressLocality", "")]
                    data["address"] = ", ".join(p for p in parts if p)
            offers = ld.get("offers", {})
            if isinstance(offers, dict) and "asking_price" not in data:
                price = offers.get("price")
                if price:
                    try:
                        data["asking_price"] = float(price)
                    except (TypeError, ValueError):
                        pass

    # ── Compute derived fields ───────────────────────────────────────────
    # Best available sqm
    data["sqm"] = data.get("sqm_primary") or data.get("sqm_usable") or data.get("sqm_gross")

    # Best available price
    if "asking_price" not in data and "total_price" in data:
        data["asking_price"] = data["total_price"]

    # Compute total price (asking + fellesgjeld) for borettslag
    if data.get("asking_price"):
        fellesgjeld = data.get("fellesgjeld", 0) or 0
        data["total_price_calc"] = data["asking_price"] + fellesgjeld
    elif data.get("total_price"):
        data["total_price_calc"] = data["total_price"]
    else:
        data["total_price_calc"] = None

    # Price per sqm
    if data.get("total_price_calc") and data.get("sqm") and data["sqm"] > 0:
        data["price_per_sqm"] = round(data["total_price_calc"] / data["sqm"], 0)

    return data


async def _fetch_listing_details(listing_urls: list[dict],
                                  concurrency: int = FINN_CONCURRENCY) -> list[dict]:
    """
    Fetch individual listing pages concurrently with httpx.
    Returns list of parsed listing dicts.
    """
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(20.0),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENTS[1], "Accept-Language": "nb-NO,nb;q=0.9,no;q=0.8"},
    ) as client:

        async def fetch_one(item: dict) -> Optional[dict]:
            async with semaphore:
                try:
                    resp = await client.get(item["url"])
                    if resp.status_code == 200:
                        return _parse_listing_html(resp.text, item["url"], item["finnkode"])
                    else:
                        logger.debug(f"[finn] HTTP {resp.status_code} for {item['finnkode']}")
                        return None
                except Exception as e:
                    logger.debug(f"[finn] Error fetching {item['finnkode']}: {e}")
                    return None

        # Process in batches for rate limiting
        batch_size = concurrency * 2
        for i in range(0, len(listing_urls), batch_size):
            batch = listing_urls[i:i + batch_size]
            tasks = [fetch_one(item) for item in batch]
            batch_results = await asyncio.gather(*tasks)

            for r in batch_results:
                if r:
                    results.append(r)

            done = min(i + batch_size, len(listing_urls))
            logger.info(f"[finn] Fetched {done}/{len(listing_urls)} listings "
                       f"({len(results)} successful)")

            if i + batch_size < len(listing_urls):
                await asyncio.sleep(FINN_DELAY_BETWEEN_BATCHES)

    return results


# ── Public API ───────────────────────────────────────────────────────────────

def scrape_listings(max_pages: int = FINN_MAX_PAGES,
                    max_listings: int = None,
                    location_params: str = "") -> pd.DataFrame:
    """
    Scrape Finn.no property listings. Two-stage:
      1. Playwright → collect listing URLs from search pages
      2. httpx + BS4 → fetch and parse each listing

    Parameters
    ----------
    max_pages : int
        Maximum search result pages to scrape (default from config)
    max_listings : int, optional
        Cap on number of listings to fetch details for

    Returns
    -------
    pd.DataFrame with one row per listing
    """
    logger.info("[finn] Starting Finn.no scraper...")
    start = time.time()

    # Stage 1: Collect URLs
    listing_urls = asyncio.run(_collect_listing_urls(max_pages=max_pages, location_params=location_params))

    if not listing_urls:
        logger.warning("[finn] No listing URLs found — check if Finn.no structure changed")
        return pd.DataFrame()

    # Cap listings if requested
    if max_listings and len(listing_urls) > max_listings:
        listing_urls = listing_urls[:max_listings]
        logger.info(f"[finn] Capped at {max_listings} listings")

    # Stage 2: Fetch details
    parsed = asyncio.run(_fetch_listing_details(listing_urls))

    if not parsed:
        logger.warning("[finn] No listings successfully parsed")
        return pd.DataFrame()

    df = pd.DataFrame(parsed)

    # Clean up — drop listings without price or sqm
    required = ["asking_price", "sqm"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    initial_count = len(df)
    df = df.dropna(subset=["asking_price"])
    df = df[df["asking_price"] > 0]

    # Filter out non-residential property types
    NON_RESIDENTIAL = {"garasje/parkering", "garasje", "parkering", "parkeringsplass",
                       "næringseiendom", "tomt", "fritidstomt"}
    if "property_type" in df.columns:
        df = df[~df["property_type"].str.lower().isin(NON_RESIDENTIAL)]

    elapsed = time.time() - start
    logger.info(f"[finn] Scraping complete: {len(df)} valid listings "
                f"(dropped {initial_count - len(df)} without price) "
                f"in {elapsed:.1f}s")

    return df
