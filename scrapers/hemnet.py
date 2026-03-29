"""
scrapers/hemnet.py — Two-stage Hemnet.se property listing scraper.

Stage 1: Playwright (headless browser) to collect listing URLs from
         JavaScript-rendered search result pages.
Stage 2: httpx (async HTTP) + BeautifulSoup to fetch and parse
         individual listing detail pages.

Returns a DataFrame with one row per listing.

NOTE: Hemnet location IDs are approximations as of early 2026 — if scraping
      returns no results, verify the location IDs via Hemnet's own search UI
      (inspect the network request to /bostader).
"""

import asyncio
import json as _json
import logging
import re
import time
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from config import (
    HEMNET_SEARCH_URL,
    HEMNET_MAX_PAGES,
    HEMNET_CONCURRENCY,
    HEMNET_DELAY_BETWEEN_PAGES,
    HEMNET_DELAY_BETWEEN_BATCHES,
)

logger = logging.getLogger(__name__)

# User-Agent rotation for respectful scraping
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
]

# Field extraction mapping: Swedish label → our field name
FIELD_MAP = {
    "utgångspris":    "asking_price",
    "pris":           "asking_price",
    "boarea":         "sqm",
    "boyta":          "sqm",
    "biarea":         "sqm_secondary",
    "antal rum":      "rooms",
    "rum":            "rooms",
    "byggnadsår":     "year_built",
    "byggår":         "year_built",
    "avgift":         "monthly_fee",
    "månadsavgift":   "monthly_fee",
    "driftkostnad":   "operating_cost",
    "tomtarea":       "plot_area",
    "tomtstorlek":    "plot_area",
    "bostadstyp":     "property_type",
    "upplåtelseform": "ownership_type",
    "energiklass":    "energy_rating",
    "våningsplan":    "floor",
    "etage":          "floor",
}

# Property types to exclude from investment analysis
_NON_RESIDENTIAL = {"tomt", "fritidshus", "fritidstomt", "garage", "parkering"}


# ── Stage 1: Collect listing URLs ────────────────────────────────────────────

async def _collect_listing_urls(max_pages: int = HEMNET_MAX_PAGES,
                                 location_params: str = "") -> list[dict]:
    """
    Use Playwright to navigate Hemnet.se search pages and extract listing URLs.
    Returns list of {url, listing_id} dicts.

    Parameters
    ----------
    max_pages : int
        Maximum search result pages to scrape.
    location_params : str
        Pre-built query string fragment, e.g. "location_ids[]=898051" for Stockholm.
        Multiple locations can be combined: "location_ids[]=898051&location_ids[]=898165"
    """
    from playwright.async_api import async_playwright

    listings = []
    seen_ids = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=USER_AGENTS[0],
            viewport={"width": 1920, "height": 1080},
            locale="sv-SE",
        )
        page = await context.new_page()

        loc_prefix = f"?{location_params}&" if location_params else "?"
        consecutive_errors = 0
        _MAX_CONSECUTIVE_ERRORS = 3

        for page_num in range(1, max_pages + 1):
            url = f"{HEMNET_SEARCH_URL}{loc_prefix}page={page_num}"
            logger.info(f"[hemnet] Scraping search page {page_num}/{max_pages}...")

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.wait_for_timeout(2000)  # let JS render

                # Extract all links pointing to individual listing pages
                # Hemnet listing URLs contain "/bostad/" path segment
                links = await page.evaluate(r"""
                    () => {
                        const results = [];
                        const seen = new Set();
                        document.querySelectorAll('a[href*="/bostad/"]').forEach(a => {
                            const href = a.href;
                            // Extract listing ID — last numeric segment in the URL path
                            const match = href.match(/\/bostad\/[^/]+-(\d+)(?:[/?#]|$)/);
                            if (match && !seen.has(match[1])) {
                                seen.add(match[1]);
                                // Normalise to canonical URL without query params
                                const cleanUrl = href.split('?')[0].split('#')[0];
                                results.push({ url: cleanUrl, listing_id: match[1] });
                            }
                        });
                        return results;
                    }
                """)

                if not links:
                    # Fallback: broader search for any /bostad/ link
                    links = await page.evaluate(r"""
                        () => {
                            const results = [];
                            const seen = new Set();
                            document.querySelectorAll('a').forEach(a => {
                                const href = a.getAttribute('href') || '';
                                if (href.includes('/bostad/')) {
                                    const match = href.match(/\/bostad\/[^/]+-(\d+)/);
                                    if (match && !seen.has(match[1])) {
                                        seen.add(match[1]);
                                        const fullUrl = href.startsWith('http')
                                            ? href.split('?')[0]
                                            : 'https://www.hemnet.se' + href.split('?')[0];
                                        results.push({ url: fullUrl, listing_id: match[1] });
                                    }
                                }
                            });
                            return results;
                        }
                    """)

                new_count = 0
                for item in links:
                    lid = item["listing_id"]
                    if lid not in seen_ids:
                        seen_ids.add(lid)
                        listings.append(item)
                        new_count += 1

                logger.info(f"[hemnet] Page {page_num}: found {new_count} new listings "
                            f"(total: {len(listings)})")

                consecutive_errors = 0
                if new_count == 0 and page_num > 1:
                    logger.info("[hemnet] No new listings on page — stopping pagination")
                    break

            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"[hemnet] Error on search page {page_num} "
                               f"(consecutive={consecutive_errors}): {e}")
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    logger.error(f"[hemnet] {_MAX_CONSECUTIVE_ERRORS} consecutive page errors — "
                                 "aborting Stage 1. Hemnet may have changed structure.")
                    break
                continue

            await asyncio.sleep(HEMNET_DELAY_BETWEEN_PAGES)

        await browser.close()

    logger.info(f"[hemnet] Stage 1 complete: {len(listings)} unique listing URLs collected")
    return listings


# ── Stage 2: Fetch and parse listing details ─────────────────────────────────

def _parse_number(text: str) -> Optional[float]:
    """
    Parse a Swedish number string to float.

    Examples:
      "4 500 000 kr"   → 4500000.0
      "12 500 kr/mån"  → 12500.0
      "85 m²"          → 85.0
    """
    if not text:
        return None
    # Remove currency/unit suffixes, non-breaking spaces, regular spaces
    cleaned = (
        text
        .replace("\xa0", "")
        .replace("\u2009", "")   # thin space
        .replace(" ", "")
        .split("kr")[0]          # strip "kr" and everything after
        .split("/")[0]           # strip "/mån" etc.
        .split("m")[0]           # strip "m²" etc.
        .strip()
    )
    # Remove any remaining non-numeric chars except comma/dot/minus
    cleaned = re.sub(r"[^\d,.-]", "", cleaned)
    # Swedish decimal comma
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _parse_listing_html(html: str, url: str, listing_id: str) -> dict:
    """Parse a single Hemnet.se listing page and extract property data."""
    soup = BeautifulSoup(html, "lxml")
    data = {"url": url, "hemnetkod": listing_id}

    # ── Extract title ────────────────────────────────────────────────────────
    title_tag = soup.find("h1")
    if title_tag:
        data["title"] = title_tag.get_text(strip=True)

    # ── Extract address — JSON-LD first (structured, stable) ─────────────────
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            ld = _json.loads(script.string or "")
            if isinstance(ld, list):
                ld = next((x for x in ld if isinstance(x, dict)), {})
            # Hemnet uses @type RealEstateListing or Residence
            addr = ld.get("address") or {}
            if isinstance(addr, dict):
                parts = [
                    addr.get("streetAddress"),
                    addr.get("postalCode"),
                    addr.get("addressLocality"),
                ]
                combined = " ".join(p for p in parts if p)
                if combined:
                    data["address"] = combined
                    break
            elif isinstance(addr, str) and addr:
                data["address"] = addr
                break
        except Exception:
            pass

    # HTML fallback: Hemnet address selectors
    if "address" not in data:
        for selector in [
            "[data-testid='property-address']",
            ".property-address",
            "h1 + p",
            ".qa-property-address",
        ]:
            addr_el = soup.select_one(selector)
            if addr_el:
                data["address"] = addr_el.get_text(" ", strip=True)
                break

    # Postal code heuristic — Swedish format is 5 digits "XXX XX Cityname"
    if "address" not in data:
        for tag in soup.find_all(["p", "span", "div"]):
            text = tag.get_text(strip=True)
            if re.search(r"\d{3}\s?\d{2}\s+\w+", text) and len(text) < 200:
                data["address"] = text
                break

    # ── Key-value pairs from <dt>/<dd> definition lists ──────────────────────
    for dt in soup.find_all("dt"):
        label = dt.get_text(strip=True).lower().rstrip(":")
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        value_text = dd.get_text(strip=True)

        if label in FIELD_MAP:
            field = FIELD_MAP[label]
            if field in ("asking_price", "sqm", "sqm_secondary", "monthly_fee",
                         "operating_cost", "plot_area", "rooms"):
                data[field] = _parse_number(value_text)
            elif field == "year_built":
                num = _parse_number(value_text)
                data[field] = int(num) if num and 1800 < num < 2030 else None
            elif field == "rooms":
                num = _parse_number(value_text)
                data[field] = int(num) if num else None
            else:
                data[field] = value_text

    # ── Table-based layouts ───────────────────────────────────────────────────
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True).lower().rstrip(":")
            value_text = cells[1].get_text(strip=True)
            if label in FIELD_MAP and FIELD_MAP[label] not in data:
                field = FIELD_MAP[label]
                if field in ("asking_price", "sqm", "sqm_secondary", "monthly_fee",
                             "operating_cost", "plot_area"):
                    data[field] = _parse_number(value_text)
                elif field == "year_built":
                    num = _parse_number(value_text)
                    data[field] = int(num) if num and 1800 < num < 2030 else None
                else:
                    data[field] = value_text

    # ── JSON-LD structured data (secondary pass for price/address) ───────────
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            ld = _json.loads(script.string or "")
            if isinstance(ld, dict) and ld.get("@type") in (
                "Product", "RealEstateListing", "Residence", "Apartment", "House"
            ):
                if "title" not in data and "name" in ld:
                    data["title"] = ld["name"]
                if "address" not in data:
                    addr = ld.get("address", {})
                    if isinstance(addr, dict):
                        parts = [
                            addr.get("streetAddress", ""),
                            addr.get("postalCode", ""),
                            addr.get("addressLocality", ""),
                        ]
                        data["address"] = " ".join(p for p in parts if p)
                offers = ld.get("offers", {})
                if isinstance(offers, dict) and "asking_price" not in data:
                    price = offers.get("price")
                    if price:
                        try:
                            data["asking_price"] = float(price)
                        except (ValueError, TypeError):
                            pass
        except (ValueError, TypeError, AttributeError):
            continue

    # ── Derived fields ────────────────────────────────────────────────────────
    # Sweden has no "fellesgjeld" equivalent — total_price_calc = asking_price
    if data.get("asking_price"):
        data["total_price_calc"] = data["asking_price"]
    else:
        data["total_price_calc"] = None

    # Alias monthly_fee → common_costs_monthly for scorer compatibility
    if "monthly_fee" in data and "common_costs_monthly" not in data:
        data["common_costs_monthly"] = data["monthly_fee"]

    # Price per sqm
    if data.get("total_price_calc") and data.get("sqm") and data["sqm"] > 0:
        data["price_per_sqm"] = round(data["total_price_calc"] / data["sqm"], 0)

    return data


async def _fetch_listing_details(listing_urls: list[dict],
                                  concurrency: int = HEMNET_CONCURRENCY) -> list[dict]:
    """
    Fetch individual listing pages concurrently with httpx.
    Returns list of parsed listing dicts.
    """
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(20.0),
        follow_redirects=True,
        headers={
            "User-Agent": USER_AGENTS[1],
            "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    ) as client:

        async def fetch_one(item: dict) -> Optional[dict]:
            async with semaphore:
                try:
                    resp = await client.get(item["url"])
                    if resp.status_code == 200:
                        return _parse_listing_html(resp.text, item["url"], item["listing_id"])
                    else:
                        logger.debug(f"[hemnet] HTTP {resp.status_code} for listing {item['listing_id']}")
                        return None
                except Exception as e:
                    logger.debug(f"[hemnet] Error fetching listing {item['listing_id']}: {e}")
                    return None

        batch_size = concurrency * 2
        for i in range(0, len(listing_urls), batch_size):
            batch = listing_urls[i:i + batch_size]
            tasks = [fetch_one(item) for item in batch]
            batch_results = await asyncio.gather(*tasks)

            for r in batch_results:
                if r:
                    results.append(r)

            done = min(i + batch_size, len(listing_urls))
            logger.info(f"[hemnet] Fetched {done}/{len(listing_urls)} listings "
                        f"({len(results)} successful)")

            if i + batch_size < len(listing_urls):
                await asyncio.sleep(HEMNET_DELAY_BETWEEN_BATCHES)

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def scrape_listings(max_pages: int = HEMNET_MAX_PAGES,
                    max_listings: int = None,
                    location_params: str = "") -> pd.DataFrame:
    """
    Scrape Hemnet.se property listings. Two-stage:
      1. Playwright → collect listing URLs from search pages
      2. httpx + BS4 → fetch and parse each listing

    Parameters
    ----------
    max_pages : int
        Maximum search result pages to scrape (default from config).
    max_listings : int, optional
        Cap on number of listings to fetch details for.
    location_params : str
        Hemnet location filter query string, e.g. "location_ids[]=898051".

    Returns
    -------
    pd.DataFrame with one row per listing.
    """
    logger.info("[hemnet] Starting Hemnet.se scraper...")
    start = time.time()

    # Stage 1: Collect URLs via Playwright
    listing_urls = asyncio.run(
        _collect_listing_urls(max_pages=max_pages, location_params=location_params)
    )

    if not listing_urls:
        logger.warning(
            "[hemnet] No listing URLs found — check if Hemnet.se structure changed "
            "or verify that location_ids are still valid."
        )
        return pd.DataFrame()

    # Cap listings if requested
    if max_listings and len(listing_urls) > max_listings:
        listing_urls = listing_urls[:max_listings]
        logger.info(f"[hemnet] Capped at {max_listings} listings")

    # Stage 2: Fetch details
    parsed = asyncio.run(_fetch_listing_details(listing_urls))

    if not parsed:
        logger.warning("[hemnet] No listings successfully parsed")
        return pd.DataFrame()

    df = pd.DataFrame(parsed)

    # Ensure required numeric columns exist
    for col in ("asking_price", "sqm"):
        if col not in df.columns:
            df[col] = np.nan

    initial_count = len(df)
    df = df.dropna(subset=["asking_price"])
    df = df[df["asking_price"] > 0]

    # Filter out non-residential property types
    if "property_type" in df.columns:
        df = df[~df["property_type"].str.lower().fillna("").isin(_NON_RESIDENTIAL)]

    elapsed = time.time() - start
    logger.info(
        f"[hemnet] Scraping complete: {len(df)} valid listings "
        f"(dropped {initial_count - len(df)} without price/non-residential) "
        f"in {elapsed:.1f}s"
    )

    return df
