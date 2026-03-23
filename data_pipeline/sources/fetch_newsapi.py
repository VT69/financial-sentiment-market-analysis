"""
fetch_newsapi.py — Pull financial news articles via NewsAPI.

Requires:
    NEWSAPI_KEY environment variable
    (free tier at newsapi.org — 100 req/day, max 1 month history)

⚠ KNOWN LIMITATION ⚠
The free NewsAPI developer plan only allows articles from the past
30 days.  This source is intentionally designed for the LIVE PIPELINE
(periodic updates) rather than historical backfill.

Queries
-------
'stock market', 'financial markets', 'bitcoin',
'NIFTY', 'market crash', 'recession', 'inflation'

Output
------
data/raw/newsapi/raw_articles.csv
  columns: published_at, source_name, title, description, url

data/raw/newsapi/daily_agg.csv
  columns: date, query, article_count, unique_sources
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _ROOT / "data" / "raw" / "newsapi"

BASE_URL = "https://newsapi.org/v2/everything"
PAGE_SIZE = 100
MAX_PAGES = 5        # stay within free tier per query
CALL_DELAY = 1       # seconds between paginated requests

QUERIES = [
    "stock market",
    "financial markets",
    "bitcoin",
    "NIFTY",
    "market crash",
    "recession",
    "inflation",
]

RAW_COLS = ["published_at", "source_name", "title", "description", "url", "query"]


def _fetch_query(api_key: str, query: str, log) -> list[dict]:
    """Pull all available pages for a single search query."""
    rows = []
    for page in range(1, MAX_PAGES + 1):
        params = {
            "q":        query,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": PAGE_SIZE,
            "page":     page,
            "apiKey":   api_key,
        }
        try:
            resp = requests.get(BASE_URL, params=params, timeout=20)
            if resp.status_code == 429:
                log.error("NewsAPI: HTTP 429 — rate limit hit for query '%s'", query)
                break
            if resp.status_code == 426:
                log.error("NewsAPI: Upgrade required — free tier may not support this request.")
                break
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "ok":
                log.error("NewsAPI error for query '%s': %s", query, data.get("message"))
                break

            articles = data.get("articles", [])
            if not articles:
                break

            for art in articles:
                rows.append({
                    "published_at": art.get("publishedAt", "")[:10],  # keep date only
                    "source_name":  (art.get("source") or {}).get("name", ""),
                    "title":        art.get("title", ""),
                    "description":  art.get("description", ""),
                    "url":          art.get("url", ""),
                    "query":        query,
                })

            total = data.get("totalResults", 0)
            fetched = page * PAGE_SIZE
            if fetched >= total:
                break

            time.sleep(CALL_DELAY)

        except requests.exceptions.RequestException as exc:
            log.error("HTTP error fetching NewsAPI query '%s' page %d: %s", query, page, exc)
            break

    return rows


def fetch_newsapi(log=None):
    """
    Pull news articles for all configured queries.

    Returns:
        List of manifest record dicts.
    """
    if log is None:
        import logging
        log = logging.getLogger("pipeline.newsapi")

    from utils.cache import is_cache_valid
    from utils.manifest import build_record

    raw_path = _RAW_DIR / "raw_articles.csv"
    agg_path = _RAW_DIR / "daily_agg.csv"

    if is_cache_valid(raw_path) and is_cache_valid(agg_path):
        log.info("CACHE HIT  — NewsAPI (raw_articles.csv + daily_agg.csv)")
        try:
            df_raw = pd.read_csv(raw_path, index_col="published_at")
            df_agg = pd.read_csv(agg_path, index_col="date")
            return [
                build_record("newsapi", raw_path, df_raw,
                             extra={"limitation": "Free tier: 1 month history, 100 req/day"}),
                build_record("newsapi", agg_path, df_agg),
            ]
        except Exception as exc:
            log.warning("Could not read cached NewsAPI files: %s", exc)

    api_key = os.getenv("NEWSAPI_KEY", "").strip()
    if not api_key:
        log.error("NEWSAPI_KEY not set — skipping NewsAPI source.")
        return []

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    quota_hit = False

    log.info("=== NewsAPI: pulling %d queries ===", len(QUERIES))

    for query in QUERIES:
        if quota_hit:
            log.warning("SKIPPED    — '%s' (quota hit)", query)
            continue

        log.info("Fetching   — '%s'", query)
        rows = _fetch_query(api_key, query, log)

        if not rows:
            log.warning("No articles returned for query '%s'", query)
        else:
            log.info("  → %d articles", len(rows))
            all_rows.extend(rows)

    if not all_rows:
        log.error("No articles collected — nothing to save.")
        return []

    records = []

    # ── Raw articles ─────────────────────────────────────────────────────────
    df_raw = pd.DataFrame(all_rows, columns=RAW_COLS)
    df_raw = df_raw.drop_duplicates(subset=["url"])
    df_raw = df_raw.sort_values("published_at")
    df_raw = df_raw.set_index("published_at")
    df_raw.index.name = "published_at"
    df_raw.to_csv(raw_path)
    log.info("SAVED raw_articles.csv (%d articles)", len(df_raw))
    records.append(build_record("newsapi", raw_path, df_raw,
                                extra={
                                    "limitation": (
                                        "Free NewsAPI tier provides only the last 30 days of "
                                        "articles. This source is designed for the live pipeline, "
                                        "not historical research."
                                    )
                                }))

    # ── Daily aggregates ─────────────────────────────────────────────────────
    df_agg = (
        df_raw.reset_index()
        .rename(columns={"published_at": "date"})
        .groupby(["date", "query"])
        .agg(
            article_count=("url", "count"),
            unique_sources=("source_name", pd.Series.nunique),
        )
        .reset_index()
        .set_index("date")
        .sort_index()
    )
    df_agg.to_csv(agg_path)
    log.info("SAVED daily_agg.csv (%d rows)", len(df_agg))
    records.append(build_record("newsapi", agg_path, df_agg))

    log.info("=== NewsAPI: complete ===")
    return records


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.logger import get_logger
    from dotenv import load_dotenv
    load_dotenv()
    fetch_newsapi(log=get_logger("newsapi"))
