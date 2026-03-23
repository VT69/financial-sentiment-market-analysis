"""
fetch_alphavantage.py — Pull macroeconomic indicators via Alpha Vantage.

Requires:
    ALPHA_VANTAGE_KEY environment variable
    (free tier: 25 requests/day, 5 requests/minute)

Indicators pulled (4 total — conserves free quota)
---------------------------------------------------
  real_gdp          Real GDP (quarterly)
  federal_funds_rate Federal Funds Rate (daily)
  cpi               Consumer Price Index (monthly)
  treasury_yield_10y Treasury yield 10Y (daily)

Rate limiting
-------------
12-second sleep between calls → ≤5 calls/minute on free tier.
On quota hit (HTTP 429 or API-level error), saves what was collected
and logs remaining series as skipped.

Output
------
data/raw/alpha_vantage/{indicator}.csv
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _ROOT / "data" / "raw" / "alpha_vantage"

BASE_URL = "https://www.alphavantage.co/query"
CALL_DELAY = 12   # seconds between API calls

INDICATORS = [
    {
        "id":       "real_gdp",
        "function": "REAL_GDP",
        "params":   {"interval": "quarterly"},
        "value_col": "value",
        "description": "Real GDP (quarterly, interpolated)",
    },
    {
        "id":       "federal_funds_rate",
        "function": "FEDERAL_FUNDS_RATE",
        "params":   {"interval": "daily"},
        "value_col": "value",
        "description": "Federal Funds Rate (daily)",
    },
    {
        "id":       "cpi",
        "function": "CPI",
        "params":   {"interval": "monthly"},
        "value_col": "value",
        "description": "Consumer Price Index (monthly)",
    },
    {
        "id":       "treasury_yield_10y",
        "function": "TREASURY_YIELD",
        "params":   {"interval": "daily", "maturity": "10year"},
        "value_col": "value",
        "description": "10-year Treasury yield (daily)",
    },
]


def _call_api(function: str, params: dict, api_key: str, log) -> dict | None:
    """Make a single Alpha Vantage API call and return the JSON body."""
    payload = {"function": function, "apikey": api_key, **params}
    try:
        resp = requests.get(BASE_URL, params=payload, timeout=30)
        if resp.status_code == 429:
            log.error("Alpha Vantage: HTTP 429 — daily or per-minute quota exceeded.")
            return None
        resp.raise_for_status()
        data = resp.json()

        # API-level quota message
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information")
            log.error("Alpha Vantage API message (likely quota): %s", msg)
            return None

        return data
    except requests.exceptions.RequestException as exc:
        log.error("HTTP request failed for function=%s: %s", function, exc)
        return None


def _parse_response(data: dict, indicator_id: str, value_col: str, log) -> pd.DataFrame | None:
    """
    Parse the Alpha Vantage JSON response into a pandas DataFrame.
    AV economic endpoints return {metadata, data: [{date, value}, ...]}.
    """
    if "data" not in data:
        log.error("Unexpected Alpha Vantage response structure for %s: keys=%s",
                  indicator_id, list(data.keys()))
        return None

    rows = data["data"]  # list of {"date": "2024-01-01", "value": "5.33"}
    df = pd.DataFrame(rows)

    if df.empty:
        log.warning("Empty data for %s", indicator_id)
        return None

    df = df.rename(columns={"date": "date", "value": value_col})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date").sort_index()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df.index.name = "date"
    df.columns = [indicator_id]
    return df


def fetch_alphavantage(log=None):
    """
    Pull all configured Alpha Vantage indicators.

    Returns:
        List of manifest record dicts.
    """
    if log is None:
        import logging
        log = logging.getLogger("pipeline.alphavantage")

    from utils.cache import is_cache_valid
    from utils.manifest import build_record

    api_key = os.getenv("ALPHA_VANTAGE_KEY", "").strip()
    if not api_key:
        log.error("ALPHA_VANTAGE_KEY not set — skipping Alpha Vantage source.")
        return []

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    quota_hit = False

    log.info("=== Alpha Vantage: pulling %d indicators ===", len(INDICATORS))

    for i, ind in enumerate(INDICATORS):
        if quota_hit:
            log.warning("SKIPPED    — %s (quota hit on previous call)", ind["id"])
            continue

        out_path = _RAW_DIR / f"{ind['id']}.csv"

        if is_cache_valid(out_path):
            log.info("CACHE HIT  — %s", ind["id"])
            try:
                df_cached = pd.read_csv(out_path, index_col="date")
                records.append(build_record("alpha_vantage", out_path, df_cached))
            except Exception as exc:
                log.warning("Could not read cached file %s: %s", out_path, exc)
            continue

        log.info("Downloading  — %s (%s)", ind["id"], ind["description"])
        data = _call_api(ind["function"], ind["params"], api_key, log)

        if data is None:
            quota_hit = True
            log.error("QUOTA/ERR  — %s skipped, stopping further calls", ind["id"])
            continue

        df = _parse_response(data, ind["id"], ind["value_col"], log)

        if df is None:
            log.error("FAILED     — %s (parse error)", ind["id"])
        else:
            try:
                df.to_csv(out_path)
                log.info("SAVED      — %s  (%d rows)", ind["id"], len(df))
                records.append(build_record("alpha_vantage", out_path, df))
            except Exception as exc:
                log.error("Could not save %s: %s", out_path, exc)

        # Rate limit: sleep between calls (skip after the last one)
        if i < len(INDICATORS) - 1 and not quota_hit:
            log.debug("Sleeping %ds for AV rate limit...", CALL_DELAY)
            time.sleep(CALL_DELAY)

    log.info("=== Alpha Vantage: complete — %d/%d indicators saved ===",
             len(records), len(INDICATORS))
    return records


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.logger import get_logger
    from dotenv import load_dotenv
    load_dotenv()
    fetch_alphavantage(log=get_logger("alphavantage"))
