"""
fetch_trends.py — Pull Google Trends search volume via pytrends.

Terms pulled (weekly, then resampled to daily)
----------------------------------------------
Finance / Fear:
  'stock market crash', 'market crash', 'recession',
  'inflation', 'interest rates', 'financial crisis'

Crypto:
  'bitcoin crash', 'crypto crash', 'bitcoin price', 'buy bitcoin'

India / NIFTY:
  'nifty', 'sensex', 'stock market india', 'share market crash india'

Output
------
data/raw/google_trends/raw_weekly/{term_clean}.csv   — raw weekly (0-100)
data/raw/google_trends/daily_resampled.csv            — all terms, forward-filled daily

Limitations
-----------
Google Trends API is unofficial/undocumented.  Rate limiting is done
via 3-second sleep between requests.  Use proxies or increase DELAY if
you hit 429 errors repeatedly.
"""

import re
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_WEEKLY_DIR = _ROOT / "data" / "raw" / "google_trends" / "raw_weekly"
_DAILY_OUT = _ROOT / "data" / "raw" / "google_trends" / "daily_resampled.csv"

TERMS = [
    # Finance / Fear
    "stock market crash",
    "market crash",
    "recession",
    "inflation",
    "interest rates",
    "financial crisis",
    # Crypto
    "bitcoin crash",
    "crypto crash",
    "bitcoin price",
    "buy bitcoin",
    # India / NIFTY
    "nifty",
    "sensex",
    "stock market india",
    "share market crash india",
]

DELAY = 3          # seconds between requests
TIMEFRAME = "2010-01-01 2026-03-21"  # max available


def _term_to_filename(term: str) -> str:
    """Convert a search term to a safe filename."""
    clean = re.sub(r"[^\w\s-]", "", term).strip()
    clean = re.sub(r"[\s]+", "_", clean).lower()
    return f"{clean}.csv"


def _pull_term(pytrends, term: str, log) -> pd.DataFrame | None:
    """Pull weekly trend data for a single term."""
    try:
        pytrends.build_payload(
            [term],
            cat=0,
            timeframe=TIMEFRAME,
            geo="",
            gprop="",
        )
        df = pytrends.interest_over_time()

        if df is None or df.empty:
            log.warning("No data for term: '%s'", term)
            return None

        # Drop the 'isPartial' column if present
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])

        df.columns = [term]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index = df.index.strftime("%Y-%m-%d")
        df.index.name = "date"
        return df

    except Exception as exc:
        log.error("pytrends failed for term '%s': %s", term, exc)
        return None


def fetch_trends(log=None):
    """
    Pull Google Trends for all configured terms.

    Returns:
        List of manifest record dicts.
    """
    if log is None:
        import logging
        log = logging.getLogger("pipeline.trends")

    from utils.cache import is_cache_valid
    from utils.manifest import build_record

    try:
        from pytrends.request import TrendReq
    except ImportError:
        log.error("pytrends not installed. Run: pip install pytrends")
        return []

    _WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    _DAILY_OUT.parent.mkdir(parents=True, exist_ok=True)

    pytrends = TrendReq(hl="en-US", tz=0)
    records = []
    all_weekly: list[pd.DataFrame] = []

    log.info("=== Google Trends: pulling %d terms (3s delay each) ===", len(TERMS))

    for term in TERMS:
        filename = _term_to_filename(term)
        out_path = _WEEKLY_DIR / filename

        if is_cache_valid(out_path):
            log.info("CACHE HIT  — '%s'", term)
            try:
                df_cached = pd.read_csv(out_path, index_col="date")
                all_weekly.append(df_cached)
                records.append(build_record("google_trends", out_path, df_cached))
            except Exception as exc:
                log.warning("Could not read cached file %s: %s", out_path, exc)
            continue

        log.info("Pulling term — '%s'", term)
        df = _pull_term(pytrends, term, log)

        if df is not None:
            try:
                df.to_csv(out_path)
                log.info("SAVED      — '%s'  (%d weekly rows)", term, len(df))
                all_weekly.append(df)
                records.append(build_record("google_trends", out_path, df))
            except Exception as exc:
                log.error("Could not save '%s': %s", out_path, exc)
        else:
            log.error("FAILED     — no data for '%s'", term)

        time.sleep(DELAY)

    # ── Build daily resampled combined file ──────────────────────────────────
    if all_weekly:
        try:
            combined_weekly = all_weekly[0]
            for s in all_weekly[1:]:
                combined_weekly = combined_weekly.join(s, how="outer")

            combined_weekly.index = pd.to_datetime(combined_weekly.index)
            combined_weekly.sort_index(inplace=True)

            # Resample weekly → daily using forward-fill
            daily_idx = pd.date_range(
                start=combined_weekly.index.min(),
                end=combined_weekly.index.max(),
                freq="D",
            )
            daily = combined_weekly.reindex(daily_idx).ffill()
            daily.index = daily.index.strftime("%Y-%m-%d")
            daily.index.name = "date"

            daily.to_csv(_DAILY_OUT)
            log.info("Daily resampled combined file saved → daily_resampled.csv (%d rows)",
                     len(daily))
            records.append(build_record("google_trends", _DAILY_OUT, daily,
                                        extra={"note": "weekly data forward-filled to daily"}))
        except Exception as exc:
            log.error("Could not build daily_resampled.csv: %s", exc)

    log.info("=== Google Trends: complete — %d terms saved ===",
             sum(1 for r in records if "raw_weekly" in r.get("file_path", "")))
    return records


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.logger import get_logger
    fetch_trends(log=get_logger("trends"))
