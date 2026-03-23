"""
fetch_fred.py — Download macroeconomic and financial series from FRED.

Requires:
    FRED_API_KEY environment variable (free at fred.stlouisfed.org)

Series pulled
-------------
VIXCLS           VIX daily close
BAMLH0A0HYM2     US HY credit spread (fragility signal)
BAMLC0A0CM       US IG credit spread
T10Y2Y           10Y-2Y yield spread (recession indicator)
T10Y3M           10Y-3M yield spread
TEDRATE          TED spread (bank stress)
DPCREDIT         Discount window borrowing
DPRIME           Prime rate
UMCSENT          U. Michigan Consumer Sentiment
USREC            NBER recession indicator (binary)
M2SL             M2 money supply
CPIAUCSL         CPI (all items)
UNRATE           Unemployment rate
DCOILWTICO       WTI crude oil price (daily)
GOLDAMGBD228NLBM Gold price (daily, London PM fix)

Output
------
data/raw/fred/{SERIES_ID}.csv   — individual series
data/raw/fred/fred_combined.csv  — all series merged on date
"""

import os
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _ROOT / "data" / "raw" / "fred"

START_DATE = "2010-01-01"

SERIES = {
    "VIXCLS":           "VIX daily close",
    "BAMLH0A0HYM2":     "US High Yield credit spread",
    "BAMLC0A0CM":       "US Investment Grade credit spread",
    "T10Y2Y":           "10Y-2Y Treasury yield spread",
    "T10Y3M":           "10Y-3M Treasury yield spread",
    "TEDRATE":          "TED spread (bank stress)",
    "DPCREDIT":         "Discount window borrowing",
    "DPRIME":           "Prime rate",
    "UMCSENT":          "UMich Consumer Sentiment",
    "USREC":            "NBER recession indicator",
    "M2SL":             "M2 money supply",
    "CPIAUCSL":         "CPI all items",
    "UNRATE":           "Unemployment rate",
    "DCOILWTICO":       "WTI crude oil daily",
    "GOLDAMGBD228NLBM": "Gold price London PM fix",
}


def _pull_series(fred, series_id: str, log) -> pd.DataFrame | None:
    """Fetch a single FRED series and return a cleaned DataFrame."""
    try:
        s = fred.get_series(series_id, observation_start=START_DATE)
        if s is None or s.empty:
            log.warning("FRED series %s returned empty data", series_id)
            return None
        df = s.to_frame(name=series_id.lower())
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index = df.index.strftime("%Y-%m-%d")
        df.index.name = "date"
        return df
    except Exception as exc:
        log.error("Failed to pull FRED series %s: %s", series_id, exc)
        return None


def fetch_fred(log=None):
    """
    Download all FRED series and save individual + combined CSVs.

    Returns:
        List of manifest record dicts.
    """
    if log is None:
        import logging
        log = logging.getLogger("pipeline.fred")

    from utils.cache import is_cache_valid
    from utils.manifest import build_record

    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        log.error("FRED_API_KEY not set — skipping FRED source entirely.")
        return []

    try:
        from fredapi import Fred
    except ImportError:
        log.error("fredapi not installed. Run: pip install fredapi")
        return []

    fred = Fred(api_key=api_key)
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    all_series: list[pd.DataFrame] = []

    log.info("=== FRED: starting downloads for %d series ===", len(SERIES))

    for series_id, description in SERIES.items():
        out_path = _RAW_DIR / f"{series_id}.csv"

        if is_cache_valid(out_path):
            log.info("CACHE HIT  — %s", series_id)
            try:
                df_cached = pd.read_csv(out_path, index_col="date")
                df_cached.columns = [series_id.lower()]
                all_series.append(df_cached)
                records.append(build_record("fred", out_path, df_cached))
            except Exception as exc:
                log.warning("Could not read cached file %s: %s", out_path, exc)
            continue

        log.info("Downloading  — %s (%s)", series_id, description)
        df = _pull_series(fred, series_id, log)

        if df is None:
            log.error("FAILED     — %s", series_id)
            continue

        try:
            df.to_csv(out_path)
            log.info("SAVED      — %s  (%d rows)", series_id, len(df))
            all_series.append(df)
            records.append(build_record("fred", out_path, df))
        except Exception as exc:
            log.error("Could not save %s: %s", out_path, exc)

    # ── Build combined file ──────────────────────────────────────────────────
    if all_series:
        combined_path = _RAW_DIR / "fred_combined.csv"
        try:
            combined = all_series[0]
            for s in all_series[1:]:
                combined = combined.join(s, how="outer")
            combined.index.name = "date"
            combined.sort_index(inplace=True)
            combined.to_csv(combined_path)
            log.info("Combined FRED file saved → fred_combined.csv (%d rows, %d cols)",
                     len(combined), len(combined.columns))
            records.append(build_record("fred", combined_path, combined,
                                        extra={"note": "merged from all individual series"}))
        except Exception as exc:
            log.error("Could not write fred_combined.csv: %s", exc)

    log.info("=== FRED: complete — %d/%d series saved ===", len(records), len(SERIES))
    return records


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.logger import get_logger
    from dotenv import load_dotenv
    load_dotenv()
    fetch_fred(log=get_logger("fred"))
