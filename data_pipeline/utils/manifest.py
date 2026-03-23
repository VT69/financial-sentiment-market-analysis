"""
manifest.py — Pipeline manifest writer.

After all data sources are fetched, the master runner calls
write_manifest() to produce data/raw/manifest.json, recording
exactly what was downloaded, when, and the data quality statistics.

Schema per record:
    {
        "source":       "yfinance",
        "file_path":    "data/raw/yfinance/BTC-USD.csv",
        "date_range":   {"start": "2010-01-01", "end": "2024-12-31"},
        "columns":      ["open", "high", ...],
        "rows":         3650,
        "pct_missing":  {"open": 0.0, "volume": 2.3},
        "pull_timestamp": "2024-01-15T10:30:00"
    }

Usage:
    from utils.manifest import build_record, write_manifest

    rec = build_record("yfinance", path_to_csv, df)
    write_manifest([rec1, rec2, ...])
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "manifest.json"


def build_record(
    source: str,
    file_path: str | Path,
    df: pd.DataFrame,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a manifest record from a pandas DataFrame.

    Args:
        source:    Short identifier, e.g. "yfinance", "fred"
        file_path: Path where the CSV was saved
        df:        The saved DataFrame (used to compute stats)
        extra:     Optional dict with additional notes (e.g. limitations)

    Returns:
        dict ready to be serialised to JSON
    """
    fp = Path(file_path)
    relative = str(fp).replace("\\", "/")

    pct_missing: dict[str, float] = {}
    date_start = date_end = None
    cols: list[str] = []

    if df is not None and not df.empty:
        pct_missing = {
            col: round(df[col].isna().mean() * 100, 2) for col in df.columns
        }
        cols = list(df.columns)
        idx = df.index
        if hasattr(idx, "min"):
            try:
                date_start = str(idx.min())[:10]
                date_end = str(idx.max())[:10]
            except Exception:
                pass

    record: dict[str, Any] = {
        "source": source,
        "file_path": relative,
        "date_range": {"start": date_start, "end": date_end},
        "columns": cols,
        "rows": len(df) if df is not None else 0,
        "pct_missing": pct_missing,
        "pull_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if extra:
        record["notes"] = extra

    return record


def write_manifest(records: list[dict[str, Any]]) -> None:
    """
    Write (overwrite) the master manifest.json with all records.

    Args:
        records: List of dicts produced by build_record().
    """
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_sources": len(records),
        "sources": records,
    }

    with open(_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def load_manifest() -> dict[str, Any]:
    """Load existing manifest or return an empty skeleton."""
    if _MANIFEST_PATH.exists():
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"generated_at": None, "total_sources": 0, "sources": []}
