"""
cache.py — Local file cache helpers.

The pipeline saves every downloaded file to data/raw/{source}/.
Before re-downloading, call is_cache_valid() to skip work and
preserve API quotas on frequent re-runs.

Usage:
    from utils.cache import is_cache_valid
    if is_cache_valid("data/raw/yfinance/BTC-USD.csv"):
        print("Cache hit — skipping download")
"""

import os
import time
from pathlib import Path


def is_cache_valid(filepath: str | Path, max_age_hours: float = 24.0) -> bool:
    """
    Return True if *filepath* exists and was last modified less than
    *max_age_hours* ago.  Returns False if the file is missing, empty,
    or older than the threshold.

    Args:
        filepath:      Absolute or relative path to the cached file.
        max_age_hours: Maximum acceptable age in hours (default 24).

    Returns:
        bool
    """
    p = Path(filepath)
    if not p.exists():
        return False
    if p.stat().st_size == 0:
        return False  # treat zero-byte files as invalid cache

    age_seconds = time.time() - p.stat().st_mtime
    return age_seconds < (max_age_hours * 3600)


def cache_path_for(base_dir: str | Path, filename: str) -> Path:
    """
    Convenience: return the full Path for a file inside *base_dir*,
    creating the directory if it doesn't exist.

    Args:
        base_dir:  Directory such as "data/raw/yfinance"
        filename:  Leaf filename such as "BTC-USD.csv"

    Returns:
        Full Path object ready for open() or pandas I/O.
    """
    d = Path(base_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / filename
