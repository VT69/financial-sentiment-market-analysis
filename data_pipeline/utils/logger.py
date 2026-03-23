"""
logger.py — Pipeline-wide logging setup.

Creates a named logger that writes to both:
  - data/raw/pipeline.log  (persistent file)
  - stdout                 (for interactive runs)

Usage:
    from utils.logger import get_logger
    log = get_logger("yfinance")
    log.info("Starting download...")
    log.error("Download failed: %s", str(e))
"""

import logging
import os
from pathlib import Path

_LOG_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "pipeline.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

_root_logger_configured = False


def _configure_root_logger() -> None:
    global _root_logger_configured
    if _root_logger_configured:
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)-8s]  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — always appends so multiple runs accumulate
    fh = logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    root = logging.getLogger("pipeline")
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)
    root.propagate = False

    _root_logger_configured = True


def get_logger(source_name: str) -> logging.Logger:
    """Return a child logger namespaced under 'pipeline.<source_name>'."""
    _configure_root_logger()
    return logging.getLogger(f"pipeline.{source_name}")
