"""
fetch_all.py — Master pipeline runner for the financial data pipeline.

Usage
-----
Run all sources:
    python fetch_all.py

Run a specific source (or comma-separated list):
    python fetch_all.py --source yfinance
    python fetch_all.py --source fred,gdelt

Available source names:
    yfinance, fred, gdelt, trends, alphavantage, reddit, newsapi, quandl

Environment
-----------
All API keys are loaded from .env in the project root.
See .env.example for the full list of required keys.

Output
------
data/raw/               — all CSVs and parquet cache files
data/raw/manifest.json  — data quality manifest (auto-written after all pulls)
data/raw/pipeline.log   — timestamped log of all successes and failures
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure the data_pipeline directory is on the path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Load .env from project root before importing anything that reads env vars
try:
    from dotenv import load_dotenv
    _env_file = _HERE.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        load_dotenv()  # fallback: search CWD / parent dirs
except ImportError:
    pass  # python-dotenv not installed; env vars must be set manually


from utils.logger import get_logger
from utils.manifest import write_manifest

log = get_logger("master")

# ── Registry of all sources ──────────────────────────────────────────────────
# Each entry: (source_name, import_path, function_name)
SOURCE_REGISTRY = [
    ("yfinance",      "sources.fetch_yfinance",      "fetch_yfinance"),
    ("fred",          "sources.fetch_fred",           "fetch_fred"),
    ("gdelt",         "sources.fetch_gdelt",          "fetch_gdelt"),
    ("trends",        "sources.fetch_trends",         "fetch_trends"),
    ("alphavantage",  "sources.fetch_alphavantage",   "fetch_alphavantage"),
    ("reddit",        "sources.fetch_reddit",         "fetch_reddit"),
    ("newsapi",       "sources.fetch_newsapi",        "fetch_newsapi"),
    ("quandl",        "sources.fetch_quandl",         "fetch_quandl"),
]

ALL_NAMES = [s[0] for s in SOURCE_REGISTRY]


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Financial data pipeline — pull from all or selected sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available sources: {', '.join(ALL_NAMES)}",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        help=(
            "Comma-separated list of sources to run, or 'all'. "
            f"Available: {', '.join(ALL_NAMES)}. Default: all"
        ),
    )
    return parser.parse_args()


# ── Source runner ────────────────────────────────────────────────────────────

def run_source(name: str, module_path: str, func_name: str) -> tuple[list, bool]:
    """
    Dynamically import and run a single source fetcher.

    Returns:
        (records, success) — records is a list of manifest dicts,
                             success is False if an unhandled exception occurred.
    """
    try:
        import importlib
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        source_log = get_logger(name)
        records = func(log=source_log)
        return records or [], True
    except Exception as exc:
        log.error("Unhandled exception in source '%s': %s", name, exc, exc_info=True)
        return [], False


# ── Summary printer ──────────────────────────────────────────────────────────

def print_summary(results: dict[str, tuple[list, bool]]) -> None:
    """Print a formatted summary table after all sources complete."""
    print("\n" + "=" * 62)
    print("  PIPELINE SUMMARY")
    print("=" * 62)
    print(f"  {'SOURCE':<18} {'STATUS':<10} {'RECORDS'}")
    print("  " + "-" * 58)

    total_records = 0
    for name, (records, success) in results.items():
        status = "OK" if success else "FAILED"
        n = len(records)
        total_records += n
        print(f"  {name:<18} {status:<10} {n} manifest entries")

    print("  " + "-" * 58)
    print(f"  {'TOTAL':<18} {'':10} {total_records} manifest entries")
    print("=" * 62)
    print()


# ── Master function ──────────────────────────────────────────────────────────

def run_pipeline(target_sources: list[str]) -> None:
    """Run the full pipeline for the given list of source names."""
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║         FINANCIAL DATA PIPELINE — STARTING               ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("Sources to run: %s", ", ".join(target_sources))

    pipeline_start = time.time()
    all_records: list[dict] = []
    results: dict[str, tuple[list, bool]] = {}

    for name, module_path, func_name in SOURCE_REGISTRY:
        if name not in target_sources:
            continue

        log.info("")
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("  Running source: %s", name.upper())
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        t0 = time.time()
        records, success = run_source(name, module_path, func_name)
        elapsed = time.time() - t0

        all_records.extend(records)
        results[name] = (records, success)
        log.info("  Finished %s in %.1fs — %d manifest entries", name, elapsed, len(records))

    # ── Write manifest ───────────────────────────────────────────────────────
    log.info("")
    log.info("Writing manifest.json (%d total entries)...", len(all_records))
    try:
        write_manifest(all_records)
        log.info("manifest.json written successfully.")
    except Exception as exc:
        log.error("Could not write manifest.json: %s", exc)

    total_elapsed = time.time() - pipeline_start
    log.info("")
    log.info("Pipeline completed in %.1f seconds.", total_elapsed)

    print_summary(results)


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.source.strip().lower() == "all":
        target = ALL_NAMES
    else:
        requested = [s.strip().lower() for s in args.source.split(",")]
        unknown = [s for s in requested if s not in ALL_NAMES]
        if unknown:
            print(f"ERROR: Unknown source(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(ALL_NAMES)}")
            sys.exit(1)
        target = requested

    run_pipeline(target)


if __name__ == "__main__":
    main()
