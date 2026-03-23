"""
fetch_reddit.py — Pull posts from finance and crypto subreddits via PRAW.

Requires:
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env
    (Register a free app at reddit.com/prefs/apps)

Subreddits
----------
r/wallstreetbets, r/investing, r/stocks,
r/bitcoin, r/cryptocurrency, r/IndiaInvestments

Pulls HOT + TOP (last 1000 available) for each subreddit.

⚠ KNOWN LIMITATION ⚠
Reddit API returns only the most recent ~1000 posts per subreddit.
Historical data (beyond ~2-4 weeks) is NOT available via the free API.
Pushshift (pushshift.io) was the standard historical source but has
become unreliable/paid. This pipeline documents the limitation clearly.

Output
------
data/raw/reddit/raw_posts.csv      — individual posts
data/raw/reddit/daily_agg.csv      — daily aggregates per subreddit

Columns (raw_posts)
-------------------
date, subreddit, title, score, num_comments, upvote_ratio,
post_id, url

Columns (daily_agg)
-------------------
date, subreddit, post_count, mean_score, mean_upvote_ratio,
mean_num_comments
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _ROOT / "data" / "raw" / "reddit"

SUBREDDITS = [
    "wallstreetbets",
    "investing",
    "stocks",
    "bitcoin",
    "cryptocurrency",
    "IndiaInvestments",
]

POST_LIMIT = 1000   # max posts per listing type per subreddit

RAW_COLS = ["date", "subreddit", "title", "score", "num_comments",
            "upvote_ratio", "post_id", "url"]
AGG_COLS = ["date", "subreddit", "post_count", "mean_score",
            "mean_upvote_ratio", "mean_num_comments"]


def _post_to_row(post, subreddit: str) -> dict:
    """Convert a PRAW submission to a flat dict."""
    ts = getattr(post, "created_utc", None)
    if ts:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d")
    else:
        date_str = None

    return {
        "date":          date_str,
        "subreddit":     subreddit,
        "title":         getattr(post, "title", ""),
        "score":         getattr(post, "score", None),
        "num_comments":  getattr(post, "num_comments", None),
        "upvote_ratio":  getattr(post, "upvote_ratio", None),
        "post_id":       post.id,
        "url":           f"https://reddit.com{post.permalink}",
    }


def _pull_subreddit(reddit, sub_name: str, log) -> list[dict]:
    """Pull HOT + TOP posts from a subreddit and return a list of rows."""
    rows = []
    seen_ids: set[str] = set()

    try:
        sub = reddit.subreddit(sub_name)

        for listing_type in ("hot", "top"):
            try:
                if listing_type == "hot":
                    posts = sub.hot(limit=POST_LIMIT)
                else:
                    posts = sub.top(time_filter="all", limit=POST_LIMIT)

                for post in posts:
                    if post.id not in seen_ids:
                        seen_ids.add(post.id)
                        rows.append(_post_to_row(post, sub_name))

                log.debug("  %s/%s: %d posts collected so far", sub_name, listing_type, len(rows))

            except Exception as exc:
                log.error("Error pulling %s/%s: %s", sub_name, listing_type, exc)

    except Exception as exc:
        log.error("Could not access r/%s: %s", sub_name, exc)

    return rows


def fetch_reddit(log=None):
    """
    Pull posts from all configured subreddits.

    Returns:
        List of manifest record dicts.
    """
    if log is None:
        import logging
        log = logging.getLogger("pipeline.reddit")

    from utils.cache import is_cache_valid
    from utils.manifest import build_record

    raw_path = _RAW_DIR / "raw_posts.csv"
    agg_path = _RAW_DIR / "daily_agg.csv"

    if is_cache_valid(raw_path) and is_cache_valid(agg_path):
        log.info("CACHE HIT  — Reddit (raw_posts.csv + daily_agg.csv)")
        try:
            df_raw = pd.read_csv(raw_path, index_col="date")
            df_agg = pd.read_csv(agg_path, index_col="date")
            return [
                build_record("reddit", raw_path, df_raw,
                             extra={"limitation": "PRAW API returns ~1000 recent posts only"}),
                build_record("reddit", agg_path, df_agg),
            ]
        except Exception as exc:
            log.warning("Could not read cached Reddit files: %s", exc)

    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    user_agent = os.getenv("REDDIT_USER_AGENT", "financial-pipeline/1.0").strip()

    if not client_id or not client_secret:
        log.error("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set — skipping Reddit.")
        return []

    try:
        import praw
    except ImportError:
        log.error("praw not installed. Run: pip install praw")
        return []

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            read_only=True,
        )
        # Verify connection
        _ = reddit.subreddits.popular(limit=1)
    except Exception as exc:
        log.error("Reddit authentication failed: %s", exc)
        return []

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    log.info("=== Reddit: pulling HOT + TOP from %d subreddits ===", len(SUBREDDITS))

    for sub in SUBREDDITS:
        log.info("Pulling r/%s ...", sub)
        rows = _pull_subreddit(reddit, sub, log)
        log.info("  r/%s → %d unique posts", sub, len(rows))
        all_rows.extend(rows)

    if not all_rows:
        log.error("No Reddit posts collected — nothing to save.")
        return []

    records = []

    # ── Save raw posts ───────────────────────────────────────────────────────
    df_raw = pd.DataFrame(all_rows, columns=RAW_COLS)
    df_raw = df_raw.sort_values("date")
    df_raw = df_raw.set_index("date")
    df_raw.to_csv(raw_path)
    log.info("SAVED raw_posts.csv (%d posts)", len(df_raw))
    records.append(build_record("reddit", raw_path, df_raw,
                                extra={
                                    "limitation": (
                                        "PRAW API returns only the most recent ~1000 posts "
                                        "per subreddit. Historical data beyond ~2-4 weeks "
                                        "is not available via the free Reddit API."
                                    )
                                }))

    # ── Daily aggregates ─────────────────────────────────────────────────────
    df_agg = (
        df_raw.reset_index()
        .groupby(["date", "subreddit"])
        .agg(
            post_count=("post_id", "count"),
            mean_score=("score", "mean"),
            mean_upvote_ratio=("upvote_ratio", "mean"),
            mean_num_comments=("num_comments", "mean"),
        )
        .reset_index()
        .set_index("date")
        .sort_index()
    )
    df_agg.to_csv(agg_path)
    log.info("SAVED daily_agg.csv (%d rows)", len(df_agg))
    records.append(build_record("reddit", agg_path, df_agg))

    log.info("=== Reddit: complete ===")
    return records


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.logger import get_logger
    from dotenv import load_dotenv
    load_dotenv()
    fetch_reddit(log=get_logger("reddit"))
