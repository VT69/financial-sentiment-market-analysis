import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def fetch_gdelt(query, start_date, end_date, asset):
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=20)

        if response.status_code != 200 or response.text.strip() == "":
            return pd.DataFrame()

        data = response.json()
        articles = data.get("articles", [])

        records = []
        for art in articles:
            records.append({
                "timestamp": art.get("seendate"),
                "text": (art.get("title", "") + " " + art.get("snippet", "")).strip(),
                "source": art.get("source"),
                "asset": asset,
            })

        return pd.DataFrame(records)

    except Exception as e:
        print(f"[ERROR] {e}")
        return pd.DataFrame()


def download_gdelt_asset(query, asset, start, end):
    all_data = []
    current = start

    while current < end:
        next_week = current + timedelta(days=7)
        print(f"[INFO] {asset} | {current.date()} → {next_week.date()}")

        df = fetch_gdelt(query, current, next_week, asset)

        if not df.empty:
            print(f"  + {len(df)} articles")
            all_data.append(df)
        else:
            print("  - No data")

        time.sleep(3)  # rate limit
        current = next_week

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)

    btc_news = download_gdelt_asset(
        query="bitcoin OR cryptocurrency",
        asset="BTC",
        start=datetime(2019, 1, 1),
        end=datetime(2024, 1, 1),
    )

    btc_news.to_csv("data/raw/gdelt_btc_news.csv", index=False)
    print(f"\n[SAVED] BTC news rows: {len(btc_news)}")
