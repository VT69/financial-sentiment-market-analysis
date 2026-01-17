import pandas as pd
from pathlib import Path

RAW_DIR = Path("../data/raw/gdelt/events")
OUT_DIR = Path("../data/processed/gdelt_filtered")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_MAP = {
    "date": 1,
    "event_code": 26,
    "event_root": 28,
    "goldstein": 30,
    "actor1_country": 37,
    "actor2_country": 44,
    "avg_tone": 55,
}

def filter_daily_file(path):
    date = path.stem.replace("events_", "")
    print(f"🔄 Processing {date}")

    try:
        chunks = pd.read_csv(
            path,
            sep="\t",
            header=None,
            engine="python",
            on_bad_lines="skip",
            chunksize=200_000,
            low_memory=False
        )

        first = True
        out_file = OUT_DIR / f"{date}.csv"

        for chunk in chunks:
            max_col = chunk.shape[1] - 1
            valid = {k: v for k, v in COL_MAP.items() if v <= max_col}

            df = chunk[list(valid.values())]
            df.columns = list(valid.keys())

            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            df.dropna(subset=["date"], inplace=True)

            df.to_csv(out_file, mode="w" if first else "a",
                      index=False, header=first)
            first = False

        print(f"  ✅ Saved {out_file.name}")

    except Exception as e:
        print(f"  ❌ Error {date}: {e}")