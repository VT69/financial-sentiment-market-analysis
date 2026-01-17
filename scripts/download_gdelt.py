import requests
import zipfile
import io
from pathlib import Path

BASE_URL = "http://data.gdeltproject.org/events"
RAW_DIR = Path("../data/raw/gdelt/events")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_quarter(year, quarter):
    qmap = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
    fname = f"{year}{qmap[quarter]}.zip"
    url = f"{BASE_URL}/{fname}"

    print(f"\n📦 Downloading {fname}")

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"✗ Failed {fname}: {e}")
        return

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for file in z.namelist():
            out = RAW_DIR / file
            z.extract(file, RAW_DIR)
            size_mb = out.stat().st_size / 1e6
            print(f"  → Extracted {file} ({size_mb:.1f} MB)")

    print(f"✅ Completed {year} {qmap[quarter]}")