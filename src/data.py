"""
Data loading and download utilities.
Downloads CSV from assignment URLs, loads and filters to selected channels.
"""
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Assignment CSV URLs (try first, then fallback)
DATA_URLS = [
    "https://pub-db5679e69cc4407a89dc460661812ec6.r2.dev/data.csv",
    "https://eu2.contabostorage.com/62824c32198b4d53a08054da7a8b4df1:novatv/data.csv",
]

# Column names from assignment
TIMESLOT_COL = "timeslot datetime from"
MAIN_INDENT_COL = "main indent"
CHANNEL_ID_COL = "channel id"
TARGET_COL = "share 15 54"
TARGET_3MO_COL = "share 15 54 3mo mean"


def download_data(save_dir: str = "data", filename: str = "data.csv") -> str:
    """Download CSV from assignment URLs; return path to saved file."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    out_path = save_path / filename

    for url in DATA_URLS:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(r.content)
            return str(out_path)
        except Exception as e:
            continue
    raise RuntimeError("Failed to download data from any URL")


def load_data(
    path: Optional[str] = None,
    data_dir: str = "data",
    use_four_channels: bool = True,
) -> pd.DataFrame:
    """
    Load CSV and optionally filter to four channels.
    If path is None, uses data_dir/data.csv (downloads if missing).
    use_four_channels: if True, keep only the four selected channel IDs
    (we use the four most frequent channel IDs in the data as the "selected" four).
    """
    if path is None:
        path = os.path.join(data_dir, "data.csv")
    if not os.path.isfile(path):
        download_data(save_dir=data_dir, filename=os.path.basename(path))
        path = os.path.join(data_dir, "data.csv")

    df = pd.read_csv(path)

    # Normalize column names (strip spaces)
    df.columns = df.columns.str.strip()

    # Map common alternate names to expected names
    renames = {}
    for c in df.columns:
        c2 = c.strip().lower().replace("  ", " ")
        if "timeslot" in c2 and "datetime" in c2:
            renames[c] = TIMESLOT_COL
        elif "main" in c2 and "indent" in c2:
            renames[c] = MAIN_INDENT_COL
        elif "channel" in c2 and "id" in c2:
            renames[c] = CHANNEL_ID_COL
        elif "share" in c2 and "15" in c2 and "54" in c2 and "3mo" not in c2:
            renames[c] = TARGET_COL
        elif "share" in c2 and "15" in c2 and "54" in c2 and "3mo" in c2:
            renames[c] = TARGET_3MO_COL
    if renames:
        df = df.rename(columns=renames)

    # Parse timeslot
    if TIMESLOT_COL in df.columns:
        df[TIMESLOT_COL] = pd.to_datetime(df[TIMESLOT_COL], errors="coerce")

    if use_four_channels and CHANNEL_ID_COL in df.columns:
        top4 = df[CHANNEL_ID_COL].value_counts().head(4).index.tolist()
        df = df[df[CHANNEL_ID_COL].isin(top4)].copy()

    return df
