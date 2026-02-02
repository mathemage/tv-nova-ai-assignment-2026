"""
Task 1: Download data, EDA, and channel analysis.
Run: python notebooks/task1_eda.py
  (from repo root; or set PYTHONPATH / run as module)
"""
import sys
from pathlib import Path

# Allow importing from src when run from repo root or notebooks/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data import (
    CHANNEL_ID_COL,
    MAIN_INDENT_COL,
    TARGET_3MO_COL,
    TIMESLOT_COL,
    TARGET_COL,
    load_data,
)

DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)


def run_eda() -> pd.DataFrame:
    """Download (if needed), load, filter to four channels, run EDA."""
    print("Loading data...")
    df = load_data(data_dir=str(ROOT / "data"), use_four_channels=True)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Ensure timeslot is datetime
    if TIMESLOT_COL in df.columns:
        df[TIMESLOT_COL] = pd.to_datetime(df[TIMESLOT_COL], errors="coerce")
        df = df.dropna(subset=[TIMESLOT_COL])
        df["hour"] = df[TIMESLOT_COL].dt.hour
        df["day_of_week"] = df[TIMESLOT_COL].dt.dayofweek
        df["month"] = df[TIMESLOT_COL].dt.month
        df["weekend"] = (df["day_of_week"] >= 5).astype(int)
        print(f"Time range: {df[TIMESLOT_COL].min()} to {df[TIMESLOT_COL].max()}")

    # Distributions of key columns
    for col in [TARGET_COL, TARGET_3MO_COL]:
        if col in df.columns:
            print(f"\n{col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
            print(df[col].describe())

    # Time patterns
    if "hour" in df.columns and TARGET_COL in df.columns:
        by_hour = df.groupby("hour")[TARGET_COL].agg(["mean", "count"])
        print("\nShare by hour (mean):")
        print(by_hour.head(10))
    if "day_of_week" in df.columns and TARGET_COL in df.columns:
        by_dow = df.groupby("day_of_week")[TARGET_COL].mean()
        print("\nShare by day of week (0=Mon..6=Sun):")
        print(by_dow)
    if "month" in df.columns and TARGET_COL in df.columns:
        by_month = df.groupby("month")[TARGET_COL].mean()
        print("\nShare by month:")
        print(by_month)

    # By channel
    if CHANNEL_ID_COL in df.columns:
        ch = df.groupby(CHANNEL_ID_COL).agg(
            rows=pd.NamedAgg(column=CHANNEL_ID_COL, aggfunc="count"),
            **{TARGET_COL: ["mean", "std"]} if TARGET_COL in df.columns else {},
        )
        print("\nBy channel:")
        print(ch)

    # Main indent (movie ID) - sample
    if MAIN_INDENT_COL in df.columns:
        n_ids = df[MAIN_INDENT_COL].nunique()
        print(f"\nUnique main indent (movie ID) count: {n_ids}")

    return df


def main():
    df = run_eda()
    print("\nEDA done. Write conclusions to docs/task1_conclusions.md")
    return df


if __name__ == "__main__":
    main()
