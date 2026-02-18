"""Task 1: Download data, EDA, and channel analysis.

This script performs exploratory data analysis on TV viewership data,
analyzing patterns across time (hours, days, months) and channels.

Usage
-----
Run from repository root:
    python notebooks/task1_eda.py

Requirements
------------
- pandas >= 2.0
- Data is downloaded automatically or uses synthetic data from data/

Example Output
--------------
When run with synthetic data (5000 rows, 4 channels):

    Loading data...
    Shape: (5000, 5)
    Columns: ['timeslot datetime from', 'main_ident', 'channel id', 'share 15 54', 'share 15 54 3mo mean']
    Time range: 2023-01-01 00:00:00 to 2023-07-28 07:00:00

    share 15 54: mean=5.9790, std=3.0325
    count    5000.000000
    mean        5.978968
    std         3.032511
    min         0.100000
    25%         3.880000
    50%         5.920000
    75%         8.000000
    max        16.470000
    Name: share 15 54, dtype: float64

    Share by hour (mean):
              mean  count
    hour                 
    0     6.108086    209
    1     5.931292    209
    2     5.982871    209
    ...

    Share by day of week (0=Mon..6=Sun):
    day_of_week
    0    5.923403
    1    6.113681
    ...

    By channel:
                rows      mean       std
    channel id                          
    ch1         1223  5.946688  3.031682
    ch2         1278  6.003858  3.007627
    ch3         1260  6.009794  3.027789
    ch4         1239  5.953810  3.066633

    Unique main indent (movie ID) count: 199
    EDA done. Write conclusions to docs/task1_conclusions.md

Notes
-----
- Creates time features: hour, day_of_week, month, weekend
- Analyzes target variable (share 15 54) and 3-month mean
- Groups statistics by time periods and channels
- Results inform feature engineering for Tasks 2 and 3
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
    """Download (if needed), load, filter to four channels, run EDA.
    
    Performs comprehensive exploratory data analysis including:
    - Data shape and structure validation
    - Time range analysis with derived features
    - Target variable distributions and statistics
    - Temporal patterns (hourly, daily, monthly)
    - Channel-specific statistics
    - Main indent (movie ID) uniqueness analysis
    
    Returns
    -------
    pd.DataFrame
        DataFrame with loaded data and engineered time features.
        Contains columns: timeslot datetime from, main_ident, channel id,
        share 15 54, share 15 54 3mo mean, hour, day_of_week, month, weekend.
    
    Notes
    -----
    - Downloads data from remote URLs if not available locally
    - Filters to four main channels for analysis
    - Adds temporal features for downstream modeling
    """
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
        )
        if TARGET_COL in df.columns:
            ch2 = df.groupby(CHANNEL_ID_COL)[TARGET_COL].agg(["mean", "std"])
            ch = ch.join(ch2)
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
