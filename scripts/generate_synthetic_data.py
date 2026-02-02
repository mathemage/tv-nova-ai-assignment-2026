"""
Generate a small synthetic CSV for testing when assignment URLs are unavailable.
Columns match assignment: timeslot datetime from, main indent, channel id, share 15 54, share 15 54 3mo mean.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)
n = 5000
t0 = pd.Timestamp("2023-01-01")
df = pd.DataFrame({
    "timeslot datetime from": [t0 + pd.Timedelta(hours=i % (365 * 24)) for i in range(n)],
    "main indent": np.random.randint(1, 200, n),
    "channel id": np.random.choice(["ch1", "ch2", "ch3", "ch4"], n),
    "share 15 54": np.clip(5 + np.random.randn(n) * 3 + np.random.rand(n) * 2, 0.1, 25).round(2),
})
df["share 15 54 3mo mean"] = df["share 15 54"].rolling(90, min_periods=1).mean().round(2)
out = DATA_DIR / "data_synthetic.csv"
df.to_csv(out, index=False)
print(f"Wrote {out} ({len(df)} rows). Use DATA_PATH={out} for training if real data unavailable.")
