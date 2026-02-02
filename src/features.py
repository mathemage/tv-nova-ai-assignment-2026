"""
Feature construction for Task 2 (no 3mo mean) and Task 3 (with 3mo mean).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.data import (
    CHANNEL_ID_COL,
    TARGET_3MO_COL,
    TIMESLOT_COL,
    TARGET_COL,
)

# Feature names produced (for inference)
TIME_FEATURES = ["hour", "day_of_week", "month", "weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
CHANNEL_FEATURE = "channel_id_enc"


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-derived features from timeslot datetime from."""
    out = df.copy()
    if TIMESLOT_COL not in out.columns:
        return out
    t = pd.to_datetime(out[TIMESLOT_COL], errors="coerce")
    out["hour"] = t.dt.hour
    out["day_of_week"] = t.dt.dayofweek
    out["month"] = t.dt.month
    out["weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def features_task2(df: pd.DataFrame, channel_encoder: LabelEncoder = None, fit: bool = True):
    """
    Build feature matrix for Task 2 (exclude share 15 54 3mo mean).
    Returns X (np.ndarray), y (np.ndarray), channel_encoder, feature_names.
    """
    df = build_time_features(df)
    # Categorical: channel id
    if CHANNEL_ID_COL in df.columns:
        if fit:
            channel_encoder = LabelEncoder()
            df = df.copy()
            df[CHANNEL_FEATURE] = channel_encoder.fit_transform(df[CHANNEL_ID_COL].astype(str))
        else:
            seen = set(channel_encoder.classes_)
            vals = df[CHANNEL_ID_COL].astype(str)
            df = df.copy()
            enc = np.full(len(vals), len(channel_encoder.classes_), dtype=np.int64)
            mask = vals.isin(seen)
            enc[mask] = channel_encoder.transform(vals[mask])
            df[CHANNEL_FEATURE] = enc
    else:
        df = df.copy()
        df[CHANNEL_FEATURE] = 0
        if fit:
            channel_encoder = LabelEncoder().fit(["default"])

    feats = [f for f in TIME_FEATURES if f in df.columns] + [CHANNEL_FEATURE]
    X = df[feats].astype(float).values
    y = df[TARGET_COL].values.astype(np.float32) if TARGET_COL in df.columns else None
    return X, y, channel_encoder, feats


def features_task3(df: pd.DataFrame, channel_encoder=None, scaler_3mo: StandardScaler = None, fit: bool = True):
    """
    Build feature matrix for Task 3 (include share 15 54 3mo mean).
    Returns X, y, channel_encoder, scaler_3mo, feature_names.
    """
    df = build_time_features(df)
    if CHANNEL_ID_COL in df.columns:
        if fit:
            channel_encoder = LabelEncoder()
            df = df.copy()
            df[CHANNEL_FEATURE] = channel_encoder.fit_transform(df[CHANNEL_ID_COL].astype(str))
        else:
            seen = set(channel_encoder.classes_)
            vals = df[CHANNEL_ID_COL].astype(str)
            df = df.copy()
            enc = np.full(len(vals), len(channel_encoder.classes_), dtype=np.int64)
            mask = vals.isin(seen)
            enc[mask] = channel_encoder.transform(vals[mask])
            df[CHANNEL_FEATURE] = enc
    else:
        df = df.copy()
        df[CHANNEL_FEATURE] = 0
        if fit:
            channel_encoder = LabelEncoder().fit(["default"])

    feats = [f for f in TIME_FEATURES if f in df.columns] + [CHANNEL_FEATURE]
    base_feats = [f for f in TIME_FEATURES if f in df.columns] + [CHANNEL_FEATURE]
    if TARGET_3MO_COL in df.columns:
        col_3mo = df[TARGET_3MO_COL].fillna(df[TARGET_COL].mean() if TARGET_COL in df.columns else 0).values.reshape(-1, 1)
        if fit:
            scaler_3mo = StandardScaler()
            col_3mo = scaler_3mo.fit_transform(col_3mo)
        else:
            col_3mo = scaler_3mo.transform(col_3mo) if scaler_3mo is not None else col_3mo
        feats = base_feats + ["share_15_54_3mo_mean"]
        X_base = df[base_feats].astype(float).values
        X = np.hstack([X_base, col_3mo])
    else:
        scaler_3mo = None
        feats = base_feats
        X = df[feats].astype(float).values
    y = df[TARGET_COL].values.astype(np.float32) if TARGET_COL in df.columns else None
    return X, y, channel_encoder, scaler_3mo, feats
