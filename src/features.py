"""
Feature construction for Task 2 (no 3mo mean) and Task 3 (with 3mo mean).

This module is responsible for transforming raw DataFrames (as loaded by
``src.data``) into numeric feature matrices suitable for training and inference.

Overview of feature engineering pipeline
-----------------------------------------
1. **Time features** (``build_time_features``): cyclical and categorical
   encodings derived from the ``timeslot datetime from`` column.
2. **Channel encoding** (both task functions): label-encodes the ``channel id``
   column, handling unseen channels at inference time.
3. **3-month mean feature** (Task 3 only): standardizes the
   ``share 15 54 3mo mean`` column and appends it to the feature matrix.

Constants
---------
TIME_FEATURES : list[str]
    Ordered list of time-derived column names produced by
    ``build_time_features``. The same ordering is used at training and
    inference, so it must not be changed without retraining.
CHANNEL_FEATURE : str
    Name of the label-encoded channel column added to every feature matrix.
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


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Ordered list of time-derived feature names produced by build_time_features.
# This list is also exported for use by downstream inference code (e.g.
# predict_task2.py) so that the expected input columns are always consistent
# with what was used during training.
TIME_FEATURES = ["hour", "day_of_week", "month", "weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

# Name of the label-encoded channel-ID column appended to every feature matrix.
CHANNEL_FEATURE = "channel_id_enc"


# ---------------------------------------------------------------------------
# Time-feature construction
# ---------------------------------------------------------------------------

def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-derived features computed from the timeslot datetime column.

    Eight features are added (see ``TIME_FEATURES``):

    * ``hour``       – hour of day (0–23, integer).
    * ``day_of_week``– day of week (0=Monday … 6=Sunday, integer).
    * ``month``      – calendar month (1–12, integer).
    * ``weekend``    – 1 if Saturday or Sunday, else 0 (binary integer).
    * ``hour_sin``   – sine encoding of hour on a 24-hour cycle.
    * ``hour_cos``   – cosine encoding of hour on a 24-hour cycle.
    * ``dow_sin``    – sine encoding of day-of-week on a 7-day cycle.
    * ``dow_cos``    – cosine encoding of day-of-week on a 7-day cycle.

    Cyclical (sin/cos) encodings preserve the continuity of periodic signals
    (e.g. hour 23 is close to hour 0) in a form that linear models and neural
    networks can exploit without manual binning.

    If the ``TIMESLOT_COL`` column is absent the DataFrame is returned
    unchanged (no features are added).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that may contain the ``TIMESLOT_COL`` column.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with eight additional columns if ``TIMESLOT_COL`` is
        present, otherwise an unchanged copy of ``df``. The original DataFrame
        is never modified in place.
    """
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


# ---------------------------------------------------------------------------
# Task-specific feature builders
# ---------------------------------------------------------------------------

def features_task2(df: pd.DataFrame, channel_encoder: LabelEncoder = None, fit: bool = True):
    """Build the feature matrix for Task 2 (without the 3-month mean feature).

    Feature set
    -----------
    * All time features from ``build_time_features`` that are present in
      ``df`` (see ``TIME_FEATURES``).
    * One label-encoded channel-ID column (``CHANNEL_FEATURE``).

    Channel encoding
    ----------------
    When ``fit=True`` a new :class:`~sklearn.preprocessing.LabelEncoder` is
    fitted on the channel IDs present in ``df``.

    When ``fit=False`` (inference mode) the supplied ``channel_encoder`` is
    used to transform channel IDs.  Unseen channel IDs (not present in the
    encoder's ``classes_``) are mapped to ``len(encoder.classes_)`` so that
    the model receives a consistent out-of-vocabulary index rather than
    raising an error.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  May optionally contain ``TIMESLOT_COL`` to generate
        time features and ``TARGET_COL`` for the target array ``y``; both
        columns are optional at inference time.
    channel_encoder : LabelEncoder, optional
        Pre-fitted encoder for the channel-ID column.  Required when
        ``fit=False``; ignored when ``fit=True``.
    fit : bool, default True
        If ``True``, fit a new ``channel_encoder`` from ``df``.
        If ``False``, transform using the provided ``channel_encoder``.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Numeric feature matrix.
    y : np.ndarray of shape (n_samples,) or None
        Target values (``share 15 54``) as ``float32``, or ``None`` if
        ``TARGET_COL`` is not present in ``df``.
    channel_encoder : LabelEncoder
        The fitted (or re-used) channel encoder.  Pass this object back in
        subsequent calls with ``fit=False`` to ensure consistent encoding.
    feature_names : list[str]
        Ordered list of column names that correspond to columns of ``X``.
        Use this list to reconstruct feature names after loading a saved model.
    """
    df = build_time_features(df)

    # Encode the categorical channel-ID column
    if CHANNEL_ID_COL in df.columns:
        if fit:
            channel_encoder = LabelEncoder()
            df = df.copy()
            df[CHANNEL_FEATURE] = channel_encoder.fit_transform(df[CHANNEL_ID_COL].astype(str))
        else:
            # Map unseen IDs to an out-of-vocabulary index
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
    """Build the feature matrix for Task 3 (with the 3-month mean feature).

    Feature set
    -----------
    * All time features from ``build_time_features`` that are present in
      ``df`` (see ``TIME_FEATURES``).
    * One label-encoded channel-ID column (``CHANNEL_FEATURE``).
    * When available, the ``share 15 54 3mo mean`` column (``TARGET_3MO_COL``),
      standardized with a :class:`~sklearn.preprocessing.StandardScaler`.
      If ``TARGET_3MO_COL`` is absent, the feature matrix falls back to the
      same feature set as Task 2 and ``scaler_3mo`` is returned as ``None``.

    Channel encoding
    ----------------
    Identical to :func:`features_task2`: unseen channel IDs are mapped to an
    out-of-vocabulary index when ``fit=False``.

    3-month mean feature
    --------------------
    Missing values in ``TARGET_3MO_COL`` are imputed with the mean of
    ``TARGET_COL`` (when available) or zero before standardization.
    When ``fit=True`` a new scaler is fitted; when ``fit=False`` the provided
    ``scaler_3mo`` is applied (or the raw values are used if it is ``None``).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  May optionally contain ``TARGET_3MO_COL`` to enable
        the 3-month mean feature.
    channel_encoder : LabelEncoder, optional
        Pre-fitted encoder for the channel-ID column.  Required when
        ``fit=False``; ignored when ``fit=True``.
    scaler_3mo : StandardScaler, optional
        Pre-fitted scaler for the 3-month mean column.  Recommended when
        ``fit=False`` and ``TARGET_3MO_COL`` is present; if omitted, the raw
        (unscaled) values are used. Ignored when ``fit=True``.
    fit : bool, default True
        If ``True``, fit new encoders/scalers from ``df``.
        If ``False``, transform using the provided objects.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Numeric feature matrix.
    y : np.ndarray of shape (n_samples,) or None
        Target values (``share 15 54``) as ``float32``, or ``None`` if
        ``TARGET_COL`` is not present in ``df``.
    channel_encoder : LabelEncoder
        The fitted (or re-used) channel encoder.
    scaler_3mo : StandardScaler or None
        The fitted (or re-used) scaler for the 3-month mean column,
        or ``None`` if ``TARGET_3MO_COL`` was absent.
    feature_names : list[str]
        Ordered list of column names that correspond to columns of ``X``.
    """
    df = build_time_features(df)

    # Encode the categorical channel-ID column
    if CHANNEL_ID_COL in df.columns:
        if fit:
            channel_encoder = LabelEncoder()
            df = df.copy()
            df[CHANNEL_FEATURE] = channel_encoder.fit_transform(df[CHANNEL_ID_COL].astype(str))
        else:
            # Map unseen IDs to an out-of-vocabulary index
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

    base_feats = [f for f in TIME_FEATURES if f in df.columns] + [CHANNEL_FEATURE]

    # Append the standardized 3-month mean feature when available
    if TARGET_3MO_COL in df.columns:
        fallback = df[TARGET_COL].mean() if TARGET_COL in df.columns else 0
        col_3mo = df[TARGET_3MO_COL].fillna(fallback).values.reshape(-1, 1)
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
