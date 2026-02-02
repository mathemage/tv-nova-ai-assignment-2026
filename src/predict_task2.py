"""
Load Task 2 model and run prediction. Used by the FastAPI service (Task 4).
"""
import json
import pickle
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch

from src.models_task2 import MLP, MLPLarge

# Default model dir when running from service
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_task2_model(model_dir: Union[str, Path] = None):
    """Load model, scaler, channel encoder, feature names from model_dir."""
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    state = torch.load(model_dir / "task2_best.pt", map_location="cpu")
    with open(model_dir / "task2_feature_names.json") as f:
        feature_names = json.load(f)
    with open(model_dir / "task2_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_dir / "task2_channel_encoder.pkl", "rb") as f:
        channel_encoder = pickle.load(f)
    input_size = state["input_size"]
    model_type = state.get("model_type", "mlp_large")
    if model_type == "mlp":
        model = MLP(input_size)
    else:
        model = MLPLarge(input_size)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, scaler, channel_encoder, feature_names


def predict_share(
    timeslot_datetime_from: List[str],
    channel_id: List[str],
    model_dir: Union[str, Path] = None,
) -> List[float]:
    """
    Predict share 15 54 for each (timeslot, channel_id).
    timeslot_datetime_from: list of datetime strings (e.g. "2024-01-15 20:00:00").
    channel_id: list of channel IDs (same length).
    """
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    model, scaler, channel_encoder, feature_names = load_task2_model(model_dir)

    df = pd.DataFrame({
        "timeslot datetime from": pd.to_datetime(timeslot_datetime_from),
        "channel id": channel_id,
    })
    from src.features import build_time_features
    df = build_time_features(df)
    # Encode channel (unknown -> last index)
    seen = set(channel_encoder.classes_)
    vals = df["channel id"].astype(str)
    enc = np.full(len(vals), len(channel_encoder.classes_), dtype=np.int64)
    mask = vals.isin(seen)
    enc[mask] = channel_encoder.transform(vals[mask])
    df["channel_id_enc"] = enc
    feats = [f for f in feature_names if f in df.columns]
    X = df[feats].astype(float).values
    X = scaler.transform(X)
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return pred.tolist()
