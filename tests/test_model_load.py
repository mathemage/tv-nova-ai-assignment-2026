"""Test that Task 2 model loads and runs one prediction."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_predict_task2_with_synthetic_model():
    """If models/task2_best.pt exists, load and predict. Otherwise create minimal compatible artifacts."""
    from src.features import TIME_FEATURES, CHANNEL_FEATURE
    feature_names = list(TIME_FEATURES) + [CHANNEL_FEATURE]
    input_size = len(feature_names)
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if not (models_dir / "task2_best.pt").exists():
        import json
        import pickle
        import torch
        from src.models_task2 import MLPLarge
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        model = MLPLarge(input_size)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_size": input_size,
            "model_type": "mlp_large",
            "feature_names": feature_names,
        }, models_dir / "task2_best.pt")
        with open(models_dir / "task2_scaler.pkl", "wb") as f:
            pickle.dump(StandardScaler(), f)
        with open(models_dir / "task2_channel_encoder.pkl", "wb") as f:
            pickle.dump(LabelEncoder().fit(["ch1", "ch2", "ch3", "ch4"]), f)
        with open(models_dir / "task2_feature_names.json", "w") as f:
            json.dump(feature_names, f)
    from src.predict_task2 import load_task2_model, predict_share
    model, scaler, enc, names = load_task2_model(models_dir)
    assert len(names) == input_size
    preds = predict_share(
        timeslot_datetime_from=["2024-01-15 20:00:00"],
        channel_id=["ch1"],
        model_dir=models_dir,
    )
    assert len(preds) == 1
    assert isinstance(preds[0], (float, int))
