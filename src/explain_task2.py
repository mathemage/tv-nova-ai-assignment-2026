"""
Task 2 explainability: gradient-based feature importance and optional SHAP.
Summarize which features influence the prediction most.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models_task2 import MLP, MLPLarge


def load_model_and_artifacts(model_dir: Path, device: torch.device):
    import json
    import pickle
    state = torch.load(model_dir / "task2_best.pt", map_location=device)
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
    model.to(device)
    model.eval()
    return model, scaler, channel_encoder, feature_names


def gradient_importance(model, X: np.ndarray, device: torch.device, n_sample: int = 500):
    """Mean absolute gradient w.r.t. inputs as feature importance."""
    if len(X) > n_sample:
        idx = np.random.default_rng(42).choice(len(X), n_sample, replace=False)
        X = X[idx]
    x = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
    out = model(x)
    out.sum().backward()
    imp = x.grad.abs().mean(dim=0).cpu().numpy()
    return imp


def run_explainability(model_dir: str = "models", data_path: str = None):
    from src.data import load_data
    from src.features import features_task2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(model_dir)
    model, scaler, channel_encoder, feature_names = load_model_and_artifacts(model_dir, device)

    df = load_data(path=data_path, data_dir=str(ROOT / "data"), use_four_channels=True)
    X, y, _, _ = features_task2(df, channel_encoder=channel_encoder, fit=False)
    X = scaler.transform(X)

    imp = gradient_importance(model, X, device)
    order = np.argsort(imp)[::-1]
    print("Gradient-based feature importance (mean |grad|):")
    for i in order:
        print(f"  {feature_names[i]}: {imp[i]:.6f}")
    return dict(zip(feature_names, imp.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--data_path", default=None)
    args = parser.parse_args()
    run_explainability(model_dir=args.model_dir, data_path=args.data_path)


if __name__ == "__main__":
    main()
