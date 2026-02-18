"""Task 2 explainability: gradient-based feature importance and optional SHAP.

This script analyzes which features most influence the Task 2 model's predictions
using gradient-based attribution methods. Results help understand model behavior
and feature relevance.

Methods
-------
- **Gradient-based importance**: Mean absolute gradient w.r.t. input features
- Computed on a sample of validation data
- Higher values indicate stronger feature influence

Usage
-----
Basic usage with defaults::

    python src/explain_task2.py

Custom configuration::

    python src/explain_task2.py --model_dir models --data_path data/data.csv

Command-line Arguments
----------------------
--model_dir : str, default='models'
    Directory containing trained model artifacts
--data_path : str, optional
    Path to data CSV (default: data/data.csv)

Example Output
--------------
Feature importance ranking (synthetic data, MLPLarge model)::

    Gradient-based feature importance (mean |grad|):
      dow_cos: 0.607786
      hour_cos: 0.571952
      weekend: 0.564178
      hour_sin: 0.562173
      dow_sin: 0.509978
      day_of_week: 0.501377
      hour: 0.498369
      month: 0.462932
      channel_id_enc: 0.350756

Interpretation
--------------
The output shows that:
- **Cyclic time features** (dow_cos, hour_cos, hour_sin, dow_sin) have the
  highest importance, indicating strong temporal patterns
- **Weekend indicator** is highly influential for predictions
- **Channel encoding** has lower importance, suggesting patterns are more
  temporal than channel-specific
- **Month** has moderate importance, capturing seasonal trends

Output Format
-------------
Returns a dictionary mapping feature names to importance scores:
    {'feature_name': importance_value, ...}
Values are normalized gradients (higher = more important).

Implementation Details
----------------------
1. Load trained model from model_dir
2. Load and preprocess data (same as training)
3. Sample up to 500 random examples
4. Compute gradients of model output w.r.t. inputs
5. Average absolute gradients across samples
6. Sort features by importance

Notes
-----
- Uses gradient magnitudes, not SHAP (faster for large models)
- Sampling reduces computation time
- Results are deterministic (fixed random seed)
- Works with both MLP and MLPLarge architectures

See Also
--------
train_task2.py : Training script for Task 2 models
models_task2.py : Model architectures
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
    """Compute mean absolute gradient w.r.t. inputs as feature importance.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model in eval mode.
    X : np.ndarray
        Input features, shape (n_samples, n_features).
    device : torch.device
        Device for computation (CPU or CUDA).
    n_sample : int, default=500
        Maximum number of samples to use (for efficiency).
    
    Returns
    -------
    np.ndarray
        Feature importance scores, shape (n_features,).
        Higher values indicate stronger influence.
    
    Notes
    -----
    - Uses backpropagation to compute input gradients
    - Averages absolute gradients across samples
    - Fixed random seed (42) for reproducibility
    """
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
