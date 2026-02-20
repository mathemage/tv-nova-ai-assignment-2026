"""Task 2: Train MLP (primary) and MLPLarge (most complex for Task 4).

This script trains deep learning models to predict TV viewership share (share 15 54)
from temporal and channel features, WITHOUT using the 3-month rolling mean feature.

Training Strategy
-----------------
- Time-based train/validation split (last 20% for validation)
- MSE loss with Adam optimizer
- Early stopping based on validation RMSE
- Saves best model checkpoint, scaler, and feature metadata

Usage
-----
Basic training with defaults::

    python src/train_task2.py

Custom configuration::

    python src/train_task2.py --model mlp --epochs 100 --patience 10 \\
                               --batch_size 64 --lr 1e-3

Command-line Arguments
----------------------
--data_path : str, optional
    Path to CSV data file (default: data/data.csv)
--model : {'mlp', 'mlp_large'}, default='mlp_large'
    Model architecture to train
--epochs : int, default=100
    Maximum training epochs
--patience : int, default=10
    Early stopping patience (epochs without improvement)
--batch_size : int, default=64
    Training batch size
--lr : float, default=1e-3
    Learning rate for Adam optimizer
--out_dir : str, default='models'
    Directory to save trained model and artifacts
--val_frac : float, default=0.2
    Fraction of data for validation

Example Output
--------------
Training run with synthetic data (5000 samples, 5 epochs)::

    Loading data...
    Features: ['hour', 'day_of_week', 'month', 'weekend', 'hour_sin', 
               'hour_cos', 'dow_sin', 'dow_cos', 'channel_id_enc'], X.shape=(5000, 9)
    Train 4000, Val 1000
    Metrics: {
        'best_epoch': 3, 
        'val_rmse': 3.035895808586932, 
        'val_mae': 2.450582981109619, 
        'train_time_sec': 0.92, 
        'n_params': 43777, 
        'inference_latency_ms_per_sample': 0.1109
    }
    Saved model to models/task2_best.pt

Outputs
-------
The script saves the following artifacts to --out_dir:

- **task2_best.pt** : Model checkpoint with state dict and metadata
- **task2_scaler.pkl** : StandardScaler for feature normalization
- **task2_channel_encoder.pkl** : LabelEncoder for channel IDs
- **task2_feature_names.json** : List of feature names in order
- **task2_metrics.json** : Training metrics and model statistics

Model Performance
-----------------
Typical metrics on synthetic data:
- Validation RMSE: ~3.0
- Validation MAE: ~2.4
- Training time: <1 second (CPU, 5000 samples)
- Inference latency: ~0.1 ms per sample
- Parameters (for input_size=9, hidden_sizes=(128, 64)): ~44K (MLPLarge), ~9.5K (MLP).
  Actual parameter counts depend on the chosen input dimensionality and hidden layer sizes.

Notes
-----
- Uses time-based split to prevent data leakage
- Features are standardized (StandardScaler)
- Model trained without share 15 54 3mo mean feature
- Inference speed measured on first sample (100 iterations)

See Also
--------
models_task2.py : Model architectures
predict_task2.py : Inference using trained models
features.py : Feature engineering functions
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_data, TIMESLOT_COL
from src.features import features_task2
from src.models_task2 import MLP, MLPLarge


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def time_based_split(X, y, test_fraction=0.2):
    """Split by time: last test_fraction of rows (assumed sorted by time)."""
    n = len(X)
    split = int(n * (1 - test_fraction))
    return X[:split], X[split:], y[:split], y[split:]


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        pred = model(Xt).cpu().numpy()
    return np.sqrt(mean_squared_error(y, pred)), mean_absolute_error(y, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Path to CSV (default: data/data.csv)")
    parser.add_argument("--model", type=str, default="mlp_large", choices=["mlp", "mlp_large"], help="Which model to train (mlp_large = most complex for Task 4)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(path=args.data_path, data_dir=str(ROOT / "data"), use_four_channels=True)
    if TIMESLOT_COL in df.columns:
        df = df.sort_values(TIMESLOT_COL).reset_index(drop=True)

    X, y, channel_encoder, feature_names = features_task2(df, fit=True)
    print(f"Features: {feature_names}, X.shape={X.shape}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.astype(np.float32)

    # Time-based split
    X_train, X_val, y_train, y_val = time_based_split(X, y, test_fraction=args.val_frac)
    print(f"Train {len(X_train)}, Val {len(X_val)}")

    device = get_device()
    input_size = X_train.shape[1]

    if args.model == "mlp":
        model = MLP(input_size, hidden_sizes=(128, 64)).to(device)
    else:
        model = MLPLarge(input_size, hidden_sizes=(256, 128, 64)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_val_rmse = float("inf")
    best_epoch = 0
    wait = 0
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_rmse, val_mae = evaluate(model, X_val, y_val, device)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            wait = 0
            state = {
                "model_state_dict": model.state_dict(),
                "input_size": input_size,
                "model_type": args.model,
                "feature_names": feature_names,
            }
            torch.save(state, out_dir / "task2_best.pt")
        else:
            wait += 1
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} train_loss={train_loss:.4f} val_rmse={val_rmse:.4f} val_mae={val_mae:.4f}")
        if wait >= args.patience:
            print(f"Early stop at epoch {epoch+1}")
            break

    train_time = time.perf_counter() - t0
    # Reload best and final metrics
    state = torch.load(out_dir / "task2_best.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    final_rmse, final_mae = evaluate(model, X_val, y_val, device)

    # Save artifacts for inference
    import pickle
    with open(out_dir / "task2_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / "task2_channel_encoder.pkl", "wb") as f:
        pickle.dump(channel_encoder, f)
    with open(out_dir / "task2_feature_names.json", "w") as f:
        json.dump(feature_names, f)

    n_params = sum(p.numel() for p in model.parameters())
    # Inference latency (single sample)
    model.eval()
    x_sample = torch.tensor(X_val[:1], dtype=torch.float32, device=device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.perf_counter()
    for _ in range(100):
        _ = model(x_sample)
    torch.cuda.synchronize() if device.type == "cuda" else None
    latency_ms = (time.perf_counter() - t1) / 100 * 1000

    metrics = {
        "best_epoch": best_epoch + 1,
        "val_rmse": float(final_rmse),
        "val_mae": float(final_mae),
        "train_time_sec": round(train_time, 2),
        "n_params": n_params,
        "inference_latency_ms_per_sample": round(latency_ms, 4),
    }
    with open(out_dir / "task2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", metrics)
    print(f"Saved model to {out_dir / 'task2_best.pt'}")
    return model, metrics


if __name__ == "__main__":
    main()
