"""
Task 2: Train MLP (primary) and MLPLarge (most complex for Task 4).
Time-based split, MSE loss, Adam, early stopping. Save best model and scaler/encoder.
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
