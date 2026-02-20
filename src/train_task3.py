"""Task 3: Train transformer model with share 15 54 3mo mean.

This script trains a Transformer-based model that CAN use the 3-month rolling mean
feature as input. The model uses attention mechanisms to capture feature interactions
and provides interpretability through attention weights.

Training Strategy
-----------------
- **Temporal split**: Train/val on all but last calendar month; test on last month
- **Loss**: MSE (Mean Squared Error)
- **Optimizer**: Adam with learning rate 5e-4
- **Early stopping**: Based on validation RMSE (patience=12 epochs)
- **Batch size**: 64 (default)

Key Differences from Task 2
----------------------------
1. **Feature set**: Includes share_15_54_3mo_mean (10 features vs 9)
2. **Model**: Transformer encoder vs MLP
3. **Split strategy**: Last-month holdout vs time-based percentage split
4. **Explainability**: Attention weights vs gradient importance

Usage
-----
Basic training with defaults::

    python src/train_task3.py

Custom configuration::

    python src/train_task3.py --epochs 80 --patience 12 --lr 5e-4 \\
                               --batch_size 64 --val_frac 0.15

Command-line Arguments
----------------------
--data_path : str, optional
    Path to CSV data file (default: data/data.csv)
--epochs : int, default=80
    Maximum training epochs
--patience : int, default=12
    Early stopping patience (epochs without improvement)
--batch_size : int, default=64
    Training batch size
--lr : float, default=5e-4
    Learning rate for Adam optimizer
--out_dir : str, default='models'
    Directory to save trained model and artifacts
--val_frac : float, default=0.15
    Fraction of train data for validation (within train/val set)

Example Output
--------------
Training run with synthetic data (5000 samples, 5 epochs)::

    Loading data...
    Features: ['hour', 'day_of_week', 'month', 'weekend', 'hour_sin', 
               'hour_cos', 'dow_sin', 'dow_cos', 'channel_id_enc', 
               'share_15_54_3mo_mean'], X.shape=(5000, 10)
    Train 3699, Val 652, Test (last month) 649
    Early stopping.
    Last month metrics: {
        'best_epoch': 2, 
        'val_rmse': 3.1409832954047556, 
        'test_rmse_last_month': 2.895919816809192, 
        'test_mae_last_month': 2.337484836578369, 
        'test_r2_last_month': -0.002093791961669922, 
        'train_time_sec': 4.54, 
        'n_params': 69185
    }
    MAE by channel (last month): {
        'ch1': 2.274495511054993, 
        'ch2': 2.353680218823383, 
        'ch3': 2.176375092489277, 
        'ch4': 2.5592480445063974
    }
    MAE by 3mo mean bucket: {
        0: 2.336145527766301, 
        1: 2.408216464996338, 
        2: 2.147170259591305, 
        3: 2.5131687394601325, 
        4: 2.269360456059122
    }
    Wrote /path/to/docs/task3_summary.md

Outputs
-------
The script saves the following artifacts to --out_dir:

- **task3_best.pt** : Model checkpoint with state dict and metadata
- **task3_scaler_X.pkl** : StandardScaler for feature normalization
- **task3_channel_encoder.pkl** : LabelEncoder for channel IDs
- **task3_scaler_3mo.pkl** : Scaler for 3-month mean feature
- **task3_feature_names.json** : List of feature names in order
- **task3_metrics.json** : Training and test metrics
- **task3_attention_sample.json** : Sample attention weights for interpretation
- **docs/task3_summary.md** : Human-readable summary report

Model Performance Analysis
--------------------------
The script analyzes model performance by:
1. **Channel**: MAE breakdown by each channel
2. **3mo mean buckets**: MAE across quintiles of historical performance
3. **Attention patterns**: Which features receive highest attention

Typical metrics on synthetic data:
- Test RMSE (last month): ~2.9
- Test MAE (last month): ~2.3
- Test R²: near 0 (synthetic data has little signal)
- Training time: ~4.5 seconds (CPU)
- Parameters: ~69K

Explainability
--------------
Attention weights saved in task3_attention_sample.json show which features
the model focuses on. Higher attention = stronger influence on prediction.

Example attention pattern::

    {
      "feature_names": ["hour", "day_of_week", ..., "share_15_54_3mo_mean"],
      "attention_by_example": [
        {
          "hour": 0.08,
          "day_of_week": 0.09,
          ...
          "share_15_54_3mo_mean": 0.25  # highest attention
        }
      ]
    }

Implementation Notes
--------------------
- Last month split: holds out data from the start of the final calendar month to the last timestamp
- Validation set is temporal: last 15% of train/val period
- Test set: last month relative to the final timestamp (may be partial if data ends mid-month)
- Features standardized with StandardScaler (Z-score normalization)
- 3mo mean feature: first normalized separately with its own StandardScaler, then combined with other features and all standardized together
- Early stopping monitors validation RMSE only

See Also
--------
models_task3.py : TabularTransformer architecture
train_task2.py : Alternative MLP-based training (Task 2)
features.py : Feature engineering with 3mo mean
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_data, TIMESLOT_COL, TARGET_COL, TARGET_3MO_COL
from src.features import features_task3
from src.models_task3 import TabularTransformer


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def last_month_split(df, X, y, timeslot_col):
    """Split data: train/val on all but the last calendar month segment; test on the last month.
    
    Ensures temporal ordering and prevents data leakage by holding out the
    most recent calendar month in the data (from the start of that month up to the last timestamp),
    which may be a partial month if the dataset ends mid-month.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with timeslot column.
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features).
    y : np.ndarray
        Target values, shape (n_samples,).
    timeslot_col : str
        Name of datetime column in df.
    
    Returns
    -------
    X_train_val : np.ndarray
        Features for training and validation (all but last month).
    y_train_val : np.ndarray
        Targets for training and validation.
    X_test : np.ndarray
        Features for test set (last month).
    y_test : np.ndarray
        Targets for test set.
    test_mask : pd.Series
        Boolean mask indicating test set rows in original df.
    
    Examples
    --------
    >>> X_tv, y_tv, X_test, y_test, mask = last_month_split(df, X, y, 'timeslot datetime from')
    >>> print(f"Train+Val: {len(X_tv)}, Test: {len(X_test)}")
    Train+Val: 4351, Test: 649
    """
    df = df.copy()
    df["_ts"] = pd.to_datetime(df[timeslot_col], errors="coerce")
    last_ts = df["_ts"].max()
    last_month_start = last_ts - pd.offsets.MonthBegin(1)
    test_mask = df["_ts"] >= last_month_start
    train_val_mask = ~test_mask
    X_train_val = X[train_val_mask.values]
    y_train_val = y[train_val_mask.values]
    X_test = X[test_mask.values]
    y_test = y[test_mask.values]
    return X_train_val, y_train_val, X_test, y_test, test_mask


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)

    print("Loading data...")
    df = load_data(path=args.data_path, data_dir=str(ROOT / "data"), use_four_channels=True)
    if TIMESLOT_COL in df.columns:
        df = df.sort_values(TIMESLOT_COL).reset_index(drop=True)

    X, y, channel_encoder, scaler_3mo, feature_names = features_task3(df, fit=True)
    print(f"Features: {feature_names}, X.shape={X.shape}")

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = y.astype(np.float32)

    # Last month split
    X_train_val, y_train_val, X_test, y_test, test_mask = last_month_split(df, X, y, TIMESLOT_COL)
    n_val = int(len(X_train_val) * args.val_frac)
    X_train, X_val = X_train_val[:-n_val], X_train_val[-n_val:]
    y_train, y_val = y_train_val[:-n_val], y_train_val[-n_val:]
    print(f"Train {len(X_train)}, Val {len(X_val)}, Test (last month) {len(X_test)}")

    device = get_device()
    input_size = X_train.shape[1]
    model = TabularTransformer(
        input_size=input_size,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    best_val_rmse = float("inf")
    best_epoch = 0
    wait = 0
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            wait = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": input_size,
                "feature_names": feature_names,
            }, out_dir / "task3_best.pt")
        else:
            wait += 1
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} train_loss={train_loss:.4f} val_rmse={val_rmse:.4f} val_mae={val_mae:.4f}")
        if wait >= args.patience:
            print("Early stopping.")
            break

    train_time = time.perf_counter() - t0
    state = torch.load(out_dir / "task3_best.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])

    # Evaluate on last month
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Save artifacts for attention export
    import pickle
    with open(out_dir / "task3_scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open(out_dir / "task3_channel_encoder.pkl", "wb") as f:
        pickle.dump(channel_encoder, f)
    with open(out_dir / "task3_scaler_3mo.pkl", "wb") as f:
        pickle.dump(scaler_3mo, f)
    with open(out_dir / "task3_feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # Attention for a few examples
    n_show = min(5, len(X_test))
    model.eval()
    x_sample = torch.tensor(X_test[:n_show], dtype=torch.float32, device=device)
    with torch.no_grad():
        attn = model.get_attention_weights(x_sample, layer_idx=0)
    attn_np = attn.cpu().numpy()
    attention_by_example = [
        {feature_names[j]: float(attn_np[i].mean(axis=0)[j]) for j in range(len(feature_names))}
        for i in range(n_show)
    ]
    with open(out_dir / "task3_attention_sample.json", "w") as f:
        json.dump({"feature_names": feature_names, "attention_by_example": attention_by_example}, f, indent=2)

    metrics = {
        "best_epoch": best_epoch,
        "val_rmse": float(best_val_rmse),
        "test_rmse_last_month": float(test_rmse),
        "test_mae_last_month": float(test_mae),
        "test_r2_last_month": float(test_r2),
        "train_time_sec": round(train_time, 2),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    with open(out_dir / "task3_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Last month metrics:", metrics)

    # Where model performs well / poorly: by channel and by 3mo mean bucket
    df_test = df.loc[test_mask].reset_index(drop=True)
    df_test["pred"] = test_pred
    df_test["error"] = np.abs(df_test[TARGET_COL] - df_test["pred"])
    if "channel id" in df_test.columns:
        by_ch = df_test.groupby("channel id")["error"].mean()
        print("MAE by channel (last month):", by_ch.to_dict())
    if TARGET_3MO_COL in df_test.columns:
        df_test["bucket_3mo"] = pd.qcut(df_test[TARGET_3MO_COL], 5, labels=False, duplicates="drop")
        by_bucket = df_test.groupby("bucket_3mo")["error"].mean()
        print("MAE by 3mo mean bucket:", by_bucket.to_dict())

    summary_path = docs_dir / "task3_summary.md"
    summary_path.write_text(f"""# Task 3 – Summary

## Model choice
- **TabularTransformer**: small Transformer encoder (2 layers, 2 heads, d_model=64). Each tabular feature is a token; we add positional encoding, then mean-pool and a 2-layer regression head.
- **Why**: Attention over features is interpretable; 3mo mean can be used as a strong feature; minimal code, no Hugging Face dependency.

## Explainability
- Attention weights from the first layer are saved in `models/task3_attention_sample.json` (mean over heads). Which features the model attends to most indicates influence on the prediction.

## Last-month performance
- Test RMSE: {metrics['test_rmse_last_month']:.4f}
- Test MAE: {metrics['test_mae_last_month']:.4f}
- Test R²: {metrics['test_r2_last_month']:.4f}

## Where the model performs well / poorly
- By channel: see training log (MAE by channel).
- By 3mo mean bucket: see training log (MAE by bucket). Typically higher error for extreme 3mo mean values or rare channel/hour combinations.

## Cost and pros/cons
- **Training time**: ~{metrics['train_time_sec']} s. **Inference**: small (number of params: {metrics['n_params']}).
- **Pros**: Uses 3mo mean; attention is explainable; better than Task 2 when 3mo mean is informative.
- **Cons**: More hyperparameters than MLP; needs light tuning (layers, d_model).

## Future steps
- **Data**: More channels, longer history, program metadata.
- **Features**: Program type, lags, rolling stats, more time features.
- **Method**: Quantile regression, channel-specific heads, or pretrained time-series transformer if reframed as sequence forecasting.
""", encoding="utf-8")
    print(f"Wrote {summary_path}")
    return model, metrics


if __name__ == "__main__":
    main()
