# Task 2 - Summary

## Model choice

- **MLPLarge**: 3 hidden layers (256, 128, 64) with ReLU + dropout (0.15), linear head.
- **Why**: Simple tabular baseline for time/channel features; fast to train and deploy.

## Features

- From `models/current/task2_feature_names.json`.
- `hour`, `day_of_week`, `month`, `weekend`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `channel_id_enc`.

## Training setup

- Time-based split (last 20% as validation).
- StandardScaler for features.
- MSE loss with Adam; early stopping (patience 10).

## Validation performance

- From `models/task2_metrics.json`.
- Best epoch: 10
- Val RMSE: 4.0569
- Val MAE: 2.9482
- Train time: 12.89 s
- Params: 43,777
- Inference latency: 0.1002 ms/sample

## Notes and limitations

- Only time + channel features; no content metadata.
- Does not use the 3mo mean baseline (reserved for Task 3).

## Future steps

- Add lag/rolling features, program metadata, or per-channel heads.
- Compare to the smaller MLP to trade accuracy for speed.
