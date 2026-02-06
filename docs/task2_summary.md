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

## Explainability notes

- `src/explain_task2.py` loads the saved Task 2 model + artifacts, rebuilds Task 2 features, scales them, then computes
  mean absolute gradients per feature to rank importance.
- Gradient-based importance = average $\left| \frac{\partial \mathrm{output}}{\partial \mathrm{feature}} \right|$ over
  sampled rows; higher means the prediction is more sensitive to that feature (in scaled input space), not causal
  influence.
- "Features sorted by importance" are simply the features ordered by those mean |grad| scores, descending.
- The sample output shows `channel_id_enc` as the most influential, followed by time-derived features (`hour_sin`,
  `hour_cos`, `dow_sin`, `dow_cos`, etc.), indicating strong channel and temporal effects.
- The `DtypeWarning` in `src/data.py` comes from `pd.read_csv` seeing mixed types in unused columns; it does not affect
  Task 2 features but can be silenced via explicit `dtype` or `low_memory=False`.
