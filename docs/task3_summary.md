# Task 3 – Summary

## Model choice
- **TabularTransformer**: small Transformer encoder (2 layers, 2 heads, d_model=64). Each tabular feature is a token; we add positional encoding, then mean-pool and a 2-layer regression head.
- **Why**: Attention over features is interpretable; 3mo mean can be used as a strong feature; minimal code, no Hugging Face dependency.

## Explainability
- Attention weights from the first layer are saved in `models/task3_attention_sample.json` (mean over heads). Which features the model attends to most indicates influence on the prediction.

## Last-month performance

- Test RMSE: 3.9342
- Test MAE: 2.9889
- Test R²: 0.7633

## Where the model performs well / poorly
- By channel: see training log (MAE by channel).
- By 3mo mean bucket: see training log (MAE by bucket). Typically higher error for extreme 3mo mean values or rare channel/hour combinations.

## Cost and pros/cons

- **Training time**: ~301.05 s. **Inference**: small (number of params: 69185).
- **Pros**: Uses 3mo mean; attention is explainable; better than Task 2 when 3mo mean is informative.
- **Cons**: More hyperparameters than MLP; needs light tuning (layers, d_model).

## Training run output

```shell
/home/mathemage/src/github/mathemage/tv-nova-ai-assignment-2026/.venv/bin/python -m src.train_task3 --out_dir models 
Loading data...
/home/mathemage/src/github/mathemage/tv-nova-ai-assignment-2026/src/data.py:62: DtypeWarning: Columns (0: ch3__f_10, 1: ch3__f_11, 2: ch54__f_10, 3: ch54__f_11, 4: ch4__f_10, 5: ch4__f_11) have mixed types. Specify dtype option on import or set low_memory=False.
  df = pd.read_csv(path)
Features: ['hour', 'day_of_week', 'month', 'weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'channel_id_enc', 'share_15_54_3mo_mean'], X.shape=(34160, 10)
Train 28657, Val 5057, Test (last month) 446
Epoch 10 train_loss=16.7049 val_rmse=4.3872 val_mae=3.3247
Epoch 20 train_loss=15.4792 val_rmse=4.2535 val_mae=3.0642
Epoch 30 train_loss=14.9163 val_rmse=4.3874 val_mae=3.2254
Epoch 40 train_loss=14.5291 val_rmse=4.2629 val_mae=3.1378
Epoch 50 train_loss=14.2721 val_rmse=4.2876 val_mae=3.1617
Early stopping.
Last month metrics: {'best_epoch': 43, 'val_rmse': 4.126121599706728, 'test_rmse_last_month': 3.9341844192898368, 'test_mae_last_month': 2.9889113903045654, 'test_r2_last_month': 0.7633025050163269, 'train_time_sec': 301.05, 'n_params': 69185}
MAE by channel (last month): {3: 5.185271077002939, 4: 2.474711570641525, 9: 2.9331202002879158, 54: 1.6632755140596183}
MAE by 3mo mean bucket: {0: 2.311480751381176, 1: 2.474711570641525, 2: 5.185271077002939}
Wrote /home/mathemage/src/github/mathemage/tv-nova-ai-assignment-2026/docs/task3_summary.md
```

## Future steps
- **Data**: More channels, longer history, program metadata.
- **Features**: Program type, lags, rolling stats, more time features.
- **Method**: Quantile regression, channel-specific heads, or pretrained time-series transformer if reframed as sequence forecasting.
