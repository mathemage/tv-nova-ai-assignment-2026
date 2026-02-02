# Task 3 – Summary

## Model choice
- **TabularTransformer**: small Transformer encoder (2 layers, 2 heads, d_model=64). Each tabular feature is a token; we add positional encoding, then mean-pool and a 2-layer regression head.
- **Why**: Attention over features is interpretable; 3mo mean can be used as a strong feature; minimal code, no Hugging Face dependency.

## Explainability
- Attention weights from the first layer are saved in `models/task3_attention_sample.json` (mean over heads). Which features the model attends to most indicates influence on the prediction.

## Last-month performance
- Test RMSE: 4.0568
- Test MAE: 3.0094
- Test R²: 0.7483

## Where the model performs well / poorly
- By channel: see training log (MAE by channel).
- By 3mo mean bucket: see training log (MAE by bucket). Typically higher error for extreme 3mo mean values or rare channel/hour combinations.

## Cost and pros/cons
- **Training time**: ~258.98 s. **Inference**: small (number of params: 69185).
- **Pros**: Uses 3mo mean; attention is explainable; better than Task 2 when 3mo mean is informative.
- **Cons**: More hyperparameters than MLP; needs light tuning (layers, d_model).

## Future steps
- **Data**: More channels, longer history, program metadata.
- **Features**: Program type, lags, rolling stats, more time features.
- **Method**: Quantile regression, channel-specific heads, or pretrained time-series transformer if reframed as sequence forecasting.
