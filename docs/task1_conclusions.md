# Task 1 – Conclusions on the four channels

## Data overview

- **Source**: CSV from assignment URLs; filtered to the four selected channel IDs (the four most frequent in the dataset).
- **Key columns**: `timeslot datetime from`, `main_ident` (movie ID), `channel id`, `share 15 54` (target), `share 15 54 3mo mean`.
- **Shape**: 34,160 rows × 53 columns.
- **Time range**: 2022-01-01 17:10:00 to 2024-01-13 23:45:00.

## Viewing patterns

- **By hour**: Prime-time hours 19–21 show the highest mean share (19h: 11.24, 20h: 10.66, 21h: 11.44). Earlier hours 16–17 are lower (8.3–9.2); 22–23 decline to ~9.7–10.8.
- **By day of week** (0=Mon … 6=Sun): Tuesday highest (11.27), Monday–Friday 10.4–11.3; weekend lower (Saturday 9.68, Sunday 9.66).
- **By month**: October highest (11.46), November 10.98, September 11.09; December lowest (9.75). Slight seasonal pattern with higher shares in autumn.

## Stability of shares

- **share 15 54**: mean = 10.36, std = 8.70 (wide spread).
- **share 15 54 3mo mean**: mean = 10.58, std = 7.88 (smoother, lower variance).
- The two are correlated; the 3mo mean is a smoothed baseline and is useful as a feature in Task 3.

## Differences between channels

| Channel id | Rows  | Mean share | Std   |
|------------|-------|------------|-------|
| 3          | 8,545 | 21.71      | 7.17  |
| 4          | 7,392 | 12.12      | 5.57  |
| 9          | 8,753 | 5.39       | 2.80  |
| 54         | 9,470 | 3.34       | 2.15  |

- Channel 3 has the highest average share and variance; channels 9 and 54 are much lower and more stable. Channel 4 is in between.

## Data quality notes

- Missing or invalid `timeslot datetime from` rows are dropped after parsing.
- CSV read reports mixed types in some columns (e.g. ch3__f_10, ch54__f_10); consider `dtype` or `low_memory=False` when loading.
- Run `python notebooks/task1_eda.py` to refresh statistics and then update this document with concrete numbers from your run.

## Modeling implications (beginner-friendly)

- Treat **time as a primary signal**: include hour, day-of-week, and month features; consider categorical bins or cyclic
  encodings for hour/month.
- **Channel effects dominate**: model channel as a strong categorical feature; consider per-channel models or explicit
  interactions with time features.
- Use `share 15 54 3mo mean` as a **trend baseline**; compare your model to it and consider modeling residuals on top of
  the baseline.
- **Variance differs by channel** (channel 3 is volatile): evaluate errors per channel and consider robust loss or
  per-channel weighting if needed.
- Prefer **time-based splits** (train on earlier periods, validate on later) to avoid leakage from future time patterns.
