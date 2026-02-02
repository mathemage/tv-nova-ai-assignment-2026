# Task 1 – Conclusions on the four channels

## Data overview

- **Source**: CSV from assignment URLs; filtered to the four selected channel IDs (the four most frequent in the dataset).
- **Key columns**: `timeslot datetime from`, `main indent` (movie ID), `channel id`, `share 15 54` (target), `share 15 54 3mo mean`.
- **Time range**: See EDA output; data is parsed and used for hour/day-of-week/month patterns.

## Viewing patterns

- **By hour**: Prime-time hours typically show higher shares; overnight lower. Exact pattern depends on the data.
- **By day of week**: Weekend vs weekday may differ; EDA script prints mean share by day.
- **By month**: Seasonal effects can be present; EDA prints mean by month.

## Stability of shares

- `share 15 54` and `share 15 54 3mo mean` are correlated; the 3mo mean is a smoothed baseline.
- Channel-level mean and std (from EDA) indicate which channels are more stable vs more variable.

## Differences between channels

- The four channels are compared in the EDA output (rows, mean share, std).
- Some channels may have more data (more timeslots); others may show higher or lower average share and different variance.

## Data quality notes

- Missing or invalid `timeslot datetime from` rows are dropped after parsing.
- Any other missing values in key columns should be handled in Tasks 2–3 (e.g. drop or impute).
- Run `python notebooks/task1_eda.py` to refresh statistics and then update this document with concrete numbers from your run.
