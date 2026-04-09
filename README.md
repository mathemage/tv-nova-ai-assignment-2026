# Nova TV AI Assignment

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://choosealicense.com/licenses/agpl-3.0/)

Nova TV Interview Assignment: data analysis, deep-learning and transformer-based prediction, containerization and orchestration.

## Requirements

- Python 3.10+
- Create a virtual environment (recommended):
  - Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`
  - Windows: `python -m venv .venv && .venv\Scripts\activate`
- Install dependencies: `pip install -r requirements.txt`

## Data

- Download from assignment URLs (done automatically by scripts if `data/data.csv` is missing):
  - https://pub-db5679e69cc4407a89dc460661812ec6.r2.dev/data.csv
  - Fallback: https://eu2.contabostorage.com/62824c32198b4d53a08054da7a8b4df1:novatv/data.csv
- For testing without network: `python scripts/generate_synthetic_data.py` then use `--data_path data/data_synthetic.csv` for training.

---

## Task 1 – Data analysis and channel conclusions

- Run EDA: `python notebooks/task1_eda.py` (from repo root).
- Conclusions are in [docs/task1_conclusions.md](docs/task1_conclusions.md). Update that file with concrete numbers after running EDA on real data.

---

## Task 2 – Predict "share 15 54" without "share 15 54 3mo mean"

- **Train** (MLP or larger MLP):  
  `python -m src.train_task2 --model mlp_large --out_dir models`  
  Use `--data_path data/data_synthetic.csv` if real data is not available.
- **Explainability**:  
  `python -m src.explain_task2 --model_dir models`
- **Artifacts**: `models/task2_best.pt`, `task2_scaler.pkl`, `task2_channel_encoder.pkl`, `task2_feature_names.json`, `task2_metrics.json`.
- **Summary**: [docs/task2_summary.md](docs/task2_summary.md) (model choice, features, validation metrics, future
  steps).
- **Training/inference cost**: See `models/task2_metrics.json` (train time, n_params, inference latency per sample). Documented in this README: small MLP/LSTM, low cost.
- **Future improvements (Task 2)**: More time features, channel-specific models, handling cold-start for new channels or timeslots; optional SHAP in addition to gradient-based importance.

---

## Task 3 – Improve prediction with "share 15 54 3mo mean" (transformer)

- **Train**:  
  `python -m src.train_task3 --out_dir models`  
  (Optionally `--data_path data/data_synthetic.csv`.)
- **Evaluation**: Last calendar month is held out; metrics in `models/task3_metrics.json`. Attention sample in `models/task3_attention_sample.json`.
- **Summary**: [docs/task3_summary.md](docs/task3_summary.md) (model choice, explainability, last-month performance, future steps).

---

## Task 4 – Containerization, service, orchestration

### Build and run the prediction service (Docker)

- **Docker**: Any recent Docker (e.g. `docker.io` 28.x on Ubuntu) is fine. If you get "permission denied", use `sudo docker` or add your user to the `docker` group: `sudo usermod -aG docker $USER` (then log out and back in).
- Build (from repo root): `docker build -f service/Dockerfile -t novatv-model .` (first build may take several minutes while dependencies download).
- Run: `docker run -p 8000:8000 -v $(pwd)/models:/app/models novatv-model`
- **Verify**: With the container running, open another terminal and run:
  - `curl -s http://localhost:8000/health` → `{"status":"ok"}`
  - `curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"timeslot_datetime_from":["2024-01-15 20:00:00"],"channel_id":["3"]}'` → `{"predictions":[<float>]}`
- The service expects a trained model in `models/` (run Task 2 first, or mount a volume with `task2_best.pt`, `task2_scaler.pkl`, `task2_channel_encoder.pkl`, `task2_feature_names.json`).
- To use versioned models (after `daily_retrain.py`), run with `-e MODEL_DIR=/app/models/current` so the app loads from the `current` symlink.

### API

- **GET /health** – Health check (for orchestration).
- **POST /predict** – (Optional) Predict "share 15 54". Body: `{"timeslot_datetime_from": ["2024-01-15 20:00:00", ...], "channel_id": ["ch1", ...]}`. Response: `{"predictions": [float, ...]}`.

### Orchestration (daily retrain, versioning, optional rollback)

- **Daily retrain**: Run `python scripts/daily_retrain.py` (e.g. via cron at 02:00). It fetches the dataset URL from an env or mock, downloads data, runs Task 2 training, saves the new model under `models/v_YYYYMMDD/`.
- **Model versions**: Stored in `models/v_YYYYMMDD/`. Keep last 7 versions.
- **Rollback**: Run `python scripts/rollback.py` to point the "current" model to the previous version (see scripts for details).
- **Prediction API**: Use the FastAPI container above; it serves predictions from the current model.

### Tests

- Run: `pytest tests/ -v`
- Tests cover: model load + one prediction, health endpoint, prediction endpoint (if implemented).

---

## Project layout

- `data/` – Downloaded CSV and optional cached data.
- `src/` – Data load, features, Task 2 models (MLP, MLPLarge), Task 3 transformer, training, prediction, explainability.
- `models/` – Saved models and versioned dirs (`v_YYYYMMDD/`).
- `service/` – FastAPI app and Dockerfile.
- `scripts/` – Download, synthetic data, daily retrain, rollback.
- `tests/` – Pytest tests for pipeline and API.
- `notebooks/` – Task 1 EDA script.
- `docs/` – Task 1 conclusions, Task 2 summary, Task 3 summary.

All work is executable locally (clone and run); no assumption on OS beyond Python and Docker.

---

## License

This project is licensed under the [GNU Affero General Public License v3.0 or later](LICENSE) (`AGPL-3.0-or-later`).
