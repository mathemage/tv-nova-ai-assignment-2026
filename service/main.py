"""
Task 4: FastAPI wrapper for Task 2 model.
Endpoints: GET /health, POST /predict (optional).
"""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="Nova TV Share Prediction", version="1.0")
MODEL_DIR = os.environ.get("MODEL_DIR", str(ROOT / "models"))


class PredictRequest(BaseModel):
    timeslot_datetime_from: list[str]
    channel_id: list[str]


class PredictResponse(BaseModel):
    predictions: list[float]


@app.get("/health")
def health():
    """Health check for orchestration."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict share 15 54 for each (timeslot, channel_id)."""
    if len(req.timeslot_datetime_from) != len(req.channel_id):
        raise HTTPException(400, "timeslot_datetime_from and channel_id must have the same length")
    try:
        from src.predict_task2 import predict_share
        preds = predict_share(
            timeslot_datetime_from=req.timeslot_datetime_from,
            channel_id=req.channel_id,
            model_dir=MODEL_DIR,
        )
        return PredictResponse(predictions=preds)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model not loaded: {e}")
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
