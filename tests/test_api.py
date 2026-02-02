"""Test FastAPI health and optional predict endpoint."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def client():
    import os
    os.environ["MODEL_DIR"] = str(ROOT / "models")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "service"))
    from main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_validation(client):
    r = client.post("/predict", json={
        "timeslot_datetime_from": ["2024-01-15 20:00:00"],
        "channel_id": ["ch1", "ch2"],
    })
    assert r.status_code == 400


def test_predict_success_or_503(client):
    """Predict either returns 200 with predictions or 503 if model not found."""
    r = client.post("/predict", json={
        "timeslot_datetime_from": ["2024-01-15 20:00:00"],
        "channel_id": ["ch1"],
    })
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert "predictions" in r.json()
        assert len(r.json()["predictions"]) == 1
