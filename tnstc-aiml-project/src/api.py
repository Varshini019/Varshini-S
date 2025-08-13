from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="TNSTC Demand Forecast API", version="0.1.0")


class DemandRequest(BaseModel):
    route_id: str = Field(..., example="R03")
    stop_id: str = Field(..., example="R03_S07")
    timestamp: datetime = Field(..., example="2025-08-15T08:30:00")
    is_holiday: Optional[bool] = Field(None, description="Override holiday flag; default inferred as 0")
    weather_index: Optional[float] = Field(None, ge=0.0, le=1.0, description="0..1 synthetic weather score; default 0.5")


@app.get("/")
def root():
    return {"status": "ok", "message": "TNSTC Demand Forecast API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


def _load_model():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "demand_model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Train it first.")
    return joblib.load(model_path)


_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = _load_model()
    return _pipeline


@app.post("/predict")
def predict(req: DemandRequest):
    pipeline = _get_pipeline()

    ts = req.timestamp
    features = {
        "route_id": req.route_id,
        "stop_id": req.stop_id,
        "hour": ts.hour,
        "day_of_week": ts.weekday(),
        "is_holiday": int(req.is_holiday) if req.is_holiday is not None else 0,
        "month": ts.month,
        "weather_index": req.weather_index if req.weather_index is not None else 0.5,
    }

    X = pd.DataFrame([features])
    y_pred = pipeline.predict(X)
    predicted_passengers = max(0, float(y_pred[0]))

    return {
        "route_id": req.route_id,
        "stop_id": req.stop_id,
        "timestamp": req.timestamp.isoformat(),
        "predicted_passengers": round(predicted_passengers, 2),
        "units": "persons",
    }