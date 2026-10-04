"""Minimal serving stub: FastAPI around IntentRouter.

    MODEL_ID=Badalt/intent-classifier-mpnet uvicorn src.serve:app --port 8000
    curl -s localhost:8000/predict -H 'content-type: application/json' -d '{"text": "eta on LD-55501?"}'

Endpoints: GET /health, POST /predict {"text"}, POST /predict_batch {"texts": [...]}.
"""
from __future__ import annotations

import os
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import IntentRouter

MODEL_ID = os.environ.get("MODEL_ID", "outputs/model_a/model")
router = IntentRouter(MODEL_ID, device=os.environ.get("DEVICE"))
app = FastAPI(title="Intent router", version="1.0")


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class BatchQuery(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "device": router.device,
            "unknown_threshold": router.threshold, "temperature": router.temperature,
            "labels": list(router.id2label.values())}


@app.post("/predict")
def predict(q: Query) -> dict:
    t0 = time.perf_counter()
    r = router.predict_batch([q.text])[0]
    return {**r, "latency_ms": round(1000 * (time.perf_counter() - t0), 2)}


@app.post("/predict_batch")
def predict_batch(q: BatchQuery) -> dict:
    t0 = time.perf_counter()
    rs = router.predict_batch(q.texts)
    return {"results": rs, "latency_ms": round(1000 * (time.perf_counter() - t0), 2)}
