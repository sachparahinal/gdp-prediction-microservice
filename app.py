from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(
    title="GDP Prediction Microservice",
    description="Predict GDP per capita using a polynomial regression model (NumPy).",
    version="1.0.0",
)

# Training data used previously
GDP_VALUES = np.array([
    135.0, 106.0, 105.0, 95.0, 88.0,
    85.0, 82.0, 81.0, 73.0, 65.0,
    63.0, 61.0, 60.0, 58.0, 57.0,
    55.0, 54.0, 52.0, 51.0, 42.0,
    51.0, 35.0, 39.0, 33.0, 54.0,
    30.0, 31.0, 28.0, 23.0, 21.0
], dtype=np.float32)

RANKS = np.arange(1, 31, dtype=np.float32)
MIN_RANK = 1
MAX_RANK = 30

# Train 3rd-degree polynomial model
coeffs = np.polyfit(RANKS, GDP_VALUES, 3)
model = np.poly1d(coeffs)

class RankRequest(BaseModel):
    ranks: list[int]

@app.get("/")
def home():
    return {
        "message": "GDP Prediction Microservice Running",
        "example_single": "/predict?rank=5",
        "example_batch": {"ranks": [1, 10, 20]},
    }

@app.get("/predict")
def predict(rank: int):
    if rank < MIN_RANK or rank > MAX_RANK:
        raise HTTPException(400, f"Rank must be between {MIN_RANK} and {MAX_RANK}.")
    prediction = model(rank)
    return {"rank": rank, "predicted_gdp": float(round(prediction, 2))}

@app.post("/predict_batch")
def predict_batch(req: RankRequest):
    results = []
    for r in req.ranks:
        if r < MIN_RANK or r > MAX_RANK:
            raise HTTPException(400, f"Rank {r} out of valid range.")
        results.append(float(round(model(r), 2)))
    return {"ranks": req.ranks, "predictions": results}
