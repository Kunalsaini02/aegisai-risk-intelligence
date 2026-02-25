from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import sys
import os
from app.agents.investigation_agent import InvestigationAgent

agent = InvestigationAgent()

# Add ml_pipeline path
sys.path.append(os.path.join(os.path.dirname(__file__), "ml_pipeline"))

from inference import FraudPredictor

app = FastAPI(title="AegisAI Risk Intelligence Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
prediction_count = 0

predictor = FraudPredictor()


class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Time: float


@app.get("/")
def read_root():
    return {"message": "AegisAI Backend Running Successfully"}


@app.post("/predict")
def predict(transaction: Transaction):

    global prediction_count
    prediction_count += 1

    input_data = transaction.dict()
    result = predictor.predict(input_data)

    return result

@app.post("/investigate")
def investigate(transaction: Transaction):

    global prediction_count
    prediction_count += 1

    input_data = transaction.dict()
    prediction = predictor.predict(input_data)

    report = agent.generate_report(prediction)

    prediction["investigation_report"] = report

    return prediction

@app.get("/health")
def health():
    return {
        "status": "running",
        "model_loaded": True,
        "llm_configured": True
    }

@app.get("/metrics")
def metrics():
    return {
        "total_predictions": prediction_count
    }