from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="NeuroBed API", version="1.0.0")

# --- 1. DATA MODELS ---
class PatientAdmission(BaseModel):
    patient_id: str
    age: int
    sex: str
    base_severity: int
    sofa: float
    cci: int
    infection_flag: int
    admission_type: str

class PredictionResponse(BaseModel):
    patient_id: str
    predicted_los: float
    predicted_discharge: datetime
    confidence_score: float
    shap_drivers: dict

# --- 2. ML ENGINE LOADER ---
# In production, load from S3 or local file
try:
    model_pipeline = joblib.load("los_model.pkl")
except:
    model_pipeline = None # Handle gracefully if missing

# --- 3. ENDPOINTS ---

@app.post("/predict_los", response_model=PredictionResponse)
async def predict_los(data: PatientAdmission):
    """
    Inference Endpoint: Calculates LOS based on clinical factors
    """
    # 1. Preprocess Input
    input_df = pd.DataFrame([data.dict()])
    
    # 2. Mock Inference (Replace with model_pipeline.predict(input_df))
    # Simulating logic based on 'synthetic_rule_based.csv' rules
    base_los = 2.0
    base_los += (data.base_severity * 1.5)
    base_los += (data.sofa * 0.4)
    if data.infection_flag: base_los += 3.0
    if data.admission_type == "Emergency": base_los += 1.2
    
    # Add noise for "Confidence Interval" simulation
    predicted_los = round(base_los, 1)
    
    # 3. Calculate Dates
    discharge_dt = datetime.now() + timedelta(days=predicted_los)
    
    return {
        "patient_id": data.patient_id,
        "predicted_los": predicted_los,
        "predicted_discharge": discharge_dt,
        "confidence_score": 0.88,
        "shap_drivers": {
            "Severity Score": data.base_severity * 1.5,
            "Infection": 3.0 if data.infection_flag else 0,
            "SOFA Score": data.sofa * 0.4
        }
    }

@app.get("/bed_status")
async def get_bed_status():
    """
    Returns real-time bed map for the dashboard
    """
    # Mock Database Query
    return [
        {"id": "ICU-01", "status": "Occupied", "patient": "P-902", "type": "ICU"},
        {"id": "ICU-02", "status": "Available", "patient": None, "type": "ICU"},
        {"id": "WARD-A1", "status": "Occupied", "patient": "P-881", "type": "General"},
    ]