from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
from src.schemas import TenantScoreRequest

# --- CONFIGURATION ---
app = FastAPI(title="Tenant Risk Scoring AI")
MODEL_PATH = "model_artifacts/tenant_risk_model.pkl"
model = None

# --- STARTUP: Load the Model ---
@app.on_event("startup")
def load_ai_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Production Model Loaded Successfully.")
    else:
        print("❌ CRITICAL WARNING: Model file not found. Please run train_model.py first.")

# --- PREDICTION ENDPOINT ---
@app.post("/predict/score")
def predict_risk_score(data: TenantScoreRequest):
    """
    Input: { "missedPeriods": 2, "totalDisputes": 1 }
    Output: { "trust_score": 12, "risk_category": "Risky", "recommendation": "Review Manually" }
    """
    if not model:
        raise HTTPException(status_code=503, detail="AI Model is not loaded.")

    try:
        # 1. Prepare Data for Model
        # The column names MUST match the training data exactly.
        features = {
            "missedPeriods": data.missedPeriods,
            "totalDisputes": data.totalDisputes
        }
        
        # Create a single-row DataFrame
        model_input = pd.DataFrame([features])

        # 2. Make Prediction
        # The model returns [Probability_Bad, Probability_Good]
        # We want the probability of "Good" (Index 1)
        probs = model.predict_proba(model_input)
        trust_probability = probs[0][1] 

        # 3. Convert to Integer Score (0-100)
        final_score = int(trust_probability * 100)

        # 4. Determine Category & Recommendation
        if final_score > 75:
            category = "Safe"
            recommendation = "Approve"
        elif final_score < 40:
            category = "Risky"
            recommendation = "Review Manually"
        else:
            category = "Moderate"
            recommendation = "Review Manually"

        # 5. Return EXACT JSON Response requested
        return {
            "trust_score": final_score,
            "risk_category": category,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")