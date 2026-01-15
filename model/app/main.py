from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
from contextlib import asynccontextmanager
from schemas import TenantScoreRequest

# --- GESTION DU CYCLE DE VIE (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code exécuté au démarrage
    MODEL_PATH = "model/models/tenant_risk_model.pkl"
    
    if os.path.exists(MODEL_PATH):
        app.state.model = joblib.load(MODEL_PATH)
        print("✅ Production Model Loaded Successfully.")
    else:
        app.state.model = None
        print(f"❌ CRITICAL: Model file not found at {MODEL_PATH}")
    
    yield
    # Code exécuté à l'arrêt (si besoin)
    print("Shutting down Tenant Risk API...")

# --- CONFIGURATION ---
app = FastAPI(title="Tenant Risk Scoring AI", lifespan=lifespan)

# --- ROUTES ---

@app.get("/")
def health_check():
    return {
        "status": "Online",
        "model_loaded": app.state.model is not None
    }

@app.post("/predict/score")
def predict_risk_score(data: TenantScoreRequest):
    if not app.state.model:
        raise HTTPException(status_code=503, detail="AI Model is not loaded on server.")

    # Approbation automatique pour les dossiers parfaits
    if data.missedPeriods == 0 and data.totalDisputes == 0:
        return {
            "trust_score": 100,
            "risk_category": "Safe",
            "recommendation": "Approve"
        }

    try:
        # 1. Préparation des données
        features = {
            "missedPeriods": data.missedPeriods,
            "totalDisputes": data.totalDisputes
        }
        model_input = pd.DataFrame([features])

        # 2. Prédiction des probabilités
        # [Probabilité_Mauvais, Probabilité_Bon] -> On prend l'index 1 (Bon)
        probs = app.state.model.predict_proba(model_input)
        trust_probability = probs[0][1] 

        # 3. Calcul du score (0-100)
        final_score = int(trust_probability * 100)

        # 4. Catégorisation
        if final_score > 75:
            category = "Safe"
            recommendation = "Approve"
        elif final_score < 40:
            category = "Risky"
            recommendation = "Review Manually"
        else:
            category = "Moderate"
            recommendation = "Review Manually"

        return {
            "trust_score": final_score,
            "risk_category": category,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# --- LANCEMENT ---
if __name__ == "__main__":
    import uvicorn
    # Port 8000 pour la cohérence avec le Dockerfile
    uvicorn.run(app, host="0.0.0.0", port=8000)
