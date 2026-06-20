import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import your unified workflow modules
from prediction_pipeline import (
    fetch_live_and_historical_data,
    preprocess_raw_data,
    engineer_live_features,
    generate_predictions
)
from alert_engine import (
    get_dynamic_alert_zones,
    generate_dispatch_alert,
    send_email_alert
)

# Initialize environment configuration
load_dotenv()

app = FastAPI(
    title="Gridlock AI - Automated Traffic Traffic Enforcement Engine",
    description="Production-grade API routing LightGBM forecasting telemetry to interactive frontends and GenAI dispatch nodes."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your Vercel deployment to communicate seamlessly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define strict payload validation frames
class AlertRequest(BaseModel):
    judge_email: str

# =====================================================================
# ENDPOINT 1: GET THE LIVE PREDICTIONS (FOR THE VISUAL MAP)
# =====================================================================
@app.get("/api/map_telemetry")
async def get_map_telemetry():
    """
    Triggers the live data pipeline on-demand, calculates spatial features,
    and returns localized prediction coordinates for map rendering.
    """
    print("\n⚡ GET /api/map_telemetry invoked. Compiling live grid maps...")
    try:
        # Run end-to-end ML prediction pipeline
        raw_data = fetch_live_and_historical_data()
        clean_data = preprocess_raw_data(raw_data)
        feature_data = engineer_live_features(clean_data)
        predictions_df = generate_predictions(feature_data)
        
        if predictions_df is None or predictions_df.empty:
            raise HTTPException(status_code=404, detail="Inference pipeline yielded zero active targets.")
            
        # FIX: Re-attach geometric coordinates derived from H3 cell centers
        import h3
        predictions_df['hex_lat'] = predictions_df['h3_cell'].apply(lambda x: h3.cell_to_latlng(x)[0])
        predictions_df['hex_lon'] = predictions_df['h3_cell'].apply(lambda x: h3.cell_to_latlng(x)[1])
        
        # Keep generic latitude/longitude naming for the frontend map compatibility
        predictions_df['latitude'] = predictions_df['hex_lat']
        predictions_df['longitude'] = predictions_df['hex_lon']
        
        # Save state locally as insurance
        predictions_df.to_csv('live_predictions.csv', index=False)
        
        # Convert matrix directly into web-ready JSON packets
        payload = predictions_df.to_dict(orient="records")
        return {"status": "success", "data": payload}
        
    except Exception as e:
        print(f"❌ Internal Pipeline Disruption: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# ENDPOINT 2: TRIGGER THE DYNAMIC GENAI ALERT EMAIL
# =====================================================================
@app.post("/api/trigger_dispatch")
async def trigger_dispatch(payload: AlertRequest):
    """
    Evaluates current risk profiles, synthesizes tactical commands via Gemini,
    and instantly transmits the resulting brief directly to the judge's email inbox.
    """
    print(f"\n⚡ POST /api/trigger_dispatch invoked for recipient: {payload.judge_email}")
    try:
        # 1. Isolate threats using Rank-Based Filtering (Dynamic K)
        critical_zones = get_dynamic_alert_zones(csv_file='live_predictions.csv')
        
        if critical_zones is None or critical_zones.empty:
            return {
                "status": "static", 
                "message": "Enforcement parameters satisfied. No severe threats require escalation."
            }
            
        # 2. Render dispatch directives using Gemini
        dispatch_brief = generate_dispatch_alert(critical_zones)
        
        if not dispatch_brief:
            raise HTTPException(status_code=502, detail="GenAI orchestration module failed to generate data stream.")
            
        # 3. Deliver automated alerting payload to target session address
        send_email_alert(dispatch_brief, receiver_email=payload.judge_email)
        
        return {
            "status": "success",
            "message": f"Tactical dispatch coordinates successfully routed to {payload.judge_email}",
            "preview": dispatch_brief.split("\n")[:5] # Returns snippet to frontend as confirmation
        }
        
    except Exception as e:
        print(f"❌ Internal Alerting Disruption: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Running local dev server directly if executed as master process
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)