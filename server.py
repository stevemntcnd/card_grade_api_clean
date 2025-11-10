# ==========================================================
# server.py — FastAPI wrapper for card grading pipeline
# ==========================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from new_card_grade import preprocess_card_image, grade_card
import json
import datetime

app = FastAPI(
    title="MintCondition Card Grader API",
    description="Runs pre-processing and grading using your OpenCV + OpenAI pipeline.",
    version="1.1"
)

# ==========================================================
# Request Schema
# ==========================================================
class CardRequest(BaseModel):
    front_url: str
    back_url: str
    quality: int = 90
    maxdim: int = 1500

# ==========================================================
# Health Check
# ==========================================================
@app.get("/")
def root():
    return {
        "status": "ok",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "message": "MintCondition Card Grader API is live."
    }

# ==========================================================
# Main Endpoint
# ==========================================================
@app.post("/grade_card")
def grade_card_endpoint(req: CardRequest):
    """
    Runs preprocessing on front/back images and submits them to OpenAI for grading.
    Returns both preprocessing JSON and grading results.
    """
    try:
        print("==========================================================")
        print(f"[INFO] Starting card grading at {datetime.datetime.now()}")
        print(f"[INFO] Front URL: {req.front_url}")
        print(f"[INFO] Back  URL: {req.back_url}")
        print("==========================================================")

        # --- Step 1: Preprocess images ---
        pre_json_front, front_b64 = preprocess_card_image(req.front_url, req.quality, req.maxdim)
        pre_json_back, back_b64 = preprocess_card_image(req.back_url, req.quality, req.maxdim)

        combined_pre = {
            "front_measurements": pre_json_front,
            "back_measurements": pre_json_back
        }

        print("[INFO] Preprocessing complete. Sending to OpenAI grading prompt...")

        # --- Step 2: Call OpenAI model ---
        result = grade_card(front_b64, back_b64, combined_pre)

        # --- Step 3: Log raw model output ---
        print("----------------------------------------------------------")
        print("[MODEL RESPONSE]")
        print(result)
        print("----------------------------------------------------------")

        return {
            "success": True,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "preprocessing": combined_pre,
            "grading_result": result
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
