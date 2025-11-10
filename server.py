import io
import os
import cv2
import base64
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from openai import OpenAI

# ==========================================================
# APP CONFIG
# ==========================================================
app = FastAPI(title="MintCondition Card Grader API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo: allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================================================
# REQUEST MODEL
# ==========================================================
class CardRequest(BaseModel):
    front_url: str | None = None
    back_url: str | None = None
    front_b64: str | None = None
    back_b64: str | None = None
    quality: int = 90
    maxdim: int = 1500


# ==========================================================
# IMAGE LOADING HELPER
# ==========================================================
def load_image(source_url: str | None = None, b64data: str | None = None):
    """Load image from URL or base64 string."""
    if source_url:
        try:
            resp = requests.get(source_url, timeout=10)
            resp.raise_for_status()
            img_arr = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise RuntimeError(f"Could not load image from URL: {e}")

    elif b64data:
        try:
            if "," in b64data:
                b64data = b64data.split(",")[1]  # remove data URL prefix
            img_bytes = base64.b64decode(b64data)
            img_arr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise RuntimeError(f"Could not decode base64 image: {e}")

    else:
        raise ValueError("No valid image source provided.")


# ==========================================================
# PREPROCESSING PLACEHOLDER
# ==========================================================
def preprocess_card_image(img):
    """Placeholder preprocessing logic."""
    if img is None:
        return {"status": "error", "message": "No image provided."}

    height, width = img.shape[:2]
    mean_intensity = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

    return {
        "dimensions": {"width": width, "height": height},
        "mean_intensity": mean_intensity,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ==========================================================
# ROUTES
# ==========================================================
@app.get("/")
def health_check():
    return {"status": "ok", "message": "MintCondition Card Grader API is live."}


@app.post("/grade_card")
def grade_card_endpoint(req: CardRequest):
    try:
        # Load images (front & back)
        front_img = load_image(req.front_url, req.front_b64)
        back_img = load_image(req.back_url, req.back_b64)

        # Run preprocessing
        front_data = preprocess_card_image(front_img)
        back_data = preprocess_card_image(back_img)

        # Combine and send to OpenAI (this can be replaced by your grading prompt)
        combined_prompt = f"""
        Given the following card preprocessing data:
        FRONT: {front_data}
        BACK: {back_data}
        Estimate condition and PSA-style grading summary.
        """

        completion = client.responses.create(
            model="gpt-5-mini",
            input=[{"role": "user", "content": combined_prompt}],
        )

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "preprocessing": {
                "front": front_data,
                "back": back_data,
            },
            "grading_result": completion.output_text,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
