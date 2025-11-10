"""
new_card_grade_v3.py
MintCondition Card Grader API (v3)
----------------------------------
• Keeps FastAPI + URL/base64 support from Render build
• Restores precise v2 image-preprocessing & centering logic
• Uses OpenAI Responses API with prompt_id (version 9)
"""

import os
import io
import cv2
import base64
import numpy as np
import requests
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ==========================================================
# INITIAL SETUP
# ==========================================================
app = FastAPI(title="MintCondition Card Grader API v3", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# IMAGE LOADING UTILITIES
# ==========================================================
def load_image(source_url: str | None = None, b64data: str | None = None):
    """Load an image from a URL or base64 string."""
    if source_url:
        try:
            resp = requests.get(source_url, timeout=10)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise RuntimeError(f"Could not load image from URL: {e}")
    elif b64data:
        try:
            if "," in b64data:
                b64data = b64data.split(",")[1]
            img_bytes = base64.b64decode(b64data)
            arr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise RuntimeError(f"Could not decode base64 image: {e}")
    else:
        raise ValueError("No valid image source provided.")


# ==========================================================
# V2 IMAGE HANDLING / CENTERING LOGIC
# ==========================================================
def refine_edges_for_vintage(img_gray):
    """Sharpen true edges and suppress noise using Sobel gradient magnitude."""
    grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Auto threshold adaptively based on brightness
    mean_val = np.mean(norm)
    thresh_val = 30 if mean_val < 100 else 40
    _, edges = cv2.threshold(norm, thresh_val, 255, cv2.THRESH_BINARY)
    return edges


def preprocess_card_image(img):
    """Extract card boundaries and compute centering metrics."""
    if img is None:
        return {"status": "error", "message": "No image provided."}

    # Resize safely for consistency
    maxdim = 1500
    h, w = img.shape[:2]
    scale = maxdim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Contrast/brightness boost before grayscale
    img = cv2.convertScaleAbs(img, alpha=1.15, beta=5)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = refine_edges_for_vintage(gray)

    # Morphological close to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours and pick largest rectangular region
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "No contour found."}

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Distances to image edges (Centering Reference Point logic)
    left = x
    right = img.shape[1] - (x + w)
    top = y
    bottom = img.shape[0] - (y + h)

    # Compute ratios
    horizontal_ratio = round(min(left, right) / max(left, right + 1e-5), 3)
    vertical_ratio = round(min(top, bottom) / max(top, bottom + 1e-5), 3)

    # PSA-style whole-number ratios
    def ratio_to_style(r):
        pct = int(round(r * 100))
        return f"{max(pct, 1)*1}/{(100 - max(pct,1))*1}"

    horiz_style = ratio_to_style(horizontal_ratio)
    vert_style = ratio_to_style(vertical_ratio)

    return {
        "dimensions": {"width": img.shape[1], "height": img.shape[0]},
        "frame_bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "margins": {
            "left": int(left),
            "right": int(right),
            "top": int(top),
            "bottom": int(bottom),
        },
        "ratios": {
            "horizontal": horizontal_ratio,
            "vertical": vertical_ratio,
            "horizontalStyle": horiz_style,
            "verticalStyle": vert_style,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


# ==========================================================
# ROUTES
# ==========================================================
@app.get("/")
def health():
    return {"status": "ok", "message": "MintCondition Card Grader API is live."}


@app.post("/grade_card")
def grade_card(req: CardRequest):
    try:
        # Load front/back images
        front_img = load_image(req.front_url, req.front_b64)
        back_img = load_image(req.back_url, req.back_b64)

        # Run preprocessing logic
        front_data = preprocess_card_image(front_img)
        back_data = preprocess_card_image(back_img)

        # Build combined prompt for OpenAI
        combined_prompt = f"""
        Perform a professional PSA-style grading analysis.
        FRONT DATA: {front_data}
        BACK DATA: {back_data}
        """

        # OpenAI Responses API call using prompt_id (version 9)
        completion = client.responses.create(
            model="gpt-5-mini",
            prompt={"id": "pmpt_690fe027d50c819782c0d8720142b0b00e6d4016c646f820"},
            input=[{"role": "user", "content": combined_prompt}],
        )

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "preprocessing": {"front": front_data, "back": back_data},
            "grading_result": completion.output_text,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


# ==========================================================
# HEAD route (for Render health checks)
# ==========================================================
@app.head("/")
def head_check():
    return {"status": "ok"}
