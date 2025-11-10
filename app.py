"""
app.py — Mint Condition Grading API (Improved Prompt Integration)
---------------------------------------------------------------
Upgrades your production Render build to send structured preprocessing
results and clear instructions to OpenAI for improved centering accuracy.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import base64, cv2, numpy as np, json, os
from openai import OpenAI

# ==========================================================
# CONFIGURATION
# ==========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-yourkeyhere")
PROMPT_ID = "pmpt_690fe027d50c819782c0d8720142b0b00e6d4016c646f820"
PROMPT_VERSION = "9"

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Mint Condition Grading API (v1.6 Refined)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# EDGE / BORDER UTILITIES
# ==========================================================
def refine_edges(img_gray):
    grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(norm, 40, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

def measure_borders(edge_img, axis="horizontal"):
    h, w = edge_img.shape
    if axis == "horizontal":
        left = next((x for x in range(w // 2) if np.count_nonzero(edge_img[:, x]) > 5), w)
        right = next((x for x in range(w - 1, w // 2, -1) if np.count_nonzero(edge_img[:, x]) > 5), w)
        return max(left, 2), max(w - right, 2)
    else:
        top = next((y for y in range(h // 2) if np.count_nonzero(edge_img[y, :]) > 0.25 * w), h)
        bottom = next((y for y in range(h - 1, h // 2, -1) if np.count_nonzero(edge_img[y, :]) > 0.25 * w), h)
        return max(top, 2), max(h - bottom, 2)

# ==========================================================
# PREPROCESSING
# ==========================================================
def preprocess_card_image(file: UploadFile, quality=90, max_dim=1500):
    contents = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(contents, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load {file.filename}")

    h, w = img.shape[:2]
    scale = max(h, w) / max_dim
    if scale > 1.0:
        img = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = refine_edges(gray)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found.")

    c = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(c)
    cropped = img[y:y+h_box, x:x+w_box]
    edges_cropped = edges[y:y+h_box, x:x+w_box]

    left_px, right_px = measure_borders(edges_cropped, "horizontal")
    top_px, bottom_px = measure_borders(edges_cropped, "vertical")

    horiz_ratio = round(min(left_px, right_px) / max(left_px, right_px + 1e-5), 3)
    vert_ratio = round(min(top_px, bottom_px) / max(top_px, bottom_px + 1e-5), 3)

    extreme_offcenter = horiz_ratio < 0.1 or vert_ratio < 0.1

    pre_json = {
        "filename": file.filename,
        "dimensions": {"width": w, "height": h},
        "borders": {"top": top_px, "bottom": bottom_px, "left": left_px, "right": right_px},
        "ratios": {"horizontal": horiz_ratio, "vertical": vert_ratio},
        "extreme_offcenter": extreme_offcenter
    }

    _, buffer = cv2.imencode(".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    b64_img = base64.b64encode(buffer).decode("utf-8")

    return pre_json, b64_img

# ==========================================================
# ROUTE: GRADE CARD
# ==========================================================
@app.post("/grade-card")
async def grade_card(front: UploadFile = File(...), back: UploadFile = File(...)):
    front_pre, front_b64 = preprocess_card_image(front)
    back_pre, back_b64 = preprocess_card_image(back)

    combined_pre = {"front_measurements": front_pre, "back_measurements": back_pre}

    # Build clear model instructions
    instruction_text = (
        "You are a PSA-style card grading assistant.\n"
        "Use the provided preprocessing measurements for centering and borders.\n"
        "Do not re-measure centering visually — trust the ratios below.\n"
        "Then evaluate corners, edges, and surface condition from the attached images.\n"
        "Output a structured JSON matching the grading schema (v7.0)."
    )

    print(f"[INFO] Sending {front.filename} / {back.filename} to OpenAI with structured preprocessing.")

    response = client.responses.create(
        prompt={"id": PROMPT_ID, "version": PROMPT_VERSION},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction_text},
                    {"type": "input_text", "text": json.dumps(combined_pre, indent=2)},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{front_b64}"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{back_b64}"}
                ]
            }
        ]
    )

    try:
        result = response.output_text
    except Exception:
        result = str(response)

    return {"grading_result": result, "measurements": combined_pre}

# ==========================================================
# HEALTH CHECK
# ==========================================================
@app.get("/")
def health():
    return {"status": "ok", "message": "MintCondition Card Grader API is live."}
