"""
Mint Condition Card Grader API (v1.8)
-------------------------------------
Refined for extreme off-center detection (90/10+).
Integrates updated Prompt Version 10.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import base64, cv2, numpy as np, json, os
from openai import OpenAI

# ==========================================================
# CONFIG
# ==========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PROMPT_ID = "pmpt_690fe027d50c819782c0d8720142b0b00e6d4016c646f820"
PROMPT_VERSION = "10"

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Mint Condition Grader API (v1.8 Extreme Centering Fix)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# EDGE UTILITIES
# ==========================================================
def refine_edges(img_gray):
    """Detect faint outer card border, even against similar backgrounds."""
    grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Lower threshold + heavier dilation to include faint dark borders
    _, edges = cv2.threshold(norm, 15, 255, cv2.THRESH_BINARY)
    edges = cv2.dilate(edges, np.ones((7,7), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return edges


def measure_borders(edge_img, axis="horizontal"):
    """Measure left/right or top/bottom border pixel thickness."""
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
    """Extract border geometry and compute PSA-style centering ratios."""
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
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x_coords = [p[0] for p in box]
    y_coords = [p[1] for p in box]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    w_box, h_box = x_max - x_min, y_max - y_min
    x, y = x_min, y_min

    cropped = img[y:y+h_box, x:x+w_box]
    edges_cropped = edges[y:y+h_box, x:x+w_box]

    left_px, right_px = measure_borders(edges_cropped, "horizontal")
    top_px, bottom_px = measure_borders(edges_cropped, "vertical")

    horiz_ratio = round(min(left_px, right_px) / max(left_px, right_px + 1e-5), 3)
    vert_ratio = round(min(top_px, bottom_px) / max(top_px, bottom_px + 1e-5), 3)

    # Identify extreme off-center (≈ 85/15 or worse)
    extreme_offcenter = horiz_ratio < 0.15 or vert_ratio < 0.15
    ceiling = 2.0 if extreme_offcenter else None

    pre_json = {
        "filename": file.filename,
        "dimensions": {"width": w, "height": h},
        "borders": {"top": top_px, "bottom": bottom_px, "left": left_px, "right": right_px},
        "ratios": {"horizontal": horiz_ratio, "vertical": vert_ratio},
        "extreme_offcenter": extreme_offcenter,
        "ceiling": ceiling,
    }

    _, buffer = cv2.imencode(".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    b64_img = base64.b64encode(buffer).decode("utf-8")
    return pre_json, b64_img


# ==========================================================
# GRADE ENDPOINT
# ==========================================================
@app.post("/grade-card")
async def grade_card(front: UploadFile = File(...), back: UploadFile = File(...)):
    """Main grading endpoint — preprocess + call OpenAI vision model."""
    front_pre, front_b64 = preprocess_card_image(front)
    back_pre, back_b64 = preprocess_card_image(back)
    combined_pre = {"front_measurements": front_pre, "back_measurements": back_pre}

    instruction_text = (
        "You are a PSA-style card grading assistant.\n"
        "Use the provided preprocessing ratios for centering; do not re-measure visually.\n"
        "Treat ratios below 0.15 as extreme off-center (≈90/10).\n"
        "Ratios near 1.0 are balanced. Use them to determine centering score and ceiling.\n"
        "Then evaluate corners, edges, and surface visually from attached images.\n"
        "Output structured JSON matching schema v7.0."
    )

    print(f"[INFO] Sending {front.filename} / {back.filename} with Prompt v10 and structured preprocessing.")

    response = client.responses.create(
        prompt={"id": PROMPT_ID, "version": PROMPT_VERSION},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction_text},
                    {"type": "input_text", "text": json.dumps(combined_pre, indent=2)},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{front_b64}"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{back_b64}"},
                ],
            }
        ],
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
    return {"status": "ok", "message": "MintCondition Card Grader API v1.8 running."}
