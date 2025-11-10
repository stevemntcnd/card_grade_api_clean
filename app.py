"""
Mint Condition Card Grader API (v2.1)
-------------------------------------
• Includes enhanced preprocessing for accurate extreme off-centering (90/10+)
• CLAHE lighting normalization
• Shadow suppression and convex-hull contour recovery
• Aspect-ratio sanity check
• Full grading endpoint with Prompt v10 integration
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
app = FastAPI(title="Mint Condition Grader API v2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# IMAGE UTILITIES
# ==========================================================
def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def remove_shadows(gray):
    dilated = cv2.dilate(gray, np.ones((15,15), np.uint8))
    bg = cv2.medianBlur(dilated, 35)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return norm

def refine_edges(gray):
    """Detect faint outer card border even with uneven lighting."""
    gray = apply_clahe(gray)
    gray = remove_shadows(gray)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(norm, 12, 255, cv2.THRESH_BINARY)
    edges = cv2.dilate(edges, np.ones((9,9), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return edges

def get_main_contour(edges, image_shape):
    """Find primary rectangular contour and repair gaps via convex hull."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found.")
    h, w = image_shape[:2]
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), _ = rect
    aspect = rw / rh if rh else 1
    if aspect < 0.6 or aspect > 1.6 or cv2.contourArea(c) < 0.2 * h * w:
        for alt in sorted(contours, key=cv2.contourArea, reverse=True)[1:]:
            r = cv2.minAreaRect(alt)
            (cx, cy), (rw, rh), _ = r
            a = rw / rh if rh else 1
            if 0.6 < a < 1.6:
                c = alt
                break
    return cv2.convexHull(c)

def measure_borders(edge_img, axis="horizontal"):
    """Return pixel thickness of borders on given axis."""
    h, w = edge_img.shape
    if axis == "horizontal":
        left = next((x for x in range(w) if np.count_nonzero(edge_img[:, x]) > 5), w)
        right = next((x for x in range(w - 1, -1, -1) if np.count_nonzero(edge_img[:, x]) > 5), w)
        return max(left, 1), max(w - right, 1)
    else:
        top = next((y for y in range(h) if np.count_nonzero(edge_img[y, :]) > 0.25 * w), h)
        bottom = next((y for y in range(h - 1, -1, -1) if np.count_nonzero(edge_img[y, :]) > 0.25 * w), h)
        return max(top, 1), max(h - bottom, 1)

# ==========================================================
# PREPROCESSING
# ==========================================================
def preprocess_card_image(file: UploadFile, quality=90, max_dim=1500):
    """Extract border geometry and compute centering ratios."""
    buf = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot load {file.filename}")

    h, w = img.shape[:2]
    scale = max(h, w) / max_dim
    if scale > 1.0:
        img = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = refine_edges(gray)
    contour = get_main_contour(edges, img.shape)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x_min, y_min = np.min(box[:,0]), np.min(box[:,1])
    x_max, y_max = np.max(box[:,0]), np.max(box[:,1])
    cropped = img[y_min:y_max, x_min:x_max]
    edges_cropped = edges[y_min:y_max, x_min:x_max]

    left, right = measure_borders(edges_cropped, "horizontal")
    top, bottom = measure_borders(edges_cropped, "vertical")

    horiz_ratio = round(min(left, right) / max(left, right + 1e-5), 3)
    vert_ratio = round(min(top, bottom) / max(top, bottom + 1e-5), 3)
    extreme = horiz_ratio < 0.15 or vert_ratio < 0.15
    ceiling = 2.0 if extreme else None

    pre_json = {
        "filename": file.filename,
        "dimensions": {"width": w, "height": h},
        "borders": {"top": top, "bottom": bottom, "left": left, "right": right},
        "ratios": {"horizontal": horiz_ratio, "vertical": vert_ratio},
        "extreme_offcenter": extreme,
        "ceiling": ceiling,
    }

    _, enc = cv2.imencode(".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    b64 = base64.b64encode(enc).decode("utf-8")
    return pre_json, b64

# ==========================================================
# ENDPOINTS
# ==========================================================
@app.post("/grade-card")
async def grade_card(front: UploadFile = File(...), back: UploadFile = File(...)):
    """Preprocess both sides and send for grading."""
    front_pre, front_b64 = preprocess_card_image(front)
    back_pre, back_b64 = preprocess_card_image(back)
    combined = {"front_measurements": front_pre, "back_measurements": back_pre}

    instruction = (
        "You are a PSA-style card grading assistant.\n"
        "Use the provided preprocessing ratios for centering; do not re-measure visually.\n"
        "Treat ratios below 0.15 as extreme off-center (~90/10).\n"
        "Ratios near 1.0 are balanced. Use them to determine centering score and ceiling.\n"
        "Then evaluate corners, edges, and surface visually from the attached images.\n"
        "Output structured JSON matching schema v10.0."
    )

    print(f"[INFO] Grading {front.filename} / {back.filename} using Prompt v10")

    response = client.responses.create(
        prompt={"id": PROMPT_ID, "version": PROMPT_VERSION},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_text", "text": json.dumps(combined, indent=2)},
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

    return {"grading_result": result, "measurements": combined}

@app.post("/preprocess")
async def preprocess(front: UploadFile = File(...), back: UploadFile = File(...)):
    """Diagnostic endpoint — returns ratios without grading."""
    front_pre, front_img = preprocess_card_image(front)
    back_pre, back_img = preprocess_card_image(back)
    return {
        "front": front_pre,
        "back": back_pre,
        "debug_front": f"data:image/jpeg;base64,{front_img}",
        "debug_back": f"data:image/jpeg;base64,{back_img}",
    }

@app.get("/")
def health():
    return {"status": "ok", "message": "MintCondition Card Grader API v2.1 running."}
