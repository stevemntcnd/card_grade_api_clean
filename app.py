"""
app.py — Mint Condition Grading API (Full Gradient-Aware Version)
----------------------------------------------------------------
Implements your local-tested preprocessing pipeline with Sobel gradient
refinement, border measurement, and OpenAI integration for grading.
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
app = FastAPI(title="Mint Condition Grading API (v1.5 Full)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# GRADIENT-AWARE EDGE REFINEMENT
# ==========================================================
def refine_edges_for_vintage(img_gray):
    grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, strong_edges = cv2.threshold(norm, 40, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(strong_edges, cv2.MORPH_CLOSE, kernel, iterations=1)

# ==========================================================
# BORDER MEASUREMENT
# ==========================================================
def detect_border_distance(edge_img, axis='horizontal'):
    h, w = edge_img.shape
    kernel = np.ones((3, 3), np.uint8)
    edge_clean = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    if axis == 'horizontal':
        left = next((x for x in range(w // 2) if np.count_nonzero(edge_clean[:, x]) > 5), w)
        right = next((x for x in range(w - 1, w // 2, -1)
                      if np.count_nonzero(edge_clean[:, x]) > 5), w)
        return max(left, 2), max(w - right, 2)
    else:
        vert_kernel = np.ones((1, 5), np.uint8)
        edge_vert = cv2.morphologyEx(edge_clean, cv2.MORPH_CLOSE, vert_kernel, iterations=2)

        def scan_top():
            for y in range(h // 2):
                if np.count_nonzero(edge_vert[y, :]) > (0.25 * w):
                    return y
            return h

        def scan_bottom():
            for y in range(h - 1, h // 2, -1):
                if np.count_nonzero(edge_vert[y, :]) > (0.25 * w):
                    return y
            return h

        top_dist = scan_top()
        bottom_edge = scan_bottom()
        bottom_dist = h - bottom_edge
        return max(top_dist, 2), max(bottom_dist, 2)

# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================
def preprocess_card_image_from_upload(file: UploadFile, quality=90, max_dim=1500):
    contents = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(contents, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {file.filename}")

    h, w = img.shape[:2]
    if h < w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]

    scale = max(h, w) / max_dim
    if scale > 1.0:
        img = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = refine_edges_for_vintage(blur)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found.")
    contour = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(contour)
    pad = int(0.02 * max(h_box, w_box))
    x, y = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + w_box + pad, img.shape[1]), min(y + h_box + pad, img.shape[0])
    cropped = img[y:y2, x:x2]
    edges_cropped = edges[y:y2, x:x2]

    left_px, right_px = detect_border_distance(edges_cropped, axis='horizontal')
    top_px, bottom_px = detect_border_distance(edges_cropped, axis='vertical')

    horizontal_ratio = round(min(left_px, right_px) / max(left_px, right_px + 1e-5), 3)
    vertical_ratio = round(min(top_px, bottom_px) / max(top_px, bottom_px + 1e-5), 3)

    min_dimension = min(h, w)
    relative_threshold = max(2, int(0.005 * min_dimension))  # 0.5%
    extreme_offcenter = (
        (left_px < relative_threshold or right_px < relative_threshold or
         top_px < relative_threshold or bottom_px < relative_threshold) and
        (horizontal_ratio < 0.10 or vertical_ratio < 0.10)
    )

    ceiling = 2.0 if extreme_offcenter else None

    pre_json = {
        "image_name": file.filename,
        "height_px": h,
        "width_px": w,
        "top_px": int(top_px),
        "bottom_px": int(bottom_px),
        "left_px": int(left_px),
        "right_px": int(right_px),
        "horizontal_ratio": horizontal_ratio,
        "vertical_ratio": vertical_ratio,
        "aspect_ratio": round(w / h, 3),
        "extreme_offcenter": extreme_offcenter,
        "ceiling": ceiling
    }

    _, buffer = cv2.imencode(".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    b64_img = base64.b64encode(buffer).decode("utf-8")

    return pre_json, b64_img

# ==========================================================
# MAIN API ENDPOINT
# ==========================================================
@app.post("/grade-card")
async def grade_card(front: UploadFile = File(...), back: UploadFile = File(...)):
    front_pre, front_b64 = preprocess_card_image_from_upload(front)
    back_pre, back_b64 = preprocess_card_image_from_upload(back)

    combined_pre = {
        "front_measurements": front_pre,
        "back_measurements": back_pre
    }

    print(f"[INFO] Sending {front.filename} / {back.filename} to OpenAI...")

    response = client.responses.create(
        prompt={"id": PROMPT_ID, "version": PROMPT_VERSION},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(combined_pre)},
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

    return {
        "grading_result": result,
        "measurements": combined_pre
    }
