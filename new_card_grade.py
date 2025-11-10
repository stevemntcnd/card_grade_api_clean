import base64, cv2, numpy as np, json, argparse
from openai import OpenAI

# ==========================================================
# CONFIG
# ==========================================================
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROMPT_ID = "pmpt_690fe027d50c819782c0d8720142b0b00e6d4016c646f820"
PROMPT_VERSION = "9"

FRONT_PATH = "/Users/stevenheiss/Mint-Condition-Fine-Tuning/defect_training_images/New Grading Prompt for Lovable/Screenshot 2025-11-09 at 4.20.33 PM.png"
BACK_PATH  = "/Users/stevenheiss/Mint-Condition-Fine-Tuning/defect_training_images/New Grading Prompt for Lovable/Screenshot 2025-11-09 at 4.20.40 PM.png"

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================================
# GRADIENT-AWARE EDGE REFINEMENT
# ==========================================================
def refine_edges_for_vintage(img_gray):
    """
    Suppresses false edges from soft gradients or fading ink typical in vintage cards.
    Uses Sobel gradient magnitude and thresholding instead of raw Canny.
    """
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
def preprocess_card_image(path, quality=90, max_dim=1500):
    import requests, numpy as np, cv2
    # --- URL support patch ---
    if path.startswith("http://") or path.startswith("https://"):
        print(f"[INFO] Fetching image from URL: {path}")
        response = requests.get(path)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch image from URL: {path}")
        image_data = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not load image: {path}")

    h, w = img.shape[:2]
    if h < w:
        print(f"[INFO] Rotated scan detected for {path} — rotating 90° to portrait.")
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
        raise ValueError(f"No contours found in {path}")
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

    print(f"[DEBUG] Borders (px) — top:{top_px} bottom:{bottom_px} left:{left_px} right:{right_px}")
    print(f"[DEBUG] Ratios — horiz:{horizontal_ratio:.2f} vert:{vertical_ratio:.2f}")

    min_dimension = min(h, w)
    relative_threshold = max(2, int(0.005 * min_dimension))  # 0.5%
    extreme_offcenter = (
        (left_px < relative_threshold or right_px < relative_threshold or
         top_px < relative_threshold or bottom_px < relative_threshold) and
        (horizontal_ratio < 0.10 or vertical_ratio < 0.10)
    )

    ceiling = 2.0 if extreme_offcenter else None

    pre_json = {
        "image_path": path,
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
# SEND TO OPENAI
# ==========================================================
def grade_card(front_b64, back_b64, pre_json):
    print("[INFO] Sending to OpenAI grading prompt...")
    response = client.responses.create(
        prompt={"id": PROMPT_ID, "version": PROMPT_VERSION},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(pre_json)},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{front_b64}"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{back_b64}"}
                ]
            }
        ]
    )
    try:
        return response.output_text
    except Exception:
        return str(response)

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and grade trading card images.")
    parser.add_argument("--front", default=FRONT_PATH, help="Path to front image")
    parser.add_argument("--back",  default=BACK_PATH,  help="Path to back image")
    parser.add_argument("--quality", type=int, default=90, help="JPEG compression quality")
    parser.add_argument("--maxdim", type=int, default=1500, help="Max dimension for resizing")
    args = parser.parse_args()

    print("[INFO] Starting preprocessing + grading pipeline...")

    pre_json_front, front_b64 = preprocess_card_image(args.front, args.quality, args.maxdim)
    pre_json_back, back_b64 = preprocess_card_image(args.back, args.quality, args.maxdim)

    combined_pre = {
        "front_measurements": pre_json_front,
        "back_measurements": pre_json_back
    }

    print("\nPreprocessing JSON:")
    print(json.dumps(combined_pre, indent=2))

    result = grade_card(front_b64, back_b64, combined_pre)

    print("\n--- MODEL RESPONSE ---")
    print(result)
