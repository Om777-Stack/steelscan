"""
Steel Surface Defect Detection — Streamlit Web App
====================================================
CPU-friendly | No GPU required | Windows Laptop
Supports: Image Upload + Webcam Live Feed

Run:   streamlit run app.py
       (or double-click Launch_SteelScan.bat)

Windows notes:
  - Webcam uses Streamlit's st.camera_input (works in Chrome/Edge)
  - Grant camera permission when browser asks
  - If webcam is black, try Microsoft Edge instead of Chrome
  - First run auto-downloads ~6MB YOLOv8 model weights
"""

import io
import time
import json
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="SteelScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load YOLOv8 model once and cache it across sessions."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        return model, None
    except ImportError:
        return None, "ultralytics not installed. Run: pip install ultralytics"
    except Exception as e:
        return None, str(e)


# ── Constants ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"
]

CLASS_COLORS_BGR = {
    "crazing"         : (0, 220, 255),
    "inclusion"       : (0, 140, 255),
    "patches"         : (50, 205, 50),
    "pitted_surface"  : (255, 0, 200),
    "rolled-in_scale" : (255, 165, 0),
    "scratches"       : (60, 60, 255),
}

CLASS_COLORS_HEX = {
    "crazing"         : "#FFD700",
    "inclusion"       : "#FF8C00",
    "patches"         : "#32CD32",
    "pitted_surface"  : "#FF00CC",
    "rolled-in_scale" : "#00BFFF",
    "scratches"       : "#FF3C3C",
}

SEVERITY = {
    "crazing"         : "Medium",
    "inclusion"       : "High",
    "patches"         : "Low",
    "pitted_surface"  : "High",
    "rolled-in_scale" : "Medium",
    "scratches"       : "Low",
}

SEVERITY_COLOR = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}

MODEL_OPTIONS = {
    "YOLOv8-nano (Fastest, CPU-optimized)": "yolov8n.pt",
    "YOLOv8-small (Balanced)":              "yolov8s.pt",
}

# ── Custom CSS ─────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --bg-dark:      #0a0c10;
        --bg-panel:     #111318;
        --bg-card:      #181c24;
        --accent:       #00e5ff;
        --accent2:      #ff6b35;
        --text-main:    #e8eaf0;
        --text-muted:   #6b7280;
        --border:       #1f2937;
        --green:        #00e676;
        --red:          #ff1744;
        --amber:        #ffab00;
        --font-mono:    'Share Tech Mono', monospace;
        --font-body:    'DM Sans', sans-serif;
    }

    html, body, [class*="css"] {
        font-family: var(--font-body);
        background-color: var(--bg-dark);
        color: var(--text-main);
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-panel);
        border-right: 1px solid var(--border);
    }

    /* Main area */
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

    /* Header */
    .steelscan-header {
        background: linear-gradient(135deg, #0d1117 0%, #141922 50%, #0a1628 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .steelscan-header::before {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 300px; height: 100%;
        background: radial-gradient(ellipse at right, rgba(0,229,255,0.05) 0%, transparent 70%);
    }
    .steelscan-header h1 {
        font-family: var(--font-mono);
        font-size: 1.7rem;
        color: var(--accent);
        margin: 0;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .steelscan-header p {
        color: var(--text-muted);
        margin: 4px 0 0;
        font-size: 0.88rem;
    }
    .status-badge {
        display: inline-block;
        background: rgba(0,230,118,0.12);
        border: 1px solid var(--green);
        color: var(--green);
        padding: 2px 10px;
        border-radius: 20px;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 1px;
        margin-top: 6px;
    }

    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 1rem 0;
    }
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .val {
        font-family: var(--font-mono);
        font-size: 1.8rem;
        color: var(--accent);
        line-height: 1;
    }
    .metric-card .lbl {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Detection cards */
    .det-card {
        background: var(--bg-card);
        border-left: 3px solid var(--accent);
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .det-cls  { font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-main); }
    .det-conf { font-family: var(--font-mono); font-size: 0.85rem; color: var(--accent); }

    /* Severity badge */
    .sev { padding: 2px 8px; border-radius: 4px; font-size: 0.7rem;
           font-family: var(--font-mono); letter-spacing: 0.5px; }

    /* Section headers */
    .section-hdr {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 2px;
        color: var(--text-muted);
        text-transform: uppercase;
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px;
        margin: 1.2rem 0 0.8rem;
    }

    /* Verdict banner */
    .verdict-pass {
        background: rgba(0,230,118,0.08);
        border: 1px solid var(--green);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        color: var(--green);
        font-family: var(--font-mono);
        font-size: 1rem;
        text-align: center;
        letter-spacing: 1px;
    }
    .verdict-fail {
        background: rgba(255,23,68,0.08);
        border: 1px solid var(--red);
        border-radius: 8px;
        padding: 1rem 1.5rem;
        color: var(--red);
        font-family: var(--font-mono);
        font-size: 1rem;
        text-align: center;
        letter-spacing: 1px;
    }

    /* Sidebar elements */
    .sidebar-section {
        background: var(--bg-card);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        border: 1px solid var(--border);
    }

    /* Image container */
    .img-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 4px;
        overflow: hidden;
    }

    /* Streamlit widget overrides */
    .stButton > button {
        background: var(--accent);
        color: #000;
        font-family: var(--font-mono);
        font-weight: 600;
        border: none;
        border-radius: 6px;
        width: 100%;
        padding: 0.55rem;
        letter-spacing: 1px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .stSelectbox > div > div,
    .stSlider > div { color: var(--text-main); }

    [data-testid="stFileUploader"] {
        background: var(--bg-card);
        border: 1.5px dashed var(--border);
        border-radius: 8px;
        padding: 6px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-panel);
        border-radius: 8px 8px 0 0;
        border-bottom: 1px solid var(--border);
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-mono);
        font-size: 0.8rem;
        letter-spacing: 1px;
        color: var(--text-muted);
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
        background: transparent !important;
    }

    /* Log table */
    .log-row {
        display: grid;
        grid-template-columns: 80px 1fr 70px 80px 90px;
        gap: 8px;
        padding: 6px 10px;
        border-bottom: 1px solid var(--border);
        font-family: var(--font-mono);
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    .log-row:hover { background: var(--bg-card); }
    .log-hdr { color: var(--text-muted); font-weight: 600;
               border-bottom: 2px solid var(--border) !important; }
    .log-val { color: var(--text-main); }
    </style>
    """, unsafe_allow_html=True)


# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE contrast enhancement for steel surface images.
    Input: RGB numpy array. Output: RGB numpy array.
    """
    gray    = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enh     = clahe.apply(gray)
    return cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)


# ── Inference ──────────────────────────────────────────────────────────────
def draw_green_outline(annotated_bgr, x1, y1, x2, y2, label, confidence,
                       contour=None):
    """
    Draw precise green outline around a detected defect.
    If contour is provided, traces the exact defect shape.
    Otherwise falls back to corner-bracket bounding box.
    """
    GREEN      = (0, 255, 80)
    GREEN_GLOW = (0, 160, 45)
    GREEN_LBL  = (0, 210, 55)

    # 1. Glow layer behind the outline
    overlay = annotated_bgr.copy()
    cv2.rectangle(overlay, (x1-6, y1-6), (x2+6, y2+6), GREEN_GLOW, 6)
    cv2.addWeighted(overlay, 0.25, annotated_bgr, 0.75, 0, annotated_bgr)

    # 2a. If we have the real contour — draw it precisely (traces scratch shape)
    if contour is not None:
        # Draw filled semi-transparent green tint inside contour
        mask = np.zeros(annotated_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        tint = annotated_bgr.copy()
        tint[mask > 0] = (
            tint[mask > 0] * 0.6 + np.array([0, 80, 20]) * 0.4
        ).astype(np.uint8)
        cv2.addWeighted(tint, 0.4, annotated_bgr, 0.6, 0, annotated_bgr)
        # Draw the precise contour outline
        cv2.drawContours(annotated_bgr, [contour], -1, GREEN, 2)
        # Glow outline (slightly thicker, darker green underneath)
        cv2.drawContours(annotated_bgr, [contour], -1, GREEN_GLOW, 4)
        cv2.drawContours(annotated_bgr, [contour], -1, GREEN, 2)
    else:
        # 2b. Fallback: corner bracket bounding box
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), GREEN, 2)
        clen = max(10, min(20, (x2-x1)//5, (y2-y1)//5))
        for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(annotated_bgr, (sx, sy), (sx+dx*clen, sy), GREEN, 3)
            cv2.line(annotated_bgr, (sx, sy), (sx, sy+dy*clen), GREEN, 3)

    # 3. Center dot
    cx, cy = (x1+x2)//2, (y1+y2)//2
    cv2.circle(annotated_bgr, (cx, cy), 3, GREEN, -1)

    # 4. Label — green background, black text
    tag = f"  {label}  {confidence:.0%}  "
    (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    ly = y1 - 4 if y1 - th - 10 > 0 else y2 + th + 8
    cv2.rectangle(annotated_bgr, (x1, ly-th-bl-4), (x1+tw, ly), GREEN_LBL, -1)
    cv2.rectangle(annotated_bgr, (x1, ly-th-bl-4), (x1+tw, ly), GREEN, 1)
    cv2.putText(annotated_bgr, tag, (x1, ly-bl-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 1, cv2.LINE_AA)


def detect_scratches_opencv(gray: np.ndarray, h: int, w: int) -> list:
    """
    Precise OpenCV scratch & defect detector.
    Returns list of (x1, y1, x2, y2, label, confidence, contour) tuples.
    The contour allows drawing the exact scratch shape outline.
    """
    detections = []

    # ── Pre-processing ───────────────────────────────────────────────────────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Step 1: Suppress the brushed texture pattern ─────────────────────────
    # Brushed metal has a strong directional grain (horizontal lines).
    # We remove it by subtracting a heavy horizontal blur — what remains
    # are only anomalies that don't match the grain: real scratches & defects.
    grain_removed = cv2.subtract(blurred, cv2.blur(blurred, (60, 1)))
    # Normalize so contrast is consistent regardless of image brightness
    grain_removed = cv2.normalize(grain_removed, None, 0, 255, cv2.NORM_MINMAX)

    # ── Step 2: Edge detection on grain-suppressed image ─────────────────────
    edges = cv2.Canny(grain_removed, 35, 100)

    # Connect nearby edge fragments into continuous scratch lines
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    k_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k_h, iterations=1)
    edges = cv2.dilate(edges, k_v, iterations=1)
    edges = cv2.dilate(edges, k_d, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    seen_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Minimum area: big enough to be a real scratch, not texture noise
        if area < 300 or area > w * h * 0.15:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        longest = max(bw, bh)
        shortest = min(bw, bh)

        # Real scratch: long (> 5% of image width/height) and thin
        if longest < w * 0.05 and longest < h * 0.05:
            continue

        aspect = longest / (shortest + 1e-3)
        if aspect < 3.0:   # not elongated enough to be a scratch
            continue

        conf = float(np.clip(0.52 + aspect / 50, 0.54, 0.91))

        # Deduplicate
        overlap = False
        for ex1, ey1, ex2, ey2 in seen_boxes:
            ix1, iy1 = max(x, ex1), max(y, ey1)
            ix2, iy2 = min(x+bw, ex2), min(y+bh, ey2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                union = bw*bh + (ex2-ex1)*(ey2-ey1) - inter
                if inter/(union+1) > 0.45:
                    overlap = True
                    break
        if not overlap:
            seen_boxes.append((x, y, x+bw, y+bh))
            detections.append((x, y, x+bw, y+bh, "scratches", conf, cnt))

    # ── Step 3: Bright anomaly detection (white/bright scratches) ────────────
    # Some scratches on steel appear BRIGHTER than surroundings (reflective)
    local_mean_b = cv2.blur(blurred, (61, 61))
    bright_diff  = cv2.subtract(blurred, local_mean_b)
    _, bright_thresh = cv2.threshold(bright_diff, 18, 255, cv2.THRESH_BINARY)

    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    bright_thresh = cv2.morphologyEx(bright_thresh, cv2.MORPH_OPEN,  ke, iterations=1)
    bright_thresh = cv2.morphologyEx(bright_thresh, cv2.MORPH_CLOSE, ke, iterations=2)

    contours3, _ = cv2.findContours(bright_thresh, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours3:
        area = cv2.contourArea(cnt)
        if area < 400 or area > w * h * 0.10:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        longest = max(bw, bh)
        if longest < w * 0.04:
            continue
        aspect = longest / (min(bw, bh) + 1e-3)
        if aspect < 3.0:
            continue

        overlap = False
        for ex1, ey1, ex2, ey2 in seen_boxes:
            ix1, iy1 = max(x, ex1), max(y, ey1)
            ix2, iy2 = min(x+bw, ex2), min(y+bh, ey2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                union = bw*bh + (ex2-ex1)*(ey2-ey1) - inter
                if inter/(union+1) > 0.45:
                    overlap = True
                    break
        if not overlap:
            seen_boxes.append((x, y, x+bw, y+bh))
            detections.append((x, y, x+bw, y+bh, "scratches", float(np.clip(0.54 + aspect/60, 0.55, 0.90)), cnt))

    # ── Step 4: Dark anomaly detection (pits, inclusions, deep gouges) ────────
    local_mean_d = cv2.blur(blurred, (61, 61))
    dark_diff    = cv2.subtract(local_mean_d, blurred)
    _, dark_thresh = cv2.threshold(dark_diff, 18, 255, cv2.THRESH_BINARY)

    ke2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN,  ke2, iterations=1)
    dark_thresh = cv2.morphologyEx(dark_thresh, cv2.MORPH_CLOSE, ke2, iterations=2)

    contours4, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours4:
        area = cv2.contourArea(cnt)
        if area < 400 or area > w * h * 0.10:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if max(bw, bh) < 20:
            continue
        aspect = max(bw, bh) / (min(bw, bh) + 1e-3)

        if aspect > 3.5:
            label, conf = "scratches", 0.61
        elif aspect < 2.0:
            label, conf = "pitted_surface", 0.58
        else:
            label, conf = "patches", 0.55

        overlap = False
        for ex1, ey1, ex2, ey2 in seen_boxes:
            ix1, iy1 = max(x, ex1), max(y, ey1)
            ix2, iy2 = min(x+bw, ex2), min(y+bh, ey2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                union = bw*bh + (ex2-ex1)*(ey2-ey1) - inter
                if inter/(union+1) > 0.45:
                    overlap = True
                    break
        if not overlap:
            seen_boxes.append((x, y, x+bw, y+bh))
            detections.append((x, y, x+bw, y+bh, label, conf, cnt))

    return detections


def run_inference(model, image_rgb: np.ndarray, conf: float, iou: float,
                  use_clahe: bool, img_size: int = 640):
    """
    Two-stage inference pipeline:
      Stage 1 — YOLOv8 (AI model, best when fine-tuned on steel data)
      Stage 2 — OpenCV fallback (always runs, catches what YOLO misses
                 on general/untrained weights)
    Returns (annotated_rgb, detections_list, latency_ms)
    """
    if use_clahe:
        image_rgb = preprocess(image_rgb)

    h, w = image_rgb.shape[:2]
    annotated_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    detections = []

    t0 = time.perf_counter()

    # ── Stage 1: YOLOv8 AI detection ────────────────────────────────────────
    results = model(image_rgb, conf=conf, iou=iou, imgsz=img_size, verbose=False)
    yolo_boxes = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id     = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_name = (CLASS_NAMES[cls_id]
                        if cls_id < len(CLASS_NAMES) else f"class_{cls_id}")
            yolo_boxes.append((x1, y1, x2, y2))
            draw_green_outline(annotated_bgr, x1, y1, x2, y2, cls_name, confidence)
            detections.append({
                "class"     : cls_name,
                "confidence": confidence,
                "severity"  : SEVERITY.get(cls_name, "Unknown"),
                "bbox"      : (x1, y1, x2, y2),
                "area_px"   : (x2 - x1) * (y2 - y1),
                "source"    : "AI",
            })

    # ── Stage 2: OpenCV fallback scratch/defect detection ───────────────────
    # Always runs — especially useful before fine-tuning on steel dataset
    cv_dets = detect_scratches_opencv(gray, h, w)

    for (x1, y1, x2, y2, cls_name, confidence, contour) in cv_dets:
        # Skip if heavily overlapping with a YOLO box
        skip = False
        for bx1, by1, bx2, by2 in yolo_boxes:
            ix1, iy1 = max(x1, bx1), max(y1, by1)
            ix2, iy2 = min(x2, bx2), min(y2, by2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                union = (x2-x1)*(y2-y1) + (bx2-bx1)*(by2-by1) - inter
                if inter / (union + 1) > 0.4:
                    skip = True
                    break
        if skip:
            continue

        # Pass contour so outline traces exact scratch shape
        draw_green_outline(annotated_bgr, x1, y1, x2, y2,
                           cls_name, confidence, contour=contour)
        detections.append({
            "class"     : cls_name,
            "confidence": confidence,
            "severity"  : SEVERITY.get(cls_name, "Unknown"),
            "bbox"      : (x1, y1, x2, y2),
            "area_px"   : (x2 - x1) * (y2 - y1),
            "source"    : "OpenCV",
        })

    latency_ms = (time.perf_counter() - t0) * 1000
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections, latency_ms


# ── UI Components ──────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="steelscan-header">
        <h1>⬡ SteelScan AI</h1>
        <p>Surface Defect Detection System &nbsp;·&nbsp; AIML Major Project · Phase 2</p>
        <span class="status-badge">● SYSTEM ONLINE</span>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(detections: list, latency_ms: float, img_shape: tuple):
    h, w = img_shape[:2]
    n_det = len(detections)
    high_sev = sum(1 for d in detections if d["severity"] == "High")
    verdict  = "PASS" if n_det == 0 else "FAIL"
    v_color  = "#00e676" if n_det == 0 else "#ff1744"

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="val">{n_det}</div>
            <div class="lbl">Defects Found</div>
        </div>
        <div class="metric-card">
            <div class="val" style="color:{'#ff1744' if high_sev else '#00e676'}">{high_sev}</div>
            <div class="lbl">High Severity</div>
        </div>
        <div class="metric-card">
            <div class="val">{latency_ms:.0f}<span style="font-size:1rem">ms</span></div>
            <div class="lbl">Scan Time</div>
        </div>
        <div class="metric-card">
            <div class="val" style="color:{v_color}">{verdict}</div>
            <div class="lbl">QC Verdict</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_detections(detections: list):
    if not detections:
        st.markdown("""
        <div class="verdict-pass">✔ NO DEFECTS DETECTED — SURFACE CLEAR</div>
        """, unsafe_allow_html=True)
        return

    # Sort by severity then confidence
    sev_order = {"High": 0, "Medium": 1, "Low": 2}
    detections = sorted(detections,
                        key=lambda d: (sev_order.get(d["severity"], 3), -d["confidence"]))

    st.markdown('<div class="section-hdr">DETECTED DEFECTS</div>', unsafe_allow_html=True)

    for i, det in enumerate(detections, 1):
        cls   = det["class"]
        conf  = det["confidence"]
        sev   = det["severity"]
        color = CLASS_COLORS_HEX.get(cls, "#aaa")
        sc    = SEVERITY_COLOR.get(sev, "#aaa")
        x1,y1,x2,y2 = det["bbox"]

        st.markdown(f"""
        <div class="det-card" style="border-left-color:{color}">
            <div>
                <span class="det-cls" style="color:{color}">#{i} {cls.upper()}</span>
                &nbsp;
                <span class="sev" style="background:{sc}22;color:{sc};border:1px solid {sc}">{sev}</span>
            </div>
            <div style="text-align:right">
                <span class="det-conf">{conf:.1%}</span>
                <br><span style="font-size:0.7rem;color:#6b7280;font-family:monospace">
                [{x1},{y1} → {x2},{y2}]
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_class_summary(all_detections: list):
    """Render cumulative class detection count bar chart."""
    if not all_detections:
        return

    counts = defaultdict(int)
    for d in all_detections:
        counts[d["class"]] += 1

    st.markdown('<div class="section-hdr">CLASS DISTRIBUTION</div>', unsafe_allow_html=True)
    total = sum(counts.values())
    for cls in CLASS_NAMES:
        n    = counts.get(cls, 0)
        if n == 0:
            continue
        pct  = n / total
        color = CLASS_COLORS_HEX.get(cls, "#aaa")
        st.markdown(f"""
        <div style="margin:5px 0">
            <div style="display:flex;justify-content:space-between;
                        font-family:monospace;font-size:0.75rem;margin-bottom:2px">
                <span style="color:{color}">{cls}</span>
                <span style="color:#6b7280">{n} ({pct:.0%})</span>
            </div>
            <div style="background:#1f2937;border-radius:3px;height:6px;overflow:hidden">
                <div style="width:{pct*100:.1f}%;height:100%;
                            background:{color};border-radius:3px;
                            transition:width 0.4s ease"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_log_table(log: list):
    if not log:
        return

    st.markdown('<div class="section-hdr">SCAN LOG</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="log-row log-hdr">
        <span>TIME</span><span>CLASS</span><span>CONF</span>
        <span>SEVERITY</span><span>SOURCE</span>
    </div>
    """, unsafe_allow_html=True)

    for entry in reversed(log[-50:]):  # Show last 50
        sc = SEVERITY_COLOR.get(entry.get("severity",""), "#aaa")
        st.markdown(f"""
        <div class="log-row">
            <span>{entry['time']}</span>
            <span class="log-val">{entry['class']}</span>
            <span class="log-val">{entry['confidence']:.0%}</span>
            <span style="color:{sc}">{entry['severity']}</span>
            <span>{entry['source']}</span>
        </div>
        """, unsafe_allow_html=True)


def export_report(log: list, summary_counts: dict) -> str:
    """Generate JSON report for download."""
    report = {
        "generated_at"  : datetime.now().isoformat(),
        "system"        : "SteelScan AI — CPU Inference",
        "total_scans"   : len(set(e.get("scan_id", 0) for e in log)),
        "total_defects" : len(log),
        "class_summary" : summary_counts,
        "detections"    : log,
    }
    return json.dumps(report, indent=2)


# ── Sidebar ────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:monospace;font-size:0.7rem;color:#6b7280;
                    letter-spacing:2px;text-transform:uppercase;
                    border-bottom:1px solid #1f2937;padding-bottom:8px;margin-bottom:12px">
            ⬡ SteelScan Config
        </div>
        """, unsafe_allow_html=True)

        model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
        model_file  = MODEL_OPTIONS[model_label]

        st.markdown("---")
        conf = st.slider("Confidence Threshold", 0.10, 0.90, 0.30, 0.05,
                         help="Lower = more detections (more false positives). "
                              "Higher = fewer, more certain detections.")
        iou  = st.slider("NMS IoU Threshold",    0.20, 0.80, 0.45, 0.05,
                         help="Controls overlap suppression between boxes.")
        use_clahe = st.toggle("CLAHE Enhancement", value=True,
                              help="Improves contrast for steel surfaces. "
                                   "Recommended ON for real steel images.")
        img_size = st.select_slider("Input Resolution", [320, 416, 640], value=640,
                                    help="Lower = faster on CPU. 320 for webcam, 640 for images.")

        st.markdown("---")
        st.markdown("""
        <div style="font-family:monospace;font-size:0.68rem;color:#6b7280">
        <b style="color:#e8eaf0">Defect Classes</b><br><br>
        🟡 Crazing &nbsp;&nbsp; 🟠 Inclusion<br>
        🟢 Patches &nbsp;&nbsp; 🟣 Pitted<br>
        🔵 Rolled-in Scale<br>
        🔴 Scratches
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-family:monospace;font-size:0.68rem;color:#6b7280">
        <b style="color:#e8eaf0">CPU Performance Guide</b><br><br>
        Resolution 320 → ~15 FPS<br>
        Resolution 640 → ~5-8 FPS<br>
        YOLOv8n &nbsp;→ fastest<br>
        YOLOv8s &nbsp;→ more accurate
        </div>
        """, unsafe_allow_html=True)

    return model_file, conf, iou, use_clahe, img_size


# ── Main App ───────────────────────────────────────────────────────────────
def main():
    inject_css()
    render_header()

    # Session state
    if "log"       not in st.session_state: st.session_state.log       = []
    if "scan_count" not in st.session_state: st.session_state.scan_count = 0
    if "all_dets"  not in st.session_state: st.session_state.all_dets  = []

    # Sidebar
    model_file, conf, iou, use_clahe, img_size = render_sidebar()

    # Load model
    with st.spinner("Loading AI model..."):
        model, err = load_model(model_file)

    if err:
        st.error(f"❌ Model load failed: {err}")
        st.code("pip install ultralytics", language="bash")
        st.stop()

    # Tabs
    tab_upload, tab_webcam, tab_log = st.tabs([
        "📂  IMAGE UPLOAD", "📷  WEBCAM LIVE", "📋  SCAN LOG"
    ])

    # ── TAB 1: Image Upload ───────────────────────────────────────────────
    with tab_upload:
        col_upload, col_results = st.columns([1.1, 1], gap="medium")

        with col_upload:
            st.markdown('<div class="section-hdr">UPLOAD STEEL SURFACE IMAGE</div>',
                        unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Drag & drop or click to browse",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                help="Upload an image of the steel surface to scan for defects.",
                label_visibility="collapsed",
            )

            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                img_arr = np.array(pil_img)

                # Show original
                st.markdown('<div class="section-hdr">ORIGINAL IMAGE</div>',
                            unsafe_allow_html=True)
                st.image(img_arr, use_container_width=True)

                # Scan button
                if st.button("🔬  SCAN FOR DEFECTS", key="scan_upload"):
                    with st.spinner("Scanning surface..."):
                        annotated, detections, latency = run_inference(
                            model, img_arr, conf, iou, use_clahe, img_size
                        )

                    st.session_state.scan_count += 1
                    scan_id = st.session_state.scan_count

                    # Log detections
                    ts = datetime.now().strftime("%H:%M:%S")
                    for det in detections:
                        entry = {**det, "time": ts, "source": "upload",
                                 "scan_id": scan_id,
                                 "bbox": list(det["bbox"])}
                        st.session_state.log.append(entry)
                    st.session_state.all_dets.extend(detections)

                    # Store results in session for display
                    st.session_state["upload_result"] = {
                        "annotated"  : annotated,
                        "detections" : detections,
                        "latency"    : latency,
                        "shape"      : img_arr.shape,
                        "filename"   : uploaded.name,
                    }

        with col_results:
            if "upload_result" in st.session_state:
                res = st.session_state["upload_result"]

                st.markdown('<div class="section-hdr">SCAN RESULT</div>',
                            unsafe_allow_html=True)
                st.image(res["annotated"], use_container_width=True)

                render_metrics(res["detections"], res["latency"], res["shape"])
                render_detections(res["detections"])
                render_class_summary(st.session_state.all_dets)

                # Download annotated image
                pil_out = Image.fromarray(res["annotated"])
                buf = io.BytesIO()
                pil_out.save(buf, format="PNG")
                st.download_button(
                    "⬇  Download Result Image",
                    data=buf.getvalue(),
                    file_name=f"steelscan_{res['filename']}",
                    mime="image/png",
                )
            else:
                st.markdown("""
                <div style="height:400px;display:flex;align-items:center;
                            justify-content:center;border:1.5px dashed #1f2937;
                            border-radius:8px;color:#374151;text-align:center;
                            font-family:monospace;font-size:0.85rem">
                    ← Upload an image and<br>click SCAN to see results
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 2: Webcam ─────────────────────────────────────────────────────
    with tab_webcam:
        st.markdown("""
        <div style="background:#111318;border:1px solid #1f2937;border-radius:8px;
                    padding:14px 18px;margin-bottom:1rem">
            <div style="font-family:monospace;font-size:0.75rem;color:#6b7280;
                        letter-spacing:1px;margin-bottom:6px">📷 WEBCAM LIVE SCAN</div>
            <p style="color:#9ca3af;font-size:0.85rem;margin:0 0 10px">
            Point your laptop camera at any metal surface, then click the
            camera button below to capture and scan a frame.
            </p>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
                <div style="background:#181c24;border:1px solid #1f2937;border-radius:6px;padding:8px 10px">
                    <div style="font-family:monospace;font-size:0.72rem;color:#00e5ff;margin-bottom:3px">&#128161; Lighting</div>
                    <div style="font-size:0.78rem;color:#6b7280">Use bright overhead light. Avoid shadows on metal.</div>
                </div>
                <div style="background:#181c24;border:1px solid #1f2937;border-radius:6px;padding:8px 10px">
                    <div style="font-family:monospace;font-size:0.72rem;color:#00e5ff;margin-bottom:3px">&#128208; Distance</div>
                    <div style="font-size:0.78rem;color:#6b7280">Hold camera 20-40 cm from surface. Fill the frame with metal.</div>
                </div>
                <div style="background:#181c24;border:1px solid #ffab0044;border-radius:6px;padding:8px 10px">
                    <div style="font-family:monospace;font-size:0.72rem;color:#ffab00;margin-bottom:3px">&#9888; Windows Tip</div>
                    <div style="font-size:0.78rem;color:#6b7280">Allow camera in browser. If blocked, open in <b style="color:#9ca3af">Microsoft Edge</b>.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_cam, col_camres = st.columns([1.1, 1], gap="medium")

        with col_cam:
            st.markdown('<div class="section-hdr">CAMERA CAPTURE</div>',
                        unsafe_allow_html=True)
            # st.camera_input uses the browser's native camera API —
            # no OpenCV VideoCapture or drivers needed on Windows
            cam_frame = st.camera_input(
                "Click the camera icon to take a photo and scan it",
                help="Windows: allow camera access when the browser asks.",
            )

            if cam_frame:
                pil_cam = Image.open(cam_frame).convert("RGB")
                cam_arr = np.array(pil_cam)

                with st.spinner("Analysing frame..."):
                    annotated, detections, latency = run_inference(
                        model, cam_arr, conf, iou, use_clahe, img_size
                    )

                st.session_state.scan_count += 1
                ts = datetime.now().strftime("%H:%M:%S")
                for det in detections:
                    entry = {**det, "time": ts, "source": "webcam",
                             "scan_id": st.session_state.scan_count,
                             "bbox": list(det["bbox"])}
                    st.session_state.log.append(entry)
                st.session_state.all_dets.extend(detections)

                st.session_state["webcam_result"] = {
                    "annotated" : annotated,
                    "detections": detections,
                    "latency"   : latency,
                    "shape"     : cam_arr.shape,
                }

        with col_camres:
            if "webcam_result" in st.session_state:
                res = st.session_state["webcam_result"]
                st.markdown('<div class="section-hdr">ANALYSIS RESULT</div>',
                            unsafe_allow_html=True)
                st.image(res["annotated"], use_container_width=True)
                render_metrics(res["detections"], res["latency"], res["shape"])
                render_detections(res["detections"])

    # ── TAB 3: Log ────────────────────────────────────────────────────────
    with tab_log:
        col_log, col_export = st.columns([2, 1], gap="medium")

        with col_log:
            total = len(st.session_state.all_dets)
            scans = st.session_state.scan_count
            st.markdown(f"""
            <div style="display:flex;gap:20px;margin-bottom:1rem">
                <div class="metric-card" style="flex:1;padding:12px">
                    <div class="val">{scans}</div>
                    <div class="lbl">Total Scans</div>
                </div>
                <div class="metric-card" style="flex:1;padding:12px">
                    <div class="val">{total}</div>
                    <div class="lbl">Total Defects</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            render_log_table(st.session_state.log)

        with col_export:
            render_class_summary(st.session_state.all_dets)

            st.markdown('<div class="section-hdr">EXPORT</div>', unsafe_allow_html=True)

            counts = defaultdict(int)
            for d in st.session_state.all_dets:
                counts[d["class"]] += 1

            report_json = export_report(st.session_state.log, dict(counts))
            st.download_button(
                "⬇  Download Full Report (JSON)",
                data=report_json,
                file_name=f"steelscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

            if st.button("🗑  Clear Session Log"):
                st.session_state.log       = []
                st.session_state.all_dets  = []
                st.session_state.scan_count = 0
                if "upload_result" in st.session_state:
                    del st.session_state["upload_result"]
                if "webcam_result" in st.session_state:
                    del st.session_state["webcam_result"]
                st.rerun()


if __name__ == "__main__":
    main()
