"""
SteelScan AI — Cloud Version (Streamlit Cloud)
================================================
Lightweight: OpenCV only, no PyTorch/YOLO needed
Works on Streamlit Cloud free tier
"""

import io
import time
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="SteelScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────
CLASS_NAMES = ["crazing", "inclusion", "patches",
               "pitted_surface", "rolled-in_scale", "scratches"]

SEVERITY = {
    "crazing"         : "Medium",
    "inclusion"       : "High",
    "patches"         : "Low",
    "pitted_surface"  : "High",
    "rolled-in_scale" : "Medium",
    "scratches"       : "Low",
}

SEVERITY_COLOR = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336"}
GREEN      = (0, 255, 80)
GREEN_GLOW = (0, 160, 45)
GREEN_LBL  = (0, 210, 55)

# ── CSS ────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500&display=swap');
    :root {
        --bg-dark:#0a0c10; --bg-panel:#111318; --bg-card:#181c24;
        --accent:#00e5ff; --text-main:#e8eaf0; --text-muted:#6b7280;
        --border:#1f2937; --green:#00e676; --red:#ff1744;
        --font-mono:'Share Tech Mono',monospace; --font-body:'DM Sans',sans-serif;
    }
    html,body,[class*="css"]{font-family:var(--font-body);background:var(--bg-dark);color:var(--text-main);}
    #MainMenu,footer,header{visibility:hidden;}
    section[data-testid="stSidebar"]{background:var(--bg-panel);border-right:1px solid var(--border);}
    .main .block-container{padding-top:1.5rem;max-width:1400px;}
    .steelscan-header{background:linear-gradient(135deg,#0d1117,#141922,#0a1628);border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:8px;padding:1.4rem 2rem;margin-bottom:1.5rem;}
    .steelscan-header h1{font-family:var(--font-mono);font-size:1.7rem;color:var(--accent);margin:0;letter-spacing:2px;}
    .steelscan-header p{color:var(--text-muted);margin:4px 0 0;font-size:.88rem;}
    .status-badge{display:inline-block;background:rgba(0,230,118,.12);border:1px solid var(--green);color:var(--green);padding:2px 10px;border-radius:20px;font-family:var(--font-mono);font-size:.72rem;letter-spacing:1px;margin-top:6px;}
    .metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:1rem 0;}
    .metric-card{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:1rem 1.2rem;text-align:center;}
    .metric-card .val{font-family:var(--font-mono);font-size:1.8rem;color:var(--accent);line-height:1;}
    .metric-card .lbl{font-size:.75rem;color:var(--text-muted);margin-top:4px;text-transform:uppercase;letter-spacing:1px;}
    .det-card{background:var(--bg-card);border-left:3px solid var(--accent);border-radius:6px;padding:10px 14px;margin:6px 0;display:flex;justify-content:space-between;align-items:center;}
    .det-cls{font-family:var(--font-mono);font-size:.85rem;}
    .det-conf{font-family:var(--font-mono);font-size:.85rem;color:var(--accent);}
    .sev{padding:2px 8px;border-radius:4px;font-size:.7rem;font-family:var(--font-mono);}
    .section-hdr{font-family:var(--font-mono);font-size:.72rem;letter-spacing:2px;color:var(--text-muted);text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:6px;margin:1.2rem 0 .8rem;}
    .verdict-pass{background:rgba(0,230,118,.08);border:1px solid var(--green);border-radius:8px;padding:1rem 1.5rem;color:var(--green);font-family:var(--font-mono);font-size:1rem;text-align:center;letter-spacing:1px;}
    .verdict-fail{background:rgba(255,23,68,.08);border:1px solid var(--red);border-radius:8px;padding:1rem 1.5rem;color:var(--red);font-family:var(--font-mono);font-size:1rem;text-align:center;letter-spacing:1px;}
    .stButton>button{background:var(--accent);color:#000;font-family:var(--font-mono);font-weight:600;border:none;border-radius:6px;width:100%;padding:.55rem;letter-spacing:1px;}
    .stTabs [data-baseweb="tab"]{font-family:var(--font-mono);font-size:.8rem;letter-spacing:1px;color:var(--text-muted);}
    .stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;background:transparent!important;}
    </style>
    """, unsafe_allow_html=True)


# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enh   = clahe.apply(gray)
    return cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)


# ── Draw outline ───────────────────────────────────────────────────────────
def draw_green_outline(bgr, x1, y1, x2, y2, label, confidence, contour=None):
    overlay = bgr.copy()
    cv2.rectangle(overlay, (x1-6, y1-6), (x2+6, y2+6), GREEN_GLOW, 6)
    cv2.addWeighted(overlay, 0.25, bgr, 0.75, 0, bgr)

    if contour is not None:
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        tint = bgr.copy()
        tint[mask > 0] = (tint[mask > 0] * 0.6 + np.array([0, 80, 20]) * 0.4).astype(np.uint8)
        cv2.addWeighted(tint, 0.4, bgr, 0.6, 0, bgr)
        cv2.drawContours(bgr, [contour], -1, GREEN_GLOW, 4)
        cv2.drawContours(bgr, [contour], -1, GREEN, 2)
    else:
        cv2.rectangle(bgr, (x1, y1), (x2, y2), GREEN, 2)
        clen = max(10, min(20, (x2-x1)//5, (y2-y1)//5))
        for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(bgr, (sx, sy), (sx+dx*clen, sy), GREEN, 3)
            cv2.line(bgr, (sx, sy), (sx, sy+dy*clen), GREEN, 3)

    cx, cy = (x1+x2)//2, (y1+y2)//2
    cv2.circle(bgr, (cx, cy), 3, GREEN, -1)

    tag = f"  {label}  {confidence:.0%}  "
    (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    ly = y1 - 4 if y1 - th - 10 > 0 else y2 + th + 8
    cv2.rectangle(bgr, (x1, ly-th-bl-4), (x1+tw, ly), GREEN_LBL, -1)
    cv2.rectangle(bgr, (x1, ly-th-bl-4), (x1+tw, ly), GREEN, 1)
    cv2.putText(bgr, tag, (x1, ly-bl-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


# ── Detection ──────────────────────────────────────────────────────────────
def detect_defects(gray: np.ndarray, h: int, w: int) -> list:
    """Pure OpenCV defect detection — no AI model needed."""
    detections = []
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Grain suppression: removes brushed metal texture ──────────────────
    grain_removed = cv2.subtract(blurred, cv2.blur(blurred, (60, 1)))
    grain_removed = cv2.normalize(grain_removed, None, 0, 255, cv2.NORM_MINMAX)

    # ── Edge detection for scratches ──────────────────────────────────────
    edges = cv2.Canny(grain_removed, 35, 100)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    k_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k_h, iterations=1)
    edges = cv2.dilate(edges, k_v, iterations=1)
    edges = cv2.dilate(edges, k_d, iterations=1)

    seen_boxes = []
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > w * h * 0.15:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        longest = max(bw, bh)
        if longest < w * 0.05 and longest < h * 0.05:
            continue
        aspect = longest / (min(bw, bh) + 1e-3)
        if aspect < 3.0:
            continue
        conf = float(np.clip(0.52 + aspect / 50, 0.54, 0.91))
        overlap = any(
            (min(x+bw,ex2)-max(x,ex1)) * (min(y+bh,ey2)-max(y,ey1)) /
            (bw*bh + (ex2-ex1)*(ey2-ey1) -
             max(0,(min(x+bw,ex2)-max(x,ex1))) * max(0,(min(y+bh,ey2)-max(y,ey1))) + 1) > 0.45
            for ex1,ey1,ex2,ey2 in seen_boxes
        )
        if not overlap:
            seen_boxes.append((x, y, x+bw, y+bh))
            detections.append((x, y, x+bw, y+bh, "scratches", conf, cnt))

    # ── Bright scratch detection ───────────────────────────────────────────
    local_mean = cv2.blur(blurred, (61, 61))
    bright_diff = cv2.subtract(blurred, local_mean)
    _, bt = cv2.threshold(bright_diff, 18, 255, cv2.THRESH_BINARY)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    bt = cv2.morphologyEx(bt, cv2.MORPH_OPEN, ke, iterations=1)
    bt = cv2.morphologyEx(bt, cv2.MORPH_CLOSE, ke, iterations=2)
    contours2, _ = cv2.findContours(bt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area < 400 or area > w * h * 0.10:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if max(bw, bh) < w * 0.04:
            continue
        aspect = max(bw, bh) / (min(bw, bh) + 1e-3)
        if aspect < 3.0:
            continue
        overlap = any(
            (min(x+bw,ex2)-max(x,ex1)) * (min(y+bh,ey2)-max(y,ey1)) /
            (bw*bh + (ex2-ex1)*(ey2-ey1) + 1) > 0.45
            for ex1,ey1,ex2,ey2 in seen_boxes
        )
        if not overlap:
            seen_boxes.append((x, y, x+bw, y+bh))
            detections.append((x, y, x+bw, y+bh, "scratches",
                               float(np.clip(0.54+aspect/60, 0.55, 0.90)), cnt))

    # ── Dark anomaly detection (pits, patches) ────────────────────────────
    dark_diff = cv2.subtract(local_mean, blurred)
    _, dt = cv2.threshold(dark_diff, 18, 255, cv2.THRESH_BINARY)
    dt = cv2.morphologyEx(dt, cv2.MORPH_OPEN, ke, iterations=1)
    dt = cv2.morphologyEx(dt, cv2.MORPH_CLOSE, ke, iterations=2)
    contours3, _ = cv2.findContours(dt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours3:
        area = cv2.contourArea(cnt)
        if area < 400 or area > w * h * 0.10:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if max(bw, bh) < 20:
            continue
        aspect = max(bw, bh) / (min(bw, bh) + 1e-3)
        label = "scratches" if aspect > 3.5 else ("pitted_surface" if aspect < 2.0 else "patches")
        conf  = 0.61 if aspect > 3.5 else (0.58 if aspect < 2.0 else 0.55)
        overlap = any(
            (min(x+bw,ex2)-max(x,ex1)) * (min(y+bh,ey2)-max(y,ey1)) /
            (bw*bh + (ex2-ex1)*(ey2-ey1) + 1) > 0.45
            for ex1,ey1,ex2,ey2 in seen_boxes
        )
        if not overlap:
            seen_boxes.append((x, y, x+bw, y+bh))
            detections.append((x, y, x+bw, y+bh, label, conf, cnt))

    return detections


# ── Main inference ─────────────────────────────────────────────────────────
def run_inference(image_rgb: np.ndarray, use_clahe: bool):
    if use_clahe:
        image_rgb = preprocess(image_rgb)

    h, w = image_rgb.shape[:2]
    bgr   = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    gray  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    t0 = time.perf_counter()
    raw_dets = detect_defects(gray, h, w)
    latency_ms = (time.perf_counter() - t0) * 1000

    detections = []
    for (x1, y1, x2, y2, cls_name, confidence, contour) in raw_dets:
        draw_green_outline(bgr, x1, y1, x2, y2, cls_name, confidence, contour)
        detections.append({
            "class"     : cls_name,
            "confidence": confidence,
            "severity"  : SEVERITY.get(cls_name, "Low"),
            "bbox"      : (x1, y1, x2, y2),
            "area_px"   : (x2-x1)*(y2-y1),
        })

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), detections, latency_ms


# ── UI helpers ─────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="steelscan-header">
        <h1>⬡ SteelScan AI</h1>
        <p>Surface Defect Detection System &nbsp;·&nbsp; AIML Major Project · Phase 2</p>
        <span class="status-badge">● SYSTEM ONLINE</span>
    </div>""", unsafe_allow_html=True)


def render_metrics(detections, latency_ms, img_shape):
    n   = len(detections)
    hi  = sum(1 for d in detections if d["severity"] == "High")
    v   = "PASS" if n == 0 else "FAIL"
    vc  = "#00e676" if n == 0 else "#ff1744"
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card"><div class="val">{n}</div><div class="lbl">Defects Found</div></div>
        <div class="metric-card"><div class="val" style="color:{'#ff1744' if hi else '#00e676'}">{hi}</div><div class="lbl">High Severity</div></div>
        <div class="metric-card"><div class="val">{latency_ms:.0f}<span style="font-size:1rem">ms</span></div><div class="lbl">Scan Time</div></div>
        <div class="metric-card"><div class="val" style="color:{vc}">{v}</div><div class="lbl">QC Verdict</div></div>
    </div>""", unsafe_allow_html=True)


def render_detections(detections):
    if not detections:
        st.markdown('<div class="verdict-pass">✔ NO DEFECTS DETECTED — SURFACE CLEAR</div>',
                    unsafe_allow_html=True)
        return
    st.markdown('<div class="section-hdr">DETECTED DEFECTS</div>', unsafe_allow_html=True)
    sev_order = {"High": 0, "Medium": 1, "Low": 2}
    for i, det in enumerate(sorted(detections,
                                   key=lambda d: (sev_order.get(d["severity"],3), -d["confidence"])), 1):
        sc = SEVERITY_COLOR.get(det["severity"], "#aaa")
        x1,y1,x2,y2 = det["bbox"]
        st.markdown(f"""
        <div class="det-card">
            <div>
                <span class="det-cls">#{i} {det['class'].upper()}</span>&nbsp;
                <span class="sev" style="background:{sc}22;color:{sc};border:1px solid {sc}">{det['severity']}</span>
            </div>
            <div style="text-align:right">
                <span class="det-conf">{det['confidence']:.1%}</span><br>
                <span style="font-size:.7rem;color:#6b7280;font-family:monospace">[{x1},{y1}→{x2},{y2}]</span>
            </div>
        </div>""", unsafe_allow_html=True)


def render_class_summary(all_dets):
    if not all_dets:
        return
    counts = defaultdict(int)
    for d in all_dets:
        counts[d["class"]] += 1
    total = sum(counts.values())
    st.markdown('<div class="section-hdr">CLASS DISTRIBUTION</div>', unsafe_allow_html=True)
    for cls in CLASS_NAMES:
        n = counts.get(cls, 0)
        if n == 0:
            continue
        pct = n / total
        st.markdown(f"""
        <div style="margin:5px 0">
            <div style="display:flex;justify-content:space-between;font-family:monospace;font-size:.75rem;margin-bottom:2px">
                <span style="color:#00e5ff">{cls}</span><span style="color:#6b7280">{n} ({pct:.0%})</span>
            </div>
            <div style="background:#1f2937;border-radius:3px;height:6px">
                <div style="width:{pct*100:.1f}%;height:100%;background:#00e5ff;border-radius:3px"></div>
            </div>
        </div>""", unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    inject_css()
    render_header()

    if "log"        not in st.session_state: st.session_state.log        = []
    if "scan_count" not in st.session_state: st.session_state.scan_count = 0
    if "all_dets"   not in st.session_state: st.session_state.all_dets   = []

    # Sidebar
    with st.sidebar:
        st.markdown('<div style="font-family:monospace;font-size:.7rem;color:#6b7280;letter-spacing:2px;text-transform:uppercase;border-bottom:1px solid #1f2937;padding-bottom:8px;margin-bottom:12px">⬡ SteelScan Config</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-family:monospace;font-size:.75rem;color:#00e5ff;margin-bottom:8px">Detection Engine: OpenCV</div>', unsafe_allow_html=True)
        use_clahe = st.toggle("CLAHE Enhancement", value=True)
        st.markdown("---")
        st.markdown('<div style="font-family:monospace;font-size:.68rem;color:#6b7280"><b style="color:#e8eaf0">Defect Classes</b><br><br>🟡 Crazing &nbsp; 🟠 Inclusion<br>🟢 Patches &nbsp; 🟣 Pitted<br>🔵 Rolled-in Scale<br>🔴 Scratches</div>', unsafe_allow_html=True)

    tab_upload, tab_webcam, tab_log = st.tabs(["📂  IMAGE UPLOAD", "📷  WEBCAM LIVE", "📋  SCAN LOG"])

    # ── Upload tab ─────────────────────────────────────────────────────────
    with tab_upload:
        col_l, col_r = st.columns([1.1, 1], gap="medium")
        with col_l:
            st.markdown('<div class="section-hdr">UPLOAD STEEL SURFACE IMAGE</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png","bmp","tiff"],
                                        label_visibility="collapsed")
            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                img_arr = np.array(pil_img)
                st.markdown('<div class="section-hdr">ORIGINAL IMAGE</div>', unsafe_allow_html=True)
                st.image(img_arr, use_container_width=True)
                if st.button("🔬  SCAN FOR DEFECTS", key="scan_upload"):
                    with st.spinner("Scanning..."):
                        annotated, detections, latency = run_inference(img_arr, use_clahe)
                    st.session_state.scan_count += 1
                    ts = datetime.now().strftime("%H:%M:%S")
                    for det in detections:
                        st.session_state.log.append({**det, "time": ts,
                            "source": "upload", "bbox": list(det["bbox"])})
                    st.session_state.all_dets.extend(detections)
                    st.session_state["upload_result"] = {
                        "annotated": annotated, "detections": detections,
                        "latency": latency, "shape": img_arr.shape,
                        "filename": uploaded.name}

        with col_r:
            if "upload_result" in st.session_state:
                res = st.session_state["upload_result"]
                st.markdown('<div class="section-hdr">SCAN RESULT</div>', unsafe_allow_html=True)
                st.image(res["annotated"], use_container_width=True)
                render_metrics(res["detections"], res["latency"], res["shape"])
                render_detections(res["detections"])
                render_class_summary(st.session_state.all_dets)
                buf = io.BytesIO()
                Image.fromarray(res["annotated"]).save(buf, format="PNG")
                st.download_button("⬇  Download Result Image", buf.getvalue(),
                                   f"steelscan_{res['filename']}", "image/png")
            else:
                st.markdown('<div style="height:400px;display:flex;align-items:center;justify-content:center;border:1.5px dashed #1f2937;border-radius:8px;color:#374151;text-align:center;font-family:monospace;font-size:.85rem">← Upload an image and<br>click SCAN to see results</div>', unsafe_allow_html=True)

    # ── Webcam tab ─────────────────────────────────────────────────────────
    with tab_webcam:
        st.markdown('<div style="background:#111318;border:1px solid #1f2937;border-radius:8px;padding:14px 18px;margin-bottom:1rem;font-size:.85rem;color:#9ca3af">📷 Point your camera at a metal surface and click the capture button to scan.</div>', unsafe_allow_html=True)
        col_c, col_cr = st.columns([1.1, 1], gap="medium")
        with col_c:
            st.markdown('<div class="section-hdr">CAMERA CAPTURE</div>', unsafe_allow_html=True)
            cam_frame = st.camera_input("Click to capture and scan")
            if cam_frame:
                pil_cam = Image.open(cam_frame).convert("RGB")
                cam_arr = np.array(pil_cam)
                with st.spinner("Analysing..."):
                    annotated, detections, latency = run_inference(cam_arr, use_clahe)
                st.session_state.scan_count += 1
                ts = datetime.now().strftime("%H:%M:%S")
                for det in detections:
                    st.session_state.log.append({**det, "time": ts,
                        "source": "webcam", "bbox": list(det["bbox"])})
                st.session_state.all_dets.extend(detections)
                st.session_state["webcam_result"] = {
                    "annotated": annotated, "detections": detections,
                    "latency": latency, "shape": cam_arr.shape}
        with col_cr:
            if "webcam_result" in st.session_state:
                res = st.session_state["webcam_result"]
                st.markdown('<div class="section-hdr">ANALYSIS RESULT</div>', unsafe_allow_html=True)
                st.image(res["annotated"], use_container_width=True)
                render_metrics(res["detections"], res["latency"], res["shape"])
                render_detections(res["detections"])

    # ── Log tab ────────────────────────────────────────────────────────────
    with tab_log:
        col_lg, col_ex = st.columns([2, 1], gap="medium")
        with col_lg:
            total = len(st.session_state.all_dets)
            scans = st.session_state.scan_count
            st.markdown(f'<div style="display:flex;gap:20px;margin-bottom:1rem"><div class="metric-card" style="flex:1;padding:12px"><div class="val">{scans}</div><div class="lbl">Total Scans</div></div><div class="metric-card" style="flex:1;padding:12px"><div class="val">{total}</div><div class="lbl">Total Defects</div></div></div>', unsafe_allow_html=True)
            if st.session_state.log:
                st.markdown('<div class="section-hdr">SCAN LOG</div>', unsafe_allow_html=True)
                for entry in reversed(st.session_state.log[-30:]):
                    sc = SEVERITY_COLOR.get(entry.get("severity",""), "#aaa")
                    st.markdown(f'<div style="display:grid;grid-template-columns:70px 1fr 60px 80px;gap:8px;padding:6px 10px;border-bottom:1px solid #1f2937;font-family:monospace;font-size:.75rem"><span style="color:#6b7280">{entry["time"]}</span><span style="color:#e8eaf0">{entry["class"]}</span><span style="color:#00e5ff">{entry["confidence"]:.0%}</span><span style="color:{sc}">{entry["severity"]}</span></div>', unsafe_allow_html=True)
        with col_ex:
            render_class_summary(st.session_state.all_dets)
            st.markdown('<div class="section-hdr">EXPORT</div>', unsafe_allow_html=True)
            counts = defaultdict(int)
            for d in st.session_state.all_dets:
                counts[d["class"]] += 1
            report = json.dumps({"generated_at": datetime.now().isoformat(),
                                 "total_scans": scans, "total_defects": total,
                                 "class_summary": dict(counts),
                                 "detections": st.session_state.log}, indent=2)
            st.download_button("⬇  Download Report (JSON)", report,
                               f"steelscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                               "application/json")
            if st.button("🗑  Clear Session"):
                for k in ["log","all_dets","upload_result","webcam_result"]:
                    if k in st.session_state: del st.session_state[k]
                st.session_state.scan_count = 0
                st.rerun()


if __name__ == "__main__":
    main()
