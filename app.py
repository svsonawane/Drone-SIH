from huggingface_hub import hf_hub_download
import os

os.makedirs("models", exist_ok=True)
if not os.path.exists("models/best.pth"):
    hf_hub_download(
        repo_id="svsonawane/drone-sih",  # ← change this to your HF repo
        filename="best.pth",
        local_dir="models"
    )

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import io
import pandas as pd

st.set_page_config(page_title="SkyMap — Land Segmentation", layout="wide", page_icon="🛰️")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit header/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Page background */
.stApp {
    background: #0b0f1a;
}

/* Main content area */
.block-container {
    padding: 2rem 3rem 4rem 3rem !important;
    max-width: 1300px;
}

/* Hero title */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #e8eaf0;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0;
}
.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    margin-top: 0.5rem;
    font-weight: 400;
    letter-spacing: 0.01em;
}
.hero-accent {
    color: #4ade80;
}
.hero-wrap {
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid #1f2937;
    margin-bottom: 2rem;
}

/* Stat cards */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 1.5rem 0;
}
.stat-card {
    border-radius: 12px;
    padding: 1rem 1.1rem;
    border: 1px solid #1f2937;
    background: #111827;
}
.stat-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 6px;
}
.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #e8eaf0;
    line-height: 1.1;
}
.stat-pct {
    font-size: 12px;
    color: #6b7280;
    margin-top: 4px;
}
.dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}

/* Section headers */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 0.8rem;
    margin-top: 2rem;
}

/* Upload zone */
.stFileUploader > div {
    border: 1.5px dashed #1f2937 !important;
    border-radius: 12px !important;
    background: #111827 !important;
    padding: 2rem !important;
    transition: border-color 0.2s;
}
.stFileUploader > div:hover {
    border-color: #4ade80 !important;
}
.stFileUploader label {
    color: #9ca3af !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1f2937 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #9ca3af !important;
    font-size: 13px !important;
}

/* Input fields */
.stTextInput input, .stNumberInput input {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    font-size: 13px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #4ade80 !important;
    box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.1) !important;
}

/* Success/error alerts */
.stSuccess {
    background: rgba(74, 222, 128, 0.08) !important;
    border: 1px solid rgba(74, 222, 128, 0.2) !important;
    border-radius: 8px !important;
    color: #4ade80 !important;
}
.stError {
    background: rgba(239, 68, 68, 0.08) !important;
    border: 1px solid rgba(239, 68, 68, 0.2) !important;
    border-radius: 8px !important;
}

/* Image containers */
.stImage img {
    border-radius: 10px !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #4ade80 !important;
}

/* Divider */
hr {
    border-color: #1f2937 !important;
    margin: 2rem 0 !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #1f2937 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Download buttons */
.stDownloadButton button {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
    color: #9ca3af !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
.stDownloadButton button:hover {
    border-color: #4ade80 !important;
    color: #4ade80 !important;
    background: rgba(74, 222, 128, 0.05) !important;
}

/* Image caption */
.stImage p {
    color: #4b5563 !important;
    font-size: 12px !important;
    text-align: center !important;
}

/* Column headers for images */
h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
}

/* Plot background */
.stPlotlyChart, .stPyplot {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Background", "Building", "Woodland", "Water", "Road"]
CLASS_COLORS = np.array([
    [100, 100, 100],
    [255,  50,  50],
    [ 50, 200,  50],
    [ 50, 150, 255],
    [200, 200, 200],
], dtype=np.uint8)

HEX_COLORS = ["#646464", "#ff3232", "#32c832", "#3296ff", "#c8c8c8"]
CLASS_ICONS = ["◼", "🏠", "🌲", "💧", "🛣"]

GSD    = 0.25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=5
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def preprocess(img_rgb, size=512):
    img  = cv2.resize(img_rgb, (size, size))
    norm = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return norm(Image.fromarray(img)).unsqueeze(0).to(DEVICE)

def predict(model, img_rgb):
    inp = preprocess(img_rgb)
    with torch.no_grad():
        out  = model(inp)
        pred = out.squeeze(0).argmax(0).cpu().numpy()
    return pred

def colorize(pred):
    return CLASS_COLORS[pred]

def area_report(pred, gsd=GSD):
    total_px = pred.size
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        px  = int((pred == i).sum())
        m2  = px * gsd ** 2
        ha  = m2 / 10000
        pct = px / total_px * 100
        rows.append({
            "Class": name,
            "Pixels": px,
            "Area (m²)": round(m2, 1),
            "Area (ha)": round(ha, 4),
            "Coverage (%)": round(pct, 1)
        })
    return rows

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    model_path = st.text_input("Model path", value="models/best.pth")
    gsd_input  = st.number_input("Ground sampling distance (m/px)", value=0.25, step=0.05, format="%.2f")

    st.markdown("---")
    st.markdown("**Legend**")
    for i, name in enumerate(CLASS_NAMES):
        c = CLASS_COLORS[i]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0;">'
            f'<span style="width:10px;height:10px;border-radius:50%;'
            f'background:rgb({c[0]},{c[1]},{c[2]});display:inline-block;flex-shrink:0;"></span>'
            f'<span style="color:#9ca3af;font-size:13px;">{name}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        '<p style="color:#374151;font-size:11px;">SkyMap v1.0 · ResNet-34 U-Net</p>',
        unsafe_allow_html=True
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <p class="hero-title">Sky<span class="hero-accent">Map</span></p>
    <p class="hero-sub">Satellite & aerial land cover segmentation — buildings, woodland, water, roads</p>
</div>
""", unsafe_allow_html=True)

# ── Model load ────────────────────────────────────────────────────────────────
if not os.path.exists(model_path):
    st.error(f"Model not found at `{model_path}`. Check your Hugging Face repo or model path.")
    st.stop()

model = load_model(model_path)
st.success(f"✓ Model ready · {DEVICE.type.upper()}")

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Input Image</p>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a satellite or aerial image here",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    label_visibility="collapsed"
)

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not decode the image. Please upload a valid JPG or PNG.")
        st.stop()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]

    with st.spinner("Analysing image..."):
        pred       = predict(model, img_rgb)
        pred_color = colorize(pred)
        rows       = area_report(pred, gsd=gsd_input)

    # ── Side by side ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Segmentation Output</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("### Original")
        st.image(img_rgb, use_column_width=True, caption=f"{w} × {h} px")
    with col2:
        st.markdown("### Predicted mask")
        st.image(pred_color, use_column_width=True, caption="Colour-coded land classes")

    # ── Overlay ───────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Overlay · 50% blend</p>', unsafe_allow_html=True)
    img_res  = cv2.resize(img_rgb,    (512, 512))
    mask_res = cv2.resize(pred_color, (512, 512), interpolation=cv2.INTER_NEAREST)
    overlay  = cv2.addWeighted(img_res, 0.5, mask_res, 0.5, 0)
    st.image(overlay, use_column_width=False, width=680)

    # ── Stat cards ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Area Report</p>', unsafe_allow_html=True)

    cards_html = '<div class="stat-grid">'
    for i, row in enumerate(rows):
        c = CLASS_COLORS[i]
        cards_html += f"""
        <div class="stat-card">
            <div class="stat-label">
                <span class="dot" style="background:rgb({c[0]},{c[1]},{c[2]})"></span>
                {row['Class']}
            </div>
            <div class="stat-value">{row['Area (ha)']:.2f} <span style="font-size:13px;font-weight:400;color:#6b7280;">ha</span></div>
            <div class="stat-pct">{row['Coverage (%)']:.1f}% · {row['Area (m²)']:,.0f} m²</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Table ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df = df[df["Pixels"] > 0]
    st.dataframe(
        df.set_index("Class"),
        use_container_width=True,
        hide_index=False
    )

    # ── Chart ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Coverage Breakdown</p>', unsafe_allow_html=True)

    labels = [r["Class"] for r in rows if r["Pixels"] > 0]
    values = [r["Area (ha)"] for r in rows if r["Pixels"] > 0]
    colors = [tuple(CLASS_COLORS[CLASS_NAMES.index(l)] / 255.0) for l in labels]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("#0b0f1a")
    ax.set_facecolor("#111827")

    bars = ax.bar(labels, values, color=colors, width=0.5,
                  edgecolor="none", zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.3f} ha",
                ha="center", va="bottom",
                fontsize=9, color="#9ca3af",
                fontfamily="monospace")

    ax.set_ylabel("Area (ha)", color="#4b5563", fontsize=11)
    ax.tick_params(colors="#6b7280", labelsize=11)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.yaxis.grid(True, color="#1f2937", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.25 if values else 1)

    for label in ax.get_xticklabels():
        label.set_color("#9ca3af")
    for label in ax.get_yticklabels():
        label.set_color("#4b5563")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Export</p>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        buf = io.BytesIO()
        Image.fromarray(pred_color).save(buf, format="PNG")
        st.download_button("↓ Mask PNG", buf.getvalue(),
                           file_name="predicted_mask.png", mime="image/png")
    with col_b:
        buf2 = io.BytesIO()
        Image.fromarray(overlay).save(buf2, format="PNG")
        st.download_button("↓ Overlay PNG", buf2.getvalue(),
                           file_name="overlay.png", mime="image/png")
    with col_c:
        st.download_button("↓ Area CSV", df.to_csv().encode("utf-8"),
                           file_name="area_report.csv", mime="text/csv")
