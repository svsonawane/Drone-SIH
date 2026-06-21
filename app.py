from huggingface_hub import hf_hub_download
import os

os.makedirs("models", exist_ok=True)
if not os.path.exists("models/best.pth"):
    hf_hub_download(
        repo_id="SnehalSonawane/Drone",
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
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
import io

st.set_page_config(page_title="Land Segmentation", layout="wide", page_icon="🛰️")

CLASS_NAMES  = ["Background", "Building", "Woodland", "Water", "Road"]
CLASS_COLORS = np.array([
    [100, 100, 100],
    [255,  50,  50],
    [ 50, 200,  50],
    [ 50, 150, 255],
    [200, 200, 200],
], dtype=np.uint8)

GSD = 0.25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    img = cv2.resize(img_rgb, (size, size))
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
        px = int((pred == i).sum())
        m2 = px * gsd ** 2
        ha = m2 / 10000
        pct = px / total_px * 100
        rows.append({"Class": name, "Pixels": px, "Area (m²)": round(m2, 1),
                     "Area (ha)": round(ha, 4), "Coverage (%)": round(pct, 1)})
    return rows

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    model_path = st.text_input("Model path", value="models/best.pth")
    gsd_input  = st.number_input("Ground Sampling Distance (m/pixel)", value=0.25, step=0.05, format="%.2f")
    st.markdown("---")
    st.markdown("**Class Legend**")
    for i, name in enumerate(CLASS_NAMES):
        c = CLASS_COLORS[i]
        st.markdown(
            f'<span style="background-color:rgb({c[0]},{c[1]},{c[2]});'
            f'padding:2px 12px;border-radius:4px;color:white;font-weight:bold">&nbsp;</span> {name}',
            unsafe_allow_html=True
        )
        st.markdown("")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🛰️ Land Segmentation & Area Estimator")
st.markdown("Upload a satellite or aerial image to detect **buildings, woodland, water, roads** and estimate their real-world area.")

if not os.path.exists(model_path):
    st.error(f"Model not found at `{model_path}`. Please check your Hugging Face repo.")
    st.stop()

model = load_model(model_path)
st.success(f"✅ Model loaded from `{model_path}`")

uploaded = st.file_uploader("Upload an image (JPG, PNG, TIF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read the image. Please upload a valid JPG or PNG file.")
        st.stop()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]

    st.markdown("---")

    with st.spinner("Running segmentation..."):
        pred       = predict(model, img_rgb)
        pred_color = colorize(pred)
        rows       = area_report(pred, gsd=gsd_input)

    # ── Side by side images ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(img_rgb, use_column_width=True, caption=f"Size: {w}×{h} px")
    with col2:
        st.subheader("🗺️ Predicted Mask")
        st.image(pred_color, use_column_width=True, caption="Colour-coded land classes")

    # ── Overlay ───────────────────────────────────────────────────────────────
    st.subheader("🔀 Overlay (50% blend)")
    img_res   = cv2.resize(img_rgb,   (512, 512))
    mask_res  = cv2.resize(pred_color,(512, 512), interpolation=cv2.INTER_NEAREST)
    overlay   = cv2.addWeighted(img_res, 0.5, mask_res, 0.5, 0)
    st.image(overlay, use_column_width=False, width=700)

    # ── Area table ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📐 Land Area Report")

    cols = st.columns(len(CLASS_NAMES))
    for i, row in enumerate(rows):
        c = CLASS_COLORS[i]
        with cols[i]:
            st.markdown(
                f'<div style="background-color:rgb({c[0]},{c[1]},{c[2]});'
                f'padding:10px;border-radius:8px;text-align:center;color:white">'
                f'<b>{row["Class"]}</b><br>'
                f'{row["Area (m²)"]:,.0f} m²<br>'
                f'{row["Area (ha)"]} ha<br>'
                f'{row["Coverage (%)"]}%</div>',
                unsafe_allow_html=True
            )

    st.markdown("###")

    import pandas as pd
    df = pd.DataFrame(rows)
    df = df[df["Pixels"] > 0]
    st.dataframe(df.set_index("Class"), use_container_width=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.subheader("📊 Area Breakdown Chart")
    fig, ax = plt.subplots(figsize=(8, 4))
    labels  = [r["Class"] for r in rows if r["Pixels"] > 0]
    values  = [r["Area (ha)"] for r in rows if r["Pixels"] > 0]
    colors  = [CLASS_COLORS[CLASS_NAMES.index(l)] / 255 for l in labels]
    bars    = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.3f} ha", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Area (hectares)")
    ax.set_title("Detected Land Cover Areas")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Download buttons ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Download Results")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        mask_pil = Image.fromarray(pred_color)
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        st.download_button("Download Mask PNG", buf.getvalue(),
                           file_name="predicted_mask.png", mime="image/png")

    with col_b:
        overlay_pil = Image.fromarray(overlay)
        buf2 = io.BytesIO()
        overlay_pil.save(buf2, format="PNG")
        st.download_button("Download Overlay PNG", buf2.getvalue(),
                           file_name="overlay.png", mime="image/png")

    with col_c:
        csv = df.to_csv().encode("utf-8")
        st.download_button("Download Area CSV", csv,
                           file_name="area_report.csv", mime="text/csv")
