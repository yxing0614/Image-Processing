import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# =========================================================
# CORE PROCESSING FUNCTIONS (From your Notebook)
# =========================================================

def get_segmentation_steps(img_rgb):
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, np.array([5, 20, 20]), np.array([95, 255, 255]))
    kernel = np.ones((7,7), np.uint8)
    holey_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    contours, _ = cv.findContours(holey_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    healed_mask = np.zeros_like(holey_mask)
    if contours:
        cnt = max(contours, key=cv.contourArea)
        cv.drawContours(healed_mask, [cnt], -1, 255, -1)
    return holey_mask, healed_mask

def apply_clahe(img_gray, clip=2.0):
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    return clahe.apply(img_gray)

def edge_strength(img, mask):
    sx = cv.Sobel(img, cv.CV_64F, 1, 0, 3)
    sy = cv.Sobel(img, cv.CV_64F, 0, 1, 3)
    mag = np.sqrt(sx**2 + sy**2)
    return np.mean(mag[mask > 0])

# =========================================================
# STREAMLIT INTERFACE
# =========================================================

st.set_page_config(page_title="Tomato Lab Diagnostic Suite", layout="wide")
st.title("🍅 Tomato Leaf Advanced Diagnostic Suite")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Tomato Leaf Image", type=['jpg','jpeg','png'])

if uploaded_file:
    # Load and Prepare
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv.imdecode(file_bytes, 1)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)

    # Sidebar Controls
    st.sidebar.header("Global Controls")
    clahe_clip = st.sidebar.slider("CLAHE Intensity", 1.0, 5.0, 2.0)
    
    # Run Core Pipeline
    holey, final_mask = get_segmentation_steps(img_rgb)
    enhanced_gray = apply_clahe(img_gray, clahe_clip)

    # TABS FOR DIFFERENT DOCUMENTATION SECTIONS
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Morphological Segmentation", 
        "2. Structural Enhancement", 
        "3. Pseudocolour Diagnostics",
        "4. Reliability Tests"
    ])

    # --- TAB 1: SEGMENTATION ---
    with tab1:
        st.header("Objective 1: Chromatic Isolation & Morphological Healing")
        c1, c2, c3 = st.columns(3)
        c1.image(img_rgb, caption="Input RGB")
        c2.image(holey, caption="Step A: Holey Mask (Pathological Gaps)")
        c3.image(final_mask, caption="Step B: Final Healed Mask (Solid Fill)")
        st.info("The algorithm uses Contour-Filling to 'heal' the gaps caused by dark necrotic lesions that fall outside the HSV range.")

    # --- TAB 2: ENHANCEMENT ---
    with tab2:
        st.header("Objective 2: Intensity Profile & Edge Strength")
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.image(enhanced_gray, caption="CLAHE Enhanced Base")
            e_orig = edge_strength(img_gray, final_mask)
            e_new = edge_strength(enhanced_gray, final_mask)
            st.metric("Mean Edge Strength", f"{e_new:.2f}", f"+{((e_new-e_orig)/e_orig)*100:.1f}%")

        with col_right:
            # Interactive Line Scan
            row_idx = st.slider("Select Scan Line (Row)", 0, img_gray.shape[0], img_gray.shape[0]//2)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(img_gray[row_idx, :], color='gray', label="Original", alpha=0.6)
            ax.plot(enhanced_gray[row_idx, :], color='green', label="CLAHE Enhanced")
            ax.set_title(f"Intensity Profile at Row {row_idx}")
            ax.legend()
            st.pyplot(fig)

    # --- TAB 3: PSEUDOCOLOUR ---
    with tab3:
        st.header("Visual Diagnostics per LUT")
        luts = {"JET": cv.COLORMAP_JET, "HOT": cv.COLORMAP_HOT, "HSV": cv.COLORMAP_HSV, "VIRIDIS": cv.COLORMAP_VIRIDIS}
        cols = st.columns(4)
        
        for i, (name, val) in enumerate(luts.items()):
            p = cv.applyColorMap(enhanced_gray, val)
            p_rgb = cv.cvtColor(p, cv.COLOR_BGR2RGB)
            masked_p = cv.bitwise_and(p_rgb, p_rgb, mask=final_mask)
            cols[i].image(masked_p, caption=f"{name} LUT")
            
    # --- TAB 4: RELIABILITY ---
    with tab4:
        st.header("Objective 3: Generalisation & Stability Tests")
        
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.subheader("Structural Fidelity (Round-Trip)")
            # HSV vs HOT Round-trip
            hsv_p = cv.applyColorMap(enhanced_gray, cv.COLORMAP_HSV)
            hsv_back = cv.cvtColor(hsv_p, cv.COLOR_BGR2GRAY)
            ssim_val = ssim(enhanced_gray, hsv_back)
            st.image(hsv_back, caption=f"HSV-to-Gray Distortion (SSIM: {ssim_val:.3f})")
            st.error("HSV creates 'Structural Hallucinations' visible as false contours above.")

        with c_right:
            st.subheader("Chromatic Reliability (Stress Test)")
            # Simulate dimming
            dimmed = (enhanced_gray.astype(float) * 0.6).astype(np.uint8)
            jet_std = cv.applyColorMap(enhanced_gray, cv.COLORMAP_JET)
            jet_dim = cv.applyColorMap(dimmed, cv.COLORMAP_JET)
            st.image(cv.cvtColor(jet_dim, cv.COLOR_BGR2RGB), caption="JET Output @ 60% Brightness")
            st.warning("Note the Color Shift: NCD instability causes the lesion to change color category in low light.")

else:
    st.image("https://images.unsplash.com/photo-1592841200221-a6898f307bac?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", use_column_width=True)
    st.info("Upload an image in the sidebar to start the Technical Diagnostic Suite.")
