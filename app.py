import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image

# Processing Functions
def segment_leaf(img_rgb):
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, np.array([5, 20, 20]), np.array([95, 255, 255]))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((7,7), np.uint8))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    res = np.zeros_like(mask)
    for cnt in contours:
        if cv.contourArea(cnt) > 500:
            cv.drawContours(res, [cnt], -1, 255, -1)
    return res

def get_pseudo(gray, cmap_val, mask):
    p = cv.applyColorMap(gray, cmap_val)
    p_rgb = cv.cvtColor(p, cv.COLOR_BGR2RGB)
    return cv.bitwise_and(p_rgb, p_rgb, mask=mask)

# UI Layout
st.set_page_config(page_title="Tomato Leaf App", layout="wide")
st.title("🍅 Tomato Leaf Pseudocolour App")

uploaded_file = st.file_uploader("Upload Leaf Image", type=['jpg','png','jpeg'])

if uploaded_file:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, 1)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Process
    mask = segment_leaf(img_rgb)
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    
    # Display Results
    col1, col2 = st.columns(2)
    col1.image(img_rgb, caption="Original RGB")
    col2.image(cv.bitwise_and(img_rgb, img_rgb, mask=mask), caption="Segmented Leaf")
    
    st.divider()
    st.subheader("Pseudocolour LUTs")
    l1, l2, l3, l4 = st.columns(4)
    l1.image(get_pseudo(clahe, cv.COLORMAP_JET, mask), caption="JET")
    l2.image(get_pseudo(clahe, cv.COLORMAP_HOT, mask), caption="HOT")
    l3.image(get_pseudo(clahe, cv.COLORMAP_HSV, mask), caption="HSV")
    l4.image(get_pseudo(clahe, cv.COLORMAP_VIRIDIS, mask), caption="VIRIDIS")
