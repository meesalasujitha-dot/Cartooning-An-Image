
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Cartoon Image Generator", layout="wide")
st.title("🎨 Cartoon Image Generator")

def preprocess(image):
    img = np.array(image)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def cartoonize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,9,9)
    color = cv2.bilateralFilter(img,9,250,250)
    return cv2.bitwise_and(color,color,mask=edges)

# ✅ EXACT strong pencil (same as good earlier version)
def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21,21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return sketch  # no extra brightness/contrast changes

def black_white(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded)
    img = preprocess(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_container_width=True)

    option = st.selectbox("Filter", ["Cartoon","Pencil Sketch","Black & White"])

    if option == "Cartoon":
        output = cartoonize(img)
    elif option == "Pencil Sketch":
        output = pencil_sketch(img)
    else:
        output = black_white(img)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB) if len(output.shape)==3 else output

    with col2:
        st.image(output_rgb, caption="Output", use_container_width=True)

    buf = io.BytesIO()
    Image.fromarray(output_rgb).save(buf, format="PNG")
    st.download_button("Download", buf.getvalue(), "output.png")

else:
    st.warning("Upload image")
