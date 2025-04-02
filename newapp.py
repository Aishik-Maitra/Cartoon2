import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_file(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def color_quantization(img, k=10):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

def soft_glow(img, intensity=0.4):
    gaussian = cv2.GaussianBlur(img, (21, 21), 0)
    glow = cv2.addWeighted(img, 1.0 + intensity, gaussian, -intensity, 0)
    return glow

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a + 15, b + 20))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

st.title("Ghibli-Style Image Filter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    
    if st.button("Classify"):
        img = read_file(image)
        img_color = color_quantization(img, k=10)
        blurred = cv2.bilateralFilter(img_color, d=9, sigmaColor=150, sigmaSpace=150)
        glow_img = soft_glow(blurred, intensity=0.5)
        final_img = enhance_contrast(glow_img)
        
        st.image(final_img, caption="Ghibli-Style Image", use_column_width=True)