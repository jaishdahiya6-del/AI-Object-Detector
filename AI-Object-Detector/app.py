"""Streamlit web app for AI Object Detector (optional deployment target).

Run locally:   streamlit run app.py
Deploy free:   Streamlit Community Cloud (streamlit.io/cloud) - see README.
"""
import os
import tempfile
import cv2
import streamlit as st
from PIL import Image

from src import config
from src.utils import load_model, draw_detections, logger

st.set_page_config(page_title="AI Object Detector", page_icon="🎯", layout="centered")


@st.cache_resource
def get_model():
    return load_model()


def main():
    st.title("🎯 AI Object Detection System")
    st.caption("YOLOv8 + OpenCV — detect objects in images and videos in your browser.")

    model = get_model()
    st.success(f"Model loaded on device: {config.DEVICE.upper()}")

    tab_img, tab_vid = st.tabs(["📷 Image Detection", "🎬 Video Detection"])

    with tab_img:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            frame = cv2_from_pil(image)
            with st.spinner("Detecting objects..."):
                results = model.predict(frame, conf=config.CONFIDENCE_THRESHOLD,
                                         iou=config.IOU_THRESHOLD, device=config.DEVICE, verbose=False)
                frame, count = draw_detections(frame, results)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"{count} object(s) detected")

    with tab_vid:
        uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            st.info("Video uploaded. Processing first pass may take a while on CPU.")
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            frame_limit = 200  # safety cap for hosted demos
            i = 0
            while cap.isOpened() and i < frame_limit:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(frame, conf=config.CONFIDENCE_THRESHOLD,
                                         iou=config.IOU_THRESHOLD, device=config.DEVICE, verbose=False)
                frame, _ = draw_detections(frame, results)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                i += 1
            cap.release()


def cv2_from_pil(pil_image):
    import numpy as np
    arr = np.array(pil_image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    main()
