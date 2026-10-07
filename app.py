"""Streamlit demo: upload an image or video, pick classes, get annotated output + counts.

Run:  streamlit run app.py
"""
import json
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from visioncount import VisionCounter

st.set_page_config(page_title="VisionCount", layout="wide")
st.title("VisionCount — YOLO Object Detection & Counting")
st.caption("YOLOv8 + ByteTrack + OpenCV. Upload an image to detect and count objects, "
           "or a video to track them and count line crossings.")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], help="n = fastest, m = most accurate")
    conf = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
    classes = st.multiselect("Only count these classes (empty = all)",
                             ["person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "bottle"],
                             default=[])
    line_pos = st.slider("Counting line height (% of frame, video only)", 10, 90, 50)


@st.cache_resource
def load(model, conf, classes):
    return VisionCounter(model, conf, list(classes) or None)


file = st.file_uploader("Image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
if file:
    suffix = Path(file.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.read())
    tmp.close()

    if suffix in {".jpg", ".jpeg", ".png"}:
        vc = load(model, conf, tuple(classes))
        img = cv2.imread(tmp.name)
        out, counts = vc.detect_image(img)
        c1, c2 = st.columns([3, 1])
        c1.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        c2.subheader("Counts")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            c2.metric(k, v)
    else:
        cap = cv2.VideoCapture(tmp.name)
        w, h = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()
        y = int(h * line_pos / 100)
        vc = VisionCounter(model, conf, classes or None, line=((0, y), (w, y)))
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        frame_box, bar = st.empty(), st.progress(0.0)
        cap = cv2.VideoCapture(tmp.name)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(5) or 25, (w, h))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = vc.process_frame(frame)
            writer.write(out)
            if i % 5 == 0:
                frame_box.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
            i += 1
            bar.progress(min(i / total, 1.0))
        cap.release()
        writer.release()
        s = vc.summary(frames=i)
        a, b, c = st.columns(3)
        a.metric("In", s["in"]); b.metric("Out", s["out"]); c.metric("Total", s["total"])
        st.json(s)
        st.download_button("Download annotated video", open(out_path, "rb"), "visioncount_output.mp4")
        st.download_button("Download summary JSON", json.dumps(s, indent=2), "summary.json")
