import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦")
st.title("🚦 Traffic Sign Classifier")

img = st.file_uploader("Upload a traffic sign image", type=["jpg", "jpeg", "png"])
if img:
    st.image(img, width=240)
    r = requests.post(f"{API}/predict", files={"file": (img.name, img.getvalue(), img.type)})
    if r.ok:
        d = r.json()
        st.metric("Prediction", d["prediction"], f"{d['confidence']:.1%} confidence")
        if d["uncertain"]:
            st.warning("Low confidence – please review this image manually.")
        st.bar_chart({t["class"]: t["confidence"] for t in d["top3"]})
    else:
        st.error(r.text)