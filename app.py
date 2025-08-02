import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
from predictor import predict

# App configuration
st.set_page_config(page_title="TruthGuard 🛡️", layout="centered")

# Title and intro
st.title("🛡️ TruthGuard - Fake News Detector")
st.markdown("Enter a news headline below, and TruthGuard will detect whether it's **Real** or **Fake**.")

# Display library versions (for debugging or transparency)
with st.expander("⚙️ Environment Info"):
    st.write("Numpy version:", np.__version__)
    st.write("Pandas version:", pd.__version__)
    st.write("Torch version:", torch.__version__)

# Headline input
headline = st.text_input("📰 Headline to verify")

# Predict on input
if st.button("Predict"):
    if not headline.strip():
        st.warning("Please enter a headline first.")
    else:
        prediction, confidence = predict(headline)
        
        st.success(f"✅ Prediction: **{prediction.upper()}**")
        st.info(f"📊 Confidence: **{confidence}%**")
