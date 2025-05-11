# app/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # <- Moved to top

import streamlit as st
from src.predict import predict
from src.explain_llm import generate_explanation


st.set_page_config(page_title="DeepFakeShield", layout="centered")
st.title("🛡️ DeepFakeShield: AI Deepfake Detection")

st.write("Upload an image (real or fake) to detect if it's AI-generated.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file
    file_path = os.path.join("temp_upload", uploaded_file.name)
    os.makedirs("temp_upload", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    # Predict and explain
    st.write("\n🔍 Running prediction...")
    label, confidence = predict(file_path, "models/simple_cnn.pth")

    st.markdown(f"### 🧠 Prediction: `{label}` ({confidence:.2f} confidence)")

    st.write("\n💬 Generating LLM explanation...")
    explanation = generate_explanation(label, confidence)
    st.markdown("**Why did the AI predict this?**")
    st.info(explanation)

    # Cleanup
    os.remove(file_path)
