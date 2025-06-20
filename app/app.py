import streamlit as st
import requests

st.set_page_config(page_title="Email Feedback Classifier")

st.title("📧 AI Feedback Classifier + Sentiment Score")

text = st.text_area("Enter feedback:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some feedback text.")
    else:
        response = requests.post(
            "http://127.0.0.1:8000/predict/",
            json={"text": text}
        )

        if response.status_code == 200:
            data = response.json()
            st.success(f"📌 Category: **{data['predicted_label']}**")
            st.info(f"💬 VADER Sentiment: **{data['vader_sentiment']['compound']}**")
            st.info(f"🤖 HF Sentiment: **{data['hf_sentiment']['label']}** (score: {data['hf_sentiment']['score']:.2f})")
        else:
            st.error("Prediction failed. Check the FastAPI server.")
