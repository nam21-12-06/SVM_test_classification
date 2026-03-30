import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

# Page config
st.set_page_config(
    page_title="AI Text Classifier",
    layout="centered"
)

# Title
st.title("AI Text Classification System")
st.markdown("Classify text into 20 topics using Machine Learning (TF-IDF + SVM)")

# Input
text = st.text_area("Enter your text here:", height=150)

# Button
if st.button("Predict"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        try:
            with st.spinner("Predicting..."):
                response = requests.post(API_URL, json={"text": text})
                data = response.json()

            predictions = data["predictions"]

            # Top prediction
            top_pred = predictions[0]
            st.subheader("Top Prediction")
            st.success(f"{top_pred['label']} — {top_pred['confidence_percent']}%")

            # Top-K predictions
            st.subheader("Top Predictions")
            for pred in predictions:
                label = pred["label"]
                conf = pred["confidence_percent"]

                st.write(f"**{label}**")
                st.progress(int(conf))

        except Exception as e:
            st.error(f"Error: {e}")