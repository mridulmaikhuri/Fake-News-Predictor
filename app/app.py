import streamlit as st
import joblib
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(ROOT_DIR))

from scripts.preprocess import preprocess_text
from scripts.explain import shap_explanation
from scripts.dataLoader import loadData

import requests
from bs4 import BeautifulSoup

def parse_url(url, timeout=10):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = " ".join(
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) > 30
    )

    return text

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pipeline.pkl"
CLASS_NAMES = ["Fake", "Real"]

# Loaders
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
@st.cache_data
def load_background_data(n_samples=100):
    X_train, _, _, _ = loadData()
    return X_train.sample(n_samples, random_state=42)


pipeline = load_model()
X_background = load_background_data()

# UI
st.title("📰 Fake News Detection App")
st.write("Check whether a news article is **Fake or Real**.")

input_mode = st.radio(
    "Choose input type",
    ["Paste Text", "Paste URL"],
    horizontal=True
)

if input_mode == "Paste Text":
    raw_text = st.text_area("News Content", height=250)
else:
    raw_text = st.text_input("News Article URL")

show_explanation = st.checkbox("Show SHAP explanation")

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    try:
        if not raw_text.strip():
            st.warning("Please provide input.")
            st.stop()

        if input_mode == "Paste URL":
            with st.spinner("Fetching article from URL..."):
                raw_text = parse_url(raw_text)

            if len(raw_text) < 200:
                st.error("Could not extract enough text from the URL.")
                st.stop()

            with st.expander("📄 Extracted Article Text"):
                st.write(raw_text[:3000] + "...")

        processed_text = preprocess_text(raw_text)

        pred = pipeline.predict([processed_text])[0]
        prob = pipeline.predict_proba([processed_text])[0]

        st.subheader("Prediction")
        st.success(f"🟢 {CLASS_NAMES[pred]}")

        st.subheader("Confidence")
        st.progress(int(prob[pred] * 100))
        st.write(f"{prob[pred]*100:.2f}% confidence")

        # ---------------- SHAP ----------------
        if show_explanation:
            with st.spinner("Generating SHAP explanation..."):
                fake_words, real_words = shap_explanation(
                    pipeline,
                    processed_text,
                    X_background
                )

            st.subheader("🔍 Model Explanation (SHAP)")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔴 Words pushing **FAKE**")
                st.dataframe(fake_words, use_container_width=True)

            with col2:
                st.markdown("### 🟢 Words pushing **REAL**")
                st.dataframe(real_words, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")