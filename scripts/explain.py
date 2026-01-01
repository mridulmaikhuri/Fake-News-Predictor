from dataLoader import loadData
from pathlib import Path
from preprocess import preprocess_text
import joblib as jb
import pandas as pd
import logging
import shap

# Config

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR/"models"
MODEL_PATH = MODEL_DIR/"pipeline.pkl"

CLASS_NAMES = ["Fake", "Real"]

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    logging.info(f"Loading model from {MODEL_PATH}")
    return jb.load(MODEL_PATH)

def shap_explanation(pipeline, text, X_background, top_k = 10):
    vectorizer = pipeline.named_steps['tfidf']
    model = pipeline.named_steps['clf']

    X_vec = vectorizer.transform([text])
    X_bg_vec = vectorizer.transform(X_background)
    masker = shap.maskers.Independent(X_bg_vec)

    explainer = shap.LinearExplainer(
        model,
        masker
    )

    shap_values = explainer.shap_values(X_vec)
    feature_names = vectorizer.get_feature_names_out()
    values = shap_values[0]
    nonzero_idx = X_vec.nonzero()[1]

    df = pd.DataFrame({
        "word": feature_names[nonzero_idx],
        "shap_value": values[nonzero_idx]
    }).sort_values("shap_value")

    return df.head(top_k), df.tail(top_k)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = loadData()
    pipeline = load_model()
    
    text = X_test.iloc[0]
    true_label = y_test.iloc[0]
    pred_label = pipeline.predict([text])[0]
    confidence = pipeline.predict_proba([text])[0]

    print(f"True Label: {CLASS_NAMES[true_label]}")
    print(f"Predicted Label: {CLASS_NAMES[pred_label]}")
    print(f"Confidence: {confidence[0]*100}%")

    # exp = lime_explanation(pipeline, text, 20)
    # print("\nLIME Explanation (Top Features):\n")
    # for word, weight in exp.as_list():
    #     direction = "↑ REAL" if weight > 0 else "↓ FAKE"
    #     print(f"{word:30} {weight:+.4f} ({direction})")

    fake_words, real_words = shap_explanation(pipeline, text, X_train.sample(100, random_state=42))
    print("\nTop words pushing the prediction towards FAKE are:\n")
    print(fake_words)
    print("\nTop words pushing the prediction towards REAL are:\n")
    print(real_words)