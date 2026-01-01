from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib as jb
from pathlib import Path
from dataLoader import loadData
from preprocess import preprocess_text
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR/"models"
MODEL_PATH = MODEL_DIR/"pipeline.pkl"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def build_pipeline():
    pipeline = Pipeline([
        (
            'tfidf', 
            TfidfVectorizer(
                preprocessor=preprocess_text,
                max_df=0.9,
                min_df=5,
                ngram_range=(1, 2)
            )
        ),
        (
            'clf', 
            LogisticRegression(
                solver='liblinear', 
                max_iter=1000,
                random_state=42
            )
        ),
    ])
    return pipeline

def train():
    X_train, X_test, y_train, y_test = loadData()

    pipeline = build_pipeline()

    logging.info("Starting model training...")
    pipeline.fit(X_train, y_train)
    logging.info("Training completed...")

    MODEL_DIR.mkdir(exist_ok=True)
    jb.dump(pipeline, MODEL_PATH)
    logging.info(f"Model saved at: {MODEL_PATH}")

if __name__ == '__main__':
    train()