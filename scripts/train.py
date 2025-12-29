from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib as jb
from pathlib import Path
from dataLoader import loadData
from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/'data'

def train():
    X_train, X_test, y_train, y_test = loadData()
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

    print ("Pipeline defined. Starting model training...")
    pipeline.fit(X_train, y_train)
    print ("Model training completed. Storing it in .pkl file for future use...")

    MODEL_DIR = BASE_DIR/"models"
    MODEL_DIR.mkdir(exist_ok=True)

    MODEL_PATH = MODEL_DIR/"pipeline.pkl"
    jb.dump(pipeline, MODEL_PATH)

    print(f"Model stored successfully at {MODEL_PATH}")

if __name__ == '__main__':
    train()