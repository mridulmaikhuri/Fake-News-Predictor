import joblib as jb
from pathlib import Path
from scripts.dataLoader import loadData
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from scripts.preprocess import preprocess_text
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR/"models"
MODEL_PATH = MODEL_DIR/"pipeline.pkl"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

CLASS_NAMES = ["Fake", "Real"]

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    logging.info(f"Loading model from {MODEL_PATH}")
    return jb.load(MODEL_PATH)

def evaluate():
    pipeline = load_model()

    X_train, X_test, y_train, y_test = loadData()
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f'Accuracy: {accuracy:.4f}')

    report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
    logging.info(
        "Classification Report:\n"
        + classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    )

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate()
