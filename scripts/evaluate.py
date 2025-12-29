import joblib as jb
from pathlib import Path
from dataLoader import loadData
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR/"models"

def evaluate():
    pipeline = jb.load(MODEL_DIR/"pipeline.pkl")

    X_train, X_test, y_train, y_test = loadData()
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
    disp.plot(cmap='Blues')
    plt.show()

if __name__ == '__main__':
    evaluate()
