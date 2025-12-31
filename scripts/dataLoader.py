import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/'data'
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

def loadData():
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Fake.csv or True.csv not found in the data directory."
        )
    
    logging.info("Loading datasets...")
    df1 = pd.read_csv(fake_path)
    df2 = pd.read_csv(true_path)
    df1['label'] = 0
    df2['label'] = 1
    logging.info(
        "Loaded Fake (%d rows) and Real (%d rows) datasets",
        len(df1),
        len(df2),
    )

    df = pd.concat([df1, df2], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['content'] = df['title'] + ' ' + df['text']
    df = df[['content', 'label']]

    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['label'], test_size=0.2, random_state=42
    )   

    logging.info(
        "Data split completed | Train: %d | Test: %d",
        len(X_train),
        len(X_test),
    )

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    loadData()