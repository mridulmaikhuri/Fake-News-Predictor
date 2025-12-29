import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/'data'

def loadData():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    df1 = pd.read_csv(DATA_DIR/'Fake.csv')
    df2 = pd.read_csv(DATA_DIR/'True.csv')
    df1['label'] = 0
    df2['label'] = 1

    print ("Both fake and real news datasets loaded successfully.")
    print (f"Fake news dataset shape: {df1.shape}")
    print (f"Real news dataset shape: {df2.shape}")

    df = pd.concat([df1, df2], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['content'] = df['title'] + ' ' + df['text']
    df = df[['content', 'label']]

    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['label'], test_size=0.2, random_state=42
    )   

    print ("Data split into training and testing sets.")
    print (f"Training set size: {X_train.shape[0]}")
    print (f"Testing set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    loadData()