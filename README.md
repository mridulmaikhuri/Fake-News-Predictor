# 📰 Fake News Predictor

A machine learning–based application that classifies news articles as **Fake** or **Real** using natural language processing (NLP) techniques. The project includes data preprocessing, model training, and a simple web interface for predictions.

---

## 🚀 Features

* Text preprocessing (cleaning, normalization, tokenization)
* TF-IDF–based feature extraction
* Supervised ML classifier (pipeline-based)
* Trained model saved and loaded using `joblib`
* Interactive web app built with **Streamlit**
* Simple and modular project structure

---

## 🧠 Tech Stack

* **Python 3.9+**
* **Scikit-learn**
* **Pandas / NumPy**
* **NLTK / regex (for preprocessing)**
* **Streamlit** (for web app)
* **Joblib** (model persistence)

---

## 📂 Project Structure

```
fake-news-predictor/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   └── pipeline.pkl
│
├── scripts/
│   ├── preprocess.py
│   └── train.py
│
├── app/
│   └── app.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/fake-news-predictor.git
cd fake-news-predictor
```

### 2️⃣ Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\\Scripts\\activate  # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Model Training (Optional)

If you want to retrain the model:

```bash
python scripts/train.py
```

This will:

* Load and merge Fake & True datasets
* Preprocess the text
* Train the ML pipeline
* Save the trained model to `models/pipeline.pkl`

---

## 🌐 Run the Web App

```bash
streamlit run app/app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 🧪 Example Prediction

Input:

```
Breaking: Scientists confirm water found on Mars!
```

Output:

```
Prediction: Real News
```

---

## 📊 Dataset

* **Fake.csv** – Fake news articles
* **True.csv** – Real news articles

Each dataset contains:

* `title`
* `text`
* `subject`
* `date`

---

## 🔍 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

(Evaluation performed during training stage)

---

## 🛡️ Limitations

* Model performance depends heavily on dataset quality
* Cannot verify real-time news authenticity
* Susceptible to adversarial or satirical content

---

## 👤 Author

**Mridul Maikhuri**
Feel free to connect and contribute!
