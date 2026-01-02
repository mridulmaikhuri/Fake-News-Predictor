# 📰 Fake News Detection App (Explainable AI)

An end-to-end **Fake News Detection** web application built using **Machine Learning**, **Streamlit**, and **SHAP**.  
The app classifies news articles as **Fake** or **Real**, provides **confidence scores**, and explains predictions using **model interpretability (XAI)** techniques.

---

## 🚀 Features

- ✅ Classifies news as **Fake / Real**
- 📊 Displays **prediction confidence**
- 🔍 **Explainable AI (SHAP)** – shows which words influenced the decision
- ✍️ Supports **manual text input**
- 🔗 Supports **URL-based article parsing**
- ⚡ Cached model & background data for fast inference
- 🌐 Ready for deployment (Streamlit Cloud / Docker)

---

## 🧠 Model & Approach

- **Text Representation:** TF-IDF Vectorization  
- **Classifier:** Linear model (Logistic Regression / Linear SVM)  
- **Explainability:** SHAP (LinearExplainer)  
- **Pipeline:** `sklearn.pipeline.Pipeline`

The same pipeline is used for:
- Training
- Inference
- Explainability

This ensures **consistency and reproducibility**.

---

## 📁 Project Structure

