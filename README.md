# 🎓 Student Career Path Prediction using ANN

## 📌 Overview

This project predicts students' **future career paths** using an **Artificial Neural Network (ANN)** based on academic performance and skills.

---

## 🚀 Features

* Data preprocessing (missing value handling)
* Feature encoding & scaling
* Artificial Neural Network (ANN)
* Dropout & L2 regularization
* Class imbalance handling
* Performance evaluation:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Visualizations:

  * Career distribution
  * Correlation heatmap
  * Confusion matrix
  * Training accuracy & loss graphs

---

## 🧠 Model Architecture

* Dense (128 neurons) + Dropout
* Dense (64 neurons) + Dropout
* Dense (32 neurons)
* Output layer (Softmax)

---

## 📂 Project Structure

```
Student-Career-Path-ANN/
│
├── data/
├── src/
├── outputs/
├── model/
├── README.md
├── requirements.txt
```

---

## ⚙️ Installation & Run

```bash
pip install -r requirements.txt
python src/career_prediction_ann.py
```

---

## 📊 Results

### 🔹 Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)

### 🔹 Training Graph

![Accuracy Plot](outputs/accuracy_plot.png)

---

## 💾 Model

Trained model saved as:

```
model/career_model.keras
```

---

## 🔥 Future Improvements

* Hyperparameter tuning
* Try other ML models (Random Forest, XGBoost)
* Deploy as a web app (Streamlit / Flask)

---

## 👨‍💻 Author
Devyanshu
