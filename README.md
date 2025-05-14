# 💳 Fraud Detection with PyTorch

This project demonstrates a simple deep learning-based binary classification model to detect fraudulent credit card transactions using PyTorch and Streamlit.

---

## 🚀 Project Highlights

- 🧠 Model: Fully connected neural network using PyTorch
- ⚖️ Class imbalance handled through proper weighting
- 📊 Evaluation: ROC-AUC, confusion matrix
- 🧪 Testing with synthetic or real transaction data
- 🌐 Streamlit dashboard for interactive fraud prediction

---

## 📁 Project Structure

```
.
├── train.py              # Train the fraud detection model
├── fraud_app.py          # Streamlit app to simulate predictions
├── EVALUATION.md         # Metrics and how to interpret them
├── requirements.txt      # All necessary Python packages
└── README.md             # You're here
```

---

## 📥 Dataset Download

This project uses the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### 🔄 Steps to Use the Real Dataset:

1. Create a Kaggle account and download the file `creditcard.csv`
   → [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

2. Place `creditcard.csv` into your project root directory (same folder as `train.py`)

---

## ⚙️ Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/fraud_detection_pytorch.git
cd fraud_detection_pytorch
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🏋️‍♀️ Model Training

Run the following to train the model:
```bash
python train.py
```

Make sure the last line of `train.py` saves the model:
```python
torch.save(model.state_dict(), "fraud_model.pth")
```

---

## 🧾 Synthetic Data Overview

The `train.py` script generates synthetic binary classification data designed to simulate real-world fraud detection scenarios, where the dataset is highly imbalanced—i.e., one class (non-fraudulent transactions) significantly outnumbers the other (fraudulent transactions). Using `sklearn.datasets.make_classification`, it creates a dataset with two classes and a strong class imbalance (e.g., 95% legitimate, 5% fraudulent). This reflects the typical challenges in fraud detection, where the minority class is the most critical to detect but also the hardest to model due to limited examples.

The generated data includes:

- 🧮 **1,000 samples** with **10 features** per sample  
- 🎯 A **binary target variable** (`y`) where `0` = legitimate, `1` = fraudulent  
- ⚖️ A **class weight distribution** of `[0.95, 0.05]` to simulate imbalance  
- 🧼 **Standardized features** using `StandardScaler` for stable neural network training  

The model is then trained on this data using a simple feedforward neural network built with PyTorch, and the trained model is saved for later inference.

---

## 📊 Evaluation

After training, evaluate the model using:
- ROC-AUC
- Confusion Matrix
- Precision, Recall, F1-Score

```bash
python evaluate.py
```

---

## 🌐 Streamlit Dashboard

Launch the interactive prediction dashboard:
```bash
python -m streamlit run fraud_app.py
```

- Input synthetic transaction features
- Get real-time fraud prediction and probability
- Model must be trained first (`fraud_model.pth` is required)

---

## 👤 Author

Doris Avedikian  
GitHub: [@dorisavedikian](https://github.com/dorisavedikian)

