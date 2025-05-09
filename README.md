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

3. Optional: Preprocess and scale:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('creditcard.csv')
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])
```

4. Optional: Handle class imbalance with weights:
```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=[0, 1], y=df['Class'])
```

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

## 📊 Evaluation

After training, evaluate the model using:
- ROC-AUC
- Confusion Matrix
- Precision, Recall, F1-Score

See `EVALUATION.md` for full examples.

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

## ❗ Common Issues

- `ModuleNotFoundError`: Run `pip install -r requirements.txt`
- `FileNotFoundError: fraud_model.pth`: You need to run `train.py` first
- Port conflict on 8501: Use `streamlit run fraud_app.py --server.port 8502`

---

## 👤 Author

Doris Avedikian  
GitHub: [@dorisavedikian](https://github.com/dorisavedikian)

---

## 🧠 Future Improvements

- Add SHAP for model interpretability
- Deploy Streamlit to Hugging Face Spaces or Render
- Use time-based sequences or LSTM for temporal fraud patterns