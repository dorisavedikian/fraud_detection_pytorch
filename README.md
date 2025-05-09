# Fraud Detection with PyTorch

This project demonstrates a simple binary classification model to detect fraudulent transactions using PyTorch. It includes:
- Data preprocessing
- Handling class imbalance
- Model training and evaluation
- (Optional) Model explainability with SHAP

To run: `python train.py`

## 📥 Real Dataset Integration: Credit Card Fraud (Kaggle)

**Dataset Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 🔄 Steps to Use:
1. Download `creditcard.csv` from the Kaggle link above.
2. Place the file in the project directory.
3. Load and preprocess:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('creditcard.csv')
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])
```
4. Handle class imbalance (optional):
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=[0, 1], y=df['Class'])
```

Use the features as inputs to your neural net, and `Class` as the label.
