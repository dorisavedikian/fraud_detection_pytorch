"""
📊 Evaluation Metrics for Fraud Detection

After training the model, it should be evaluated using key classification metrics that are especially
relevant for imbalanced datasets like fraud detection.

Metrics:

✅ Confusion Matrix:
- True Positives (TP): correctly predicted fraudulent transactions
- False Positives (FP): non-fraud transactions incorrectly classified as fraud
- True Negatives (TN): correctly predicted non-fraudulent transactions
- False Negatives (FN): fraudulent transactions missed by the model

✅ Best Practices:
- Use ROC-AUC (Receiver Operating Characteristic - Area Under Curve) to evaluate performance on imbalanced data
- Monitor Precision, Recall, and F1-score to understand trade-offs between catching fraud and avoiding false alarms
"""

import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from train import FraudDetector

# Recreate test data to match training setup
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.95], random_state=42)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

# Load trained model
model = FraudDetector(input_dim=10)
model.load_state_dict(torch.load("fraud_model.pth"))
model.eval()

# Inference
with torch.no_grad():
    y_pred_prob = model(X_test_tensor).numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluation
cm = confusion_matrix(y_test_tensor.numpy(), y_pred)
roc_auc = roc_auc_score(y_test_tensor.numpy(), y_pred_prob)

print("Confusion Matrix:\n", cm)
print("ROC-AUC:", roc_auc)