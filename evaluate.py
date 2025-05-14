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
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
from train import FraudDetector, X_test_tensor, y_test_tensor  # or load saved tensors

# Load model
model = FraudDetector(input_dim=20)
model.load_state_dict(torch.load("fraud_model.pth"))
model.eval()

# Predict
y_pred_prob = model(X_test_tensor).detach().numpy()
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluation metrics
cm = confusion_matrix(y_test_tensor.numpy(), y_pred)
roc_auc = roc_auc_score(y_test_tensor.numpy(), y_pred_prob)

print("Confusion Matrix:\n", cm)
print("ROC-AUC:", roc_auc)
