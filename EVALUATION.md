
## 📊 Evaluation Metrics (Fraud Detection)

After training your model, evaluate it using these metrics:

### ✅ Confusion Matrix
- True Positives (TP): correctly predicted fraud
- False Positives (FP): wrongly predicted fraud
- True Negatives (TN): correctly predicted non-fraud
- False Negatives (FN): missed fraud cases

```python
from sklearn.metrics import confusion_matrix, roc_auc_score

# Predict on test set
y_pred_prob = model(X_test_tensor).detach().numpy()
y_pred = (y_pred_prob > 0.5).astype(int)

# Confusion matrix
print(confusion_matrix(y_test_tensor.numpy(), y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test_tensor.numpy(), y_pred_prob)
print("ROC-AUC:", roc_auc)
```

### ✅ Best Practices
- Use **ROC-AUC** for imbalanced datasets
- Monitor **Precision, Recall, and F1-score** to evaluate trade-offs
