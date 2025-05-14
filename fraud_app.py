import streamlit as st
import torch
import numpy as np
from train import FraudDetector  # assumes model definition is in train.py

"""
This Streamlit web application allows users to input transaction data and predicts whether
the transaction is fraudulent using a pre-trained PyTorch model.

Modules used:
- streamlit: Web app UI
- torch: Model loading and inference
- numpy: Numerical input processing
"""

st.title("💳 Fraud Detection Demo")

st.markdown("""
This demo allows you to simulate a credit card transaction using 10 features and predict whether it might be fraudulent.
""")

st.markdown("""
Each of the 10 features below is represented as a slider ranging from -5.0 to 5.0.

Why this range? During training, all input features were standardized using **z-score normalization** (via `StandardScaler`), which transforms data to have:
- A **mean of 0**
- A **standard deviation of 1**

This means most real-world values fall between **-3 and 3** after scaling. The wider **-5 to 5** range lets you experiment with edge cases or extreme transaction behaviors while keeping inputs consistent with what the model was trained on.
""")

# Use real-like feature names (10 only)
FEATURE_NAMES = [
    "Transaction Amount",
    "Transaction Time",
    "Cardholder Age",
    "Merchant Category",
    "Card Type",
    "Account Tenure",
    "Transaction Frequency",
    "Geo Distance",
    "Device Score",
    "Historical Fraud Rate"
]

inputs = []
for name in FEATURE_NAMES:
    val = st.slider(f"{name}", -5.0, 5.0, 0.0, step=0.1)
    inputs.append(val)

# Load model (still trained on 20 features, so we need to adapt model or retrain with 10)
model = FraudDetector(input_dim=10)
model.load_state_dict(torch.load("fraud_model.pth", map_location=torch.device('cpu')))
model.eval()

# Run prediction
if st.button("Predict Fraud"):
    with torch.no_grad():
        x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        prob = model(x).item()
        prediction = "FRAUDULENT" if prob > 0.5 else "LEGITIMATE"
        st.write(f"🔍 Prediction: **{prediction}**")
        st.progress(prob)
        st.write(f"Probability of fraud: {prob:.2%}")