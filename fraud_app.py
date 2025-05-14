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
This demo allows you to simulate a credit card transaction and predict whether it might be fraudulent.
""")

# User inputs for 20 features (as in the synthetic data or Kaggle dataset)
inputs = []
for i in range(20):
    val = st.slider(f"Feature {i+1}", -5.0, 5.0, 0.0, step=0.1)
    inputs.append(val)

# Load model (make sure model is saved as fraud_model.pth)
model = FraudDetector(input_dim=20)
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
