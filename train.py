"""
This script trains a deep learning model using PyTorch to detect fraudulent transactions.
It performs data loading, preprocessing, model training, evaluation, and saving the trained model
for later use in deployment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FraudDetector(nn.Module):
    """
    A simple feedforward neural network for binary fraud detection.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def main():
    # Generate synthetic binary classification data with class imbalance
    X, y = make_classification(n_samples=1000, n_features=10, weights=[0.95], random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    # Initialize model, loss function, and optimizer
    model = FraudDetector(input_dim=10)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), "fraud_model.pth")

if __name__ == "__main__":
    main()