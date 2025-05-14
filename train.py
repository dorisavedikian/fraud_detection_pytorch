"""
This script trains a deep learning model using PyTorch to detect fraudulent transactions.
It performs data loading, preprocessing, model training, evaluation, and saving the trained model
for later use in deployment.
"""

import torch                                                    # Neural network modeling and training
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic binary classification data with class imbalance
X, y = make_classification(n_samples=1000, n_features=20, weights=[0.95], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

class FraudDetector(nn.Module):
    """
    A simple feedforward neural network for binary fraud detection.

    Args:
        input_dim (int): The number of input features.
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
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, 1)
        """
        return self.net(x)

# Initialize model, loss function, and optimizer
model = FraudDetector(input_dim=20)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()                    # Clear previous gradients
    output = model(X_train_tensor)           # Forward pass
    loss = criterion(output, y_train_tensor) # Compute binary cross-entropy loss
    loss.backward()                          # Backpropagation
    optimizer.step()                         # Update model weights
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save the trained model to file
torch.save(model.state_dict(), "fraud_model.pth")