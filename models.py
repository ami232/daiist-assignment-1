"""PyTorch model definition, shared by train.py and app.py.

The app needs this class to rebuild the model before loading its saved weights.
"""
from torch import nn


class LinearRegressionModel(nn.Module):
    """Linear regression packaged as a standard PyTorch module: y_hat = X @ w + b."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)  # holds w (weight) and b (bias)

    def forward(self, x):
        return self.linear(x)
