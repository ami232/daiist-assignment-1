import torch
from torch import nn


class LogisticRegressionModel(nn.Module):
    """Logistic regression as a PyTorch module: one linear layer.
    It outputs logits (raw scores); sigmoid turns them into probabilities."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)
