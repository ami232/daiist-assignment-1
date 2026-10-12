"""Shared logistic model definition and inference. No training on import."""
import torch
from torch import nn


class LogisticModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, dtype=torch.float64)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # logits; sigmoid only for inference


def tensor_probabilities(x, weights, bias):
    with torch.no_grad():
        return torch.sigmoid(torch.as_tensor(x, dtype=torch.float64) @ weights + bias).numpy()
