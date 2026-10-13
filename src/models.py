"""
Model definitions and training/inference helpers, shared by train.py
(which trains and saves) and app.py (which loads and predicts -- it
must NEVER retrain, only run a forward pass with saved weights).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# 1. scikit-learn
# ---------------------------------------------------------------------------
def train_sklearn(X_train, y_train, C, seed=42):
    model = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    model.fit(X_train, y_train)
    return model


def predict_sklearn(model, X):
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# 2. From-scratch PyTorch loop (manual forward/backward/update, no
#    torch.nn, no torch.optim -- Session 5's manual approach)
# ---------------------------------------------------------------------------
def train_pytorch_manual(X_train, y_train, C, lr, n_epochs, seed=42):
    torch.manual_seed(seed)
    X = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    n_features = X.shape[1]

    w = torch.zeros((n_features, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    weight_decay = 1.0 / (C * len(X))  # matched to sklearn's C for a fair comparison

    for _ in range(n_epochs):
        logits = X @ w + b
        probs = torch.sigmoid(logits)
        eps = 1e-7
        bce = -(y * torch.log(probs + eps) + (1 - y) * torch.log(1 - probs + eps)).mean()
        loss = bce + weight_decay * (w ** 2).sum()

        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

    return {"w": w.detach(), "b": b.detach()}


def predict_pytorch_manual(params, X):
    Xt = torch.tensor(X.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(Xt @ params["w"] + params["b"])
    return probs.numpy().ravel()


# ---------------------------------------------------------------------------
# 3. Standard torch.nn.Module + torch.optim workflow
# ---------------------------------------------------------------------------
class LogisticRegressionModule(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


def train_pytorch_module(X_train, y_train, C, lr, n_epochs, seed=42):
    torch.manual_seed(seed)
    X = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    model = LogisticRegressionModule(X.shape[1])
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1.0 / (C * len(X)))

    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

    return model


def predict_pytorch_module(model, X):
    Xt = torch.tensor(X.to_numpy(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xt))
    return probs.numpy().ravel()


def load_pytorch_module(state_dict_path, n_features):
    model = LogisticRegressionModule(n_features)
    model.load_state_dict(torch.load(state_dict_path, weights_only=True))
    model.eval()
    return model


def load_pytorch_manual(params_path):
    return torch.load(params_path, weights_only=True)