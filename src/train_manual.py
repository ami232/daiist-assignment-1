"""Manual PyTorch logistic regression. OWNER WRITES THIS FILE.

Contract:
    fit_manual(X: np.ndarray, y: np.ndarray, lr: float, l2: float, epochs: int) -> dict
        returns {"w": torch.Tensor[n_features], "b": torch.Tensor[1], "loss_curve": list[float]}
    predict_manual(params: dict, X: np.ndarray) -> np.ndarray   # probabilities in [0, 1]

Rules of the game (from the Session 5 notebook):
  - no torch.nn layers, no torch.optim; only raw tensors, autograd and a hand-written update
  - w, b are tensors with requires_grad=True
  - loss = mean binary cross-entropy on the logits + l2 * (w**2).sum()
  - loss.backward(); then inside torch.no_grad(): w -= lr * w.grad; b -= lr * b.grad
  - zero the grads every epoch
  - full batch, record loss every epoch so train.py can save the curve

The model: p(dead | x) = sigmoid(x . w + b). The quantity z = x . w + b is the "logit";
sigmoid squashes it into (0, 1). Training finds w, b that make p high on dead rows (y = 1) and
low on live rows (y = 0) by minimising cross-entropy, which is -log of the probability the
model assigned to the correct class, averaged over the rows.
"""
import numpy as np
import torch
import torch.nn.functional as F


def fit_manual(X: np.ndarray, y: np.ndarray, lr: float = 0.1, l2: float = 0.0, epochs: int = 2000) -> dict:
    # float64 to match sklearn's precision, so the three models can be compared coefficient
    # by coefficient. These tensors are plain data — they do NOT need gradients.
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)  # 0.0 / 1.0 labels, must be float for BCE

    # The parameters we DO learn. Start at zero: with w = 0, b = 0 every logit is 0 and every
    # probability is 0.5, a neutral starting point. requires_grad=True makes autograd track them.
    w = torch.zeros(X_t.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    loss_curve: list[float] = []
    for _ in range(epochs):
        # Forward pass, step 1: the logits z = X w + b, one per row. NOT yet probabilities.
        logits = X_t @ w + b

        # Forward pass, step 2: mean binary cross-entropy between logits and labels.
        # Mathematically this is  mean( -y*log(sigmoid(z)) - (1-y)*log(1-sigmoid(z)) ).
        # We do NOT write it that way. For a large negative z, sigmoid(z) underflows to exactly
        # 0.0 in floating point and log(0) = -inf; for a large positive z, 1 - sigmoid(z) does
        # the same. One such row poisons the whole mean with inf/NaN and training dies.
        # binary_cross_entropy_with_logits computes the SAME quantity from z directly using the
        # identity  -log(sigmoid(z)) = log(1 + exp(-z)) = max(z,0) - z*y + log(1 + exp(-|z|)),
        # which never takes a log of 0 and never exponentiates a large positive number.
        bce = F.binary_cross_entropy_with_logits(logits, y_t)

        # Add the L2 penalty on w (not b): large weights are taxed, which keeps the model from
        # chasing training-set coincidences. l2 is matched to sklearn's C in train.py.
        loss = bce + l2 * (w ** 2).sum()

        # Backward pass: autograd walks the graph it recorded during the forward pass and
        # fills w.grad and b.grad with d(loss)/d(w) and d(loss)/d(b).
        loss.backward()

        # Manual gradient-descent step: move each parameter a little AGAINST its gradient.
        # no_grad() so this update isn't itself recorded as part of the next graph.
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

        # Grads accumulate by default (backward ADDS to .grad), so clear them before the next epoch.
        w.grad.zero_()
        b.grad.zero_()

        loss_curve.append(float(loss.item()))

    # Detach so the saved params are pure data (no autograd graph hanging off them).
    return {"w": w.detach(), "b": b.detach(), "loss_curve": loss_curve}


def predict_manual(params: dict, X: np.ndarray) -> np.ndarray:
    """Probability of dead for each row: sigmoid of the logit. At prediction time the
    overflow concern above does not apply: torch.sigmoid is itself computed stably, and a
    probability of exactly 0.0 or 1.0 is a fine answer here (we are not taking its log)."""
    X_t = torch.tensor(X, dtype=torch.float64)
    with torch.no_grad():
        return torch.sigmoid(X_t @ params["w"] + params["b"]).numpy()
