"""
Assignment 1 — training pipeline.

You're using a script for this stage. If you'd rather use a notebook,
write train.ipynb instead and delete this file — main.py refuses to run if
it finds both train.py and train.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via `uv run python main.py train`, this file must, start to finish, with no
manual steps in between:

- Load your committed dataset.
- Apply the feature engineering, preprocessing, and train/val/test split
  consistent with the business framing you wrote in REPORT.md.
- Train the same model — linear regression for a regression target,
  logistic regression for a binary one — three ways, on the same split:
  scikit-learn, a from-scratch PyTorch loop (Session 5's manual approach:
  raw tensors, autograd, a manual gradient-step update), and the standard
  torch.nn.Module + torch.optim workflow.
- Evaluate all three against each other and against a naive baseline.
- Save whatever app.py needs to build its dashboard without retraining
  anything.

How you structure the code beyond that — file layout, function
boundaries, naming — is your call to make and be able to defend.
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from data_prep import clean, load_raw
from features import add_features, get_X_y
from preprocessing import build_preprocessor, time_split

MODELS_DIR = Path("models")
RANDOM_STATE = 42

# Regularization strengths to try for sklearn (C = 1 / strength):
# small C = strong regularization (simpler model), big C = weak regularization
C_VALUES = [0.01, 0.1, 1.0, 10.0]

# PyTorch settings
LEARNING_RATES = [0.1, 0.5, 1.0]  # tried on validation, best one kept
EPOCHS = 2000                      # full passes over the training data


def evaluate(name, y_true, prob, threshold=0.5):
    """Print and return the main metrics for one model."""
    pred = (prob >= threshold).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, prob) if len(np.unique(prob)) > 1 else 0.5,
        "log_loss": log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7)),
    }
    print(f"  {name:28s} accuracy {metrics['accuracy']:.3f} | "
          f"ROC-AUC {metrics['roc_auc']:.3f} | log loss {metrics['log_loss']:.3f}")
    return metrics


# ===================================================================
# Manual PyTorch: we write the model, the loss and the update ourselves
# ===================================================================
def train_manual_torch(Xtr, ytr, C, lr, epochs=EPOCHS):
    """Logistic regression with plain gradient descent, written by hand."""
    X = torch.tensor(Xtr, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    n, d = X.shape

    # 1. Parameters: one weight per column + one bias, all starting at 0.
    #    requires_grad=True -> PyTorch tracks them to compute gradients.
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # Same L2 penalty as sklearn's C, rescaled because we use the MEAN loss
    l2 = 1.0 / (C * n)

    losses = []
    for epoch in range(epochs):
        # 2. Forward pass: score -> probability
        z = X @ w + b              # weighted sum for every booking
        p = torch.sigmoid(z)       # squash into a probability 0..1

        # 3. Loss: binary cross-entropy (= log loss) + L2 penalty
        eps = 1e-7                 # avoids log(0)
        bce = -(y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).mean()
        loss = bce + 0.5 * l2 * (w ** 2).sum()

        # 4. Backward pass: PyTorch computes d(loss)/dw and d(loss)/db
        loss.backward()

        # 5. Gradient descent step: move a little AGAINST the gradient
        with torch.no_grad():      # this update itself must not be tracked
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()         # reset, otherwise gradients add up
            b.grad.zero_()

        losses.append(loss.item())

    return w.detach(), b.detach(), losses


def predict_manual(w, b, X):
    """Probability of cancellation from the hand-made model."""
    with torch.no_grad():
        z = torch.tensor(X, dtype=torch.float32) @ w + b
        return torch.sigmoid(z).numpy()


# ===================================================================
# Standard PyTorch: nn.Module + loss function + optimizer
# ===================================================================
class LogisticRegressionNet(nn.Module):
    """The same model, written the standard PyTorch way."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)   # weights + bias in one layer
        nn.init.zeros_(self.linear.weight)       # start at 0, like the manual
        nn.init.zeros_(self.linear.bias)         # version -> fair comparison

    def forward(self, x):
        # Returns the raw score z (a "logit"); sigmoid is applied in the loss
        return self.linear(x).squeeze(1)


def train_standard_torch(Xtr, ytr, C, lr, epochs=EPOCHS):
    """Same logistic regression, using PyTorch's ready-made building blocks."""
    X = torch.tensor(Xtr, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    n, d = X.shape

    model = LogisticRegressionNet(d)
    loss_fn = nn.BCEWithLogitsLoss()                        # sigmoid + log loss
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # plain gradient descent
    l2 = 1.0 / (C * n)                                      # same brake as sklearn

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()                  # 1. reset old gradients
        z = model(X)                           # 2. forward pass
        loss = loss_fn(z, y) + 0.5 * l2 * (model.linear.weight ** 2).sum()  # 3. loss
        loss.backward()                        # 4. backward pass
        optimizer.step()                       # 5. update the weights
        losses.append(loss.item())

    return model, losses


def predict_standard(model, X):
    """Probability of cancellation from the nn.Module model."""
    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(z).numpy()


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    # ---------- 1. Data: clean -> features -> time split ----------
    data = add_features(clean(load_raw(), verbose=False))
    train, val, test = time_split(data)
    X_train, y_train = get_X_y(train)
    X_val, y_val = get_X_y(val)
    X_test, y_test = get_X_y(test)

    # ---------- 2. Preprocessing: learn on train only ----------
    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_train)
    Xva = pre.transform(X_val)
    Xte = pre.transform(X_test)
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")
    print(f"Train {Xtr.shape} | Val {Xva.shape} | Test {Xte.shape}")

    # ---------- 3. Naive baseline ----------
    # "Always predict the training cancellation rate" -> predicts 0 (no cancel)
    base_rate = y_train.mean()
    print(f"\nNaive baseline: always predict P(cancel) = {base_rate:.3f}")

    # ---------- 4. sklearn: choose C on VALIDATION ----------
    print("\nChoosing regularization strength C on validation:")
    best_C, best_loss = None, np.inf
    for C in C_VALUES:
        model = LogisticRegression(C=C, max_iter=2000)
        model.fit(Xtr, y_train)
        loss = log_loss(y_val, model.predict_proba(Xva)[:, 1])
        print(f"  C = {C:<6} validation log loss = {loss:.4f}")
        if loss < best_loss:
            best_C, best_loss = C, loss
    print(f"  -> best C = {best_C}")

    sk_model = LogisticRegression(C=best_C, max_iter=2000)
    sk_model.fit(Xtr, y_train)
    joblib.dump(sk_model, MODELS_DIR / "sklearn_logreg.joblib")

    # ---------- 5. Manual PyTorch: choose learning rate on VALIDATION ----------
    torch.manual_seed(RANDOM_STATE)
    ytr = y_train.to_numpy(dtype=np.float32)
    print(f"\nManual PyTorch ({EPOCHS} epochs, same C = {best_C}): choosing learning rate")
    best_lr, best_loss = None, np.inf
    for lr in LEARNING_RATES:
        w, b, _ = train_manual_torch(Xtr, ytr, C=best_C, lr=lr)
        loss = log_loss(y_val, predict_manual(w, b, Xva))
        print(f"  lr = {lr:<4} validation log loss = {loss:.4f}")
        if loss < best_loss:
            best_lr, best_loss = lr, loss
    print(f"  -> best learning rate = {best_lr}")

    w_manual, b_manual, manual_losses = train_manual_torch(Xtr, ytr, C=best_C, lr=best_lr)
    torch.save({"w": w_manual, "b": b_manual, "lr": best_lr, "C": best_C,
                "losses": manual_losses}, MODELS_DIR / "torch_manual.pt")
    print(f"  training loss: start {manual_losses[0]:.4f} -> end {manual_losses[-1]:.4f}")

    # ---------- 6. Standard PyTorch: same C, same learning rate, same epochs ----------
    print(f"\nStandard PyTorch (nn.Module + SGD, lr = {best_lr}, {EPOCHS} epochs)")
    std_model, std_losses = train_standard_torch(Xtr, ytr, C=best_C, lr=best_lr)
    torch.save({"state_dict": std_model.state_dict(), "n_features": Xtr.shape[1],
                "lr": best_lr, "C": best_C, "losses": std_losses},
               MODELS_DIR / "torch_standard.pt")
    print(f"  training loss: start {std_losses[0]:.4f} -> end {std_losses[-1]:.4f}")

    # ---------- 7. Results on VALIDATION (test is kept for the end) ----------
    p_sk = sk_model.predict_proba(Xva)[:, 1]
    p_manual = predict_manual(w_manual, b_manual, Xva)
    p_std = predict_standard(std_model, Xva)
    print("\nValidation results (threshold 0.5):")
    evaluate("Naive baseline", y_val, np.full(len(y_val), base_rate))
    evaluate("sklearn LogisticRegression", y_val, p_sk)
    evaluate("Manual PyTorch", y_val, p_manual)
    evaluate("Standard PyTorch", y_val, p_std)

    # Do the three versions agree?
    print("\nAgreement between the three versions (validation):")
    for name, a, b_ in [("sklearn vs manual", p_sk, p_manual),
                        ("sklearn vs standard", p_sk, p_std),
                        ("manual vs standard", p_manual, p_std)]:
        print(f"  {name:20s} avg prob difference {np.abs(a - b_).mean():.4f} | "
              f"same decision {((a >= 0.5) == (b_ >= 0.5)).mean():.1%}")

    # ---------- 8. Which features matter most? ----------
    names = pre.get_feature_names_out()
    coefs = sk_model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1][:10]
    print("\nTop 10 features by size of coefficient (+ = more likely to cancel):")
    for i in order:
        print(f"  {names[i]:40s} {coefs[i]:+.3f}")

    # Where do the two sets of weights differ most? (and how many examples?)
    diff = np.abs(coefs - w_manual.numpy())
    print("\nBiggest weight differences sklearn vs manual (rows with a 1 in train):")
    for i in np.argsort(diff)[::-1][:5]:
        n_ones = int((Xtr[:, i] == 1).sum())
        print(f"  {names[i]:40s} sklearn {coefs[i]:+.3f} | manual {w_manual[i]:+.3f} "
              f"| {n_ones} training rows")

    # Manual vs standard should be (almost) identical: same algorithm
    w_std = std_model.linear.weight.detach().numpy()[0]
    print(f"\nLargest weight difference manual vs standard: "
          f"{np.abs(w_manual.numpy() - w_std).max():.6f}")


if __name__ == "__main__":
    main()
    
    from evaluate import run_evaluation
    run_evaluation()