"""
Training pipeline (run with: uv run python main.py train).

1. Load data, engineer features, time-based split.
2. Tune the regularization strength and the decision threshold on a
   time-based validation slice (last 6 months of the training years).
3. Train the SAME L2-regularized logistic regression three ways:
   scikit-learn, a manual PyTorch loop, and nn.Module + torch.optim.
4. Compare them on the 2015 test set against a naive baseline.
5. Save everything app.py needs to artifacts/.
"""
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch import nn

from pipeline import (
    ARTIFACTS, LOSS_IF_DEFAULT, PROFIT_IF_REPAID, TEST_YEAR, VALIDATION_START,
    LogisticRegressionTorch, business_cost, engineer, load_data,
    make_preprocessor, time_split,
)

SEED = 42
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1]  # L2 strength candidates (only thing tuned)
LR = 0.5                             # learning rate for both PyTorch versions
EPOCHS = 30_000                      # full-batch gradient descent steps
# Why so many: the one-hot columns of each categorical always sum to 1, just
# like the intercept, so weight can shift between them while barely changing
# predictions. Gradient descent crawls along that flat direction: at 3,000
# epochs predictions differed from sklearn by up to 0.0066 (intercept off by
# 0.16); at 30,000 they agree to ~1e-6. See artifacts/loss_history.csv.
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

torch.manual_seed(SEED)
np.random.seed(SEED)
ARTIFACTS.mkdir(exist_ok=True)

# Objective shared by all three implementations:
#   mean binary cross-entropy + (lambda / 2) * ||w||^2   (intercept not penalized)
# scikit-learn writes the same objective as C * sum(loss) + 0.5 * ||w||^2,
# so dividing by C * n gives C = 1 / (lambda * n).


def fit_sklearn(X, y, lam):
    model = LogisticRegression(C=1 / (lam * len(y)), max_iter=10_000, tol=1e-10)
    return model.fit(X, y)


def fit_manual_torch(X, y, lam):
    """Session 5 style: raw tensors, hand-written loss, manual update step."""
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    history = []
    for _ in range(EPOCHS):
        p = torch.sigmoid(X @ w + b).clamp(1e-12, 1 - 1e-12)
        bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
        loss = bce + lam / 2 * (w ** 2).sum()
        loss.backward()                  # autograd computes the gradients
        with torch.no_grad():            # manual gradient-descent step
            w -= LR * w.grad
            b -= LR * b.grad
            w.grad.zero_()
            b.grad.zero_()
        history.append(loss.item())
    return w.detach(), b.detach(), history


def fit_module_torch(X, y, lam):
    """Standard workflow: nn.Module + loss object + torch.optim optimizer."""
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    model = LogisticRegressionTorch(X.shape[1])
    nn.init.zeros_(model.linear.weight)
    nn.init.zeros_(model.linear.bias)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    history = []
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y) + lam / 2 * (model.linear.weight ** 2).sum()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return model, history


def evaluate(y, p, loan_amnt, threshold):
    return {
        "roc_auc": roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5,
        "log_loss": log_loss(y, p),
        "accuracy_at_0.5": accuracy_score(y, p >= 0.5),
        "cost_usd": business_cost(y, p, loan_amnt, threshold),
        "rejection_rate": float((p >= threshold).mean()),
    }


# 1) Data -------------------------------------------------------------------
df = load_data()
train_df, test_df = time_split(df)
y_train = train_df["Default"].to_numpy()
y_test = test_df["Default"].to_numpy()
print(f"Train: {len(train_df)} loans (2012-2014), default rate {y_train.mean():.3f}")
print(f"Test:  {len(test_df)} loans ({TEST_YEAR}), default rate {y_test.mean():.3f}")

# 2) Tuning on a time-based validation slice --------------------------------
is_val = train_df["issue_d"] >= VALIDATION_START
fit_part, val_part = train_df[~is_val], train_df[is_val]
pre_val = make_preprocessor().fit(engineer(fit_part))
X_fit = pre_val.transform(engineer(fit_part))
X_val = pre_val.transform(engineer(val_part))
y_fit, y_val = fit_part["Default"].to_numpy(), val_part["Default"].to_numpy()

val_losses = {}
for lam in LAMBDAS:
    p_val = fit_sklearn(X_fit, y_fit, lam).predict_proba(X_val)[:, 1]
    val_losses[lam] = log_loss(y_val, p_val)
    print(f"  lambda={lam:g}: validation log loss {val_losses[lam]:.4f}")
best_lambda = min(val_losses, key=val_losses.get)

p_val = fit_sklearn(X_fit, y_fit, best_lambda).predict_proba(X_val)[:, 1]
val_costs = [business_cost(y_val, p_val, val_part["loan_amnt"], t) for t in THRESHOLDS]
best_threshold = float(THRESHOLDS[int(np.argmin(val_costs))])
print(f"Chosen lambda: {best_lambda:g} | cost-minimizing threshold: {best_threshold:.2f}")

# 3) Final fit on all training years ----------------------------------------
preprocessor = make_preprocessor().fit(engineer(train_df))
X_train = preprocessor.transform(engineer(train_df))
X_test = preprocessor.transform(engineer(test_df))
feature_names = preprocessor.get_feature_names_out()

sk_model = fit_sklearn(X_train, y_train, best_lambda)
w_manual, b_manual, hist_manual = fit_manual_torch(X_train, y_train, best_lambda)
torch_model, hist_module = fit_module_torch(X_train, y_train, best_lambda)

X_test_t = torch.tensor(X_test, dtype=torch.float64)
with torch.no_grad():
    probs = {
        "sklearn": sk_model.predict_proba(X_test)[:, 1],
        "torch_manual": torch.sigmoid(X_test_t @ w_manual + b_manual).numpy(),
        "torch_module": torch.sigmoid(torch_model(X_test_t)).numpy(),
    }

# 4) Comparison ---------------------------------------------------------------
loan_test = test_df["loan_amnt"].to_numpy()
baseline_p = np.full(len(y_test), y_train.mean())  # same risk for everyone
results = {"baseline_approve_all": evaluate(y_test, baseline_p, loan_test, threshold=1.01)}
for name, p in probs.items():
    results[name] = evaluate(y_test, p, loan_test, best_threshold)

coefs = pd.DataFrame({
    "feature": list(feature_names) + ["intercept"],
    "sklearn": np.append(sk_model.coef_[0], sk_model.intercept_[0]),
    "torch_manual": np.append(w_manual.numpy(), b_manual.numpy()),
    "torch_module": np.append(
        torch_model.linear.weight.detach().numpy()[0],
        torch_model.linear.bias.detach().numpy(),
    ),
})
agreement = {
    "max_abs_prob_diff_manual_vs_sklearn": float(np.abs(probs["torch_manual"] - probs["sklearn"]).max()),
    "max_abs_prob_diff_module_vs_sklearn": float(np.abs(probs["torch_module"] - probs["sklearn"]).max()),
    "max_abs_prob_diff_manual_vs_module": float(np.abs(probs["torch_manual"] - probs["torch_module"]).max()),
    "max_abs_coef_diff_manual_vs_sklearn": float((coefs["torch_manual"] - coefs["sklearn"]).abs().max()),
}

table = pd.DataFrame(results).T
print("\nTest set (2015) results:")
print(table.round(4).to_string())
print("\nAgreement between implementations:")
for k, v in agreement.items():
    print(f"  {k}: {v:.2e}")

# 5) Save artifacts (app.py only loads these, never retrains) ----------------
joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
joblib.dump(sk_model, ARTIFACTS / "sklearn_model.joblib")
torch.save({"w": w_manual, "b": b_manual}, ARTIFACTS / "torch_manual.pt")
torch.save(torch_model.state_dict(), ARTIFACTS / "torch_module.pt")
coefs.to_csv(ARTIFACTS / "coefficients.csv", index=False)
pd.DataFrame({"torch_manual": hist_manual, "torch_module": hist_module}).to_csv(
    ARTIFACTS / "loss_history.csv", index_label="epoch"
)
with open(ARTIFACTS / "metrics.json", "w") as f:
    json.dump({
        "best_lambda": best_lambda,
        "validation_log_loss": {str(k): v for k, v in val_losses.items()},
        "best_threshold": best_threshold,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "loss_if_default": LOSS_IF_DEFAULT,
        "profit_if_repaid": PROFIT_IF_REPAID,
        "test_results": results,
        "agreement": agreement,
    }, f, indent=2)
print(f"\nSaved artifacts to {ARTIFACTS}/")
