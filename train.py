"""
Assignment 1 — training pipeline.

Goal: predict the market value of a FIFA 20 outfield player with a linear
regression, trained three ways (scikit-learn, manual PyTorch loop, standard
PyTorch), so a low-budget club can spot players that look cheap for what
they are.

Run with:  uv run python main.py train
Everything the dashboard needs is saved in the models/ folder.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import ALL_FEATURES, BASIC_FEATURES, add_features, clean_data

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "fifa20_players.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED)

# Hyperparameters (the only ones we tune: regularization and learning rate).
ALPHAS_TO_TRY = [0.0, 1.0, 10.0, 100.0, 1000.0]
MANUAL_LR = 0.05
MANUAL_EPOCHS = 5000
ADAM_LR = 0.02
ADAM_EPOCHS = 3000


# ---------------------------------------------------------------------------
# 1. Load, clean and build features
# ---------------------------------------------------------------------------
raw = pd.read_csv(DATA_PATH)
print("rows in the file:", len(raw))

data = clean_data(raw)
data = add_features(data)
print("rows after removing goalkeepers and free agents:", len(data))


# ---------------------------------------------------------------------------
# 2. Split: 70% train, 15% validation, 15% test
# ---------------------------------------------------------------------------
# The data is one single picture of FIFA 20 (no dates), so there is no
# "future" that could leak into training. A random split is fine.
train_df, rest_df = train_test_split(data, test_size=0.30, random_state=SEED)
val_df, test_df = train_test_split(rest_df, test_size=0.50, random_state=SEED)
print("train / val / test rows:", len(train_df), len(val_df), len(test_df))

y_train = train_df["log_value"].values
y_val = val_df["log_value"].values
y_test = test_df["log_value"].values


# ---------------------------------------------------------------------------
# 3. Scale the features
# ---------------------------------------------------------------------------
# Features have very different sizes (age ~25, skills ~60, flags 0/1).
# Gradient descent works badly with that, so we put them all on the same
# scale. The scaler only learns from the training rows.
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[ALL_FEATURES])
X_val = scaler.transform(val_df[ALL_FEATURES])
X_test = scaler.transform(test_df[ALL_FEATURES])


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def evaluate(y_true_log, y_pred_log):
    """Metrics on the log scale and, undoing the log, in euros."""
    euros_true = np.exp(y_true_log)
    euros_pred = np.exp(y_pred_log)
    percent_error = np.abs(euros_pred - euros_true) / euros_true * 100
    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "mae_euros": float(mean_absolute_error(euros_true, euros_pred)),
        "median_percent_error": float(np.median(percent_error)),
    }


# ---------------------------------------------------------------------------
# 5. Did the engineered features help? (checked on validation, never on test)
# ---------------------------------------------------------------------------
feature_sets = {
    "basic features only": BASIC_FEATURES,
    "basic + age_from_25_squared": BASIC_FEATURES + ["age_from_25_squared"],
    "basic + best_skill": BASIC_FEATURES + ["best_skill"],
    "basic + position flags": BASIC_FEATURES + ["is_midfielder", "is_attacker"],
    "basic + is_left_footed": BASIC_FEATURES + ["is_left_footed"],
    "all features": ALL_FEATURES,
}
feature_comparison = {}
print("\nValidation RMSE (log) for different feature sets:")
for name, columns in feature_sets.items():
    small_scaler = StandardScaler()
    Xa = small_scaler.fit_transform(train_df[columns])
    Xb = small_scaler.transform(val_df[columns])
    check_model = Ridge(alpha=0.0).fit(Xa, y_train)
    rmse = evaluate(y_val, check_model.predict(Xb))["rmse_log"]
    feature_comparison[name] = rmse
    print(f"  {name:28s} {rmse:.4f}")


# ---------------------------------------------------------------------------
# 6. Naive baseline: always predict the average of the training set
# ---------------------------------------------------------------------------
baseline_value = float(y_train.mean())
baseline_test_pred = np.full(len(y_test), baseline_value)


# ---------------------------------------------------------------------------
# 7. Model 1: scikit-learn (Ridge = linear regression + L2 penalty)
# ---------------------------------------------------------------------------
# Pick the regularization strength (alpha) with the validation set.
alpha_results = {}
print("\nValidation RMSE (log) for each alpha:")
for alpha in ALPHAS_TO_TRY:
    candidate = Ridge(alpha=alpha).fit(X_train, y_train)
    rmse = evaluate(y_val, candidate.predict(X_val))["rmse_log"]
    alpha_results[alpha] = rmse
    print(f"  alpha={alpha:7.1f}  {rmse:.5f}")

best_alpha = min(alpha_results, key=alpha_results.get)
print("best alpha:", best_alpha)

sklearn_model = Ridge(alpha=best_alpha).fit(X_train, y_train)


# ---------------------------------------------------------------------------
# Tensors for the two PyTorch versions (same data, same split)
# ---------------------------------------------------------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

n_samples, n_features = X_train_t.shape

# scikit-learn's Ridge minimises:  sum of squared errors + alpha * sum(w^2).
# Our PyTorch loss uses the MEAN of squared errors, so to keep it the same
# problem we also divide the penalty by the number of samples.
l2_strength = best_alpha / n_samples


# ---------------------------------------------------------------------------
# 8. Model 2: manual PyTorch loop (Session 5 style)
# ---------------------------------------------------------------------------
w = torch.zeros(n_features, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def forward(X):
    return X @ w + b


def mse_loss(y_pred, y):
    return ((y_pred - y) ** 2).mean()


manual_loss_history = []
for epoch in range(MANUAL_EPOCHS):
    y_pred = forward(X_train_t)
    loss = mse_loss(y_pred, y_train_t) + l2_strength * (w ** 2).sum()

    loss.backward()

    with torch.no_grad():
        w -= MANUAL_LR * w.grad
        b -= MANUAL_LR * b.grad

    w.grad.zero_()
    b.grad.zero_()

    manual_loss_history.append(loss.item())
    if epoch % 1000 == 0:
        print(f"manual   epoch {epoch:4d}  loss {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 9. Model 3: standard PyTorch (nn.Module + torch.optim)
# ---------------------------------------------------------------------------
torch_model = nn.Linear(n_features, 1)
optimizer = torch.optim.Adam(torch_model.parameters(), lr=ADAM_LR)
loss_fn = nn.MSELoss()

torch_loss_history = []
for epoch in range(ADAM_EPOCHS):
    optimizer.zero_grad()
    y_pred = torch_model(X_train_t).squeeze(-1)
    loss = loss_fn(y_pred, y_train_t) + l2_strength * (torch_model.weight ** 2).sum()
    loss.backward()
    optimizer.step()

    torch_loss_history.append(loss.item())
    if epoch % 1000 == 0:
        print(f"standard epoch {epoch:4d}  loss {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 10. Compare the three models and the baseline on the test set
# ---------------------------------------------------------------------------
sklearn_test_pred = sklearn_model.predict(X_test)
with torch.no_grad():
    manual_test_pred = forward(X_test_t).numpy()
    torch_test_pred = torch_model(X_test_t).squeeze(-1).numpy()

metrics = {
    "Naive baseline (train mean)": evaluate(y_test, baseline_test_pred),
    "scikit-learn": evaluate(y_test, sklearn_test_pred),
    "PyTorch manual loop": evaluate(y_test, manual_test_pred),
    "PyTorch nn.Module + optim": evaluate(y_test, torch_test_pred),
}

print("\nTest set results:")
print(pd.DataFrame(metrics).T.round(4).to_string())

# Do the three methods agree? Compare their weights one by one.
sklearn_weights = sklearn_model.coef_
manual_weights = w.detach().numpy()
torch_weights = torch_model.weight.detach().numpy().flatten()

weights_table = pd.DataFrame({
    "feature": ALL_FEATURES,
    "sklearn": sklearn_weights,
    "manual": manual_weights,
    "standard": torch_weights,
})
print("\nWeights learned by each method:")
print(weights_table.round(4).to_string(index=False))

agreement = {
    "max_weight_diff_manual_vs_sklearn": float(np.abs(manual_weights - sklearn_weights).max()),
    "max_weight_diff_standard_vs_sklearn": float(np.abs(torch_weights - sklearn_weights).max()),
    "max_prediction_diff_manual_vs_sklearn": float(np.abs(manual_test_pred - sklearn_test_pred).max()),
    "max_prediction_diff_standard_vs_sklearn": float(np.abs(torch_test_pred - sklearn_test_pred).max()),
}
print("\nAgreement between methods:")
for name, value in agreement.items():
    print(f"  {name}: {value:.6f}")


# ---------------------------------------------------------------------------
# 11. Save everything the dashboard needs
# ---------------------------------------------------------------------------
joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
joblib.dump(sklearn_model, MODELS_DIR / "sklearn_model.joblib")
torch.save({"w": w.detach(), "b": b.detach()}, MODELS_DIR / "manual_model.pt")
torch.save(torch_model.state_dict(), MODELS_DIR / "torch_model.pt")

# The test rows (never used for training) so the app can predict on them.
test_columns = ["short_name", "overall", "value_eur", "player_positions",
                "preferred_foot", "position_group", "log_value"] + ALL_FEATURES
test_df[test_columns].to_csv(MODELS_DIR / "test_data.csv", index=False)

results = {
    "features": ALL_FEATURES,
    "best_alpha": best_alpha,
    "alpha_results": {str(k): v for k, v in alpha_results.items()},
    "feature_comparison": feature_comparison,
    "baseline_value": baseline_value,
    "metrics": metrics,
    "agreement": agreement,
    "weights": weights_table.to_dict(orient="list"),
    "manual_loss_history": manual_loss_history,
    "torch_loss_history": torch_loss_history,
    "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
}
with open(MODELS_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved models and results in", MODELS_DIR)
