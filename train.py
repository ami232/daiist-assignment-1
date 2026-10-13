from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/telco_churn.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
SEED = 42

COST_FP = 50    # wasted retention offer
COST_FN = 184   # missed churner: 0.30 * 780 - 50

# 1. Load and clean
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Blank TotalCharges:", df["TotalCharges"].isna().sum(),
      "| their tenure values:", df.loc[df["TotalCharges"].isna(), "tenure"].unique())
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customerID"])

addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
df[addon_cols + ["MultipleLines"]] = df[addon_cols + ["MultipleLines"]].replace(
    {"No internet service": "No", "No phone service": "No"})

# 2. Feature engineering
df["num_addons"] = (df[addon_cols] == "Yes").sum(axis=1)
df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                            labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                "SeniorCitizen", "num_addons"]
categorical_cols = [c for c in df.columns
                    if c not in numeric_cols + ["Churn"]]

# 3. Split
X = df.drop(columns=["Churn"])
y = df["Churn"].to_numpy()
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Train: {len(y_train)} rows, churn rate {y_train.mean():.3f}")
print(f"Test:  {len(y_test)} rows, churn rate {y_test.mean():.3f}")

# 4. Preprocess (fit on train only)
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                          sparse_output=False), categorical_cols),
])
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)
feature_names = list(preprocessor.get_feature_names_out())
print("Features after preprocessing:", X_train.shape[1])

# 5. Baseline: predict nobody churns
baseline_acc = (y_test == 0).mean()
baseline_cost = y_test.sum() * COST_FN
print(f"Baseline accuracy: {baseline_acc:.3f}  "
      f"| baseline cost on test set: EUR {baseline_cost:,}")

# 6. Save for later stages
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
X_test_raw.assign(Churn=y_test).to_csv(MODELS_DIR / "test_set.csv", index=False)
with open(MODELS_DIR / "config.json", "w") as f:
    json.dump({"cost_fp": COST_FP, "cost_fn": COST_FN,
               "feature_names": feature_names,
               "baseline_accuracy": float(baseline_acc),
               "baseline_cost": int(baseline_cost)}, f, indent=2)
print("Part 1 done.")

# ===============================================================
# PART 2: the same logistic regression, three ways
# ===============================================================
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score

from model_def import LogisticRegressionModel

torch.manual_seed(SEED)
LR = 1.0        # learning rate for both PyTorch versions
EPOCHS = 5000   # full-batch gradient descent steps


def cost_at(y_true, probs, threshold):
    """Business cost: false positives * 50 + false negatives * 184."""
    pred = (probs >= threshold).astype(int)
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    return int(fp * COST_FP + fn * COST_FN)


def best_threshold(y_true, probs):
    """Threshold with the lowest business cost (searched on TRAIN data)."""
    grid = np.linspace(0.01, 0.99, 99)
    costs = [cost_at(y_true, probs, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


# --- 7. Choose regularization strength C (the only sklearn tuning) ---
# 5-fold cross-validation on the TRAINING set only, scored by ROC-AUC.
# AUC is almost flat across C, so I take the STRONGEST regularization
# (smallest C) within 0.002 of the best score: same performance, more
# stable coefficients (MonthlyCharges is nearly a sum of the service columns).
cv_scores = {}
for c in [0.01, 0.1, 1, 10]:
    cv_scores[c] = cross_val_score(LogisticRegression(C=c, max_iter=5000),
                                   X_train, y_train, cv=5,
                                   scoring="roc_auc").mean()
    print(f"  C={c:<5} mean CV ROC-AUC = {cv_scores[c]:.4f}")
best = max(cv_scores.values())
C = min(c for c, s in cv_scores.items() if s >= best - 0.002)
n = len(y_train)
lam = 1 / (C * n)   # same penalty strength, written the way PyTorch needs it
print(f"Chosen C = {C}  ->  PyTorch L2 lambda = 1/(C*n) = {lam:.6f}")

# --- 8. Method 1: scikit-learn ---
sk_model = LogisticRegression(C=C, max_iter=5000)
sk_model.fit(X_train, y_train)

# Tensors for PyTorch
Xtr = torch.tensor(X_train, dtype=torch.float32)
ytr = torch.tensor(y_train, dtype=torch.float32)
Xte = torch.tensor(X_test, dtype=torch.float32)

# --- 9. Method 2: manual PyTorch loop (no nn.Module, no optimizer) ---
w = torch.zeros(Xtr.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for epoch in range(EPOCHS):
    z = Xtr @ w + b                                   # logits
    p = torch.sigmoid(z).clamp(1e-7, 1 - 1e-7)        # probabilities
    bce = -(ytr * torch.log(p) + (1 - ytr) * torch.log(1 - p)).mean()
    loss = bce + lam / 2 * (w ** 2).sum()             # L2 on weights, not bias
    loss.backward()                                   # compute gradients
    with torch.no_grad():                             # update by hand
        w -= LR * w.grad
        b -= LR * b.grad
        w.grad.zero_()
        b.grad.zero_()
    if epoch % 1000 == 0:
        print(f"  manual  epoch {epoch:4d}  loss {loss.item():.5f}")

# --- 10. Method 3: standard nn.Module + torch.optim ---
nn_model = LogisticRegressionModel(Xtr.shape[1])
optimizer = torch.optim.SGD(nn_model.parameters(), lr=LR)
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    logits = nn_model(Xtr)
    loss = loss_fn(logits, ytr) + lam / 2 * (nn_model.linear.weight ** 2).sum()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"  nn      epoch {epoch:4d}  loss {loss.item():.5f}")


# --- 11. Compare everything on the same test set ---
def predict_all(X_np, X_t):
    with torch.no_grad():
        return {
            "sklearn": sk_model.predict_proba(X_np)[:, 1],
            "manual_pytorch": torch.sigmoid(X_t @ w + b).numpy(),
            "nn_pytorch": torch.sigmoid(nn_model(X_t)).numpy(),
        }

train_probs = predict_all(X_train, Xtr)
test_probs = predict_all(X_test, Xte)

results = {"baseline": {"roc_auc": 0.5, "log_loss": None,
                        "accuracy@0.5": round(float(baseline_acc), 4),
                        "threshold": None, "test_cost": int(baseline_cost)}}
for name in test_probs:
    t = best_threshold(y_train, train_probs[name])   # chosen on TRAIN
    pt = test_probs[name]
    results[name] = {
        "roc_auc": round(roc_auc_score(y_test, pt), 4),
        "log_loss": round(log_loss(y_test, pt), 4),
        "accuracy@0.5": round(accuracy_score(y_test, pt >= 0.5), 4),
        "threshold": round(t, 2),
        "test_cost": cost_at(y_test, pt, t),          # evaluated on TEST
    }

print("\n=== Test-set comparison ===")
print(pd.DataFrame(results).T.to_string())

# Do the three agree? Compare learned coefficients directly.
coefs = pd.DataFrame({
    "sklearn": np.append(sk_model.coef_.ravel(), sk_model.intercept_),
    "manual_pytorch": np.append(w.detach().numpy(), b.detach().numpy()),
    "nn_pytorch": np.append(nn_model.linear.weight.detach().numpy().ravel(),
                            nn_model.linear.bias.detach().numpy()),
}, index=feature_names + ["intercept"])
print("\nMax |coef difference| vs sklearn:")
print("  manual PyTorch:", f"{(coefs['manual_pytorch'] - coefs['sklearn']).abs().max():.5f}")
print("  nn PyTorch:    ", f"{(coefs['nn_pytorch'] - coefs['sklearn']).abs().max():.5f}")
print("Max |probability difference| on test set:",
      f"{max(np.abs(test_probs[k] - test_probs['sklearn']).max() for k in test_probs):.5f}")

# --- 12. Save the three trained models and the results ---
joblib.dump(sk_model, MODELS_DIR / "sklearn_model.joblib")
torch.save({"w": w.detach(), "b": b.detach()}, MODELS_DIR / "manual_pytorch.pt")
torch.save(nn_model.state_dict(), MODELS_DIR / "nn_pytorch.pt")
coefs.to_csv(MODELS_DIR / "coefficients.csv")
with open(MODELS_DIR / "results.json", "w") as f:
    json.dump({"C": C, "lambda": lam, "lr": LR, "epochs": EPOCHS,
               "n_features": int(Xtr.shape[1]), "results": results}, f, indent=2)
print("\nPart 2 done. Saved 3 models + results to models/")
