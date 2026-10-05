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
import pandas as pd
import numpy as np

# Load
df = pd.read_csv("data/Telco-Customer-Churn.csv")

customer_ids = df["customerID"].copy()

# Clean
df = df.drop(columns=["customerID"])

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["Churn"] = (df["Churn"] == "Yes").astype(int)

df = df.replace("No phone service", "No")
df = df.replace("No internet service", "No")

print(df.isnull().sum()[df.isnull().sum() > 0])
print(df.dtypes)
print(df.shape)
print(df["Churn"].value_counts())
print(df.head())

# Feature engineering
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, np.inf],
    labels=["0-12", "13-24", "25-48", "49+"],
)

service_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines",
]
df["total_services"] = (df[service_cols] == "Yes").sum(axis=1)

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df["contract_risk"] = df["Contract"].map(contract_map)

df["is_new_customer"] = (df["tenure"] <= 3).astype(int)

expected_total = df["MonthlyCharges"] * df["tenure"]
df["spend_mismatch"] = (df["TotalCharges"] - expected_total).abs()

print(df.shape)
print(df[["tenure_group", "total_services", "contract_risk", "is_new_customer", "spend_mismatch"]].head())

# Encode, split, scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

y = df["Churn"]
X = df.drop(columns=["Churn"])
X = pd.get_dummies(X, drop_first=True)

print(X.shape)
print(X.columns.tolist())

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, customer_ids, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train.shape, X_test.shape)
print(y_train.mean(), y_test.mean())
print(ids_test.head())

# Model 1: scikit-learn logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

sk_model = LogisticRegression(max_iter=1000)
sk_model.fit(X_train_scaled, y_train)

sk_preds = sk_model.predict(X_test_scaled)
sk_probs = sk_model.predict_proba(X_test_scaled)[:, 1]

print("Sklearn Accuracy:", accuracy_score(y_test, sk_preds))
print("Sklearn ROC AUC:", roc_auc_score(y_test, sk_probs))

# Naive baseline
baseline_preds = np.zeros_like(y_test)
print("Baseline Accuracy:", accuracy_score(y_test, baseline_preds))

# Model 2: manual PyTorch loop
import torch
torch.manual_seed(14)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

n_features = X_train_t.shape[1]

weights = torch.zeros(n_features, 1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

learning_rate = 0.1
n_epochs = 300

for epoch in range(n_epochs):
    z = X_train_t @ weights + bias
    preds = torch.sigmoid(z)

    loss = -(y_train_t * torch.log(preds + 1e-8) + (1 - y_train_t) * torch.log(1 - preds + 1e-8)).mean()

    loss.backward()

    with torch.no_grad():
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad
        weights.grad.zero_()
        bias.grad.zero_()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss {loss.item():.4f}")

with torch.no_grad():
    test_probs_manual = torch.sigmoid(X_test_t @ weights + bias)
    test_preds = (test_probs_manual >= 0.5).float()

manual_acc = (test_preds == y_test_t).float().mean().item()
print("Manual PyTorch Accuracy:", manual_acc)
print("Manual PyTorch ROC AUC:", roc_auc_score(y_test, test_probs_manual.numpy()))

# Model 3: standard PyTorch (nn.Module + torch.optim)
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

nn_model = LogisticRegressionModel(n_features)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.1)

for epoch in range(n_epochs):
    optimizer.zero_grad()

    preds = nn_model(X_train_t)
    loss = loss_fn(preds, y_train_t)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss {loss.item():.4f}")

with torch.no_grad():
    test_probs = nn_model(X_test_t)
    test_preds = (test_probs >= 0.5).float()

nn_acc = (test_preds == y_test_t).float().mean().item()
print("Standard PyTorch Accuracy:", nn_acc)
print("Standard PyTorch ROC AUC:", roc_auc_score(y_test, test_probs.numpy()))

# Save models and test results
import os
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(sk_model, "models/sklearn_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

torch.save(weights, "models/manual_weights.pt")
torch.save(bias, "models/manual_bias.pt")

torch.save(nn_model.state_dict(), "models/nn_model_state.pt")

np.savez(
    "models/test_results.npz",
    ids_test=ids_test.values,
    y_test=y_test.values,
    sk_probs=sk_probs,
    manual_probs=test_probs_manual.numpy(),
    nn_probs=test_probs.numpy(),
    tenure=X_test["tenure"].values,
    monthly_charges=X_test["MonthlyCharges"].values,
    total_services=X_test["total_services"].values,
    contract_risk=X_test["contract_risk"].values,
)

print("Saved all models and test results to /models")