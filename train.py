import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "sp500_raw.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

print("=" * 60)
print("1. Loading data …")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
print(f"   Raw rows: {len(df)}, columns: {list(df.columns)}")

print("2. Engineering features …")


def safe_div(a, b):
    return np.where(b != 0, a / b, 0.0)


df["Return_1d"] = df["Close"].pct_change(1)
df["Return_5d"] = df["Close"].pct_change(5)
df["Return_10d"] = df["Close"].pct_change(10)
df["Return_21d"] = df["Close"].pct_change(21)

for window in [5, 10, 21, 50]:
    df[f"SMA_{window}"] = df["Close"].rolling(window).mean()
    df[f"Close_over_SMA_{window}"] = safe_div(df["Close"].values, df[f"SMA_{window}"].values)

df["SMA_5_over_21"] = safe_div(df["SMA_5"].values, df["SMA_21"].values)
df["SMA_10_over_50"] = safe_div(df["SMA_10"].values, df["SMA_50"].values)

df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
df["Volatility_21d"] = df["Return_1d"].rolling(21).std()
df["Volatility_ratio"] = safe_div(df["Volatility_5d"].values, df["Volatility_21d"].values)

df["Intraday_range"] = safe_div((df["High"] - df["Low"]).values, df["Close"].values)
df["Intraday_range_5d"] = df["Intraday_range"].rolling(5).mean()

delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = safe_div(gain.values, loss.values)
df["RSI_14"] = 100 - 100 / (1 + rs)

ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema_12 - ema_26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

df["Volume_SMA_5"] = df["Volume"].rolling(5).mean()
df["Volume_SMA_21"] = df["Volume"].rolling(21).mean()
df["Volume_ratio"] = safe_div(df["Volume"].values, df["Volume_SMA_21"].values)
df["Volume_change_1d"] = df["Volume"].pct_change(1)

df["Distance_from_high_52w"] = safe_div(
    df["Close"].values,
    df["High"].rolling(252).max().values,
)
df["Distance_from_low_52w"] = safe_div(
    df["Close"].values,
    df["Low"].rolling(252).min().values,
)

df["DayOfWeek"] = df["Date"].dt.dayofweek

bb_mid = df["Close"].rolling(20).mean()
bb_std = df["Close"].rolling(20).std()
df["BB_position"] = safe_div((df["Close"] - bb_mid).values, (2 * bb_std).values)

df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

df = df.dropna().reset_index(drop=True)
print(f"   Rows after feature engineering: {len(df)}")
print(f"   Target balance: {df['Target'].value_counts().to_dict()}")

FEATURE_COLS = [
    "Return_1d", "Return_5d", "Return_10d", "Return_21d",
    "Close_over_SMA_5", "Close_over_SMA_10", "Close_over_SMA_21", "Close_over_SMA_50",
    "SMA_5_over_21", "SMA_10_over_50",
    "Volatility_5d", "Volatility_21d", "Volatility_ratio",
    "Intraday_range", "Intraday_range_5d",
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "Volume_ratio", "Volume_change_1d",
    "Distance_from_high_52w", "Distance_from_low_52w",
    "DayOfWeek",
    "BB_position",
]
print(f"   Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

print("3. Splitting data (time-based) …")

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print(f"   Train: {len(df_train)} rows ({df_train['Date'].min().date()} → {df_train['Date'].max().date()})")
print(f"   Val:   {len(df_val)} rows ({df_val['Date'].min().date()} → {df_val['Date'].max().date()})")
print(f"   Test:  {len(df_test)} rows ({df_test['Date'].min().date()} → {df_test['Date'].max().date()})")

X_train = df_train[FEATURE_COLS].values.astype(np.float64)
X_val = df_val[FEATURE_COLS].values.astype(np.float64)
X_test = df_test[FEATURE_COLS].values.astype(np.float64)
y_train = df_train["Target"].values
y_val = df_val["Target"].values
y_test = df_test["Target"].values

for arr in [X_train, X_val, X_test]:
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

for arr in [X_train_s, X_val_s, X_test_s]:
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

print("\n4. Naive baseline …")
majority_class = int(y_train.mean() >= 0.5)
y_baseline = np.full_like(y_test, majority_class)
baseline_acc = accuracy_score(y_test, y_baseline)
print(f"   Majority class in train: {majority_class}  |  Baseline test accuracy: {baseline_acc:.4f}")

print("\n5a. Training scikit-learn logistic regression …")
sklearn_model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
sklearn_model.fit(X_train_s, y_train)

y_pred_sklearn = sklearn_model.predict(X_test_s)
y_prob_sklearn = sklearn_model.predict_proba(X_test_s)[:, 1]
print(f"   Test accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
print(f"   Test AUC:      {roc_auc_score(y_test, y_prob_sklearn):.4f}")

print("\n5b. Training manual PyTorch logistic regression …")

X_tr_t = torch.tensor(X_train_s, dtype=torch.float32)
y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_te_t = torch.tensor(X_test_s, dtype=torch.float32)

n_features = X_tr_t.shape[1]

torch.manual_seed(42)
W_manual = torch.randn(n_features, 1, requires_grad=True)
b_manual = torch.zeros(1, requires_grad=True)

lr_manual = 0.01
n_epochs_manual = 200
batch_size = 256

train_losses_manual = []
val_losses_manual = []

for epoch in range(n_epochs_manual):
    perm = torch.randperm(X_tr_t.size(0))
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, X_tr_t.size(0), batch_size):
        idx = perm[i : i + batch_size]
        X_batch = X_tr_t[idx]
        y_batch = y_tr_t[idx]

        logits = X_batch @ W_manual + b_manual
        probs = torch.sigmoid(logits)

        loss = -torch.mean(
            y_batch * torch.log(probs + 1e-8)
            + (1 - y_batch) * torch.log(1 - probs + 1e-8)
        )

        loss.backward()

        with torch.no_grad():
            W_manual -= lr_manual * W_manual.grad
            b_manual -= lr_manual * b_manual.grad
            W_manual.grad.zero_()
            b_manual.grad.zero_()

        epoch_loss += loss.item()
        n_batches += 1

    train_losses_manual.append(epoch_loss / n_batches)

    with torch.no_grad():
        val_logits = X_val_t @ W_manual + b_manual
        val_probs = torch.sigmoid(val_logits)
        val_loss = -torch.mean(
            y_val_t * torch.log(val_probs + 1e-8)
            + (1 - y_val_t) * torch.log(1 - val_probs + 1e-8)
        )
        val_losses_manual.append(val_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"   Epoch {epoch+1:3d} | train loss: {train_losses_manual[-1]:.4f} | val loss: {val_losses_manual[-1]:.4f}")

with torch.no_grad():
    logits_test_manual = X_te_t @ W_manual + b_manual
    y_prob_manual = torch.sigmoid(logits_test_manual).squeeze().numpy()
    y_pred_manual = (y_prob_manual >= 0.5).astype(int)

print(f"   Test accuracy: {accuracy_score(y_test, y_pred_manual):.4f}")
print(f"   Test AUC:      {roc_auc_score(y_test, y_prob_manual):.4f}")

print("\n5c. Training standard PyTorch logistic regression (nn.Module + optim) …")


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


torch.manual_seed(42)
std_model = LogisticRegressionModel(n_features)
optimizer = torch.optim.Adam(std_model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

n_epochs_std = 200
train_losses_std = []
val_losses_std = []

for epoch in range(n_epochs_std):
    std_model.train()
    perm = torch.randperm(X_tr_t.size(0))
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, X_tr_t.size(0), batch_size):
        idx = perm[i : i + batch_size]
        X_batch = X_tr_t[idx]
        y_batch = y_tr_t[idx]

        logits = std_model(X_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    train_losses_std.append(epoch_loss / n_batches)

    std_model.eval()
    with torch.no_grad():
        val_logits = std_model(X_val_t)
        val_loss = criterion(val_logits, y_val_t)
        val_losses_std.append(val_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"   Epoch {epoch+1:3d} | train loss: {train_losses_std[-1]:.4f} | val loss: {val_losses_std[-1]:.4f}")

std_model.eval()
with torch.no_grad():
    logits_test_std = std_model(X_te_t)
    y_prob_std = torch.sigmoid(logits_test_std).squeeze().numpy()
    y_pred_std = (y_prob_std >= 0.5).astype(int)

print(f"   Test accuracy: {accuracy_score(y_test, y_pred_std):.4f}")
print(f"   Test AUC:      {roc_auc_score(y_test, y_prob_std):.4f}")

print("\n" + "=" * 60)
print("RESULTS COMPARISON (Test set)")
print("=" * 60)

results = {}
for name, y_pred, y_prob in [
    ("Naive baseline", y_baseline, np.full_like(y_test, float(majority_class))),
    ("Scikit-learn LR", y_pred_sklearn, y_prob_sklearn),
    ("Manual PyTorch LR", y_pred_manual, y_prob_manual),
    ("Standard PyTorch LR", y_pred_std, y_prob_std),
]:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}
    print(f"  {name:22s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

print("\n6. Saving artifacts …")

with open(ARTIFACTS / "sklearn_model.pkl", "wb") as f:
    pickle.dump(sklearn_model, f)

torch.save({"W": W_manual.detach(), "b": b_manual.detach()}, ARTIFACTS / "manual_pytorch.pt")

torch.save(std_model.state_dict(), ARTIFACTS / "std_pytorch.pt")

with open(ARTIFACTS / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

test_artifacts = {
    "X_test": X_test_s.tolist(),
    "y_test": y_test.tolist(),
    "y_prob_sklearn": y_prob_sklearn.tolist(),
    "y_prob_manual": y_prob_manual.tolist(),
    "y_prob_std": y_prob_std.tolist(),
    "dates_test": df_test["Date"].dt.strftime("%Y-%m-%d").tolist(),
    "feature_names": FEATURE_COLS,
    "n_features": n_features,
    "majority_class": majority_class,
    "results": results,
}
with open(ARTIFACTS / "test_data.json", "w") as f:
    json.dump(test_artifacts, f)

curves = {
    "manual_train": train_losses_manual,
    "manual_val": val_losses_manual,
    "std_train": train_losses_std,
    "std_val": val_losses_std,
}
with open(ARTIFACTS / "training_curves.json", "w") as f:
    json.dump(curves, f)

dist_data = {
    "features": {},
    "target_train": y_train.tolist(),
    "target_test": y_test.tolist(),
}
for col in FEATURE_COLS:
    dist_data["features"][col] = {
        "train": df_train[col].tolist(),
        "test": df_test[col].tolist(),
    }
with open(ARTIFACTS / "distributions.json", "w") as f:
    json.dump(dist_data, f)

print("   All artifacts saved to ./artifacts/")
print(f"   Files: {[p.name for p in sorted(ARTIFACTS.iterdir())]}")
print("\nDone.")
