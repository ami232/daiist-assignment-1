"""
Training pipeline: Seoul bike rentals, linear regression trained three ways.

Run with:  uv run python main.py train
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

# ------------------------------------------------------------------ 1. Load and clean
df = pd.read_csv("data/SeoulBikeData.csv", encoding="latin-1")  # not UTF-8 (°C in names)
df.columns = ["date", "rented_bike_count", "hour", "temperature", "humidity",
              "wind_speed", "visibility", "dew_point", "solar_radiation",
              "rainfall", "snowfall", "seasons", "holiday", "functioning_day"]
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

# Remove non-functioning hours: the system was closed, so 0 rentals isn't demand.
df = df[df["functioning_day"] == "Yes"].reset_index(drop=True)

# Weekends and public holidays are both "day off" days (no commuting peaks).
df["day_off"] = (df["date"].dt.dayofweek >= 5) | (df["holiday"] == "Holiday")

# ------------------------------------------------------------------ 2. Features
# Dew point dropped (99% explained by temperature + humidity).
# Seasons dropped (the training period contains no autumn).
weather_cols = ["temperature", "humidity", "wind_speed", "visibility",
                "solar_radiation", "rainfall", "snowfall"]
features = df[weather_cols].astype(float).copy()

# Temperature squared lets the temperature effect bend: more rentals as it gets
# warmer, then fewer when it gets too hot (a straight line can't do that).
features["temperature_sq"] = df["temperature"] ** 2

# Any rain puts cyclists off, even light rain, so a yes/no flag next to the mm amount.
features["is_raining"] = (df["rainfall"] > 0).astype(float)

# Hour x day-type interaction, one-hot: one column per (working day, hour) and
# (day off, hour). Working-day hour 0 is left out as the reference category,
# because all 48 columns together always sum to 1 and would duplicate the intercept.
for day_type, is_off in [("work", False), ("off", True)]:
    for h in range(24):
        if day_type == "work" and h == 0:
            continue
        features[f"{day_type}_h{h:02d}"] = ((df["day_off"] == is_off) & (df["hour"] == h)).astype(float)

y = df["rented_bike_count"].to_numpy(dtype=np.float32)

# Log target: the models predict log(1 + rentals) instead of rentals. Effects become
# multiplicative ("rush hour is 3x busier") instead of fixed ("+800 bikes"), so the
# peaks can be small in January and big in June. Predictions are converted back to
# bikes with expm1 before any evaluation.
y_log = np.log1p(y)

# ------------------------------------------------------------------ 3. Chronological split
# Train Dec 2017-Aug 2018, validate Sep 2018, test Oct-Nov 2018.
df["split"] = np.where(df["date"] < "2018-09-01", "train",
              np.where(df["date"] < "2018-10-01", "val", "test"))
train, val, test = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy(), (df["split"] == "test").to_numpy()

# Scale using the training period only, so nothing from val/test leaks in.
scaler = StandardScaler().fit(features[train])
X = scaler.transform(features).astype(np.float32)
print(f"Rows: train {train.sum()}, val {val.sum()}, test {test.sum()} | features: {X.shape[1]}")

# Weekday rush hours, for the rush-hour RMSE.
rush = ((~df["day_off"]) & df["hour"].isin([7, 8, 9, 17, 18, 19])).to_numpy()

# ------------------------------------------------------------------ 4a. scikit-learn
sk_model = LinearRegression().fit(X[train], y_log[train])

# ------------------------------------------------------------------ 4b. Manual PyTorch loop
LR = 0.1
EPOCHS = 1000
X_train_t = torch.tensor(X[train])
y_train_t = torch.tensor(y_log[train])

def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

w = torch.zeros(X.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def forward(X_in):
    return X_in @ w + b

for epoch in range(EPOCHS):
    loss = mse_loss(forward(X_train_t), y_train_t)
    loss.backward()
    with torch.no_grad():          # don't track the update itself
        w -= LR * w.grad
        b -= LR * b.grad
    w.grad.zero_()                 # reset, or gradients accumulate
    b.grad.zero_()
print(f"Manual loop final training MSE (log scale):   {loss.item():.5f}")

# ------------------------------------------------------------------ 4c. Standard PyTorch
model = nn.Linear(X.shape[1], 1)
nn.init.zeros_(model.weight)       # same starting point as the manual loop
nn.init.zeros_(model.bias)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)  # same algorithm as the manual loop
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    loss = loss_fn(model(X_train_t).squeeze(-1), y_train_t)
    loss.backward()
    optimizer.step()
print(f"Standard loop final training MSE (log scale): {loss.item():.5f}")

# ------------------------------------------------------------------ 5. Evaluate
X_all_t = torch.tensor(X)
with torch.no_grad():
    log_preds = {
        "scikit-learn": sk_model.predict(X),
        "PyTorch manual": forward(X_all_t).numpy(),
        "PyTorch standard": model(X_all_t).squeeze(-1).numpy(),
    }
# Back from log scale to bikes, so all metrics are in bikes.
preds = {name: np.expm1(p) for name, p in log_preds.items()}
preds["Naive baseline (train mean)"] = np.full(len(y), y[train].mean())

def evaluate(mask):
    rows = {}
    for name, p in preds.items():
        rows[name] = {
            "RMSE": np.sqrt(mean_squared_error(y[mask], p[mask])),
            "Rush-hour RMSE": np.sqrt(mean_squared_error(y[mask & rush], p[mask & rush])),
            "MAE": mean_absolute_error(y[mask], p[mask]),
            "R2": r2_score(y[mask], p[mask]),
        }
    return pd.DataFrame(rows).T.round(3)

print("\nValidation (bikes):\n", evaluate(val))
print("\nTest (bikes):\n", evaluate(test))

for name in ["PyTorch manual", "PyTorch standard"]:
    diff = np.abs(preds[name] - preds["scikit-learn"])[test].max()
    print(f"Max test prediction difference, {name} vs scikit-learn: {diff:.4f} bikes")

# ------------------------------------------------------------------ 6. Save for the app
joblib.dump(sk_model, ARTIFACTS / "sklearn_model.joblib")
torch.save({"w": w.detach(), "b": b.detach()}, ARTIFACTS / "torch_manual.pt")
torch.save(model.state_dict(), ARTIFACTS / "torch_standard.pt")
joblib.dump({
    "info": df[["date", "hour", "day_off", "split", "rented_bike_count"] + weather_cols],
    "X": X,                                   # already scaled, so the app only predicts
    "feature_names": list(features.columns),
    "baseline_mean": float(y[train].mean()),
    "log_target": True,                       # model outputs must go through np.expm1
}, ARTIFACTS / "data.joblib")
print("\nSaved models and data to artifacts/")
