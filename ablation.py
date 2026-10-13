"""
Feature ablation on the VALIDATION set (fit on train only; test untouched).
Self-contained (needs no artifacts):
    uv run python ablation.py

Each row changes ONE decision of the final pipeline and reports what happens,
so every feature choice in REPORT.md is backed by a number.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from pipeline import TARGET, build_features, fit_preprocessing, load_data, split_masks

df = load_data()
masks = split_masks(df)
train_df, val_df = df[masks["train"]], df[masks["val"]]
prep = fit_preprocessing(train_df)
HOUR_CELLS = [c for c in build_features(train_df.head(1), prep).columns if c.startswith("h")]


def final(d):
    return build_features(d, prep)


def variant(drop=(), add=None):
    def make(d):
        X = final(d).drop(columns=list(drop))
        if add:
            for name, func in add.items():
                X[name] = func(d)
        return X
    return make


def plain_hour_onehot(d):
    X = final(d).drop(columns=HOUR_CELLS)
    for h in range(1, 24):
        X[f"hour_{h:02d}"] = (d["hr"] == h).astype(float)
    X["workingday"] = d["workingday"]
    return X


def no_cleaning(d):
    X = final(d)
    X["hum"] = d["hum"]                       # keep the sensor-fault zeros
    X["weathersit"] = d["weathersit"]         # keep level 4
    X["atemp"] = d["atemp"]                   # keep the near-duplicate of temp
    return X


VARIANTS = {
    "Final pipeline": (final, True),
    "Target in raw bikes instead of log(1+cnt)": (final, False),
    "Hour x day type -> plain hour one-hot + workingday": (plain_hour_onehot, True),
    "Hour x day type -> numeric hr + workingday": (
        variant(drop=HOUR_CELLS, add={"hr": lambda d: d["hr"], "workingday": lambda d: d["workingday"]}), True),
    "Without temp^2": (variant(drop=["temp_sq"]), True),
    "Without cyclical month (sin/cos)": (variant(drop=["month_sin", "month_cos"]), True),
    "Without data cleaning (hum zeros, weathersit 4, atemp)": (no_cleaning, True),
    "Plus numeric weekday (0-6)": (variant(add={"weekday": lambda d: d["weekday"]}), True),
}

rows = []
y_val = val_df[TARGET].to_numpy()
for name, (make, log_target) in VARIANTS.items():
    X_tr, X_va = make(train_df), make(val_df)
    scaler = StandardScaler().fit(X_tr)
    y_tr = np.log1p(train_df[TARGET]) if log_target else train_df[TARGET]
    pred = LinearRegression().fit(scaler.transform(X_tr), y_tr).predict(scaler.transform(X_va))
    pred = np.expm1(pred) if log_target else pred
    rows.append({"variant": name, "n_features": X_tr.shape[1],
                 "val_MAE": mean_absolute_error(y_val, pred),
                 "val_RMSE": float(np.sqrt(mean_squared_error(y_val, pred))),
                 "negative_predictions": int((pred < 0).sum())})

table = pd.DataFrame(rows)
table["MAE_change_vs_final"] = table["val_MAE"] - table.loc[0, "val_MAE"]
print(table.round(2).to_string(index=False))
