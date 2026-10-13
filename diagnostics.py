"""
Why does the model under-predict on the test set?  Run AFTER main.py train:
    uv run python diagnostics.py

Loads the saved sklearn model (no retraining) and tests two hypotheses:
  H1  Re-transformation bias: expm1(E[log y]) estimates the MEDIAN, not the mean.
  H2  Growth inside 2012: `yr` is a single 2011->2012 step, so continued
      growth during 2012 cannot be represented.
"""

import joblib
import numpy as np
import pandas as pd

from pipeline import ARTIFACTS_DIR, TARGET, build_features, load_data, split_masks, to_bikes

df = load_data()
masks = split_masks(df)
fit_df, test_df = df[masks["train"] | masks["val"]], df[masks["test"]]
pre = joblib.load(ARTIFACTS_DIR / "preprocessing.joblib")
model = joblib.load(ARTIFACTS_DIR / "sklearn_model.joblib")


def predict_log(d):
    return model.predict(pre["scaler"].transform(build_features(d, pre["prep"])))


y_test = test_df[TARGET].to_numpy()
pred_test = to_bikes(predict_log(test_df))
print(f"Test: actual mean {y_test.mean():.1f}, predicted mean {pred_test.mean():.1f}, "
      f"bias {np.mean(pred_test - y_test):+.1f}, MAE {np.mean(np.abs(pred_test - y_test)):.1f}")

# --- H1: Duan's smearing correction, estimated on the FIT rows only -----------
resid_log = np.log1p(fit_df[TARGET].to_numpy()) - predict_log(fit_df)
smear = np.mean(np.exp(resid_log))
pred_smear = (pred_test + 1) * smear - 1
print(f"\nH1 smearing factor (from train+val residuals): {smear:.3f}")
print(f"   corrected: predicted mean {pred_smear.mean():.1f}, "
      f"bias {np.mean(pred_smear - y_test):+.1f}, MAE {np.mean(np.abs(pred_smear - y_test)):.1f}")

# --- H2: does growth continue within 2012? -----------------------------------
monthly = df.groupby(["yr", "mnth"])[TARGET].mean().unstack(0)
monthly.columns = ["2011", "2012"]
monthly["ratio_2012_vs_2011"] = monthly["2012"] / monthly["2011"]
print("\nH2 year-over-year ratio by month (2012 mean / 2011 mean):")
print(monthly.round(2).to_string())

fit_resid = fit_df.assign(resid_log=resid_log)
in_2012 = fit_resid[fit_resid["yr"] == 1].groupby("mnth")["resid_log"].mean()
print("\n   Mean log-residual by month on 2012 FIT rows (positive = under-predicting):")
print(in_2012.round(3).to_string())

# --- H1 again, the legitimate way: would VALIDATION have told us the same? ---
# (fit on train only, estimate smearing on train, check on val; test untouched)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pipeline import fit_preprocessing, to_model_space

train_df, val_df = df[masks["train"]], df[masks["val"]]
prep_tr = fit_preprocessing(train_df)
X_tr_raw = build_features(train_df, prep_tr)
scaler_tr = StandardScaler().fit(X_tr_raw)
m_tr = LinearRegression().fit(scaler_tr.transform(X_tr_raw), to_model_space(train_df[TARGET]))
smear_tr = np.mean(np.exp(to_model_space(train_df[TARGET]) - m_tr.predict(scaler_tr.transform(X_tr_raw))))
p_val = to_bikes(m_tr.predict(scaler_tr.transform(build_features(val_df, prep_tr))))
p_val_s = (p_val + 1) * smear_tr - 1
y_val = val_df[TARGET].to_numpy()
print(f"\nH1 on validation (smearing {smear_tr:.3f} from train):")
print(f"   plain      bias {np.mean(p_val - y_val):+.1f}  MAE {np.mean(np.abs(p_val - y_val)):.1f}")
print(f"   corrected  bias {np.mean(p_val_s - y_val):+.1f}  MAE {np.mean(np.abs(p_val_s - y_val)):.1f}")
