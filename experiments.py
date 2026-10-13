"""Scratch experiments on VALIDATION data only. Base: log1p(tenure) fixed."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train import load_and_clean, split_data

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
CAT = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
       "InternetService"] + ADDONS + ["Contract", "PaperlessBilling", "PaymentMethod"]


def make_features(df, pay_flag=False, total=False, collapse=False,
                  addon_count=False, contract_ord=False):
    X = pd.DataFrame(index=df.index)
    X["tenure_log"] = np.log1p(df["tenure"])
    X["MonthlyCharges"] = df["MonthlyCharges"]
    X["SeniorCitizen"] = df["SeniorCitizen"]
    num = ["tenure_log", "MonthlyCharges", "SeniorCitizen"]
    cat = list(CAT)
    for c in CAT:
        X[c] = df[c]

    if collapse:  # "No internet/phone service" -> "No" (duplicate information)
        for c in ADDONS + ["MultipleLines"]:
            X[c] = X[c].replace({"No internet service": "No", "No phone service": "No"})
    if pay_flag:  # one electronic-check flag instead of 4 payment categories
        X["is_echeck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
        cat.remove("PaymentMethod")
        num.append("is_echeck")
    if contract_ord:  # 0/1/2 instead of one-hot
        X["contract_ord"] = df["Contract"].map(
            {"Month-to-month": 0, "One year": 1, "Two year": 2})
        cat.remove("Contract")
        num.append("contract_ord")
    if total:
        X["TotalCharges"] = df["TotalCharges"]
        num.append("TotalCharges")
    if addon_count:
        X["n_addons"] = (df[ADDONS] == "Yes").sum(axis=1)
        num.append("n_addons")
    return X, num, cat


def run(train_df, val_df, **opts):
    X_tr, num, cat = make_features(train_df, **opts)
    X_va, _, _ = make_features(val_df, **opts)
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X_tr, train_df["Churn"])  # fitted on TRAIN only
    p = model.predict_proba(X_va)[:, 1]
    return roc_auc_score(val_df["Churn"], p), log_loss(val_df["Churn"], p)


if __name__ == "__main__":
    train_df, val_df, _ = split_data(load_and_clean())
    variants = [
        ("base", {}),
        ("pay_flag", {"pay_flag": True}),
        ("collapse", {"collapse": True}),
        ("contract_ord", {"contract_ord": True}),
        ("+TotalCharges", {"total": True}),
        ("+addon_count", {"addon_count": True}),
    ]
    base_auc, base_ll = run(train_df, val_df)
    print(f"{'variant':<15}{'AUC':>9}{'log-loss':>10}{'d_logloss':>11}")
    for name, opts in variants:
        auc, ll = run(train_df, val_df, **opts)
        print(f"{name:<15}{auc:>9.4f}{ll:>10.4f}{ll - base_ll:>+11.4f}")