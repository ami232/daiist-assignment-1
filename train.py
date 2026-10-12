"""
train.py — Assignment 1 training pipeline. Run: uv run python main.py train
The business framing and the pipeline decisions it drives are explained in REPORT.md
"""
 
from __future__ import annotations
 
import json
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
RANDOM_STATE = 42
DATA_PATH = Path("data/telco_churn.csv")
ARTIFACT_DIR = Path("artifacts")
 
# Business assumptions
# All money is measured in months of the customer's OWN bill (MonthlyCharges), so a customer who pays more is worth more
OFFER_MONTHS = 1 # the offer: 1 free month; ONLY costs revenue if the customer would otherwise have paid
MONTHS_KEPT = 12 # a saved churner stays 12 more months
SAVE_RATE = 0.30 # share of contacted would-be churners the offer keeps (uncertain)
SAVE_RATES_TO_TEST = [0.10, 0.20, 0.30, 0.40, 0.50] # sensitivity analysis
 
# Hyperparameters (the only ones tuned)
C_CANDIDATES = [0.01, 0.1, 1.0, 10.0] # sklearn C; same penalty is reused in PyTorch
MANUAL_LRS = [0.01, 0.1, 1.0] # gradient descent
ADAM_LRS = [0.001, 0.01, 0.1]
N_EPOCHS = 3000
 
# Features
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
          "StreamingTV", "StreamingMovies"]
YES_NO = ["Partner", "Dependents", "PaperlessBilling", "PhoneService", "MultipleLines"] + ADDONS
CONTRACT_MONTHS = {"Month-to-month": 1, "One year": 12, "Two year": 24}
 
NUMERIC = (["log_tenure", "contract_months", "electronic_check", "SeniorCitizen"]
           + [f"{c}_yes" for c in YES_NO])
CATEGORICAL = ["InternetService"]
 

# DATA
def load_and_clean(path: Path = DATA_PATH):
    raw = pd.read_csv(path)
    df = raw[raw["tenure"] > 0].copy() # target definition: tenure-0 customers cannot have churned yet (none did).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return raw, df.reset_index(drop=True)
 
 
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # one 0/1 column per service; "No internet service" / "No phone service" become 0 (InternetService and PhoneService already hold that fact ∴ encoding it again would create 7 duplicate columns)
    for col in YES_NO:
        df[f"{col}_yes"] = (df[col] == "Yes").astype(int)
 
    df["log_tenure"] = np.log1p(df["tenure"]) # because churn falls steeply in the first months, then flattens
 
    df["contract_months"] = df["Contract"].map(CONTRACT_MONTHS)
 
    df["electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int) # electronic check customers churn at 45%; the other 3 payment methods are all 15-19%, so 1 flag carries all the payment information
 
    # Candidates tested in the ablation; NOT used in the final model
    df["is_male"] = (df["gender"] == "Male").astype(int)
    df["is_new_customer"] = (df["tenure"] <= 3).astype(int)
    df["num_addon_services"] = df[[f"{c}_yes" for c in ADDONS]].sum(axis=1)
    df["fiber_month_to_month"] = ((df["InternetService"] == "Fiber optic")
                                  & (df["Contract"] == "Month-to-month")).astype(int)
    return df
 
 
def split_data(df: pd.DataFrame):
    # Stratified ON Churn
    train, rest = train_test_split(df, test_size=0.4, stratify=df["Churn"],
                                   random_state=RANDOM_STATE)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["Churn"],
                                 random_state=RANDOM_STATE)
    return [d.reset_index(drop=True) for d in (train, val, test)]
 
 
def build_preprocessor(numeric: list[str], categorical: list[str]) -> Pipeline:
    encode = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical),
    ], verbose_feature_names_out=False)
    return Pipeline([("encode", encode), ("scale", StandardScaler())])
 
 
# BUSINESS VALUE
def campaign_value(y, probs, bills, threshold, save_rate=SAVE_RATE):
    contacted = probs >= threshold
    gained = save_rate * (MONTHS_KEPT - OFFER_MONTHS) * bills[contacted & (y == 1)].sum()
    wasted = OFFER_MONTHS * bills[contacted & (y == 0)].sum()
    return float(gained - wasted)
 
 
def break_even(save_rate=SAVE_RATE):
    return OFFER_MONTHS / (OFFER_MONTHS + save_rate * (MONTHS_KEPT - OFFER_MONTHS))
 
 
def best_threshold(y, probs, bills, save_rate=SAVE_RATE):
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    values = [campaign_value(y, probs, bills, t, save_rate) for t in grid]
    return float(grid[int(np.argmax(values))])

 
def objective(X, y, w, b, lam):
    z = X @ w + b
    return float(np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * np.sum(w ** 2))
 
 
def train_sklearn(X_train, y_train, X_val, y_val):
    best = None
    for C in C_CANDIDATES:
        model = LogisticRegression(C=C, max_iter=10_000, tol=1e-10).fit(X_train, y_train)
        val_ll = log_loss(y_val, model.predict_proba(X_val)[:, 1])
        print(f"  C={C:<5} validation log-loss={val_ll:.4f}")
        if best is None or val_ll < best[0]:
            best = (val_ll, C, model)
    print(f"  selected C={best[1]}")
    return best[2], best[1]
 
 
def train_manual(X, y, lr, lam):
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    history = []
    for _ in range(N_EPOCHS):
        p = torch.sigmoid(X @ w + b).clamp(1e-7, 1 - 1e-7)     # clamp avoids log(0)
        loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean() + 0.5 * lam * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        history.append(loss.item())
    return {"w": w.detach(), "b": b.detach()}, history
 
 
class ChurnModel(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
 
    def forward(self, x):
        return self.linear(x).squeeze(-1) # logits
 
 
def train_standard(X, y, lr, lam):
    torch.manual_seed(RANDOM_STATE)
    model = ChurnModel(X.shape[1]).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y) + 0.5 * lam * (model.linear.weight ** 2).sum() # Penalty
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return model, history
 
 
def pick_learning_rate(name, train_fn, lrs, X, y, lam):
    best = None
    for lr in lrs:
        model, history = train_fn(X, y, lr, lam)
        print(f"  [{name}] lr={lr:<6} final objective={history[-1]:.9f}")
        if best is None or history[-1] < best[1][-1]:
            best = (lr, history, model)
    print(f"  [{name}] selected lr={best[0]}")
    return best[0], best[2], best[1]
 

# Evaluation
def evaluate(name, y, probs, bills, threshold):
    preds = (probs >= threshold).astype(int)
    return {"model": name,
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "roc_auc": roc_auc_score(y, probs),
            "log_loss": log_loss(y, probs, labels=[0, 1]),
            "campaign_value": campaign_value(y, probs, bills, threshold)}
 
 
def run_ablation(dev_df, C):
    """'final wins k/5' shows whether a choice helps consistently OR only by luck on one split"""
    def swap(old, new):
        return [new if f == old else f for f in NUMERIC]
    raw_num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    raw_cat = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", *ADDONS, "Contract", "PaperlessBilling", "PaymentMethod"]
    no_contract = [f for f in NUMERIC if f != "contract_months"]
    no_payment = [f for f in NUMERIC if f != "electronic_check"]
    setups = [
        ("Raw columns, plain one-hot", raw_num, raw_cat),
        ("Final feature set", NUMERIC, CATEGORICAL),
        ("Raw tenure instead of log_tenure", swap("log_tenure", "tenure"), CATEGORICAL),
        ("Contract dummies instead of contract_months", no_contract, CATEGORICAL + ["Contract"]),
        ("PaymentMethod dummies instead of electronic_check", no_payment, CATEGORICAL + ["PaymentMethod"]),
        ("+ MonthlyCharges and TotalCharges (dropped)", NUMERIC + ["MonthlyCharges", "TotalCharges"], CATEGORICAL),
        ("+ gender (dropped)", NUMERIC + ["is_male"], CATEGORICAL),
        ("+ is_new_customer (rejected)", NUMERIC + ["is_new_customer"], CATEGORICAL),
        ("+ num_addon_services (rejected)", NUMERIC + ["num_addon_services"], CATEGORICAL),
        ("+ fiber_month_to_month (rejected)", NUMERIC + ["fiber_month_to_month"], CATEGORICAL),
    ]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fold_auc = [], {}
    for name, num, cat in setups:
        pipe = Pipeline([("prep", build_preprocessor(num, cat)), # re-fitted inside each fold
                         ("model", LogisticRegression(C=C, max_iter=10_000))])
        scores = cross_validate(pipe, dev_df[num + cat], dev_df["Churn"], cv=folds,
                                scoring=["roc_auc", "neg_log_loss"])
        fold_auc[name] = scores["test_roc_auc"]
        X = build_preprocessor(num, cat).fit_transform(dev_df[num + cat])
        design = np.c_[X, np.ones(len(X))]
        rows.append({"setup": name, "n_columns": X.shape[1],
                     "duplicate_columns": design.shape[1] - np.linalg.matrix_rank(design),
                     "cv_roc_auc": scores["test_roc_auc"].mean(),
                     "cv_log_loss": -scores["test_neg_log_loss"].mean()})
    for row in rows:
        wins = int((fold_auc["Final feature set"] > fold_auc[row["setup"]]).sum())
        row["final_wins_folds"] = f"{wins}/5"
    return pd.DataFrame(rows)
 
 
# Main
def main():
    for sub in ["models", "data", "metrics"]:
        (ARTIFACT_DIR / sub).mkdir(parents=True, exist_ok=True)
 
    print("Data")
    raw, df = load_and_clean()
    df = engineer_features(df)
    train_df, val_df, test_df = split_data(df)
    print(f"  kept {len(df)} of {len(raw)} rows (removed {len(raw) - len(df)} with tenure 0)")
    print(f"  train={len(train_df)}  validation={len(val_df)}  test={len(test_df)}")
 
    features = NUMERIC + CATEGORICAL
    prep = build_preprocessor(NUMERIC, CATEGORICAL)
    X_train = prep.fit_transform(train_df[features])
    X_val, X_test = prep.transform(val_df[features]), prep.transform(test_df[features])
    y_train, y_val, y_test = (d["Churn"].to_numpy() for d in (train_df, val_df, test_df))
    bills_val, bills_test = val_df["MonthlyCharges"].to_numpy(), test_df["MonthlyCharges"].to_numpy()
    feature_names = list(prep.get_feature_names_out())
    print(f"  {len(feature_names)} model columns")
 
    print("\n[1] scikit-learn")
    sk_model, C = train_sklearn(X_train, y_train, X_val, y_val)
    lam = 1.0 / (C * len(X_train))
 
    X_t = torch.tensor(X_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    print("\n[2] manual PyTorch (gradient descent)")
    manual_lr, manual, manual_hist = pick_learning_rate("manual", train_manual, MANUAL_LRS, X_t, y_t, lam)
    print("\n[3] standard PyTorch (nn.Module + Adam)")
    adam_lr, standard, standard_hist = pick_learning_rate("standard", train_standard, ADAM_LRS, X_t, y_t, lam)
 
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)

    # DO THEY AGREE?
    weights = {"scikit-learn": (sk_model.coef_.ravel(), float(sk_model.intercept_[0])),
               "Manual PyTorch": (manual["w"].numpy(), float(manual["b"].item())),
               "Standard PyTorch": (standard.linear.weight.detach().numpy().ravel(),
                                    float(standard.linear.bias.item()))}
    def predict(X, w, b):
        return 1 / (1 + np.exp(-(X @ w + b)))
    val_probs = {n: predict(X_val, *wb) for n, wb in weights.items()}
    test_probs = {n: predict(X_test, *wb) for n, wb in weights.items()}
 
    agreement = pd.DataFrame([
        {"pair": f"{a} vs {b}",
         "max_weight_difference": float(np.abs(weights[a][0] - weights[b][0]).max()),
         "intercept_difference": abs(weights[a][1] - weights[b][1]),
         "max_test_probability_difference": float(np.abs(test_probs[a] - test_probs[b]).max())}
        for a, b in [("scikit-learn", "Manual PyTorch"), ("scikit-learn", "Standard PyTorch"),
                     ("Manual PyTorch", "Standard PyTorch")]])
    objectives = {n: objective(X_train, y_train, w, b, lam) for n, (w, b) in weights.items()}
    print("\nAgreement")
    print(agreement.to_string(index=False))
    print("  training objective:", {k: round(v, 9) for k, v in objectives.items()})
 
    # Threshold: maximise campaign value on VALIDATION
    threshold = best_threshold(y_val, val_probs["scikit-learn"], bills_val)
    print(f"\nThreshold chosen on validation: {threshold} (break-even in theory: {break_even():.3f})")
 
    # Test set
    baseline_probs = baseline.predict_proba(X_test)[:, 1]
    comparison = pd.DataFrame(
        [evaluate(n, y_test, p, bills_test, threshold) for n, p in test_probs.items()]
        + [evaluate("Naive baseline", y_test, baseline_probs, bills_test, threshold)]
    ).set_index("model")
    month_to_month = (test_df["Contract"] == "Month-to-month").to_numpy().astype(float)
    policies = pd.DataFrame([
        {"policy": "Contact nobody", "campaign_value": 0.0},
        {"policy": "Contact everyone", "campaign_value": campaign_value(y_test, np.ones(len(y_test)), bills_test, 0.5)},
        {"policy": "Contact all month-to-month customers", "campaign_value": campaign_value(y_test, month_to_month, bills_test, 0.5)},
        {"policy": f"Model (threshold {threshold})", "campaign_value": campaign_value(y_test, test_probs["scikit-learn"], bills_test, threshold)},
    ])
    print(f"\nTest set @ threshold {threshold}")
    print(comparison.round(4).to_string())
    print(policies.round(0).to_string(index=False))
 
    # Sensitivity to the most uncertain assumption
    sensitivity = []
    for s in SAVE_RATES_TO_TEST:
        t = best_threshold(y_val, val_probs["scikit-learn"], bills_val, s)
        sensitivity.append({
            "save_rate": s, "break_even": break_even(s), "threshold_on_validation": t,
            "model_value": campaign_value(y_test, test_probs["scikit-learn"], bills_test, t, s),
            "contact_everyone_value": campaign_value(y_test, np.ones(len(y_test)), bills_test, 0.5, s),
            "month_to_month_rule_value": campaign_value(y_test, month_to_month, bills_test, 0.5, s)})
    sensitivity = pd.DataFrame(sensitivity)
    print("\nSensitivity to the save rate (test set)")
    print(sensitivity.round(3).to_string(index=False))
 
    print("\nAblation (5-fold cross-validation on train + validation)")
    ablation = run_ablation(pd.concat([train_df, val_df], ignore_index=True), C)
    print(ablation.round(4).to_string(index=False))
 
    # Save what app.py needs
    joblib.dump(sk_model, ARTIFACT_DIR / "models" / "sklearn_logreg.joblib")
    joblib.dump(baseline, ARTIFACT_DIR / "models" / "naive_baseline.joblib")
    torch.save(manual, ARTIFACT_DIR / "models" / "manual_pytorch_logreg.pt")
    torch.save(standard.state_dict(), ARTIFACT_DIR / "models" / "standard_pytorch_logreg.pt")
    np.savez(ARTIFACT_DIR / "data" / "test_processed.npz", X_test=X_test, y_test=y_test)
    test_df.to_csv(ARTIFACT_DIR / "data" / "test_customers.csv", index=False)
    train_df.to_csv(ARTIFACT_DIR / "data" / "train_customers.csv", index=False)
    comparison.to_csv(ARTIFACT_DIR / "metrics" / "comparison.csv")
    policies.to_csv(ARTIFACT_DIR / "metrics" / "policies.csv", index=False)
    agreement.to_csv(ARTIFACT_DIR / "metrics" / "agreement.csv", index=False)
    ablation.to_csv(ARTIFACT_DIR / "metrics" / "ablation.csv", index=False)
    sensitivity.to_csv(ARTIFACT_DIR / "metrics" / "sensitivity.csv", index=False)
    pd.DataFrame({n: wb[0] for n, wb in weights.items()}, index=feature_names).to_csv(
        ARTIFACT_DIR / "metrics" / "coefficients.csv")
    with open(ARTIFACT_DIR / "metrics" / "loss_history.json", "w") as f:
        json.dump({"Manual PyTorch": manual_hist, "Standard PyTorch": standard_hist,
                   "scikit-learn optimum": objectives["scikit-learn"]}, f)
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump({"feature_names": feature_names, "C": C, "lambda": lam,
                   "manual_lr": manual_lr, "adam_lr": adam_lr, "n_epochs": N_EPOCHS,
                   "threshold": threshold, "break_even": break_even(),
                   "offer_months": OFFER_MONTHS, "months_kept": MONTHS_KEPT,
                   "save_rate": SAVE_RATE, "training_objectives": objectives}, f, indent=2)
    print(f"\nAll artifacts saved under {ARTIFACT_DIR}/.")
 
 
if __name__ == "__main__":
    main()