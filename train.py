"""
Assignment 1 — training pipeline (Telco customer churn, logistic regression).

Run with:  uv run python main.py train

Steps:
  1. Load data/telco_churn.csv and clean it
  2. Engineer features
  3. Stratified 60/20/20 train/val/test split, scale numbers using train only
  4. Train logistic regression three ways (sklearn, manual PyTorch, nn.Module)
  5. Pick the decision threshold that minimises business cost on validation
  6. Evaluate everything on test, save models + data the app needs to artifacts/
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
ART = Path("artifacts")
ART.mkdir(exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Business costs from REPORT.md (euros per customer)
OFFER_COST = 150      # cost of the retention offer
MONTHS_LOST = 12      # a churner = 12 months of THEIR OWN bill lost
OFFER_SUCCESS = 0.5   # share of contacted churners the offer keeps


def business_cost(y_true, y_pred, monthly_bill):
    """Total euros lost for a set of yes/no 'contact this customer' decisions.

    Contacted customers always cost the offer; churners we don't keep cost
    12 months of their own monthly bill (so losing a €110/month customer
    hurts more than losing a €20/month one).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred).astype(int)
    loss = MONTHS_LOST * np.asarray(monthly_bill)
    contacted = y_pred * (OFFER_COST + y_true * (1 - OFFER_SUCCESS) * loss)
    not_contacted = (1 - y_pred) * y_true * loss
    return float(np.sum(contacted + not_contacted))


# ---------------------------------------------------------------------------
# 1. Load and clean
# ---------------------------------------------------------------------------
df = pd.read_csv("data/telco_churn.csv")
df = df.drop(columns="customerID")  # an ID carries no information about churn

# TotalCharges is stored as text; 11 rows are blank. All of them have
# tenure = 0 (brand-new customers not billed yet), so the true value is 0.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
service_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

# "No internet service" / "No phone service" just repeat what InternetService
# and PhoneService already say, so each service becomes a simple 1/0 flag.
for col in yes_no_cols + service_cols:
    df[col] = (df[col] == "Yes").astype(int)
df["gender_male"] = (df.pop("gender") == "Male").astype(int)

# (I first added num_services = sum of the 7 service flags. It is an exact
# linear combination of columns already in the model, so a linear model gains
# nothing and the coefficients stop being unique -> removed. See REPORT.md.)

# New feature 1: internet customer with neither security nor tech support
# (no one to call when something breaks -> frustration)
df["no_support"] = ((df["InternetService"] != "No")
                    & (df["OnlineSecurity"] == 0) & (df["TechSupport"] == 0)).astype(int)

# New feature 2: is the current bill higher than what they paid on average?
# TotalCharges ~= tenure x MonthlyCharges, so keeping it would just duplicate
# those two. Instead I use the gap between today's bill and the historical
# average bill (a price increase is a classic churn trigger).
avg_past_bill = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"])
df["bill_increase"] = df["MonthlyCharges"] - avg_past_bill
df = df.drop(columns="TotalCharges")

# MonthlyCharges is itself removed as a feature: the services explain 99.9% of it (R^2 =
# 0.9988, it's basically a price list), so keeping both gives the model two
# ways to say the same thing and the coefficients become unstable.
monthly_bill = df.pop("MonthlyCharges")  # kept aside: used by the business cost, not as a feature

# New feature 3: churn risk falls fast in the first months then flattens,
# so log(tenure) fits a linear model better than raw tenure.
df["log_tenure"] = np.log1p(df.pop("tenure"))

# Remaining text columns -> one-hot (drop_first avoids a redundant column)
df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True, dtype=int)

X = df.drop(columns="Churn")
y = df["Churn"].values
feature_names = list(X.columns)
print(f"{len(feature_names)} features: {feature_names}")

# ---------------------------------------------------------------------------
# 3. Split (stratified, no dates -> random split is safe) and scale
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)
print(f"train {len(X_train)}  val {len(X_val)}  test {len(X_test)}")
bill_val, bill_test = monthly_bill.loc[X_val.index].values, monthly_bill.loc[X_test.index].values

# Fit the scaler on train only so val/test stay unseen. Scaling matters for
# gradient descent: without it the big-valued columns dominate the updates.
scaler = StandardScaler().fit(X_train)
Xtr, Xva, Xte = (scaler.transform(d) for d in (X_train, X_val, X_test))

# ---------------------------------------------------------------------------
# 4a. scikit-learn — tune only C (regularisation) on validation log loss
# ---------------------------------------------------------------------------
best_C, best_loss = None, np.inf
for C in [0.01, 0.1, 1, 10]:
    m = LogisticRegression(C=C, max_iter=5000).fit(Xtr, y_train)
    loss = log_loss(y_val, m.predict_proba(Xva)[:, 1])
    print(f"  C={C:<5} val log loss {loss:.4f}")
    if loss < best_loss:
        best_C, best_loss = C, loss
sk_model = LogisticRegression(C=best_C, max_iter=5000).fit(Xtr, y_train)
print(f"sklearn: best C = {best_C}")

# sklearn minimises  C * sum(log loss) + 0.5 * ||w||^2.
# Dividing by C*n gives  mean(log loss) + (lambda/2) * ||w||^2  with
# lambda = 1 / (C * n). Using this lambda in PyTorch makes all three
# implementations minimise exactly the same function (bias not penalised).
LAMBDA = 1 / (best_C * len(Xtr))

Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
ytr_t = torch.tensor(y_train, dtype=torch.float32)
Xva_t = torch.tensor(Xva, dtype=torch.float32)
n_features = Xtr.shape[1]
EPOCHS = 3000


# ---------------------------------------------------------------------------
# 4b. Manual PyTorch loop — raw tensors, autograd, hand-written update
# ---------------------------------------------------------------------------
def train_manual(lr):
    w = torch.zeros(n_features, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for _ in range(EPOCHS):
        p = torch.sigmoid(Xtr_t @ w + b)
        p = p.clamp(1e-7, 1 - 1e-7)  # avoid log(0)
        loss = -(ytr_t * torch.log(p) + (1 - ytr_t) * torch.log(1 - p)).mean() + LAMBDA / 2 * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    return w.detach(), b.detach()


def manual_proba(w, b, X):
    return torch.sigmoid(torch.tensor(X, dtype=torch.float32) @ w + b).numpy()


# ---------------------------------------------------------------------------
# 4c. Standard PyTorch — nn.Module + torch.optim
# ---------------------------------------------------------------------------
class LogReg(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = torch.nn.Linear(n, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)  # raw score (logit); sigmoid applied in the loss


def train_standard(lr):
    torch.manual_seed(SEED)
    model = LogReg(n_features)
    # weight_decay = L2 penalty, applied to the weights only (like sklearn)
    opt = torch.optim.SGD([
        {"params": model.linear.weight, "weight_decay": LAMBDA},
        {"params": model.linear.bias, "weight_decay": 0.0},
    ], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
    return model


def standard_proba(model, X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


# Tune only the learning rate, on validation log loss
best = {}
for lr in [0.01, 0.1, 1.0]:
    w, b = train_manual(lr)
    l1 = log_loss(y_val, manual_proba(w, b, Xva))
    m = train_standard(lr)
    l2 = log_loss(y_val, standard_proba(m, Xva))
    print(f"  lr={lr:<5} val log loss  manual {l1:.4f}  standard {l2:.4f}")
    if l1 < best.get("manual", (np.inf,))[0]:
        best["manual"] = (l1, lr, (w, b))
    if l2 < best.get("standard", (np.inf,))[0]:
        best["standard"] = (l2, lr, m)

manual_lr, (w_manual, b_manual) = best["manual"][1], best["manual"][2]
standard_lr, std_model = best["standard"][1], best["standard"][2]
print(f"manual PyTorch: best lr = {manual_lr}   standard PyTorch: best lr = {standard_lr}")

# ---------------------------------------------------------------------------
# 5. Probabilities from each model + naive baseline
# ---------------------------------------------------------------------------
def all_probas(X):
    return {
        "sklearn": sk_model.predict_proba(X)[:, 1],
        "manual_pytorch": manual_proba(w_manual, b_manual, X),
        "standard_pytorch": standard_proba(std_model, X),
        # Naive baseline: every customer gets the training churn rate
        "baseline": np.full(len(X), y_train.mean()),
    }

val_p, test_p = all_probas(Xva), all_probas(Xte)

# Threshold that minimises business cost on VALIDATION (sklearn model;
# the other two give the same probabilities, checked below)
thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
val_costs = [business_cost(y_val, val_p["sklearn"] >= t, bill_val) for t in thresholds]
best_threshold = float(thresholds[int(np.argmin(val_costs))])
print(f"best threshold on validation: {best_threshold}")

# ---------------------------------------------------------------------------
# 6. Evaluate on TEST
# ---------------------------------------------------------------------------
rows = []
for name, p in test_p.items():
    rows.append({
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, p), 4),
        "log_loss": round(log_loss(y_test, p), 4),
        "accuracy@0.5": round(accuracy_score(y_test, p >= 0.5), 4),
        f"cost@{best_threshold} (€)": business_cost(y_test, p >= best_threshold, bill_test),
    })
results = pd.DataFrame(rows)
print("\nTest results:\n", results.to_string(index=False))

# Do the three implementations agree?
coefs = pd.DataFrame({
    "feature": feature_names,
    "sklearn": sk_model.coef_[0],
    "manual_pytorch": w_manual.numpy(),
    "standard_pytorch": std_model.linear.weight.detach().numpy()[0],
})
max_coef_diff = float(max((coefs["sklearn"] - coefs["manual_pytorch"]).abs().max(),
                          (coefs["sklearn"] - coefs["standard_pytorch"]).abs().max()))
max_prob_diff = float(max(np.abs(test_p["sklearn"] - test_p["manual_pytorch"]).max(),
                          np.abs(test_p["sklearn"] - test_p["standard_pytorch"]).max()))
print(f"\nmax |coef difference| vs sklearn: {max_coef_diff:.5f}")
print(f"max |probability difference| vs sklearn on test: {max_prob_diff:.5f}")
print("\nCoefficients (sorted by sklearn):\n",
      coefs.sort_values("sklearn").round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Save everything the app needs (it never retrains)
# ---------------------------------------------------------------------------
joblib.dump(sk_model, ART / "sklearn_model.joblib")
torch.save({"w": w_manual, "b": b_manual}, ART / "manual_pytorch.pt")
torch.save(std_model.state_dict(), ART / "standard_pytorch.pt")

test_df = pd.DataFrame(Xte, columns=feature_names)
test_df["Churn"] = y_test
test_df.to_csv(ART / "test_set_scaled.csv", index=False)       # model inputs
X_test.assign(MonthlyCharges=bill_test, Churn=y_test).to_csv(ART / "test_set_raw.csv", index=False)  # for readable plots
results.to_csv(ART / "results.csv", index=False)
coefs.to_csv(ART / "coefficients.csv", index=False)
json.dump({
    "best_C": best_C, "lambda": LAMBDA, "manual_lr": manual_lr, "standard_lr": standard_lr,
    "epochs": EPOCHS, "best_threshold": best_threshold, "n_features": n_features,
    "max_coef_diff": max_coef_diff, "max_prob_diff": max_prob_diff,
    "offer_cost": OFFER_COST, "months_lost": MONTHS_LOST, "offer_success": OFFER_SUCCESS,
}, open(ART / "config.json", "w"), indent=2)
print(f"\nSaved models and data to {ART}/")
