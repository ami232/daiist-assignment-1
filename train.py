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

How you structure the code beyond that — file layout, functions
boundaries, naming — is your call to make and be able to defend.
"""

"""
Assignment 1 — training pipeline.

Business framing (see REPORT.md): early-warning outreach. The collections team
calls flagged clients before the October due date; the decision threshold is
the one with the lowest total expected cost on the validation set.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/credit_default.csv")
ARTIFACTS = Path("artifacts")  # everything app.py needs is saved here
TARGET = "default"
RANDOM_STATE = 1

# Business assumptions (explained in REPORT.md)
CALL_COST = 50       # NT$ per outreach call
LOSS_SHARE = 0.5     # share of a defaulted bill the bank never recovers
CALL_SUCCESS = 0.2   # share of a defaulter's loss that a call prevents

# Monthly columns get readable month names, most recent first (Sep ... Apr).
# The original dataset numbers them PAY_0, PAY_2..PAY_6 / BILL_AMT1..6 / PAY_AMT1..6,
# where 0 or 1 = September and 6 = April.
MONTHS = ["sep", "aug", "jul", "jun", "may", "apr"]
ORIGINAL_PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
RENAME_MONTHS = {}
for i, month in enumerate(MONTHS):
    RENAME_MONTHS[ORIGINAL_PAY[i]] = f"months_late_{month}"   # repayment status
    RENAME_MONTHS[f"BILL_AMT{i + 1}"] = f"bill_{month}"        # bill amount
    RENAME_MONTHS[f"PAY_AMT{i + 1}"] = f"paid_{month}"         # amount paid

PAY_COLS = [f"months_late_{m}" for m in MONTHS]
BILL_COLS = [f"bill_{m}" for m in MONTHS]
PAID_COLS = [f"paid_{m}" for m in MONTHS]
SENSITIVE = ["SEX", "AGE", "MARRIAGE"]


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["ID"])
df = df.rename(columns={"default payment next month": TARGET, **RENAME_MONTHS})


# ---------------------------------------------------------------------------
# 2. Split 60 / 20 / 20, stratified on the target
# ---------------------------------------------------------------------------
# First cut: 60% train, 40% held out. Second cut: the 40% into two halves.
train_df, temp_df = train_test_split(
    df, test_size=0.4, stratify=df[TARGET], random_state=RANDOM_STATE
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df[TARGET], random_state=RANDOM_STATE
)

print("Split sizes and default rates:")
for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"  {name:5s}: {len(part):6,d} clients, default rate {part[TARGET].mean():.1%}")


# ---------------------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------------------
# These are fixed row-by-row rules: nothing is learned from the data, so
# applying them to every split cannot leak information between splits.
def clean(part):
    part = part.copy()

    # Categories are labels, not quantities, so give them names.
    # EDUCATION 0/5/6 and MARRIAGE 0 are undocumented -> their own "unknown" group.
    part["EDUCATION"] = part["EDUCATION"].map(
        {1: "graduate_school", 2: "university", 3: "high_school", 4: "other"}
    ).fillna("unknown")
    part["MARRIAGE"] = part["MARRIAGE"].map(
        {1: "married", 2: "single", 3: "other"}
    ).fillna("unknown")
    part["SEX"] = part["SEX"].map({1: "male", 2: "female"})

    # PAY codes: -2 = card not used, -1 = paid in full, 0 = paid part (revolving).
    # None of these is late, so all become 0 -> every months_late_* column
    # now really means "months late".
    part[PAY_COLS] = part[PAY_COLS].clip(lower=0)
    return part


# ---------------------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------------------
def add_features(part):
    part = part.copy()
    late = part[PAY_COLS] > 0  # True where the client was late that month

    # Track record: the most recent month (September) kept separate from the
    # earlier history, because a recent slip says more about October.
    # (A total "late months" count is left out: it equals late_sep +
    # late_months_apr_aug, so it would add nothing new.)
    part["late_sep"] = late["months_late_sep"].astype(int)
    part["late_months_apr_aug"] = late[PAY_COLS[1:]].sum(axis=1)

    # Credit usage: how full the card is in September (negative bills = 0).
    part["credit_usage"] = part["bill_sep"].clip(lower=0) / part["LIMIT_BAL"]

    # Payment ratio: share of the bills actually paid off.
    # Each month's payment settles the PREVIOUS month's bill, so payments
    # paid_sep..paid_may are matched with bills bill_aug..bill_apr.
    paid = part[PAID_COLS[:5]].to_numpy().sum(axis=1)
    owed = part[BILL_COLS[1:]].clip(lower=0).to_numpy().sum(axis=1)
    ratio = np.where(owed > 0, paid / np.maximum(owed, 1), 1.0)  # nothing owed = fully paid
    part["payment_ratio"] = np.clip(ratio, 0, 1)  # overpaying counts as paying in full

    # Debt trend: change in the bill from April to September, relative to the
    # credit limit so clients with big and small limits are comparable.
    part["debt_trend"] = (part["bill_sep"] - part["bill_apr"]) / part["LIMIT_BAL"]
    return part


train_df, val_df, test_df = (add_features(clean(p)) for p in (train_df, val_df, test_df))

ENGINEERED = [
    "late_sep", "late_months_apr_aug",
    "credit_usage", "payment_ratio", "debt_trend",
]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + PAID_COLS + ENGINEERED

# Two versions of the feature set, to compare with vs without sensitive columns
FEATURE_SETS = {
    "with_sensitive": (CATEGORICAL, NUMERIC),
    "without_sensitive": (
        [c for c in CATEGORICAL if c not in SENSITIVE],
        [c for c in NUMERIC if c not in SENSITIVE],
    ),
}


# ---------------------------------------------------------------------------
# 5. Encoding + scaling (fitted on TRAIN only, then applied to val and test)
# ---------------------------------------------------------------------------
ARTIFACTS.mkdir(exist_ok=True)
y_train = train_df[TARGET].to_numpy()
y_val = val_df[TARGET].to_numpy()
y_test = test_df[TARGET].to_numpy()

datasets = {}  # processed arrays for each feature set, used by the models in step 4
for version, (cat_cols, num_cols) in FEATURE_SETS.items():
    preprocessor = ColumnTransformer([
        # One yes/no column per category; drop="first" removes one redundant
        # column per variable (it is implied when all the others are 0).
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
        # Mean 0, standard deviation 1: puts NT$ amounts and month counts on
        # the same scale, which gradient descent in PyTorch needs.
        ("num", StandardScaler(), num_cols),
    ])
    X_train = preprocessor.fit_transform(train_df)  # learns means/categories from train
    X_val = preprocessor.transform(val_df)          # only applies them
    X_test = preprocessor.transform(test_df)

    datasets[version] = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "feature_names": list(preprocessor.get_feature_names_out()),
    }
    joblib.dump(preprocessor, ARTIFACTS / f"preprocessor_{version}.joblib")
    print(f"\nFeature set '{version}': {X_train.shape[1]} columns after encoding")


# ---------------------------------------------------------------------------
# 6. Business cost
# ---------------------------------------------------------------------------
def total_cost(y_true, flagged, bill):
    """Total expected cost in NT$ of a set of call decisions.

    y_true  : 1 if the client defaulted (would default if left alone), else 0
    flagged : 1 if the client gets a call, else 0
    bill    : September bill (bill_sep); negative bills count as zero loss
    """
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged)
    loss = np.maximum(np.asarray(bill), 0) * LOSS_SHARE

    call_costs = flagged * CALL_COST
    # A called defaulter loses (1 - CALL_SUCCESS) of the loss; an uncalled one loses all of it
    defaulter_losses = y_true * loss * np.where(flagged == 1, 1 - CALL_SUCCESS, 1.0)
    return float(np.sum(call_costs + defaulter_losses))


# ---------------------------------------------------------------------------
# 7. Naive baselines on the validation set
# ---------------------------------------------------------------------------
bill_val = val_df["bill_sep"].to_numpy()

cost_call_nobody = total_cost(y_val, np.zeros_like(y_val), bill_val)
cost_call_everyone = total_cost(y_val, np.ones_like(y_val), bill_val)

print("\nNaive baselines (validation set):")
print(f"  Call nobody:   NT$ {cost_call_nobody:,.0f}")
print(f"  Call everyone: NT$ {cost_call_everyone:,.0f}")

# ===========================================================================
# STEP 4 — Train logistic regression three ways, compare, save
# ===========================================================================
torch.manual_seed(RANDOM_STATE)
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)  # candidate cut-offs: 1%, 2%, ... 99%


def best_threshold(y_true, probs, bill):
    """Try every candidate threshold on a set; return the one with the lowest total cost."""
    costs = [total_cost(y_true, (probs >= t).astype(int), bill) for t in THRESHOLDS]
    i = int(np.argmin(costs))
    return float(THRESHOLDS[i]), float(costs[i])


# ---------------------------------------------------------------------------
# 8. Tune regularization (sklearn, validation set) and pick the feature set
# ---------------------------------------------------------------------------
# C = inverse regularization strength: small C = simpler model, large C = fits
# the training data more closely. Chosen by validation LOG LOSS (how good the
# predicted probabilities are): with unlimited calls the validation cost barely
# changes between settings, so it can't separate them; the threshold is then
# chosen by cost separately.
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
best = {}
print("\nRegularization tuning (validation log loss, lower = better):")
for version, d in datasets.items():
    scores = {}
    for C in C_VALUES:
        model = LogisticRegression(C=C, max_iter=5000).fit(d["X_train"], y_train)
        scores[C] = log_loss(y_val, model.predict_proba(d["X_val"])[:, 1])
    best_C = min(scores, key=scores.get)
    best[version] = (best_C, scores[best_C])
    print(f"  {version:18s} " + "  ".join(f"C={C}: {s:.4f}" for C, s in scores.items()))
    print(f"  {'':18s} -> best C = {best_C}")

# Rule: use the version WITHOUT sex/age/marital status unless including them
# lowers the validation log loss by more than 1% (a clear, real improvement).
with_loss, without_loss = best["with_sensitive"][1], best["without_sensitive"][1]
OFFICIAL = "without_sensitive" if without_loss <= with_loss * 1.01 else "with_sensitive"
C_BEST = best[OFFICIAL][0]
print(f"\nLog loss with sensitive columns {with_loss:.4f} vs without {without_loss:.4f}")
print(f"-> official feature set: {OFFICIAL} (C = {C_BEST})")

d = datasets[OFFICIAL]
X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
n_train, n_features = X_train.shape

# sklearn minimizes  C * sum(log loss) + 0.5 * ||w||^2  (bias not penalized).
# Dividing by C * n gives the same optimum as  mean(log loss) + 0.5 * LAMBDA * ||w||^2,
# which is what the two PyTorch versions minimize -> all three solve the SAME problem.
LAMBDA = 1.0 / (C_BEST * n_train)


# ---------------------------------------------------------------------------
# 9. Model A — scikit-learn
# ---------------------------------------------------------------------------
sk_model = LogisticRegression(C=C_BEST, max_iter=5000).fit(X_train, y_train)


# ---------------------------------------------------------------------------
# 10. Model B — PyTorch, manual (raw tensors, autograd, manual gradient step)
# ---------------------------------------------------------------------------
Xt_train = torch.tensor(X_train, dtype=torch.float32)
yt_train = torch.tensor(y_train, dtype=torch.float32)
EPOCHS = 3000  # full-batch gradient descent: every step uses all training clients


def train_manual(lr):
    w = torch.zeros(n_features, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for _ in range(EPOCHS):
        z = Xt_train @ w + b                                   # logits
        # log loss written out by hand; logsigmoid(z) = log(sigmoid(z)), but stable
        loss = -(yt_train * nn.functional.logsigmoid(z)
                 + (1 - yt_train) * nn.functional.logsigmoid(-z)).mean()
        loss = loss + 0.5 * LAMBDA * (w ** 2).sum()            # L2 penalty, weights only
        loss.backward()                                        # autograd fills w.grad, b.grad
        with torch.no_grad():                                  # the manual update step
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return w.detach(), b.detach(), loss.item()


# ---------------------------------------------------------------------------
# 11. Model C — PyTorch, standard (nn.Module + torch.optim)
# ---------------------------------------------------------------------------
class LogisticModel(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)  # logits; sigmoid is applied for probabilities


def train_standard(lr):
    model = LogisticModel(n_features)
    optimizer = torch.optim.SGD([
        {"params": [model.linear.weight], "weight_decay": LAMBDA},  # same L2 penalty
        {"params": [model.linear.bias], "weight_decay": 0.0},       # bias not penalized
    ], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(Xt_train), yt_train)
        loss.backward()
        optimizer.step()
    # Report the same objective as the manual version (log loss + L2 penalty),
    # so the two printed losses are directly comparable.
    with torch.no_grad():
        objective = loss_fn(model(Xt_train), yt_train) + 0.5 * LAMBDA * (model.linear.weight ** 2).sum()
    return model, objective.item()


# Learning rate: tried sparingly; keep the one that reaches the lowest training
# loss (best converged). It only affects HOW the optimum is found, not what it is.
LR_VALUES = [0.1, 0.5, 1.0]
print("\nLearning-rate tuning (final training objective, lower = better converged):")
manual_runs = {lr: train_manual(lr) for lr in LR_VALUES}
LR_BEST = min(manual_runs, key=lambda lr: manual_runs[lr][2])
print("  manual:   " + "  ".join(f"lr={lr}: {r[2]:.5f}" for lr, r in manual_runs.items()))
w_manual, b_manual, _ = manual_runs[LR_BEST]
torch.manual_seed(RANDOM_STATE)
std_model, std_loss = train_standard(LR_BEST)
print(f"  standard: lr={LR_BEST}: {std_loss:.5f}   -> chosen lr = {LR_BEST}")


# ---------------------------------------------------------------------------
# 12. Predictions, thresholds (validation) and final results (test)
# ---------------------------------------------------------------------------
def predict_proba(model_name, X):
    if model_name == "sklearn":
        return sk_model.predict_proba(X)[:, 1]
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        z = Xt @ w_manual + b_manual if model_name == "torch_manual" else std_model(Xt)
    return torch.sigmoid(z).numpy()


MODELS = ["sklearn", "torch_manual", "torch_standard"]
bill_test = test_df["bill_sep"].to_numpy()
results = {}

for name in MODELS:
    p_val, p_test = predict_proba(name, X_val), predict_proba(name, X_test)
    threshold, _ = best_threshold(y_val, p_val, bill_val)  # chosen on VALIDATION
    flagged = (p_test >= threshold).astype(int)              # applied to TEST
    results[name] = {
        "threshold": threshold,
        "test_cost": total_cost(y_test, flagged, bill_test),
        "called": int(flagged.sum()),
        "defaulters_caught": float(flagged[y_test == 1].mean()),
        "accuracy": float(accuracy_score(y_test, flagged)),
        "auc": float(roc_auc_score(y_test, p_test)),
        "log_loss": float(log_loss(y_test, p_test)),
    }

# Naive baselines on the same test set
for name, flagged in [("call_nobody", np.zeros_like(y_test)), ("call_everyone", np.ones_like(y_test))]:
    results[name] = {
        "threshold": None,
        "test_cost": total_cost(y_test, flagged, bill_test),
        "called": int(flagged.sum()),
        "defaulters_caught": float(flagged[y_test == 1].mean()),
        "accuracy": float(accuracy_score(y_test, flagged)),
        "auc": None, "log_loss": None,
    }

print(f"\nResults on the TEST set ({len(y_test):,} clients):")
print(f"  {'model':15s} {'threshold':>9s} {'cost (NT$)':>13s} {'called':>7s} {'caught':>7s} {'accuracy':>9s} {'AUC':>6s}")
for name, r in results.items():
    thr = f"{r['threshold']:.2f}" if r["threshold"] is not None else "-"
    auc = f"{r['auc']:.3f}" if r["auc"] is not None else "-"
    print(f"  {name:15s} {thr:>9s} {r['test_cost']:>13,.0f} {r['called']:>7,d} "
          f"{r['defaulters_caught']:>7.1%} {r['accuracy']:>9.1%} {auc:>6s}")

# Do the three implementations agree? Compare their predicted probabilities.
p = {name: predict_proba(name, X_test) for name in MODELS}
print("\nAgreement between implementations (test set):")
for a, b in [("sklearn", "torch_manual"), ("sklearn", "torch_standard"), ("torch_manual", "torch_standard")]:
    print(f"  {a} vs {b}: max probability difference {np.max(np.abs(p[a] - p[b])):.4f}")


# ---------------------------------------------------------------------------
# 13. Save everything app.py needs (it must never retrain)
# ---------------------------------------------------------------------------
joblib.dump(sk_model, ARTIFACTS / "model_sklearn.joblib")
torch.save({"w": w_manual, "b": b_manual}, ARTIFACTS / "model_torch_manual.pt")
torch.save(std_model.state_dict(), ARTIFACTS / "model_torch_standard.pt")

np.savez(ARTIFACTS / "test_data.npz", X_test=X_test, y_test=y_test, bill_test=bill_test)
test_df.to_csv(ARTIFACTS / "test_clients.csv", index=False)  # readable features for distributions

with open(ARTIFACTS / "results.json", "w") as f:
    json.dump({
        "official_feature_set": OFFICIAL,
        "feature_names": d["feature_names"],
        "C": C_BEST, "learning_rate": LR_BEST, "epochs": EPOCHS,
        "costs": {"call_cost": CALL_COST, "loss_share": LOSS_SHARE, "call_success": CALL_SUCCESS},
        "results": results,
    }, f, indent=2)
print(f"\nSaved models and test data to '{ARTIFACTS}/'")