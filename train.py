"""
Assignment 1 — training pipeline: Madrid Airbnb "fair price" model.

Business use: an overpricing detector for entire-home listings in Madrid.
The model predicts the fair nightly price of a listing from what it is
(location, size, amenities, quality, rules). A listing whose real price is
far above that prediction gets flagged for review.

Run with:  uv run python main.py train

Stages (each one is a section below):
  1. Load the committed raw data
  2. Scope + target  (entire homes, target = log(price))
  3. Feature engineering
  4. Split by host (train / validation / test)
  5. Preprocessing (fit on train only)
  6. Naive baseline
  7. Model A: scikit-learn Ridge
  8. Model B: manual PyTorch loop (Session 5 style)
  9. Model C: standard PyTorch (nn.Module + torch.optim)
 10. Compare the three + save everything the dashboard needs
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "listings.csv.gz"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Puerta del Sol, the "kilometre zero" of Madrid
SOL_LAT, SOL_LON = 40.4169, -3.7035


# ---------------------------------------------------------------------------
# 1. Load the committed raw data
# ---------------------------------------------------------------------------
raw = pd.read_csv(DATA_PATH)
print(f"[1] raw data: {raw.shape[0]:,} listings x {raw.shape[1]} columns")


# ---------------------------------------------------------------------------
# 2. Scope + target
# ---------------------------------------------------------------------------
# Scope: entire homes only (a room and a whole flat follow different price logic).
df = raw[raw["room_type"] == "Entire home/apt"].copy()

# Price comes as text like "$1,234.00" -> float (the currency is actually EUR).
df["price_eur"] = (
    df["price"].astype(str).str.replace(r"[$,]", "", regex=True).replace("nan", np.nan).astype(float)
)
n_before = len(df)
df = df[df["price_eur"].notna() & (df["price_eur"] > 0)].copy()
print(f"[2] entire homes: {n_before:,} -> {len(df):,} with a usable price "
      f"(dropped {n_before - len(df):,} without a price)")

# Decision: keep ALL prices (including extreme ones like EUR 15,000/night).
# Target = log(price): prices are very right-skewed, and working in logs
# means the model's error is a PERCENTAGE error, which matches the business
# rule "flag if X% above fair price".
df["log_price"] = np.log(df["price_eur"])


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def amenity_flag(amenities: pd.Series, keywords: list[str]) -> pd.Series:
    """1 if ANY amenity label contains ANY keyword (case-insensitive).
    Needed because the same amenity appears under many labels, e.g.
    'Air conditioning' vs 'AC - split type ductless system'."""
    lowered = amenities.str.lower()
    hit = pd.Series(False, index=amenities.index)
    for kw in keywords:
        hit |= lowered.str.contains(kw.lower(), regex=False)
    return hit.astype(int)


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)

    # --- Location -----------------------------------------------------------
    # Straight-line km to Puerta del Sol (1 deg lat ~ 111 km, 1 deg lon ~ 85 km in Madrid).
    dist = np.sqrt(((d["latitude"] - SOL_LAT) * 111) ** 2 + ((d["longitude"] - SOL_LON) * 85) ** 2)
    f["log_dist_to_sol_km"] = np.log1p(dist)  # log: the first km matter more than the 10th
    f["district"] = d["neighbourhood_group_cleansed"]  # 21 districts -> one-hot later

    # --- Size / capacity ----------------------------------------------------
    f["accommodates"] = d["accommodates"]
    f["bedrooms"] = d["bedrooms"]
    f["beds"] = d["beds"]
    # 'bathrooms' is missing for ~9% of rows but 'bathrooms_text' ("1.5 baths") often has it
    baths_from_text = pd.to_numeric(
        d["bathrooms_text"].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce"
    )
    baths_from_text[d["bathrooms_text"].astype(str).str.lower().str.contains("half")] = 0.5
    f["bathrooms"] = d["bathrooms"].fillna(baths_from_text)

    # --- Property type (the rare ones grouped into "Other") ----------------
    ptype = d["property_type"].fillna("Other")
    common = ["Entire rental unit", "Entire condo", "Entire loft", "Entire serviced apartment", "Entire home"]
    f["property_type"] = ptype.where(ptype.isin(common), "Other")

    # --- Amenities ------------------------------------------------------------
    am = d["amenities"].fillna("[]")
    f["n_amenities"] = am.apply(lambda s: len(json.loads(s)))
    f["has_air_conditioning"] = amenity_flag(am, ["air conditioning", '"ac -', "central air"])
    f["has_dishwasher"] = amenity_flag(am, ["dishwasher"])
    f["has_elevator"] = amenity_flag(am, ["elevator"])
    f["has_balcony_terrace"] = amenity_flag(am, ["balcony", "patio", "terrace", "rooftop"])
    f["has_pool"] = amenity_flag(am, ["pool"])
    f["has_parking"] = amenity_flag(am, ["free parking", "paid parking on premises"])
    f["has_washer"] = amenity_flag(am, ["washer"])
    f["has_workspace"] = amenity_flag(am, ["workspace"])
    f["has_self_checkin"] = amenity_flag(am, ["self check-in"])

    # --- Rules / legal status ----------------------------------------------
    # Monthly rentals (32+ night minimum) are a different market in Madrid
    f["is_monthly_rental"] = (d["minimum_nights"] >= 32).astype(int)
    f["log_minimum_nights"] = np.log1p(d["minimum_nights"])
    lic = d["license"].astype(str).str.lower()
    f["has_license"] = (d["license"].notna() & ~lic.str.contains("exempt|proceso")).astype(int)

    # --- Host ---------------------------------------------------------------
    f["is_superhost"] = (d["host_is_superhost"] == "t").astype(int)
    # Professional operators (many listings) price differently from individuals
    f["log_host_listings"] = np.log1p(d["calculated_host_listings_count"])

    # --- Reviews / quality ---------------------------------------------------
    f["has_reviews"] = (d["number_of_reviews"] > 0).astype(int)
    f["log_n_reviews"] = np.log1p(d["number_of_reviews"])
    f["reviews_per_month"] = d["reviews_per_month"].fillna(0)  # no reviews -> 0 per month
    # Scores are missing exactly when there are no reviews: imputed later
    # (median of TRAIN) and 'has_reviews' tells the model they were missing.
    for col in ["review_scores_rating", "review_scores_cleanliness",
                "review_scores_location", "review_scores_value"]:
        f[col] = d[col]

    # --- Demand signals (decision: included) -------------------------------
    # NOTE: 'estimated_revenue_l365d' is deliberately NOT used: it is computed
    # as price x occupancy, i.e. it contains the target itself (leakage).
    f["availability_365"] = d["availability_365"]
    f["estimated_occupancy_l365d"] = d["estimated_occupancy_l365d"]
    return f


X_all = build_features(df)
y_all = df["log_price"].to_numpy(dtype=np.float32)
groups = df["host_id"].to_numpy()

CATEGORICAL = ["district", "property_type"]
NUMERIC = [c for c in X_all.columns if c not in CATEGORICAL]
print(f"[3] engineered {len(NUMERIC)} numeric + {len(CATEGORICAL)} categorical features")


# ---------------------------------------------------------------------------
# 4. Split BY HOST: 70% train / 15% validation / 15% test
# ---------------------------------------------------------------------------
# One host can have 300+ near-identical flats. A random split would put copies
# of the same flat in train and test and make the test score too optimistic.
# GroupShuffleSplit keeps all listings of a host on the same side.
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, temp_idx = next(gss.split(X_all, y_all, groups))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_rel, test_rel = next(gss2.split(X_all.iloc[temp_idx], y_all[temp_idx], groups[temp_idx]))
val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

assert not set(groups[train_idx]) & set(groups[test_idx]), "a host is in both train and test!"
print(f"[4] split by host -> train {len(train_idx):,} | val {len(val_idx):,} | test {len(test_idx):,}")

X_tr_df, X_va_df, X_te_df = X_all.iloc[train_idx], X_all.iloc[val_idx], X_all.iloc[test_idx]
y_tr, y_va, y_te = y_all[train_idx], y_all[val_idx], y_all[test_idx]


# ---------------------------------------------------------------------------
# 5. Preprocessing — fitted on TRAIN only, then applied to val/test
# ---------------------------------------------------------------------------
# Numeric: fill missing with the train median, then standardise (mean 0, sd 1).
# Standardising matters for gradient descent: with raw scales (availability up
# to 365 vs. flags 0/1) one learning rate cannot suit every weight.
# Categorical: one-hot; categories unseen in train become all-zeros.
preprocessor = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), CATEGORICAL),
])
X_tr = preprocessor.fit_transform(X_tr_df).astype(np.float32)
X_va = preprocessor.transform(X_va_df).astype(np.float32)
X_te = preprocessor.transform(X_te_df).astype(np.float32)
feature_names = [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]
n_features = X_tr.shape[1]
print(f"[5] model matrix: {n_features} columns after one-hot")


def evaluate(y_true_log, y_pred_log) -> dict:
    """Metrics on the log scale (what the model optimises) and in euros."""
    eur_true, eur_pred = np.exp(y_true_log), np.exp(y_pred_log)
    return {
        "RMSE_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "R2_log": float(r2_score(y_true_log, y_pred_log)),
        "MAE_eur": float(mean_absolute_error(eur_true, eur_pred)),
        "MedianAPE_pct": float(np.median(np.abs(eur_pred - eur_true) / eur_true) * 100),
    }


# ---------------------------------------------------------------------------
# 6. Naive baseline: predict the average train log-price for every listing
# ---------------------------------------------------------------------------
baseline_value = float(y_tr.mean())
pred_base_te = np.full_like(y_te, baseline_value)
print(f"[6] naive baseline predicts EUR {np.exp(baseline_value):.0f} for everyone")


# ---------------------------------------------------------------------------
# 7. Model A — scikit-learn Ridge (linear regression + L2 regularisation)
# ---------------------------------------------------------------------------
# Ridge minimises  sum((y - Xw - b)^2) + alpha * sum(w^2).
# Tuned sparingly: a handful of alphas, chosen on VALIDATION, never on test.
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
val_rmse_by_alpha = {}
for a in alphas:
    m = Ridge(alpha=a).fit(X_tr, y_tr)
    val_rmse_by_alpha[a] = evaluate(y_va, m.predict(X_va))["RMSE_log"]
best_alpha = min(val_rmse_by_alpha, key=val_rmse_by_alpha.get)
sk_model = Ridge(alpha=best_alpha).fit(X_tr, y_tr)
pred_sk_te = sk_model.predict(X_te).astype(np.float32)
print(f"[7] sklearn Ridge: val RMSE by alpha {val_rmse_by_alpha} -> best alpha={best_alpha}")

# Did the feature engineering help? Same Ridge on only the "obvious" raw
# columns (size + raw coordinates), same split, scored on VALIDATION.
raw_cols = ["accommodates", "bedrooms", "beds", "latitude", "longitude"]
raw_imp = SimpleImputer(strategy="median").fit(df.iloc[train_idx][raw_cols])
raw_scaler = StandardScaler().fit(raw_imp.transform(df.iloc[train_idx][raw_cols]))
prep_raw = lambda idx: raw_scaler.transform(raw_imp.transform(df.iloc[idx][raw_cols]))
raw_ridge = Ridge(alpha=best_alpha).fit(prep_raw(train_idx), y_tr)
ablation = {
    "raw_columns_only_val_RMSE": evaluate(y_va, raw_ridge.predict(prep_raw(val_idx)))["RMSE_log"],
    "engineered_features_val_RMSE": val_rmse_by_alpha[best_alpha],
}
print(f"    ablation (val RMSE): raw columns only {ablation['raw_columns_only_val_RMSE']:.4f} "
      f"vs engineered features {ablation['engineered_features_val_RMSE']:.4f}")

# To train the SAME model in PyTorch we need the SAME objective. PyTorch uses
# the MEAN squared error, so dividing Ridge's objective by n gives:
#     MSE + (alpha / n) * sum(w^2)
# -> the L2 strength for PyTorch is lambda = alpha / n_train (bias not penalised).
l2_lambda = best_alpha / len(y_tr)

Xtr_t, ytr_t = torch.tensor(X_tr), torch.tensor(y_tr)
Xva_t, Xte_t = torch.tensor(X_va), torch.tensor(X_te)


# ---------------------------------------------------------------------------
# 8. Model B — manual PyTorch loop (exactly the Session 5 pattern)
# ---------------------------------------------------------------------------
def mse_loss(y_pred, y):
    return ((y_pred - y) ** 2).mean()


def train_manual(lr: float, n_epochs: int):
    torch.manual_seed(SEED)
    w = torch.zeros(n_features, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for epoch in range(n_epochs):
        y_pred = Xtr_t @ w + b                                   # forward pass
        loss = mse_loss(y_pred, ytr_t) + l2_lambda * (w ** 2).sum()  # loss (+ L2)
        loss.backward()                                          # autograd
        with torch.no_grad():                                    # manual update
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                                           # reset gradients
        b.grad.zero_()
        history.append(loss.item())
    return w.detach(), b.detach(), history


N_EPOCHS = 3000
lrs = [0.01, 0.05, 0.1]
manual_runs = {}
for lr in lrs:
    w_, b_, h_ = train_manual(lr, N_EPOCHS)
    manual_runs[lr] = (w_, b_, h_, evaluate(y_va, (Xva_t @ w_ + b_).numpy())["RMSE_log"])
best_lr_manual = min(manual_runs, key=lambda k: manual_runs[k][3])
w_manual, b_manual, hist_manual, _ = manual_runs[best_lr_manual]
pred_manual_te = (Xte_t @ w_manual + b_manual).numpy()
print(f"[8] manual PyTorch: val RMSE by lr "
      f"{ {k: round(v[3], 4) for k, v in manual_runs.items()} } -> best lr={best_lr_manual}")


# ---------------------------------------------------------------------------
# 9. Model C — standard PyTorch: nn.Module + torch.optim
# ---------------------------------------------------------------------------
class LinearRegressionModel(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_standard(lr: float, n_epochs: int):
    torch.manual_seed(SEED)
    model = LinearRegressionModel(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(Xtr_t)
        # same objective as sklearn/manual: MSE + lambda * sum(w^2), bias not penalised
        loss = loss_fn(y_pred, ytr_t) + l2_lambda * (model.linear.weight ** 2).sum()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return model, history


std_runs = {}
for lr in lrs:
    m_, h_ = train_standard(lr, N_EPOCHS)
    with torch.no_grad():
        std_runs[lr] = (m_, h_, evaluate(y_va, m_(Xva_t).numpy())["RMSE_log"])
best_lr_std = min(std_runs, key=lambda k: std_runs[k][2])
std_model, hist_std, _ = std_runs[best_lr_std]
with torch.no_grad():
    pred_std_te = std_model(Xte_t).numpy()
print(f"[9] standard PyTorch (Adam): val RMSE by lr "
      f"{ {k: round(v[2], 4) for k, v in std_runs.items()} } -> best lr={best_lr_std}")


# ---------------------------------------------------------------------------
# 10. Compare + save everything the dashboard needs
# ---------------------------------------------------------------------------
results = {
    "Naive baseline (mean log-price)": evaluate(y_te, pred_base_te),
    "scikit-learn Ridge": evaluate(y_te, pred_sk_te),
    "Manual PyTorch loop": evaluate(y_te, pred_manual_te),
    "Standard PyTorch (nn + optim)": evaluate(y_te, pred_std_te),
}
results_df = pd.DataFrame(results).T
print("\n[10] TEST-SET RESULTS")
print(results_df.round(4).to_string())

# Do the three implementations agree?
coef = pd.DataFrame({
    "sklearn": sk_model.coef_,
    "manual_torch": w_manual.numpy(),
    "standard_torch": std_model.linear.weight.detach().numpy().ravel(),
}, index=feature_names)
intercepts = {"sklearn": float(sk_model.intercept_), "manual_torch": float(b_manual.item()),
              "standard_torch": float(std_model.linear.bias.item())}
agreement = {
    "max_abs_pred_diff_manual_vs_sklearn": float(np.abs(pred_manual_te - pred_sk_te).max()),
    "max_abs_pred_diff_standard_vs_sklearn": float(np.abs(pred_std_te - pred_sk_te).max()),
    "max_abs_coef_diff_manual_vs_sklearn": float((coef.manual_torch - coef.sklearn).abs().max()),
    "max_abs_coef_diff_standard_vs_sklearn": float((coef.standard_torch - coef.sklearn).abs().max()),
}
print("\nAgreement between implementations (0 = identical):")
for k, v in agreement.items():
    print(f"  {k}: {v:.5f}")
print("\nLargest coefficients (sklearn), in log-price units per 1 SD / category:")
print(coef.reindex(coef.sklearn.abs().sort_values(ascending=False).index).head(12).round(4).to_string())

# --- Save the trained models (the app only LOADS these) --------------------
joblib.dump(preprocessor, ARTIFACTS / "preprocessor.joblib")
joblib.dump(sk_model, ARTIFACTS / "sklearn_ridge.joblib")
torch.save({"w": w_manual, "b": b_manual}, ARTIFACTS / "manual_torch.pt")
torch.save(std_model.state_dict(), ARTIFACTS / "standard_torch.pt")

# --- Save data the dashboard displays --------------------------------------
test_out = df.iloc[test_idx][["id", "name", "neighbourhood_group_cleansed", "neighbourhood_cleansed",
                              "accommodates", "bedrooms", "price_eur", "latitude", "longitude"]].copy()
test_out["pred_baseline_eur"] = np.exp(pred_base_te)
test_out["pred_sklearn_eur"] = np.exp(pred_sk_te)
test_out["pred_manual_eur"] = np.exp(pred_manual_te)
test_out["pred_standard_eur"] = np.exp(pred_std_te)
test_out.to_csv(ARTIFACTS / "test_predictions.csv", index=False)

train_view = X_tr_df.copy()
train_view["price_eur"] = np.exp(y_tr)
train_view.to_csv(ARTIFACTS / "train_features.csv", index=False)
# Raw (un-preprocessed) test features: the app pushes these through the saved
# preprocessor + saved models itself, so what it shows comes from the models on disk.
test_view = X_te_df.copy()
test_view["price_eur"] = np.exp(y_te)
test_view.to_csv(ARTIFACTS / "test_features.csv", index=False)

coef.to_csv(ARTIFACTS / "coefficients.csv")
with open(ARTIFACTS / "metrics.json", "w") as fh:
    json.dump({
        "test_results": results,
        "agreement": agreement,
        "feature_engineering_ablation": ablation,
        "intercepts": intercepts,
        "chosen_hyperparameters": {"ridge_alpha": best_alpha, "l2_lambda_torch": l2_lambda,
                                   "lr_manual": best_lr_manual, "lr_standard": best_lr_std,
                                   "epochs": N_EPOCHS},
        "val_rmse_by_alpha": {str(k): v for k, v in val_rmse_by_alpha.items()},
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "loss_history": {"manual_torch": hist_manual, "standard_torch": hist_std},
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
    }, fh)
print(f"\nSaved models + dashboard data to {ARTIFACTS.relative_to(ROOT)}/")
