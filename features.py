"""
Shared feature engineering for train.py and app.py.

Everything that turns a raw Telco row into model inputs lives here, so the
dashboard applies exactly the transformation the models were trained on.
The only *fitted* step (one-hot categories + standardisation statistics)
is learned from the training split alone and saved by train.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
ARTIFACTS = ROOT / "artifacts"

TARGET = "Churn"
SEED = 42

# Business costs (see REPORT.md, "Business / real-life framing").
# Every customer flagged as a churn risk gets a retention offer; a churner we
# miss walks away with their future revenue.
OFFER_COST = 100.0   # € per retention offer sent (TP or FP)
CHURN_COST = 500.0   # € of lost future margin per missed churner (FN)

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_FEATURES = [
    "tenure", "log_tenure", "MonthlyCharges", "log_total_charges",
    "price_change", "n_addons", "n_protection",
]
BINARY_FEATURES = [
    "SeniorCitizen", "has_partner", "has_dependents", "paperless",
    "is_new_customer", "fiber_month_to_month",
    "single_service",
]
CATEGORICAL_FEATURES = [
    "Contract", "InternetService", "PaymentMethod", "MultipleLines",
    "tenure_band",
]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Data-quality fixes that do not depend on any statistic of the data."""
    df = df.copy()
    # 11 customers have TotalCharges = " ". All of them have tenure 0: they
    # signed up but have not been billed yet, so the correct value is 0,
    # not a mean/median imputation.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    # "No internet service" / "No phone service" duplicate information that is
    # already in InternetService / PhoneService. Collapsing them to "No" avoids
    # perfectly collinear dummy columns.
    for col in ADDON_COLS:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise engineered features (no information from other rows)."""
    df = clean(df)
    out = pd.DataFrame(index=df.index)

    # Tenure: churn risk drops very fast in the first months and then flattens,
    # which a linear model can only follow through a log transform + bands.
    out["tenure"] = df["tenure"]
    out["log_tenure"] = np.log1p(df["tenure"])
    out["tenure_band"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"],
    ).astype(str)
    out["is_new_customer"] = (df["tenure"] <= 1).astype(int)

    # Charges: TotalCharges is ~ tenure x MonthlyCharges, so it is mostly
    # redundant. What is *not* redundant is how today's bill compares with the
    # customer's historical average bill: a positive gap means a recent price
    # increase / upsell, a classic churn trigger.
    out["MonthlyCharges"] = df["MonthlyCharges"]
    out["log_total_charges"] = np.log1p(df["TotalCharges"])
    avg_bill = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1), df["MonthlyCharges"])
    out["price_change"] = df["MonthlyCharges"] - avg_bill

    # Product engagement: customers with more add-ons are more "locked in";
    # security/tech-support add-ons in particular signal a support relationship.
    out["n_addons"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    out["n_protection"] = (df[["OnlineSecurity", "TechSupport"]] == "Yes").sum(axis=1)
    # Phone-only or internet-only, with no add-ons: nothing ties them to us.
    one_product = (df["PhoneService"] == "Yes") != (df["InternetService"] != "No")
    out["single_service"] = (one_product & (out["n_addons"] == 0)).astype(int)

    # Payment & contract behaviour. (An "auto_pay" flag was tried and removed:
    # it is exactly 1 - ElectronicCheck - MailedCheck, i.e. a linear
    # combination of the PaymentMethod dummies + intercept, which made the
    # optimum non-unique and the three implementations disagree.)
    out["paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)
    # Interaction: fibre customers on month-to-month contracts are by far the
    # riskiest segment (expensive product, no lock-in). A linear model cannot
    # discover this interaction on its own.
    out["fiber_month_to_month"] = (
        (df["InternetService"] == "Fiber optic") & (df["Contract"] == "Month-to-month")
    ).astype(int)

    # Demographics. Gender is deliberately excluded: it carries no signal here
    # and the retention team should not target offers on it.
    out["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    out["has_partner"] = (df["Partner"] == "Yes").astype(int)
    out["has_dependents"] = (df["Dependents"] == "Yes").astype(int)

    for col in ["Contract", "InternetService", "PaymentMethod", "MultipleLines"]:
        out[col] = df[col]
    return out


def target(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == "Yes").astype(int).to_numpy()


def make_preprocessor() -> ColumnTransformer:
    """Fitted on the training split only (scaling stats + category levels)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def business_cost(y_true, y_pred, offer_cost=OFFER_COST, churn_cost=CHURN_COST) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_offers = int(y_pred.sum())
    n_missed = int(((y_true == 1) & (y_pred == 0)).sum())
    return offer_cost * n_offers + churn_cost * n_missed
