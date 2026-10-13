"""
Shared data loading, target definition and feature engineering.

Imported by both train.py and app.py so the dashboard sees exactly the same
transformations the models were trained on.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "brfss_2022_new_york.csv"
ARTIFACTS = ROOT / "artifacts"

SEED = 42

# --- Business framing (see REPORT.md) -------------------------------------
# NY State health department mails free HbA1c (glucose) test invitations.
# Budget covers tests for 10% of the surveyed population.
CAPACITY = 0.10
COST_WASTED_TEST = 30      # EUR: invitation + test for someone without diabetes
COST_MISSED_CASE = 1000    # EUR: undetected diabetes -> later, costlier complications

# --- Columns removed on purpose -------------------------------------------
# Reverse causation: these are (at least partly) *consequences* of already
# being diagnosed, not risk factors. A diagnosed diabetic sees a doctor
# yearly, gets vaccinated and scanned, and is often told to stop drinking.
# Keeping them teaches the model to recognise people who are already
# diagnosed, which is the opposite of what a screening campaign needs.
CONSEQUENCE_COLS = [
    "LastCheckupTime",
    "ChestScan",
    "FluVaxLast12",
    "PneumoVaxEver",
    "TetanusLast10Tdap",
    "HIVTesting",
    "AlcoholDrinkers",
]
# No plausible link to diabetes risk; kept out to avoid noise.
IRRELEVANT_COLS = ["CovidPos", "HighRiskLastYear"]

COMORBIDITIES = [
    "HadHeartAttack", "HadAngina", "HadStroke", "HadAsthma", "HadSkinCancer",
    "HadCOPD", "HadDepressiveDisorder", "HadKidneyDisease", "HadArthritis",
]
DIFFICULTIES = [
    "DeafOrHardOfHearing", "BlindOrVisionDifficulty", "DifficultyConcentrating",
    "DifficultyWalking", "DifficultyDressingBathing", "DifficultyErrands",
]

AGE_MIDPOINT = {
    "Age 18 to 24": 21, "Age 25 to 29": 27, "Age 30 to 34": 32,
    "Age 35 to 39": 37, "Age 40 to 44": 42, "Age 45 to 49": 47,
    "Age 50 to 54": 52, "Age 55 to 59": 57, "Age 60 to 64": 62,
    "Age 65 to 69": 67, "Age 70 to 74": 72, "Age 75 to 79": 77,
    "Age 80 or older": 82,
}
GENERAL_HEALTH = {"Poor": 1, "Fair": 2, "Good": 3, "Very good": 4, "Excellent": 5}
REMOVED_TEETH = {"None of them": 0, "1 to 5": 1, "6 or more, but not all": 2, "All": 3}


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnosed diabetes = 1, no diabetes = 0.

    Dropped: pre-diabetes (not diabetes, but not a clean "no" either),
    pregnancy-only diabetes (gestational, a different condition) and the
    few non-answers.
    """
    df = df[df["HadDiabetes"].isin(["Yes", "No"])].copy()
    df["target"] = (df["HadDiabetes"] == "Yes").astype(int)
    return df.drop(columns="HadDiabetes").reset_index(drop=True)


def _yes(s: pd.Series) -> pd.Series:
    """Yes -> 1, No -> 0, missing stays NaN."""
    return s.map({"Yes": 1.0, "No": 0.0})


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering. Uses no statistics from other rows, so
    it can't leak test information into training."""
    out = pd.DataFrame(index=df.index)

    # Age: ordered bands -> numeric midpoint, plus a squared term because
    # risk rises with age and then flattens/drops after 80. The square is
    # centred (around a fixed 50, not a data statistic, so nothing leaks):
    # raw age and age**2 are ~0.99 correlated, which makes gradient descent
    # crawl; (age-50)**2 carries the same curvature without that overlap.
    age = df["AgeCategory"].map(AGE_MIDPOINT)
    out["age"] = age
    out["age_sq"] = (age - 50) ** 2

    # BMI: recompute from height/weight when BMI itself is missing, then
    # treat physically implausible values as missing. Risk climbs faster at
    # high BMI, so a squared term and the obesity cut-off (30) are added.
    bmi = df["BMI"].copy()
    recomputed = df["WeightInKilograms"] / df["HeightInMeters"] ** 2
    bmi = bmi.fillna(recomputed)
    bmi = bmi.where(bmi.between(14, 70))
    out["bmi"] = bmi
    out["bmi_sq"] = (bmi - 27) ** 2   # centred for the same reason as age
    out["obese"] = (bmi >= 30).astype(float).where(bmi.notna())

    # Ordinal answers -> numbers that keep their order.
    out["general_health"] = df["GeneralHealth"].map(GENERAL_HEALTH)
    out["removed_teeth"] = df["RemovedTeeth"].map(REMOVED_TEETH)

    # Sleep: both short and long sleep are linked to metabolic risk
    # (U-shape), which a single linear term can't express.
    sleep = df["SleepHours"].where(df["SleepHours"].between(3, 14))
    out["sleep_hours"] = sleep
    out["short_sleep"] = (sleep < 6).astype(float).where(sleep.notna())
    out["long_sleep"] = (sleep > 9).astype(float).where(sleep.notna())

    # CDC "frequent distress": 14+ bad days in the last 30.
    out["frequent_physical_distress"] = (df["PhysicalHealthDays"] >= 14).astype(float).where(df["PhysicalHealthDays"].notna())
    out["frequent_mental_distress"] = (df["MentalHealthDays"] >= 14).astype(float).where(df["MentalHealthDays"].notna())

    out["physically_active"] = _yes(df["PhysicalActivities"])

    # Comorbidities. Heart and kidney disease are themselves ADA screening
    # criteria, so they are legitimate inputs. No "count of conditions"
    # feature: a linear model can already add these columns up itself, so a
    # sum is redundant (and makes the weights non-unique). "Any of heart
    # attack / angina / stroke" is a max, which a linear model can't build.
    for col in COMORBIDITIES:
        out[col] = _yes(df[col])
    out["cardiovascular_history"] = out[["HadHeartAttack", "HadAngina", "HadStroke"]].max(axis=1)

    for col in DIFFICULTIES:
        out[col] = _yes(df[col])

    # Nominal categories: missing becomes its own level.
    for col in ["Sex", "RaceEthnicityCategory", "SmokerStatus", "ECigaretteUsage"]:
        out[col] = df[col].fillna("Missing")

    return out


CATEGORICAL = ["Sex", "RaceEthnicityCategory", "SmokerStatus", "ECigaretteUsage"]
# One level per category is dropped as the reference. Without this the
# dummies of each category always sum to 1, duplicating the intercept: many
# weight vectors then give identical predictions, and sklearn and gradient
# descent settle on different ones. References are chosen to read naturally
# ("compared with a never-smoker").
REFERENCE_LEVELS = ["Female", "White only, Non-Hispanic", "Never smoked",
                    "Never used e-cigarettes in my entire life"]
DERIVED = ["age_sq", "bmi_sq", "obese", "short_sleep", "long_sleep"]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Learned preprocessing (median, mean/std, category list).

    Must only ever be fitted on training rows. Numeric NaNs get the training
    median plus an indicator column, so "didn't answer" stays visible to the
    model.
    """
    # Derived columns (age_sq, bmi_sq, ...) are missing exactly when their base
    # column is, so their "was missing" indicator would be an identical
    # duplicate column. They get imputed but no indicator of their own.
    derived = [c for c in DERIVED if c in X.columns]
    base = [c for c in X.columns if c not in CATEGORICAL and c not in derived]
    cat_pipe = OneHotEncoder(handle_unknown="ignore", drop=REFERENCE_LEVELS, sparse_output=False)
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), base),
            ("derived", Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), derived),
            ("cat", cat_pipe, CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )


def raw_features(df: pd.DataFrame, include_consequences: bool = False) -> pd.DataFrame:
    """Minimal-effort feature set used only for the ablation comparison:
    original columns, one-hot everything that is text, no engineering."""
    drop = ["target", *IRRELEVANT_COLS, "WeightInKilograms", "HeightInMeters"]
    if not include_consequences:
        drop += CONSEQUENCE_COLS
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return X


def build_raw_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes("number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])


def business_cost(y_true: np.ndarray, selected: np.ndarray) -> dict:
    """Confusion matrix + EUR cost for a given set of invited people."""
    y_true = np.asarray(y_true).astype(bool)
    selected = np.asarray(selected).astype(bool)
    tp = int((selected & y_true).sum())
    fp = int((selected & ~y_true).sum())
    fn = int((~selected & y_true).sum())
    tn = int((~selected & ~y_true).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "invited": tp + fp,
        "invited_share": (tp + fp) / len(y_true),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "cost_eur": fp * COST_WASTED_TEST + fn * COST_MISSED_CASE,
    }
