"""
Shared pieces used by both train.py and app.py: data loading, feature
engineering, preprocessing and the torch model definition. Keeping them in
one place guarantees the app transforms data exactly like training did.

Decision supported: at the END OF A STUDENT'S 1ST SEMESTER, decide who is
offered a mentoring programme to prevent dropout.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "students.csv"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

SEED = 42
COST_FN = 3000  # missed dropout: remaining tuition revenue lost
COST_FP = 300   # mentoring offered to a student who would have graduated anyway

# The three macro columns take only 10 distinct combinations, and they match
# Portugal's yearly unemployment / inflation / GDP growth for 2008-2017, so
# they identify the student's enrollment cohort.
UNEMPLOYMENT_TO_COHORT = {
    7.6: 2008, 9.4: 2009, 10.8: 2010, 12.7: 2011, 15.5: 2012,
    16.2: 2013, 13.9: 2014, 12.4: 2015, 11.1: 2016, 8.9: 2017,
}
TRAIN_COHORTS = list(range(2008, 2014))  # 2008-2013
VAL_COHORTS = [2014, 2015]               # C / threshold selection
TEST_COHORTS = [2016, 2017]              # untouched until the end

# Parent qualification codes (UCI data dictionary) that mean higher education.
HIGHER_ED_CODES = {2, 3, 4, 5, 6, 40, 41, 42, 43, 44}

NUMERIC = [
    "s1_enrolled", "s1_approval_rate", "s1_failed_units", "s1_evals_per_unit",
    "s1_grade", "s1_units_without_eval", "admission_grade", "prev_qual_grade",
    "log_age_at_enrollment", "application_order", "parents_higher_ed",
]
BINARY = [
    "s1_no_grade", "s1_no_evaluations", "s1_has_credits",
    "scholarship", "mature_student", "evening", "male", "displaced",
    "special_needs", "non_portuguese", "not_single",
]
CATEGORICAL = ["course", "application_mode", "prev_qualification"]
FEATURES = NUMERIC + BINARY + CATEGORICAL


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df["cohort"] = df["Unemployment rate"].map(UNEMPLOYMENT_TO_COHORT)
    if df["cohort"].isna().any():
        raise ValueError("Unknown macro combination: cohort mapping is incomplete")
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """'Enrolled' = still studying at the end of the normal course duration:
    the final outcome is unknown, so those rows are excluded."""
    df = df[df["Target"] != "Enrolled"].copy()
    df["dropout"] = (df["Target"] == "Dropout").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Only information available at the end of the 1st semester.
    All 2nd-semester columns are dropped on purpose (they don't exist yet
    when the decision is made)."""
    s1 = "Curricular units 1st sem ({})".format
    enrolled = df[s1("enrolled")]
    approved = df[s1("approved")]
    evaluations = df[s1("evaluations")]
    grade = df[s1("grade")]
    safe_enrolled = enrolled.where(enrolled > 0)

    out = pd.DataFrame(index=df.index)
    # 1st-semester academic performance
    out["s1_enrolled"] = enrolled
    out["s1_approval_rate"] = (approved / safe_enrolled).fillna(0.0)
    out["s1_failed_units"] = (enrolled - approved).clip(lower=0)
    out["s1_evals_per_unit"] = (evaluations / safe_enrolled).fillna(0.0)
    # grade == 0 means "nothing approved, no grade", not a 0/20 grade:
    # keep a flag and let the imputer fill the grade with the train median.
    out["s1_no_grade"] = (grade == 0).astype(int)
    out["s1_grade"] = grade.where(grade > 0)
    out["s1_no_evaluations"] = (evaluations == 0).astype(int)
    out["s1_units_without_eval"] = df[s1("without evaluations")]
    out["s1_has_credits"] = (df[s1("credited")] > 0).astype(int)
    # financial situation. "Tuition fees up to date" and "Debtor" are NOT used:
    # the dataset doesn't say when they were recorded, and 94% of students not
    # up to date are dropouts, so they likely reflect the dropout itself
    # (a student who left stops paying) rather than precede it.
    out["scholarship"] = df["Scholarship holder"]
    # background at admission
    out["admission_grade"] = df["Admission grade"]
    out["prev_qual_grade"] = df["Previous qualification (grade)"]
    out["log_age_at_enrollment"] = np.log(df["Age at enrollment"])  # right-skewed
    out["mature_student"] = (df["Age at enrollment"] >= 23).astype(int)
    out["application_order"] = df["Application order"]  # 1 = first choice
    out["evening"] = 1 - df["Daytime/evening attendance"]
    out["male"] = df["Gender"]
    out["displaced"] = df["Displaced"]
    out["special_needs"] = df["Educational special needs"]
    out["non_portuguese"] = (df["Nacionality"] != 1).astype(int)
    out["not_single"] = (df["Marital status"] != 1).astype(int)
    out["parents_higher_ed"] = (
        df["Mother's qualification"].isin(HIGHER_ED_CODES).astype(int)
        + df["Father's qualification"].isin(HIGHER_ED_CODES).astype(int)
    )
    # high-cardinality codes: one-hot, rare levels grouped by the encoder
    out["course"] = df["Course"].astype(str)
    out["application_mode"] = df["Application mode"].astype(str)
    out["prev_qualification"] = df["Previous qualification"].astype(str)
    return out[FEATURES]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(
        min_frequency=30, handle_unknown="infrequent_if_exist", sparse_output=False
    )
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", categorical, CATEGORICAL),
    ])


class LogisticRegressionNet(torch.nn.Module):
    """Logistic regression as a torch module: one linear layer, returns logits."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def business_cost(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                  cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> dict:
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "cost": fn * cost_fn + fp * cost_fp}
