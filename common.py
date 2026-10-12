"""
Code shared by train.py and app.py.

Why a separate module: app.py must rebuild the exact same features and model
class that train.py used, but it must never *run* training. If app.py imported
train.py, Python would execute the whole training script on import. Putting the
shared pieces here avoids that.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart_2022_sample.csv"
ARTIFACTS = ROOT / "artifacts"

TARGET = "HadHeartAttack"
SEED = 42

# ---------------------------------------------------------------------------
# YOUR DECISIONS GO HERE
# ---------------------------------------------------------------------------
# Columns to exclude from the model. Every entry needs a reason you can defend
# in REPORT.md (e.g. "only known after the event we're trying to detect").
#
# EMPTY ON PURPOSE. HadAngina and ChestScan were both tested as leakage
# candidates in notebook 03 section 8. HadAngina is kept: angina is chest pain
# caused by coronary artery disease, which is the condition that causes heart
# attacks, so it is a genuine clinical risk factor and not a record created by
# the event we are predicting. Dropping it costs 38% of PR-AUC (0.4315 ->
# 0.2677 on the baseline); that measured cost is reported in REPORT.md as a
# sensitivity analysis so the decision is visible rather than hidden.
DROP_COLUMNS: list[str] = []


# --- Fixed lookup tables -----------------------------------------------------
# Written by hand, never computed from the data, so applying them row by row
# cannot leak anything from the validation or test rows.
HEALTH_RANK = {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}

AGE_ORDER = ["Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39", "Age 40 to 44",
             "Age 45 to 49", "Age 50 to 54", "Age 55 to 59", "Age 60 to 64", "Age 65 to 69",
             "Age 70 to 74", "Age 75 to 79", "Age 80 or older"]
AGE_RANK = {age: i for i, age in enumerate(AGE_ORDER)}
AGE_YEARS = {"Age 18 to 24": 21, "Age 25 to 29": 27, "Age 30 to 34": 32, "Age 35 to 39": 37,
             "Age 40 to 44": 42, "Age 45 to 49": 47, "Age 50 to 54": 52, "Age 55 to 59": 57,
             "Age 60 to 64": 62, "Age 65 to 69": 67, "Age 70 to 74": 72, "Age 75 to 79": 77,
             "Age 80 or older": 85}

RECENT_CHECKUP = "Within past year (anytime less than 12 months ago)"

SMOKING_RANK = {"Never smoked": 0, "Former smoker": 1,
                "Current smoker - now smokes some days": 2,
                "Current smoker - now smokes every day": 3}
TEETH_RANK = {"None of them": 0, "1 to 5": 1, "6 or more, but not all": 2, "All": 3}

DISABILITY_COLUMNS = ["DifficultyWalking", "DifficultyErrands", "DifficultyDressingBathing",
                      "DifficultyConcentrating", "DeafOrHardOfHearing", "BlindOrVisionDifficulty"]

# US Census regions. Replaces 53 near-empty State columns with 5 groups.
REGION = {}
for _s in ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont",
           "New Jersey", "New York", "Pennsylvania"]:
    REGION[_s] = "Northeast"
for _s in ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota",
           "Missouri", "Nebraska", "North Dakota", "South Dakota"]:
    REGION[_s] = "Midwest"
for _s in ["Delaware", "District of Columbia", "Florida", "Georgia", "Maryland", "North Carolina",
           "South Carolina", "Virginia", "West Virginia", "Alabama", "Kentucky", "Mississippi",
           "Tennessee", "Arkansas", "Louisiana", "Oklahoma", "Texas"]:
    REGION[_s] = "South"
for _s in ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming",
           "Alaska", "California", "Hawaii", "Oregon", "Washington"]:
    REGION[_s] = "West"
for _s in ["Guam", "Puerto Rico", "Virgin Islands"]:
    REGION[_s] = "Territory"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw survey columns into model features. THIS IS YOUR 30%.

    Ten ideas, chosen in notebook 03. Each was measured against the plain
    baseline on validation PR-AUC; the chosen set beat the baseline in 20 of 20
    reshuffled validation splits (+0.0240 on average). Columns drop from 121 to
    66, so the gain comes with a simpler model, not a bigger one.

    Every rule below is row-by-row and uses only fixed lookup tables, never a
    statistic computed from the dataset (those belong in the sklearn
    preprocessor, which is fit on the training split only).

    Order matters: new columns are all computed first, blanks are labelled
    second, and only then are the replaced raw columns dropped. That way an
    idea can still read a raw column that a previous idea has replaced (for
    example male_x_age reads AgeCategory after age_rank replaced it).
    """
    df = df.copy()
    replaced: list[str] = []

    # 1. health_rank: 5 ordered categories -> one number. Each step up the
    #    scale roughly halves the heart attack rate (notebook 01 section 5).
    df["health_rank"] = df["GeneralHealth"].map(HEALTH_RANK)
    replaced.append("GeneralHealth")

    # 2. age_rank: 13 one-hot columns -> one number. The youngest bands have
    #    only 3-11 heart attacks each, so separate coefficients are mostly noise.
    df["age_rank"] = df["AgeCategory"].map(AGE_RANK)
    replaced.append("AgeCategory")

    # 3. male_x_age: an interaction. After 45 the men's risk climbs faster than
    #    the women's; without this column logistic regression is forced to use
    #    one age slope for both sexes. Centred on a fixed 50, not the data mean.
    #
    #    The explicit zero for women matters. Without it, a woman with a blank
    #    AgeCategory gets NaN * 0 = NaN rather than 0, and the median imputer
    #    then invents a value for a feature that is 0 for every woman by
    #    construction. On this sample it happens to be harmless (women are 53%
    #    of rows and all sit at 0, so the median IS 0), but that is a property
    #    of the sex balance, not of the feature. A man with a blank age keeps
    #    NaN and is imputed, which is the intended behaviour: his age is unknown.
    is_male = (df["Sex"] == "Male").astype(int)
    df["male_x_age"] = (df["AgeCategory"].map(AGE_YEARS) - 50) * is_male
    df.loc[is_male == 0, "male_x_age"] = 0.0

    # 4. checkup_recent: the EDA showed a jump at "within the past year", not a
    #    steady trend, so one flag fits better than an evenly spaced rank.
    df["checkup_recent"] = (df["LastCheckupTime"] == RECENT_CHECKUP).astype(float).where(
        df["LastCheckupTime"].notna())
    replaced.append("LastCheckupTime")

    # 5. smoking_rank: never < former < some days < every day.
    df["smoking_rank"] = df["SmokerStatus"].map(SMOKING_RANK)
    replaced.append("SmokerStatus")

    # 6. teeth_rank: a clean, steady climb (2.6% -> 6.2% -> 12.7% -> 18.0%)
    #    that survives the within-age check.
    df["teeth_rank"] = df["RemovedTeeth"].map(TEETH_RANK)
    replaced.append("RemovedTeeth")

    # 7. difficulties_count: the six disability questions point the same way, so
    #    six coefficients become one count. Replacing beats adding: the "add"
    #    variant was negative in 7 of 10 shuffles, because in a linear model a
    #    sum of yes/no columns is something the model can already build itself.
    df["n_difficulties"] = (df[DISABILITY_COLUMNS] == "Yes").sum(axis=1)
    replaced.extend(DISABILITY_COLUMNS)

    # 8. region: 53 State columns (about 15 heart attacks each) -> 5 regions.
    df["region"] = df["State"].map(REGION)
    replaced.append("State")

    # 9. drop_height_weight: BMI = weight / height^2, so the three columns carry
    #    two pieces of information. BMI and weight correlate at 0.86.
    replaced.extend(["HeightInMeters", "WeightInKilograms"])

    # 10. blank_as_category: a blank becomes its own level instead of being
    #     filled with the most common answer, which would erase it. Must run
    #     after every new column exists and before the drop below.
    #     The label matters: OneHotEncoder(drop="first") drops the
    #     alphabetically first category, so "(no answer)" would sort first and
    #     silently change the reference level of every Yes/No column.
    #     "Not answered" sorts after "No", leaving the baseline's reference intact.
    for col in df.columns:
        if col != TARGET and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna("Not answered")

    return df.drop(columns=list(dict.fromkeys(replaced)))


# ---------------------------------------------------------------------------
# Plumbing: you shouldn't need to change much below this line
# ---------------------------------------------------------------------------
def make_xy(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Raw rows -> (feature dataframe, 0/1 target). Used identically by train and app."""
    df = engineer_features(raw)
    y = (df[TARGET] == "Yes").astype(int).to_numpy()
    X = df.drop(columns=[TARGET, *DROP_COLUMNS])
    return X, y


def column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric columns get scaled; everything else is treated as categorical."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Impute -> encode -> standardize everything.

    Standardizing *all* columns (one-hot ones included) gives gradient descent
    a well-conditioned problem, which is what lets the two PyTorch versions
    converge to the same answer as sklearn. Coefficients are then "change in
    log-odds per 1 standard deviation" for every feature.

    The imputation strategies are generic defaults. If you use the
    with_nans file, deciding how to treat missing answers is a real
    feature-engineering choice worth revisiting.
    """
    encode = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        # drop="first" avoids the dummy-variable trap (k one-hot
                        # columns + an intercept are perfectly collinear).
                        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",
                                                 sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("encode", encode), ("scale", StandardScaler())])


class LogisticRegressionNet(nn.Module):
    """Logistic regression as a PyTorch module: one linear layer.

    forward() returns logits (raw scores), not probabilities. The sigmoid is
    folded into BCEWithLogitsLoss during training and applied explicitly at
    prediction time.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))
