"""
Code shared by train.ipynb and app.py.

Keeping the feature engineering in one module means the app transforms an
applicant exactly the way the training set was transformed.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

# Levels of each categorical column, in natural order. The first level is the
# reference category: one-hot encoding drops it, so every other level's
# coefficient reads as "compared with the first level".
CATEGORIES = {
    "AGE": ["16-25", "26-39", "40-64", "65+"],
    "DRIVING_EXPERIENCE": ["0-9y", "10-19y", "20-29y", "30y+"],
    "EDUCATION": ["none", "high school", "university"],
    "INCOME": ["poverty", "working class", "middle class", "upper class"],
    "GENDER": ["female", "male"],
    "RACE": ["majority", "minority"],
    "VEHICLE_YEAR": ["before 2015", "after 2015"],
    "VEHICLE_TYPE": ["sedan", "sports car"],
    "POSTAL_CODE": [10238, 21217, 32765, 92101],
}
BINARY_COLS = ["VEHICLE_OWNERSHIP", "MARRIED", "CHILDREN"]
COUNT_COLS = ["SPEEDING_VIOLATIONS", "DUIS", "PAST_ACCIDENTS"]
NUMERIC_COLS = ["CREDIT_SCORE", "ANNUAL_MILEAGE"]
ALL_INPUTS = list(CATEGORIES) + BINARY_COLS + COUNT_COLS + NUMERIC_COLS
TARGET = "OUTCOME"

# Raw columns each model version may use
FEATURE_SETS = {
    "full": ALL_INPUTS,
    "legal": [c for c in ALL_INPUTS if c not in ("GENDER", "RACE")],
    "strict": [c for c in ALL_INPUTS
               if c not in ("GENDER", "RACE", "POSTAL_CODE", "CREDIT_SCORE")],
}


class Preprocessor:
    """
    Feature engineering, fitted on the training set only.

    fit() learns the credit-score median of each income group, the mileage
    median of each (married, children) group, the 99th-percentile cap of each
    count column, and the mean and standard deviation of every final column.
    transform() applies exactly those values to any data frame.
    """

    def __init__(self, columns):
        self.columns = list(columns)

    def fit(self, df):
        self.credit_by_income_ = df.groupby("INCOME")["CREDIT_SCORE"].median().to_dict()
        self.mileage_by_family_ = (
            df.groupby(["MARRIED", "CHILDREN"])["ANNUAL_MILEAGE"].median().to_dict()
        )
        self.caps_ = {col: float(df[col].quantile(0.99)) for col in COUNT_COLS}
        engineered = self.engineer(df)
        self.mean_ = engineered.mean()
        self.std_ = engineered.std(ddof=0).replace(0, 1.0)
        self.feature_names_ = engineered.columns.tolist()
        return self

    def engineer(self, df):
        """Fill missing values, cap the counts and one-hot encode. No scaling."""
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            if col in CATEGORIES:
                for level in CATEGORIES[col][1:]:
                    out[f"{col}={level}"] = (df[col] == level).astype(float)
            elif col in COUNT_COLS:
                out[col] = df[col].clip(upper=self.caps_[col]).astype(float)
            elif col == "CREDIT_SCORE":
                out[col] = df[col].fillna(df["INCOME"].map(self.credit_by_income_))
            elif col == "ANNUAL_MILEAGE":
                family = pd.Series(list(zip(df["MARRIED"], df["CHILDREN"])), index=df.index)
                out[col] = df[col].fillna(family.map(self.mileage_by_family_))
            else:  # binary 0/1 columns
                out[col] = df[col].astype(float)
        return out

    def transform(self, df):
        """
        Engineered features standardised with the training mean and std.

        Returned as float64, which scikit-learn needs to fit reproducibly;
        PyTorch callers convert to float32.
        """
        engineered = self.engineer(df)[self.feature_names_]
        return ((engineered - self.mean_) / self.std_).to_numpy(np.float64)


def outcome_values(y, p, threshold, values):
    """
    Value in EUR of each applicant's outcome, declining when p >= threshold.

    values holds one amount per confusion-matrix cell: TP, FP, FN, TN.
    """
    declined = p >= threshold
    claimed = y == 1
    return np.select(
        [declined & claimed, declined & ~claimed, ~declined & claimed],
        [values["TP"], values["FP"], values["FN"]],
        default=values["TN"],
    )


def profit_per_applicant(y, p, threshold, values):
    return outcome_values(y, p, threshold, values).mean()


class LogisticRegressionNet(nn.Module):
    """Logistic regression as a torch module: one linear layer, then a sigmoid."""

    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)
